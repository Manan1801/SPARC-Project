#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6–9: Cluster on PCA-run outputs → pick best model → label test → summaries & plots.

NEW FUNCTIONALITY
  • --algos lets you choose which algorithms to run (e.g., kmeans,dbscan).
  • --subsample-for-agglo N will fit/evaluate Agglomerative on a stratified subsample of N train rows,
    then assign labels to full train/test by nearest centroid.
  • --subsample-for-dbscan N will fit/evaluate DBSCAN on a stratified subsample of N train rows,
    then assign labels to full train/test by nearest-core rule.
  • --dbscan-percentiles allows tuning of eps grid (defaults 50 60 70 80).
  • K-Means controls: --kmeans-max-iter, --kmeans-n-init, --kmeans-verbose.
    - We now **suppress scikit-learn’s per-iteration logs** and instead run `n_init` manual restarts
      with `n_init=1` each, printing only: `[split_xx] KMeans k=<k> init <i>/<n_init>`.
  • Progress bars across candidate settings via tqdm for KMeans (k sweep), Agglo (k sweep), and DBSCAN (eps grid).
  • All metrics (silhouette/DBI/CH) are computed on the *training set used to fit* each model:
      - full train for KMeans (no subsample),
      - subsample for Agglo/DBSCAN when subsampling is enabled.
    Labels are still produced for the full train/test for reporting and downstream use.

INPUT (expects the folder produced by your PCA-only script):
  <pca_root>/
    split_00/
      train_scaled.csv           # cols: participant_id, frame, <z-move...>, frame_norm_x_alpha
      test_scaled.csv
    split_01/
      ...

USAGE EXAMPLES
  # KMeans only, scaled space, with minimal per-init prints
  python -u scripts/cluster.py \
      --pca-root ~/Desktop/Cluster/pca_runs \
      --outdir ~/Desktop/Cluster/cluster_runs/km_02 \
      --space scaled \
      --splits 0 \
      --algos kmeans \
      --kmin 2 --kmax 8 \
      --kmeans-max-iter 100 \
      --kmeans-n-init 20 \
      --kmeans-verbose 1

  # KMeans + DBSCAN on PCA(8), subsample DBSCAN to 30k
  python cluster.py \
      --pca-root pca_runs \
      --outdir cluster_runs_pca8 \
      --space pca --pca-k 8 \
      --splits all \
      --algos kmeans,dbscan \
      --dbscan-min 10 20 30 \
      --subsample-for-dbscan 30000

  # Agglomerative on subsample of 20k (scaled space)
  python cluster.py \
      --pca-root pca_runs \
      --outdir cluster_runs_agglo_sub \
      --space scaled \
      --splits 0 \
      --algos agglo \
      --kmin 2 --kmax 8 \
      --subsample-for-agglo 20000
"""

from __future__ import annotations
import argparse, json, math, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors

# ----------------------------- I/O helpers -----------------------------

def list_splits(pca_root: Path, splits_arg: str):
    if splits_arg == "all":
        split_dirs = sorted([p for p in pca_root.glob("split_*") if p.is_dir()])
        if not split_dirs:
            raise SystemExit(f"No split_* directories found under {pca_root}")
        return split_dirs
    idxs = set()
    for token in re.split(r"[,\s]+", splits_arg.strip()):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            idxs.update(range(int(a), int(b) + 1))
        else:
            idxs.add(int(token))
    split_dirs = []
    for i in sorted(idxs):
        d = pca_root / f"split_{i:02d}"
        if not d.is_dir():
            raise SystemExit(f"Missing split dir: {d}")
        split_dirs.append(d)
    return split_dirs

def load_scaled_csv(path: Path):
    """Return meta (pid, frame), feature matrix (numpy), feature names, and temporal index if present."""
    df = pd.read_csv(path)
    if "participant_id" not in df.columns or "frame" not in df.columns:
        raise SystemExit(f"{path} missing meta columns.")
    feat_cols = [c for c in df.columns if c not in ("participant_id", "frame")]
    X = df[feat_cols].values.astype(float)
    meta = df[["participant_id", "frame"]].copy()
    fidx = feat_cols.index("frame_norm_x_alpha") if "frame_norm_x_alpha" in feat_cols else None
    return meta, X, feat_cols, fidx

# ----------------------------- feature space -----------------------------

def fit_pca_for_space(X_train: np.ndarray, pca_k: int|None, pca_target: float|None, seed: int):
    pca = PCA(n_components=None, random_state=seed)
    pca.fit(X_train)
    if pca_k is not None and pca_k > 0:
        k = min(pca_k, X_train.shape[1])
    elif pca_target is not None and 0 < pca_target < 1:
        cum = np.cumsum(pca.explained_variance_ratio_)
        idx = np.where(cum >= pca_target)[0]
        k = int(idx[0] + 1) if len(idx) else X_train.shape[1]
    else:
        k = X_train.shape[1]
    pca_k_model = PCA(n_components=k, random_state=seed).fit(X_train)
    return pca_k_model, k, pca.explained_variance_ratio_.tolist()

# ----------------------------- subsampling -----------------------------

def stratified_subsample_indices(meta: pd.DataFrame, n_target: int, seed: int) -> np.ndarray:
    """
    Stratified by participant_id; approximate proportional allocation; ensure at least 1 per participant if possible.
    Returns array of indices into meta (0..len(meta)-1).
    """
    n = len(meta)
    if n_target >= n:
        return np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    counts = meta["participant_id"].value_counts().sort_index()
    parts = counts.index.tolist()
    sizes = counts.values.astype(int)
    props = sizes / sizes.sum()
    alloc = np.floor(props * n_target).astype(int)
    need = (alloc == 0) & (sizes > 0)
    alloc[need] = 1
    diff = n_target - alloc.sum()
    if diff > 0:
        fracs = (props * n_target) - np.floor(props * n_target)
        order = np.argsort(fracs)[::-1]
        for idx in order[:diff]:
            alloc[idx] += 1
    elif diff < 0:
        fracs = (props * n_target) - np.floor(props * n_target)
        order = np.argsort(fracs)
        for idx in order[:abs(diff)]:
            if alloc[idx] > 1:
                alloc[idx] -= 1
            else:
                for j in reversed(order):
                    if alloc[j] > 1:
                        alloc[j] -= 1
                        break
    per_part_idx = {pid: meta.index[meta["participant_id"] == pid].to_numpy() for pid in parts}
    chosen = []
    for pid, k in zip(parts, alloc):
        idxs = per_part_idx[pid]
        if k >= len(idxs):
            chosen.append(idxs)
        else:
            chosen.append(np.sort(rng.choice(idxs, size=k, replace=False)))
    return np.concatenate(chosen, axis=0)

# ----------------------------- clustering utils -----------------------------

def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X[mask], labels[mask], metric="euclidean"))
    except Exception:
        return float("nan")

def safe_davies_bouldin(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(X[mask], labels[mask]))
    except Exception:
        return float("nan")

def safe_calinski_harabasz(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(X[mask], labels[mask]))
    except Exception:
        return float("nan")

def dbscan_eps_grid(X, min_samples_list, percentiles=(50, 60, 70, 80)):
    candidates = []
    for ms in min_samples_list:
        k = max(ms, 2)
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nn.kneighbors(X)
        kth = dists[:, -1]
        for q in percentiles:
            eps = float(np.percentile(kth, q))
            if eps > 0:
                candidates.append((ms, eps))
    uniq = []
    seen = set()
    for ms, eps in candidates:
        key = (ms, round(eps, 6))
        if key not in seen:
            seen.add(key)
            uniq.append((ms, eps))
    return uniq

def assign_by_nearest_centroid(X_ref: np.ndarray, labels_ref: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    labs = np.asarray(labels_ref).astype(int)
    uniq = sorted(set(labs))
    centroids = []
    for lab in uniq:
        centroids.append(X_ref[labs == lab].mean(axis=0))
    centroids = np.vstack(centroids)
    d = ((X_new[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(int)

def assign_by_nearest_core(DB_model: DBSCAN, X_fit: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    core_idx = getattr(DB_model, "core_sample_indices_", None)
    if core_idx is None or len(core_idx) == 0:
        return np.full(len(X_new), -1, dtype=int)
    core_pts = X_fit[core_idx]
    core_labs = DB_model.labels_[core_idx]
    nn = NearestNeighbors(n_neighbors=1).fit(core_pts)
    dist, idx = nn.kneighbors(X_new)
    return np.where(dist.ravel() <= DB_model.eps, core_labs[idx.ravel()], -1).astype(int)

# ----------------------------- reporting & plots -----------------------------

def save_cluster_profiles(out_dir: Path, meta_train, X_train, feat_cols, labels, fidx_temporal=None):
    df = pd.DataFrame(X_train, columns=feat_cols)
    df.insert(0, "participant_id", meta_train["participant_id"].values)
    df.insert(1, "frame", meta_train["frame"].values)
    df["cluster_id"] = labels

    prof_mean = df.groupby("cluster_id").mean(numeric_only=True)
    prof_median = df.groupby("cluster_id").median(numeric_only=True)
    prof_mean.to_csv(out_dir / "cluster_profiles_mean.csv")
    prof_median.to_csv(out_dir / "cluster_profiles_median.csv")

    mix = df.groupby(["cluster_id","participant_id"]).size().reset_index(name="count")
    mix_piv = mix.pivot(index="cluster_id", columns="participant_id", values="count").fillna(0).astype(int)
    mix_piv.to_csv(out_dir / "cluster_participant_mix.csv")

    temporal_cols = []
    if fidx_temporal is not None and fidx_temporal >= 0:
        temporal_cols.append("frame_norm_x_alpha")
    temporal_cols.append("frame")
    df_temporal = df[["cluster_id"] + temporal_cols]
    stats = df_temporal.groupby("cluster_id").agg(["min","median","max"])
    stats.to_csv(out_dir / "cluster_temporal_summary.csv")

def plot_cluster_sizes(labels, out_path: Path):
    uniq, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar([str(int(u)) for u in uniq], counts)
    plt.xlabel("Cluster ID")
    plt.ylabel("Train count")
    plt.title("Cluster sizes (train)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def plot_silhouette_hist(X_train, labels, out_path: Path):
    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return
    s = silhouette_samples(X_train[mask], labels[mask])
    plt.figure(figsize=(6,4))
    plt.hist(s, bins=40)
    plt.xlabel("Silhouette (train, non-noise)")
    plt.ylabel("Count")
    plt.title("Silhouette sample distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def plot_temporal_box(meta_train, X_train, feat_cols, labels, out_path: Path):
    if "frame_norm_x_alpha" not in feat_cols:
        return
    idx = feat_cols.index("frame_norm_x_alpha")
    vals = X_train[:, idx]
    df = pd.DataFrame({"cluster_id": labels, "frame_norm_x_alpha": vals})
    plt.figure(figsize=(7,4.5))
    df.boxplot(by="cluster_id", column="frame_norm_x_alpha")
    plt.title("Temporal distribution per cluster (frame_norm×α)")
    plt.suptitle("")
    plt.xlabel("Cluster ID"); plt.ylabel("frame_norm×α")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

# ----------------------------- core per-split -----------------------------

def run_split(split_dir: Path, out_root: Path, space: str, pca_k: int|None, pca_target: float|None,
              kmin: int, kmax: int, dbscan_min_list, dbscan_percentiles, algos: set[str],
              sub_agglo: int, sub_dbscan: int, sub_seed: int, seed: int,
              kmeans_max_iter: int = 100, kmeans_n_init: int = 20, kmeans_verbose: int = 0):

    train_path = split_dir / "train_scaled.csv"
    test_path  = split_dir / "test_scaled.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing train/test scaled CSVs in {split_dir}")

    meta_tr, Xtr_scaled, feat_cols_scaled, fidx_temporal = load_scaled_csv(train_path)
    meta_te, Xte_scaled, _, _ = load_scaled_csv(test_path)

    # Select feature space
    used_space = space
    if space == "scaled":
        Xtr = Xtr_scaled; Xte = Xte_scaled
        used_dims = Xtr.shape[1]
        pca_info = None
        feat_cols_space = [c for c in feat_cols_scaled]
    elif space == "pca":
        pca_model, used_dims, full_evr = fit_pca_for_space(Xtr_scaled, pca_k=pca_k, pca_target=pca_target, seed=seed)
        Xtr = pca_model.transform(Xtr_scaled)
        Xte = pca_model.transform(Xte_scaled)
        used_space = f"pca_{used_dims}"
        pca_info = {"used_dims": used_dims, "full_explained_variance_ratio": full_evr}
        feat_cols_space = [f"pc{i+1}" for i in range(used_dims)]
    else:
        raise SystemExit("--space must be 'scaled' or 'pca'")

    n_train_total = Xtr.shape[0]
    candidates = []

    # ---------- KMeans (manual n_init loop; no per-iteration spam) ----------
    if "kmeans" in algos:
        print(f"[{split_dir.name}] KMeans sweep k={kmin}..{kmax} (n_init={kmeans_n_init}, max_iter={kmeans_max_iter})", flush=True)
        for k in tqdm(range(kmin, kmax+1), desc=f"{split_dir.name} | KMeans k", leave=False):
            best_km = None
            best_labels = None
            best_inertia = float("inf")
            best_iters = -1
            # Manual multi-starts with quiet sklearn; optional minimal prints
            for i in range(kmeans_n_init):
                if kmeans_verbose:
                    print(f"[{split_dir.name}] KMeans k={k} init {i+1}/{kmeans_n_init}", flush=True)
                km_i = KMeans(
                    n_clusters=k,
                    n_init=1,                  # one init per loop
                    max_iter=kmeans_max_iter,
                    random_state=seed + i,     # different seed per init
                    verbose=0                  # suppress internal iteration logs
                )
                labs_i = km_i.fit_predict(Xtr)
                inertia_i = float(getattr(km_i, "inertia_", float("inf")))
                if inertia_i < best_inertia:
                    best_inertia = inertia_i
                    best_km = km_i
                    best_labels = labs_i
                    best_iters = int(getattr(km_i, "n_iter_", -1))
            # Store candidate from best init
            cand = {
                "algo": "kmeans",
                "model": best_km,
                "labels_train_eval": best_labels,    # for metrics
                "labels_train_full": best_labels,    # for saving
                "params": {"k": k, "n_init": kmeans_n_init, "max_iter": kmeans_max_iter, "best_inertia": best_inertia},
                "train_effective_n": int(n_train_total),
                "fit_iterations": best_iters
            }
            cand["silhouette"] = safe_silhouette(Xtr, cand["labels_train_eval"])
            cand["dbi"] = safe_davies_bouldin(Xtr, cand["labels_train_eval"])
            cand["ch"] = safe_calinski_harabasz(Xtr, cand["labels_train_eval"])
            candidates.append(cand)

    # ---------- Agglomerative (Ward) ----------
    if "agglo" in algos:
        if sub_agglo and n_train_total > sub_agglo:
            sub_idx = stratified_subsample_indices(meta_tr, sub_agglo, seed=sub_seed)
            X_fit = Xtr[sub_idx]
            subnote = f"subsample({len(sub_idx)})"
        else:
            X_fit = Xtr
            subnote = "full"
        print(f"[{split_dir.name}] Agglo sweep k={kmin}..{kmax} on {subnote}", flush=True)
        for k in tqdm(range(kmin, kmax+1), desc=f"{split_dir.name} | Agglo k", leave=False):
            ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labs_fit = ag.fit_predict(X_fit)
            sil = safe_silhouette(X_fit, labs_fit)
            dbi = safe_davies_bouldin(X_fit, labs_fit)
            ch  = safe_calinski_harabasz(X_fit, labs_fit)
            labs_full = assign_by_nearest_centroid(X_fit, labs_fit, Xtr)
            candidates.append({
                "algo": "agglo",
                "model": ag,
                "labels_train_eval": labs_fit,
                "labels_train_full": labs_full,
                "params": {"k": k, "fit_on": subnote},
                "silhouette": sil, "dbi": dbi, "ch": ch,
                "train_effective_n": int(len(X_fit))
            })

    # ---------- DBSCAN ----------
    if "dbscan" in algos:
        if sub_dbscan and n_train_total > sub_dbscan:
            sub_idx = stratified_subsample_indices(meta_tr, sub_dbscan, seed=sub_seed)
            X_fit = Xtr[sub_idx]
            subnote = f"subsample({len(sub_idx)})"
        else:
            X_fit = Xtr
            subnote = "full"
        grid = dbscan_eps_grid(X_fit, dbscan_min_list, percentiles=tuple(dbscan_percentiles))
        print(f"[{split_dir.name}] DBSCAN grid on {subnote}: {len(grid)} combos", flush=True)
        for ms, eps in tqdm(grid, desc=f"{split_dir.name} | DBSCAN grid", leave=False):
            db = DBSCAN(eps=eps, min_samples=ms)
            labs_fit = db.fit_predict(X_fit)
            sil = safe_silhouette(X_fit, labs_fit)
            dbi = safe_davies_bouldin(X_fit, labs_fit)
            ch  = safe_calinski_harabasz(X_fit, labs_fit)
            labs_full = assign_by_nearest_core(db, X_fit, Xtr)
            candidates.append({
                "algo": "dbscan",
                "model": db,
                "labels_train_eval": labs_fit,
                "labels_train_full": labs_full,
                "params": {"min_samples": ms, "eps": float(eps), "fit_on": subnote},
                "silhouette": sil, "dbi": dbi, "ch": ch,
                "train_effective_n": int(len(X_fit))
            })

    if not candidates:
        raise SystemExit("No algorithms selected. Use --algos kmeans,agglo,dbscan (any subset).")

    # Select best
    def score_key(c):
        sil = c["silhouette"]; dbi = c["dbi"]; ch = c["ch"]
        sil_k = -1e9 if (sil is None or math.isnan(sil)) else sil
        dbi_k =  1e9 if (dbi is None or math.isnan(dbi)) else dbi
        ch_k  = -1e9 if (ch  is None or math.isnan(ch))  else ch
        return (sil_k, -dbi_k, ch_k)

    best = sorted(candidates, key=score_key, reverse=True)[0]

    # Assign test labels
    if best["algo"] == "kmeans":
        yte = best["model"].predict(Xte).astype(int)
    elif best["algo"] == "agglo":
        yte = assign_by_nearest_centroid(Xtr, best["labels_train_full"], Xte)
    else:  # dbscan
        if best["params"]["fit_on"].startswith("subsample"):
            proxy_core_pts = Xtr[best["labels_train_full"] != -1]
            proxy_core_labs = best["labels_train_full"][best["labels_train_full"] != -1]
            if len(proxy_core_pts) == 0:
                yte = np.full(len(Xte), -1, dtype=int)
            else:
                nn = NearestNeighbors(n_neighbors=1).fit(proxy_core_pts)
                dist, idx = nn.kneighbors(Xte)
                eps = best["params"]["eps"]
                yte = np.where(dist.ravel() <= eps, proxy_core_labs[idx.ravel()], -1).astype(int)
        else:
            yte = assign_by_nearest_core(best["model"], Xtr, Xte)

    # Output directory for this split
    out_split = out_root / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)

    # Save labels
    pd.DataFrame({
        "participant_id": meta_tr["participant_id"],
        "frame": meta_tr["frame"],
        "cluster_id": best["labels_train_full"].astype(int)
    }).to_csv(out_split / "train_labels.csv", index=False)
    pd.DataFrame({
        "participant_id": meta_te["participant_id"],
        "frame": meta_te["frame"],
        "cluster_id": yte.astype(int)
    }).to_csv(out_split / "test_labels.csv", index=False)

    # Save metrics & model info
    metrics = {
        "space": used_space,
        "dims": int(used_dims),
        "algo": best["algo"],
        "params": best["params"],
        "silhouette": best["silhouette"],
        "davies_bouldin": best["dbi"],
        "calinski_harabasz": best["ch"],
        "n_clusters_train": int(len(set(best["labels_train_full"])) - (1 if -1 in best["labels_train_full"] else 0)),
        "n_noise_train": int(np.sum(best["labels_train_full"] == -1)) if best["algo"] == "dbscan" else 0,
        "train_total_n": int(n_train_total),
        "train_effective_n": int(best["train_effective_n"]) if "train_effective_n" in best else int(n_train_total),
        "fit_iterations": int(best.get("fit_iterations", -1))
    }
    if pca_info is not None:
        metrics["pca_info"] = pca_info
    with open(out_split / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Profiles & summaries (use the feature names of current space)
    feat_cols = feat_cols_space
    Xtr_for_report = Xtr
    save_cluster_profiles(out_split, meta_tr, Xtr_for_report, feat_cols, best["labels_train_full"], fidx_temporal=fidx_temporal)

    # Plots
    plot_cluster_sizes(best["labels_train_full"], out_split / "cluster_sizes.png")
    plot_silhouette_hist(Xtr_for_report, best["labels_train_full"], out_split / "silhouette_hist.png")
    plot_temporal_box(meta_tr, Xtr_for_report, feat_cols, best["labels_train_full"], out_split / "temporal_box.png")

    return metrics

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Clustering pipeline consuming PCA-run splits.")
    ap.add_argument("--pca-root", type=Path, required=True, help="Folder that contains split_XX from PCA script")
    ap.add_argument("--outdir", type=Path, required=True, help="Output folder for clustering results")
    ap.add_argument("--splits", type=str, default="all", help="'all' or a list/range like '0,2,5-7'")
    ap.add_argument("--space", choices=["scaled","pca"], default="scaled", help="Feature space to cluster in")
    ap.add_argument("--pca-k", type=int, default=0, help="If --space pca: fixed n_components (override target)")
    ap.add_argument("--pca-target", type=float, default=0.0, help="If --space pca and --pca-k=0: cumulative variance target (e.g., 0.90)")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--dbscan-min", type=int, nargs="+", default=[10,20,30])
    ap.add_argument("--dbscan-percentiles", type=int, nargs="+", default=[50,60,70,80],
                    help="Percentiles for k-distance eps grid (e.g., 60 70 80)")
    ap.add_argument("--algos", type=str, default="kmeans,agglo,dbscan",
                    help="Comma-separated subset of algorithms to run: kmeans,agglo,dbscan")
    ap.add_argument("--subsample-for-agglo", type=int, default=0,
                    help="If >0, fit/evaluate Agglo on a stratified subsample of this many train rows")
    ap.add_argument("--subsample-for-dbscan", type=int, default=0,
                    help="If >0, fit/evaluate DBSCAN on a stratified subsample of this many train rows")
    ap.add_argument("--subsample-seed", type=int, default=2025, help="RNG seed for subsampling")
    ap.add_argument("--seed", type=int, default=123, help="General RNG seed")

    # KMeans controls + minimalist logging
    ap.add_argument("--kmeans-max-iter", type=int, default=100, help="KMeans max_iter")
    ap.add_argument("--kmeans-n-init", type=int, default=20, help="KMeans manual restarts")
    ap.add_argument("--kmeans-verbose", type=int, default=0, help="Print '[split] KMeans k=<k> init i/n' if >0")

    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    algos = set(a.strip().lower() for a in args.algos.split(",") if a.strip())
    allowed = {"kmeans","agglo","dbscan"}
    unknown = algos - allowed
    if unknown:
        raise SystemExit(f"Unknown algos in --algos: {sorted(unknown)}")
    if not algos:
        raise SystemExit("No algorithms selected. Use --algos kmeans,agglo,dbscan (any subset).")

    split_dirs = list_splits(args.pca_root, args.splits)

    all_metrics = []
    for d in split_dirs:
        print(f"[split] {d.name}")
        m = run_split(
            split_dir=d,
            out_root=args.outdir,
            space=args.space,
            pca_k=(args.pca_k if args.pca_k>0 else None),
            pca_target=(args.pca_target if args.pca_k==0 and args.pca_target>0 else None),
            kmin=args.kmin,
            kmax=args.kmax,
            dbscan_min_list=args.dbscan_min,
            dbscan_percentiles=args.dbscan_percentiles,
            algos=algos,
            sub_agglo=args.subsample_for_agglo,
            sub_dbscan=args.subsample_for_dbscan,
            sub_seed=args.subsample_seed,
            seed=args.seed,
            kmeans_max_iter=args.kmeans_max_iter,
            kmeans_n_init=args.kmeans_n_init,
            kmeans_verbose=args.kmeans_verbose
        )
        all_metrics.append(m)

    def safe_mean(vals):
        v = np.array([float(x) if x is not None else np.nan for x in vals], dtype=float)
        return float(np.nanmean(v)) if v.size else float("nan")

    summary = {
        "n_splits": len(all_metrics),
        "space": args.space,
        "pca_k": args.pca_k,
        "pca_target": args.pca_target,
        "k_range": [args.kmin, args.kmax],
        "dbscan_min": args.dbscan_min,
        "dbscan_percentiles": args.dbscan_percentiles,
        "algos": sorted(list(algos)),
        "subsample_for_agglo": args.subsample_for_agglo,
        "subsample_for_dbscan": args.subsample_for_dbscan,
        "silhouette_mean": safe_mean([m["silhouette"] for m in all_metrics]),
        "davies_bouldin_mean": safe_mean([m["davies_bouldin"] for m in all_metrics]),
        "calinski_harabasz_mean": safe_mean([m["calinski_harabasz"] for m in all_metrics]),
        "algo_counts": {a: sum(1 for m in all_metrics if m["algo"] == a) for a in ["kmeans","agglo","dbscan"]}
    }
    with open(args.outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Wrote", args.outdir / "summary.json")
    print("Algo counts:", summary["algo_counts"])
    print("Mean silhouette:", summary["silhouette_mean"])

if __name__ == "__main__":
    main()
