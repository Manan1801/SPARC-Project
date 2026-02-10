#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3–5 ONLY: Train/Test split → Scaling → PCA (with metrics & visualizations)

INPUT
  pooled_features_raw.csv  (columns: participant_id, frame, frame_norm, and *_move features)

WHAT THIS SCRIPT DOES
  • Repeated random splits (stratified by participant)
  • Scale movement features on TRAIN only (StandardScaler); append frame_norm × α
  • Fit PCA on TRAIN; transform TRAIN/TEST
  • Save:
      - train_scaled.csv, test_scaled.csv (per split)
      - pca_explained_variance.csv (per split)
      - pca_loadings.csv (per split)
      - train_pca2.csv, test_pca2.csv  (PC1/PC2 for plots)
      - plots: scree.png, cumulative_variance.png, pc_scatter_frame_norm.png, pc_scatter_participant.png, biplot.png
      - metrics.json (explained variance summaries)

USAGE (example)
  python pca.py \
      --input build/pooled_features_raw.csv \
      --outdir pca_runs \
      --splits 5 \
      --test-size 0.2 \
      --alpha 0.25 \
      --max-points 25000 \
      --seed 42
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------- helpers -----------------------------

def load_dataset(path: Path):
    df = pd.read_csv(path)
    required = {"participant_id", "frame", "frame_norm"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SystemExit(f"Input missing columns: {missing}")
    move_cols = [c for c in df.columns if c.endswith("_move")]
    if len(move_cols) == 0:
        raise SystemExit("No movement columns (*_move) found in input.")
    # enforce dtypes
    df["participant_id"] = df["participant_id"].astype(str)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["frame"]).copy()
    df["frame"] = df["frame"].astype(int)
    df["frame_norm"] = pd.to_numeric(df["frame_norm"], errors="coerce").astype(float).clip(0, 1)
    for c in move_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=move_cols + ["frame_norm"]).reset_index(drop=True)
    return df, move_cols

def build_features(df, move_cols, alpha: float):
    X_moves = df[move_cols].values
    frame_alpha = (df["frame_norm"].values.reshape(-1, 1) * float(alpha))
    # Keep meta separate for saving/plotting
    meta = df[["participant_id", "frame", "frame_norm"]].copy()
    return X_moves, frame_alpha, meta

def subsample_for_plot(meta_df: pd.DataFrame, arr2d: np.ndarray, max_points: int, seed: int):
    n = len(meta_df)
    if n <= max_points:
        return meta_df, arr2d
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return meta_df.iloc[idx].reset_index(drop=True), arr2d[idx, :]

def save_scaled(split_dir: Path, meta, X_scaled, move_cols):
    # Name columns: z-scored movement features + the temporal scalar last
    feat_cols = [f"{c}_z" for c in move_cols] + ["frame_norm_x_alpha"]
    out = pd.DataFrame(X_scaled, columns=feat_cols)
    out.insert(0, "participant_id", meta["participant_id"].values)
    out.insert(1, "frame", meta["frame"].values)
    out.to_csv(split_dir, index=False)

def make_scree_plot(explained_ratio, out_path: Path):
    k = len(explained_ratio)
    x = np.arange(1, k+1)
    plt.figure(figsize=(7,4.5))
    plt.plot(x, explained_ratio, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def make_cumvar_plot(explained_ratio, out_path: Path):
    cum = np.cumsum(explained_ratio)
    x = np.arange(1, len(cum)+1)
    plt.figure(figsize=(7,4.5))
    plt.plot(x, cum, marker='o')
    plt.axhline(0.90, linestyle='--', linewidth=1)
    plt.axhline(0.95, linestyle='--', linewidth=1)
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Variance (90%/95% reference)")
    plt.ylim(0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def make_pc_scatter_color_by_frame(meta_df, pca2, out_path: Path):
    # color by raw frame_norm (0→1)
    c = meta_df["frame_norm"].values
    plt.figure(figsize=(6.5,5.5))
    sc = plt.scatter(pca2[:,0], pca2[:,1], c=c, s=6, alpha=0.7)
    cb = plt.colorbar(sc)
    cb.set_label("frame_norm (0→1)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: PC1 vs PC2 (colored by frame_norm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def make_pc_scatter_color_by_participant(meta_df, pca2, out_path: Path, max_legend=15):
    # color top-N frequent participants; others gray
    value_counts = meta_df["participant_id"].value_counts()
    top = list(value_counts.head(max_legend).index)
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    if colors is None or len(colors) < len(top):
        colors = None  # let matplotlib cycle default
    plt.figure(figsize=(7.5,6))
    for i, pid in enumerate(top):
        mask = (meta_df["participant_id"].values == pid)
        plt.scatter(pca2[mask,0], pca2[mask,1], s=8, alpha=0.7, label=pid)
    # others
    other_mask = ~meta_df["participant_id"].isin(top).values
    if other_mask.any():
        plt.scatter(pca2[other_mask,0], pca2[other_mask,1], s=6, alpha=0.3, label="others", c="gray")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA: PC1 vs PC2 (colored by participant, top {len(top)})")
    plt.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def make_biplot(pca, pca2, feature_names, out_path: Path, topn=8):
    # Choose top features by 2D loading magnitude
    loadings = pca.components_[:2, :]  # (2, n_features)
    magnitudes = np.sqrt(loadings[0]**2 + loadings[1]**2)
    top_idx = np.argsort(magnitudes)[::-1][:topn]

    # Scale arrows for visibility
    xs = pca2[:,0]; ys = pca2[:,1]
    x_scale = (xs.max() - xs.min())
    y_scale = (ys.max() - ys.min())
    scale = 0.25 * max(x_scale, y_scale)  # heuristic

    plt.figure(figsize=(7.5,6))
    plt.scatter(xs, ys, s=6, alpha=0.5)
    for i in top_idx:
        x_vec = loadings[0, i] * scale
        y_vec = loadings[1, i] * scale
        plt.arrow(0, 0, x_vec, y_vec, head_width=0.03*scale, head_length=0.05*scale, alpha=0.8, length_includes_head=True)
        plt.text(x_vec*1.05, y_vec*1.05, feature_names[i], fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Biplot (top-loading features)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def explained_variance_metrics(explained_ratio):
    cum = np.cumsum(explained_ratio)
    def n_to(thresh):
        idx = np.where(cum >= thresh)[0]
        return int(idx[0] + 1) if len(idx) else len(cum)
    return {
        "pc1_var": float(explained_ratio[0]) if len(explained_ratio)>0 else float('nan'),
        "pc2_var": float(explained_ratio[1]) if len(explained_ratio)>1 else float('nan'),
        "cum_var_pc1_pc2": float(cum[1]) if len(cum)>1 else (float(cum[0]) if len(cum)>0 else float('nan')),
        "n_components_90pct": n_to(0.90),
        "n_components_95pct": n_to(0.95)
    }

# ----------------------------- main per-split -----------------------------

def run_one_split(split_id, df, move_cols, test_size, alpha, outdir: Path, seed, max_points):
    split_dir = outdir / f"split_{split_id:02d}"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Stratify by participant
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed,
        stratify=df["participant_id"].astype(str)
    )
    tr = df.iloc[train_idx].reset_index(drop=True)
    te = df.iloc[test_idx].reset_index(drop=True)

    # Build & scale
    Xtr_moves, tr_falpha, tr_meta = build_features(tr, move_cols, alpha)
    Xte_moves, te_falpha, te_meta = build_features(te, move_cols, alpha)

    scaler = StandardScaler().fit(Xtr_moves)
    Xtr_scaled = np.hstack([scaler.transform(Xtr_moves), tr_falpha])
    Xte_scaled = np.hstack([scaler.transform(Xte_moves), te_falpha])

    # Save scaled features
    save_scaled(split_dir / "train_scaled.csv", tr_meta, Xtr_scaled, move_cols)
    save_scaled(split_dir / "test_scaled.csv",  te_meta, Xte_scaled, move_cols)

    # PCA on TRAIN (all components)
    pca = PCA(n_components=None, svd_solver="auto", random_state=seed)
    pca.fit(Xtr_scaled)
    explained = pca.explained_variance_ratio_
    loadings = pca.components_.T  # shape (n_features, n_components)

    feature_names = [f"{c}_z" for c in move_cols] + ["frame_norm_x_alpha"]

    # Save explained variance
    ev_df = pd.DataFrame({
        "component": np.arange(1, len(explained)+1, dtype=int),
        "explained_variance_ratio": explained,
        "cumulative": np.cumsum(explained)
    })
    ev_df.to_csv(split_dir / "pca_explained_variance.csv", index=False)

    # Save loadings
    load_cols = [f"PC{i}" for i in range(1, loadings.shape[1]+1)]
    load_df = pd.DataFrame(loadings, index=feature_names, columns=load_cols)
    load_df.to_csv(split_dir / "pca_loadings.csv")

    # Transform TRAIN/TEST to first two PCs (for plots)
    tr_scores = pca.transform(Xtr_scaled)
    te_scores = pca.transform(Xte_scaled)
    tr_pca2 = tr_scores[:, :2]
    te_pca2 = te_scores[:, :2]

    # Save 2D PCA scores for external plotting if needed
    tr_pca2_df = pd.DataFrame({"pc1": tr_pca2[:,0], "pc2": tr_pca2[:,1]})
    tr_pca2_df.insert(0, "participant_id", tr_meta["participant_id"])
    tr_pca2_df.insert(1, "frame", tr_meta["frame"])
    tr_pca2_df.to_csv(split_dir / "train_pca2.csv", index=False)

    te_pca2_df = pd.DataFrame({"pc1": te_pca2[:,0], "pc2": te_pca2[:,1]})
    te_pca2_df.insert(0, "participant_id", te_meta["participant_id"])
    te_pca2_df.insert(1, "frame", te_meta["frame"])
    te_pca2_df.to_csv(split_dir / "test_pca2.csv", index=False)

    # --------- Visualizations (TRAIN only for stability) ---------
    # Scree & cumulative
    make_scree_plot(explained, split_dir / "scree.png")
    make_cumvar_plot(explained, split_dir / "cumulative_variance.png")

    # Subsample for scatter plots (avoid huge PNGs)
    tr_meta_sub, tr_pca2_sub = subsample_for_plot(tr_meta, tr_pca2, max_points=max_points, seed=seed)

    # PC scatter colored by frame_norm
    make_pc_scatter_color_by_frame(tr_meta_sub, tr_pca2_sub, split_dir / "pc_scatter_frame_norm.png")

    # PC scatter colored by participant (top-N)
    make_pc_scatter_color_by_participant(tr_meta_sub, tr_pca2_sub, split_dir / "pc_scatter_participant.png", max_legend=15)

    # Biplot (top-loading features)
    make_biplot(pca, tr_pca2_sub, feature_names, split_dir / "biplot.png", topn=8)

    # Metrics JSON for this split
    m = explained_variance_metrics(explained)
    m.update({
        "n_features": int(len(feature_names)),
        "alpha": float(alpha),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te))
    })
    with open(split_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    # Print brief summary
    print(f"[split {split_id:02d}] PC1={m['pc1_var']:.3f}, PC2={m['pc2_var']:.3f}, "
          f"PC1+PC2={m['cum_var_pc1_pc2']:.3f}, n90={m['n_components_90pct']}, n95={m['n_components_95pct']}")

    return m

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="PCA-only pipeline (split→scale→PCA) with metrics & plots.")
    ap.add_argument("--input", type=Path, required=True, help="Path to pooled_features_raw.csv")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory for PCA runs")
    ap.add_argument("--splits", type=int, default=5, help="Number of random splits")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction per split")
    ap.add_argument("--alpha", type=float, default=0.25, help="Weight for frame_norm (added as one feature)")
    ap.add_argument("--max-points", type=int, default=25000, help="Max train points plotted (subsample for PNGs)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df, move_cols = load_dataset(args.input)

    # Run splits with varying seeds for diversity
    all_metrics = []
    rng = np.random.RandomState(args.seed)
    for s in range(args.splits):
        seed = int(rng.randint(0, 10**9))
        m = run_one_split(
            split_id=s,
            df=df,
            move_cols=move_cols,
            test_size=args.test_size,
            alpha=args.alpha,
            outdir=args.outdir,
            seed=seed,
            max_points=args.max_points
        )
        all_metrics.append(m)

    # Aggregate PCA summaries across splits
    def safe_mean(vals):
        v = np.array([float(x) for x in vals], dtype=float)
        return float(np.nanmean(v)) if v.size else float("nan")

    summary = {
        "splits": args.splits,
        "alpha": args.alpha,
        "test_size": args.test_size,
        "n_features": all_metrics[0]["n_features"] if all_metrics else None,
        "pc1_var_mean": safe_mean([m["pc1_var"] for m in all_metrics]),
        "pc2_var_mean": safe_mean([m["pc2_var"] for m in all_metrics]),
        "pc1_pc2_cum_mean": safe_mean([m["cum_var_pc1_pc2"] for m in all_metrics]),
        "n_components_90pct_mean": safe_mean([m["n_components_90pct"] for m in all_metrics]),
        "n_components_95pct_mean": safe_mean([m["n_components_95pct"] for m in all_metrics])
    }
    with open(args.outdir / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] PCA-only pipeline complete. Wrote:", args.outdir / "pca_summary.json")

if __name__ == "__main__":
    main()
