# #!/usr/bin/env python3
# # plot_clusters.py
# """
# Minimal cluster scatter plotter.

# Usage:
#   python plot_clusters.py --split-dir /path/to/cluster_runs/split_00 [--space pca|scaled] [--feat-x <name> --feat-y <name>]

# Notes:
# - Default is PCA to 2D for clean visuals.
# - If you choose --space scaled, it will plot the first two feature columns unless you provide --feat-x/--feat-y.
# - Expects these files inside --split-dir:
#     train_scaled.csv, test_scaled.csv, train_labels.csv, test_labels.csv
# """

# from __future__ import annotations
# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# META_COLS = ("participant_id", "frame")

# def load_scaled(csv_path: Path):
#     df = pd.read_csv(csv_path)
#     if not set(META_COLS).issubset(df.columns):
#         raise SystemExit(f"{csv_path} missing {META_COLS}")
#     feat_cols = [c for c in df.columns if c not in META_COLS]
#     return df, feat_cols

# def merge_labels(df_scaled: pd.DataFrame, labels_path: Path):
#     labs = pd.read_csv(labels_path)
#     if not set(META_COLS + ("cluster_id",)).issubset(labs.columns):
#         raise SystemExit(f"{labels_path} missing columns")
#     merged = df_scaled.merge(labs[META_COLS + ("cluster_id",)], on=list(META_COLS), how="left")
#     missing = merged["cluster_id"].isna().sum()
#     if missing:
#         print(f"[WARN] {missing} rows in {labels_path.name} not matched; dropping.")
#     merged = merged.dropna(subset=["cluster_id"]).copy()
#     merged["cluster_id"] = merged["cluster_id"].astype(int)
#     return merged

# def build_space(df: pd.DataFrame, feat_cols, space: str, feat_x: str|None, feat_y: str|None):
#     X = df[feat_cols].to_numpy(dtype=float)
#     used_cols = None
#     if space == "pca":
#         p = PCA(n_components=2, random_state=0).fit(X)
#         Z = p.transform(X)
#         labels = [f"pc1 (viz)", f"pc2 (viz)"]
#         return Z, labels
#     else:
#         if feat_x is None or feat_y is None:
#             if len(feat_cols) < 2:
#                 raise SystemExit("Need at least 2 feature columns in scaled space.")
#             fx, fy = feat_cols[0], feat_cols[1]
#         else:
#             fx, fy = feat_x, feat_y
#             if fx not in feat_cols or fy not in feat_cols:
#                 raise SystemExit(f"--feat-x/--feat-y must be among feature columns. Got {fx}, {fy}.")
#         Z = df[[fx, fy]].to_numpy(dtype=float)
#         return Z, [fx, fy]

# def plot_scatter(Z, y, title, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6,5))
#     else:
#         fig = ax.figure
#     y = np.asarray(y, dtype=int)
#     uniq = np.unique(y)
#     # Map each cluster to a color via matplotlib default cycle; handle noise (-1) specially.
#     for u in uniq:
#         mask = (y == u)
#         if u == -1:
#             ax.scatter(Z[mask,0], Z[mask,1], s=6, marker='x', c='k', alpha=0.6, label='noise (-1)')
#         else:
#             ax.scatter(Z[mask,0], Z[mask,1], s=6, alpha=0.6, label=str(u))
#     ax.set_title(title)
#     ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
#     # Show compact legend if few clusters
#     if len(uniq) <= 15:
#         ax.legend(markerscale=3, fontsize=8, frameon=False, ncol=3)
#     ax.grid(True, linewidth=0.3, alpha=0.3)
#     return fig

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--split-dir", type=Path, required=True, help="Folder with train/test scaled+labels")
#     ap.add_argument("--space", choices=["pca","scaled"], default="pca", help="Visualize in PCA(2) or pick two scaled features")
#     ap.add_argument("--feat-x", type=str, default=None, help="Only for --space scaled")
#     ap.add_argument("--feat-y", type=str, default=None, help="Only for --space scaled")
#     ap.add_argument("--show", action="store_true", help="Also show interactive window")
#     args = ap.parse_args()

#     sd = args.split_dir
#     paths = {
#         "train_scaled": sd / "train_scaled.csv",
#         "test_scaled":  sd / "test_scaled.csv",
#         "train_labels": sd / "train_labels.csv",
#         "test_labels":  sd / "test_labels.csv",
#     }
#     for k,p in paths.items():
#         if not p.exists():
#             raise SystemExit(f"Missing: {p}")

#     # Load + merge labels
#     train_scaled, feat_cols = load_scaled(paths["train_scaled"])
#     test_scaled,  _         = load_scaled(paths["test_scaled"])
#     train = merge_labels(train_scaled, paths["train_labels"])
#     test  = merge_labels(test_scaled,  paths["test_labels"])

#     # Build visualization space
#     Ztr, labels = build_space(train, feat_cols, args.space, args.feat_x, args.feat_y)
#     Zte, _      = build_space(test,  feat_cols, args.space, args.feat_x, args.feat_y)

#     # Scatter: train
#     fig1 = plot_scatter(Ztr, train["cluster_id"].to_numpy(), title=f"Train clusters ({args.space} 2D)")
#     out1 = sd / f"train_clusters_{args.space}.png"
#     fig1.tight_layout(); fig1.savefig(out1, dpi=160)
#     print(f"[saved] {out1}")

#     # Scatter: test
#     fig2 = plot_scatter(Zte, test["cluster_id"].to_numpy(), title=f"Test clusters ({args.space} 2D)")
#     out2 = sd / f"test_clusters_{args.space}.png"
#     fig2.tight_layout(); fig2.savefig(out2, dpi=160)
#     print(f"[saved] {out2}")

#     if args.show:
#         plt.show()

# if __name__ == "__main__":
#     main()

# app_cluster_explorer.py
# Interactive clustering explorer for your PCA-run splits (scaled/PCA space).
# Usage:
#   streamlit run app_cluster_explorer.py -- --pca-root /path/to/pca_runs
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Clustering Explorer", layout="wide")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

# --------------------- CLI (passed after -- in streamlit run) ---------------------
@st.cache_resource
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-root", type=str, required=True, help="Folder containing split_XX with train_scaled.csv/test_scaled.csv")
    return parser.parse_args()

# --------------------- Data I/O ---------------------
@st.cache_data(show_spinner=False)
def list_splits(pca_root: str) -> list[str]:
    root = Path(pca_root)
    return sorted([p.name for p in root.glob("split_*") if p.is_dir()])

@st.cache_data(show_spinner=False)
def load_scaled_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "participant_id" not in df.columns or "frame" not in df.columns:
        raise ValueError(f"{csv_path} missing required columns.")
    feat_cols = [c for c in df.columns if c not in ("participant_id", "frame")]
    X = df[feat_cols].values.astype(float)
    meta = df[["participant_id", "frame"]].copy()
    fidx = feat_cols.index("frame_norm_x_alpha") if "frame_norm_x_alpha" in feat_cols else None
    return meta, X, feat_cols, fidx

# --------------------- Metrics (safe) ---------------------
def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    ok = mask.sum() >= 2 and len(np.unique(labels[mask])) >= 2
    return float(silhouette_score(X[mask], labels[mask])) if ok else float("nan")

def safe_davies_bouldin(X, labels):
    labels = np.asarray(labels); mask = labels != -1
    ok = len(np.unique(labels[mask])) >= 2
    return float(davies_bouldin_score(X[mask], labels[mask])) if ok else float("nan")

def safe_calinski_harabasz(X, labels):
    labels = np.asarray(labels); mask = labels != -1
    ok = len(np.unique(labels[mask])) >= 2
    return float(calinski_harabasz_score(X[mask], labels[mask])) if ok else float("nan")

# --------------------- Assign helpers ---------------------
def assign_by_nearest_centroid(X_ref: np.ndarray, labels_ref: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    labs = np.asarray(labels_ref).astype(int)
    uniq = sorted(set(labs))
    cents = np.vstack([X_ref[labs == u].mean(axis=0) for u in uniq])
    d = ((X_new[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(int)

def assign_by_nearest_core(db: DBSCAN, X_fit: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    core_idx = getattr(db, "core_sample_indices_", None)
    if core_idx is None or len(core_idx) == 0:
        return np.full(len(X_new), -1, dtype=int)
    core_pts = X_fit[core_idx]; core_labs = db.labels_[core_idx]
    nn = NearestNeighbors(n_neighbors=1).fit(core_pts)
    dist, idx = nn.kneighbors(X_new)
    return np.where(dist.ravel() <= db.eps, core_labs[idx.ravel()], -1).astype(int)

# --------------------- Visualization helpers (matplotlib only, single-plot) ---------------------
def plot_cluster_sizes(labels):
    uniq, counts = np.unique(labels, return_counts=True)
    fig = plt.figure(figsize=(6,4))
    plt.bar([str(int(u)) for u in uniq], counts)
    plt.xlabel("Cluster ID"); plt.ylabel("Count"); plt.title("Cluster sizes")
    plt.tight_layout()
    return fig

def plot_silhouette_hist(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return None
    s = silhouette_samples(X[mask], labels[mask])
    fig = plt.figure(figsize=(6,4))
    plt.hist(s, bins=40)
    plt.xlabel("Silhouette (non-noise)"); plt.ylabel("Count")
    plt.title("Silhouette distribution")
    plt.tight_layout()
    return fig

def plot_temporal_box(meta, X_space, feat_cols, labels):
    if "frame_norm_x_alpha" not in feat_cols:
        return None
    idx = feat_cols.index("frame_norm_x_alpha")
    vals = X_space[:, idx]
    df = pd.DataFrame({"cluster_id": labels, "frame_norm_x_alpha": vals})
    fig = plt.figure(figsize=(7,4.5))
    df.boxplot(by="cluster_id", column="frame_norm_x_alpha")
    plt.title("Temporal distribution per cluster (frame_norm×α)")
    plt.suptitle(""); plt.xlabel("Cluster ID"); plt.ylabel("frame_norm×α")
    plt.tight_layout()
    return fig

def plot_2d_embedding(X_space, labels, title):
    # 2D PCA for visualization only (computed on the same space)
    p = PCA(n_components=2, random_state=0).fit(X_space)
    emb = p.transform(X_space)
    fig = plt.figure(figsize=(6,5))
    for u in np.unique(labels):
        m = labels == u
        plt.scatter(emb[m,0], emb[m,1], s=4, label=str(int(u)) if int(u) != -1 else "noise")
    plt.xlabel("viz PC1"); plt.ylabel("viz PC2"); plt.title(title)
    # no legend to keep clean; uncomment to show:
    # plt.legend(markerscale=3)
    plt.tight_layout()
    return fig

# --------------------- Main app ---------------------
def main():
    args = get_args()
    
    st.title("🔎 Clustering Explorer")

    splits = list_splits(args.pca_root)
    if not splits:
        st.error("No split_* directories found under --pca-root.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        split = st.selectbox("Split", splits, index=0)
        split_dir = Path(args.pca_root) / split

        # Load train/test
        meta_tr, Xtr_scaled, feat_cols_scaled, fidx = load_scaled_csv(split_dir / "train_scaled.csv")
        meta_te, Xte_scaled, _, _ = load_scaled_csv(split_dir / "test_scaled.csv")

        st.markdown(f"**Train rows:** {len(meta_tr):,} • **Dims (scaled):** {Xtr_scaled.shape[1]}")

        st.header("Feature space")
        space = st.radio("Space", ["scaled", "pca"], horizontal=True)
        used_dims = Xtr_scaled.shape[1]
        pca_model = None
        feat_cols_space = list(feat_cols_scaled)

        if space == "pca":
            pca_mode = st.radio("PCA mode", ["fixed K", "target variance"], horizontal=True)
            if pca_mode == "fixed K":
                k_comp = st.slider("n_components (K)", min_value=2, max_value=min(64, Xtr_scaled.shape[1]), value=8, step=1)
                pca_model = PCA(n_components=k_comp, random_state=0).fit(Xtr_scaled)
            else:
                target = st.slider("cumulative variance target", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
                _p = PCA(n_components=None, random_state=0).fit(Xtr_scaled)
                cum = np.cumsum(_p.explained_variance_ratio_)
                k_comp = int(np.searchsorted(cum, target) + 1)
                pca_model = PCA(n_components=k_comp, random_state=0).fit(Xtr_scaled)
                st.caption(f"Auto K = {k_comp} to reach ≥ {target:.2f} variance")
            Xtr_space = pca_model.transform(Xtr_scaled)
            Xte_space = pca_model.transform(Xte_scaled)
            used_dims = Xtr_space.shape[1]
            feat_cols_space = [f"pc{i+1}" for i in range(used_dims)]
        else:
            Xtr_space = Xtr_scaled
            Xte_space = Xte_scaled

        st.header("Clustering")
        algo = st.radio("Algorithm", ["kmeans", "agglo", "dbscan"], horizontal=True)

        k = None
        if algo in ("kmeans", "agglo"):
            k = st.slider("k (number of clusters)", min_value=2, max_value=12, value=4, step=1)

        db_ms = db_eps = None
        db_sub = 0
        if algo == "dbscan":
            db_ms = st.slider("min_samples", min_value=5, max_value=100, value=20, step=1)
            # Suggest eps via k-distance percentiles
            k_for_nn = max(db_ms, 2)
            nn = NearestNeighbors(n_neighbors=k_for_nn).fit(Xtr_space)
            dists, _ = nn.kneighbors(Xtr_space)
            kth = dists[:, -1]
            p50, p60, p70, p80, p90 = np.percentile(kth, [50,60,70,80,90])
            db_eps = st.select_slider("eps (suggested percentiles)", options=[round(x,6) for x in [p50,p60,p70,p80,p90]], value=float(round(p70,6)))
            db_sub = st.number_input("Optional subsample for fitting (0 = full)", min_value=0, value=0, step=1000)

        st.header("Run")
        run = st.button("Compute clusters")

    if not run:
        st.info("← Configure options in the sidebar, then click **Compute clusters**.")
        st.stop()

    # --------------------- Fit on TRAIN, evaluate, assign TEST ---------------------
    if algo == "kmeans":
        km = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=0, verbose=0)
        ytr = km.fit_predict(Xtr_space)
        yte = km.predict(Xte_space)
        params = {"k": k, "n_init": 20, "max_iter": 300, "inertia": float(km.inertia_), "fit_iterations": int(getattr(km, "n_iter_", -1))}
    elif algo == "agglo":
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        ytr = ag.fit_predict(Xtr_space)
        yte = assign_by_nearest_centroid(Xtr_space, ytr, Xte_space)
        params = {"k": k}
    else:  # dbscan
        if db_sub and len(Xtr_space) > db_sub:
            rng = np.random.RandomState(0)
            idx = np.sort(rng.choice(len(Xtr_space), size=db_sub, replace=False))
            Xfit = Xtr_space[idx]
        else:
            Xfit = Xtr_space
        db = DBSCAN(eps=float(db_eps), min_samples=int(db_ms))
        yfit = db.fit_predict(Xfit)
        # propagate to full train via nearest core
        ytr = assign_by_nearest_core(db, Xfit, Xtr_space)
        yte = assign_by_nearest_core(db, Xfit, Xte_space)
        params = {"min_samples": int(db_ms), "eps": float(db_eps), "fit_on": f"{'subsample('+str(len(Xfit))+')' if db_sub else 'full'}"}

    sil = safe_silhouette(Xtr_space, ytr)
    dbi = safe_davies_bouldin(Xtr_space, ytr)
    ch  = safe_calinski_harabasz(Xtr_space, ytr)

    # --------------------- Layout ---------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dims (space)", f"{used_dims}")
    c2.metric("Silhouette (train)", f"{sil:.3f}" if np.isfinite(sil) else "—")
    c3.metric("DBI (train) ↓", f"{dbi:.3f}" if np.isfinite(dbi) else "—")
    c4.metric("CH (train) ↑", f"{ch:.0f}" if np.isfinite(ch) else "—")

    st.write("**Params:**", json.dumps(params, indent=2))

    # Train figures
    st.subheader("Train — Visualizations")
    colA, colB = st.columns(2)
    with colA:
        fig_sizes = plot_cluster_sizes(ytr)
        st.pyplot(fig_sizes, clear_figure=True)
    with colB:
        fig_sil = plot_silhouette_hist(Xtr_space, ytr)
        if fig_sil is not None:
            st.pyplot(fig_sil, clear_figure=True)

    fig_viz = plot_2d_embedding(Xtr_space, ytr, title="Train embedding (2D PCA of selected space)")
    st.pyplot(fig_viz, clear_figure=True)

    # Temporal box (if available)
    if "frame_norm_x_alpha" in (feat_cols_scaled if space=="scaled" else feat_cols_space):
        fig_tmp = plot_temporal_box(meta_tr, Xtr_space, (feat_cols_scaled if space=="scaled" else feat_cols_space), ytr)
        if fig_tmp is not None:
            st.pyplot(fig_tmp, clear_figure=True)

    # Participant mix table
    st.subheader("Train — Participant composition")
    df_mix = pd.DataFrame({"participant_id": meta_tr["participant_id"].values, "cluster_id": ytr})
    mix = df_mix.groupby(["cluster_id","participant_id"]).size().reset_index(name="count")
    mix_piv = mix.pivot(index="cluster_id", columns="participant_id", values="count").fillna(0).astype(int)
    st.dataframe(mix_piv)

    # --------------------- Outputs / Downloads ---------------------
    st.subheader("Assignments & Downloads")
    train_labels = pd.DataFrame({"participant_id": meta_tr["participant_id"], "frame": meta_tr["frame"], "cluster_id": ytr.astype(int)})
    test_labels  = pd.DataFrame({"participant_id": meta_te["participant_id"], "frame": meta_te["frame"], "cluster_id": yte.astype(int)})

    st.write("**Train labels (head)**")
    st.dataframe(train_labels.head(20))
    st.write("**Test labels (head)**")
    st.dataframe(test_labels.head(20))

    st.download_button("Download train_labels.csv", data=train_labels.to_csv(index=False), file_name=f"{split}_train_labels.csv")
    st.download_button("Download test_labels.csv",  data=test_labels.to_csv(index=False),  file_name=f"{split}_test_labels.csv")

if __name__ == "__main__":
    main()
