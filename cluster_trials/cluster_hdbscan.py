#!/usr/bin/env python3
"""
Visualize clustering with PCA/UMAP (2D & 3D) colored by GMM labels,
and write an output CSV with the cluster number for each row.

IMPORTANT (as requested):
For visualization coordinates ONLY, we compute:
    score_pc_i(row) = sum_j [ loading(pc_i, feature_j) * exact_raw_value(row, feature_j) ]
i.e., (PC loading) × (raw exact values) with NO centering/scaling/z-scores.

Clustering still uses PCA on standardized features (unchanged).

NEW:
- PCA variance plot to pick dims reaching >= target cumulative variance (default 90%).
- GMM model-selection plots (BIC/AIC vs k for diag/full), minima highlighted.
- Silhouette plot for the CURRENT GMM labels: per cluster, show top-N inliers and top-N outliers.
- Combined curve: Mean Silhouette (dashed) + Inertia/SSE (solid) vs k (uses chosen --gmm-cov).
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_args():
    p = argparse.ArgumentParser(description="PCA/UMAP visualization of GMM clusters + CSV export")
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--drop-cols", nargs="*", default=["participant_id", "frame"],
                   help="Columns to drop before clustering")
    p.add_argument("--pca-dims", type=int, default=9,
                   help="Target PCA dimensions used for clustering (default 9)")
    p.add_argument("--gmm-k", type=int, default=4,
                   help="Number of GMM components (use 0 to auto-select via BIC over k=2..20 and cov in {diag,full})")
    p.add_argument("--gmm-cov", default="full", choices=["full","diag"],
                   help="GMM covariance_type if --gmm-k>0 (fixed-k mode)")
    # Visualization controls
    p.add_argument("--viz", choices=["pca","umap"], default="pca",
                   help="Embedding to visualize (default: pca)")
    p.add_argument("--plot-dims", type=int, choices=[2,3], default=3,
                   help="Number of dimensions to plot (2 or 3). Default 3.")
    # UMAP params (used only if --viz umap)
    p.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for PCA/UMAP/GMM")
    p.add_argument("--out-csv", default=None, help="Path to output CSV with appended 'cluster' column")

    # Helper plots knobs
    p.add_argument("--pca-target-var", type=float, default=0.90,
                   help="Target cumulative variance for PCA variance plot (default 0.90 = 90%)")
    p.add_argument("--model-kmin", type=int, default=2,
                   help="Min k for GMM model-selection curves (default 2)")
    p.add_argument("--model-kmax", type=int, default=7,
                   help="Max k for GMM model-selection curves (default 20)")

    # PC weights × exact values visualization (bar charts per row & PC)
    p.add_argument("--pc-weights", action="store_true",
                   help="Make bar charts of per-feature contributions = (PC loading) × (raw exact values) for a chosen row")
    p.add_argument("--pc-weights-row", type=int, default=0,
                   help="Row index (0-based) whose exact values to use for the PC-weights visualization")
    p.add_argument("--pc-weights-top", type=int, default=20,
                   help="Show top-|contribution| features (default 20)")

    # Silhouette selection (for current GMM only)
    p.add_argument("--sil-top-n", type=int, default=5,
                   help="Number of top inliers (highest silhouette) to plot per cluster (default 5)")
    p.add_argument("--sil-bottom-n", type=int, default=5,
                   help="Number of top outliers (lowest silhouette) to plot per cluster (default 5)")
    return p.parse_args()

def cluster_summary(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}

def default_out_csv_path(in_csv):
    base = os.path.basename(in_csv)
    stem, ext = os.path.splitext(base)
    if ext == "":
        ext = ".csv"
    out = stem + "_clusters" + ext
    return os.path.join(os.path.dirname(in_csv), out)

def html_base_from_out(out_csv, in_csv):
    path = out_csv or default_out_csv_path(in_csv)
    return path.rsplit(".", 1)[0]

def print_pca_logs(pca_obj, feature_names):
    print("\nExplained variance ratio per component (for clustering PCA on standardized data):")
    for i, ratio in enumerate(pca_obj.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.4f}")
    print("\nPrincipal components (feature loadings):")
    for i, comp in enumerate(pca_obj.components_):
        print(f"  PC{i+1}: {comp}")
    if feature_names is not None:
        print("\nTop |loadings| per PC (with feature names):")
        for i, comp in enumerate(pca_obj.components_):
            idx = np.argsort(np.abs(comp))[::-1][:5]
            items = [f"{feature_names[j]}:{comp[j]:+.3f}" for j in idx]
            print(f"  PC{i+1}: " + ", ".join(items))

def build_hover_columns(df, n_rows):
    pid = df["participant_id"] if "participant_id" in df.columns else pd.Series(["NA"]*n_rows)
    frame = df["frame"] if "frame" in df.columns else pd.Series(["NA"]*n_rows)
    return pid, frame

def add_legend_toggles(fig, title_text):
    fig.update_layout(
        legend_title_text="Cluster",
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        title=title_text
    )
    n_traces = len(fig.data)
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.02, y=1.15, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Show All", method="update",
                     args=[{"visible": [True]*n_traces}]),
                dict(label="Hide All", method="update",
                     args=[{"visible": [False]*n_traces}]),
            ],
            pad={"r": 4, "t": 4}
        )]
    )
    return fig

def make_pc_weights_figs(pca_obj, feature_names, x_raw_row, top_k, html_base, plot_dims):
    comps_to_plot = min(plot_dims, pca_obj.components_.shape[0])
    figs = []
    for pc_ix in range(comps_to_plot):
        loadings = pca_obj.components_[pc_ix]
        contrib = loadings * x_raw_row
        idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        feat_sel = [feature_names[i] for i in idx]
        contrib_sel = contrib[idx]
        raw_vals_sel = x_raw_row[idx]
        load_sel = loadings[idx]
        bar_df = pd.DataFrame({
            "feature": feat_sel,
            "contribution": contrib_sel,
            "raw_value": raw_vals_sel,
            "loading": load_sel
        })
        fig = px.bar(
            bar_df,
            x="feature", y="contribution",
            hover_data={"raw_value": True, "loading": True, "feature": True, "contribution": True},
            title=f"PC{pc_ix+1}: Top {len(feat_sel)} |loading × raw value| features (no z-scores)"
        )
        fig.update_layout(xaxis_tickangle=-45, bargap=0.2)
        figs.append(fig)
        fig.write_html(f"{html_base}_pcweights_PC{pc_ix+1}.html", include_plotlyjs="cdn", full_html=True)
    return figs

def compute_pc_weight_scores(X_raw_imp, pca_obj, n_dims=None):
    comps = pca_obj.components_
    if n_dims is not None:
        n_dims = min(n_dims, comps.shape[0])
        comps = comps[:n_dims, :]
    return X_raw_imp @ comps.T

# ----------------------- NEW: helper plots -----------------------

def pca_variance_plot(X_std, target, html_path):
    pca_full = PCA(n_components=None, random_state=0)
    pca_full.fit(X_std)
    var = pca_full.explained_variance_ratio_
    cum = np.cumsum(var)
    n_hit = int(np.searchsorted(cum, target) + 1)

    x = np.arange(1, len(var) + 1)
    fig = go.Figure()
    fig.add_bar(x=x, y=var, name="Per-component variance ratio")
    fig.add_trace(go.Scatter(x=x, y=cum, mode="lines+markers", name="Cumulative variance"))
    fig.add_hline(y=target, line_dash="dash", annotation_text=f"Target {target:.0%}",
                  annotation_position="top left")
    fig.add_vline(x=n_hit, line_dash="dot",
                  annotation_text=f"n={n_hit} (≥{target:.0%})",
                  annotation_position="top right")
    fig.update_layout(
        title=f"PCA Variance & Cumulative (target ≥ {target:.0%})",
        xaxis_title="Number of components",
        yaxis_title="Variance ratio / Cumulative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    print(f"[plots] Saved PCA variance plot to: {html_path}")
    return n_hit, var, cum

def gmm_model_selection_plots(X_pca, kmin, kmax, random_state, html_base):
    k_vals = np.arange(kmin, kmax + 1)
    cov_types = ["diag", "full"]
    bic = {cov: [] for cov in cov_types}
    aic = {cov: [] for cov in cov_types}
    for cov in cov_types:
        for k in k_vals:
            gm = GaussianMixture(n_components=k, covariance_type=cov, random_state=random_state).fit(X_pca)
            bic[cov].append(gm.bic(X_pca))
            aic[cov].append(gm.aic(X_pca))
    for cov in cov_types:
        bic[cov] = np.array(bic[cov])
        aic[cov] = np.array(aic[cov])

    fig_bic = go.Figure()
    for cov in cov_types:
        fig_bic.add_trace(go.Scatter(x=k_vals, y=bic[cov], mode="lines+markers", name=f"BIC ({cov})"))
        k_best = int(k_vals[np.argmin(bic[cov])]); y_best = float(np.min(bic[cov]))
        fig_bic.add_vline(x=k_best, line_dash="dot", annotation_text=f"{cov}: k={k_best}",
                          annotation_position="top right")
        fig_bic.add_trace(go.Scatter(x=[k_best], y=[y_best], mode="markers",
                                     name=f"min BIC ({cov})", marker_symbol="star", marker_size=12))
    fig_bic.update_layout(title="GMM Model Selection: BIC vs k",
                          xaxis_title="k (number of components)", yaxis_title="BIC (lower is better)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    bic_path = f"{html_base}_gmm_bic.html"
    fig_bic.write_html(bic_path, include_plotlyjs="cdn", full_html=True)
    print(f"[plots] Saved GMM BIC plot to: {bic_path}")

    fig_aic = go.Figure()
    for cov in cov_types:
        fig_aic.add_trace(go.Scatter(x=k_vals, y=aic[cov], mode="lines+markers", name=f"AIC ({cov})"))
        k_best = int(k_vals[np.argmin(aic[cov])]); y_best = float(np.min(aic[cov]))
        fig_aic.add_vline(x=k_best, line_dash="dot", annotation_text=f"{cov}: k={k_best}",
                          annotation_position="top right")
        fig_aic.add_trace(go.Scatter(x=[k_best], y=[y_best], mode="markers",
                                     name=f"min AIC ({cov})", marker_symbol="star", marker_size=12))
    fig_aic.update_layout(title="GMM Model Selection: AIC vs k",
                          xaxis_title="k (number of components)", yaxis_title="AIC (lower is better)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    aic_path = f"{html_base}_gmm_aic.html"
    fig_aic.write_html(aic_path, include_plotlyjs="cdn", full_html=True)
    print(f"[plots] Saved GMM AIC plot to: {aic_path}")

# ----------------------- NEW: Silhouette (current GMM only) -----------------------

def silhouette_topbot_plot(X_for_sil, labels, pid_col, frame_col, top_n, bot_n, html_path):
    unique_clusters = sorted(np.unique(labels).tolist())
    try:
        sil_samples = silhouette_samples(X_for_sil, labels, metric="euclidean")
        sil_mean = silhouette_score(X_for_sil, labels, metric="euclidean")
    except Exception as e:
        print(f"[silhouette] Could not compute silhouette samples: {e}")
        return

    fig = make_subplots(rows=len(unique_clusters), cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=[f"Cluster {c}" for c in unique_clusters])

    any_plotted = False
    for r, c in enumerate(unique_clusters, start=1):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) < 2:
            fig.add_annotation(row=r, col=1, xref=f"x{r}", yref=f"y{r}",
                               text=f"Cluster {c}: too few samples for silhouette", showarrow=False)
            continue

        s_c = sil_samples[idx_c]
        order_in = np.argsort(-s_c)                   # best
        order_out = np.argsort(s_c)                   # worst

        n_in = min(top_n, len(idx_c))
        n_out = min(bot_n, max(0, len(idx_c) - n_in))
        in_sel = idx_c[order_in[:n_in]]
        out_sel = idx_c[order_out[:n_out]]

        def labels_for(indices):
            vals = []
            for i in indices:
                pid = str(pid_col.iloc[i]) if hasattr(pid_col, "iloc") else str(pid_col[i])
                frm = str(frame_col.iloc[i]) if hasattr(frame_col, "iloc") else str(frame_col[i])
                vals.append(f"{pid}:{frm}")
            return vals

        in_scores = sil_samples[in_sel]
        out_scores = sil_samples[out_sel]
        in_y = labels_for(in_sel)
        out_y = labels_for(out_sel)

        fig.add_trace(
            go.Bar(x=out_scores, y=out_y, orientation="h",
                   name=f"C{c} outliers", legendgroup=f"C{c}",
                   marker_line_width=0, hovertemplate="sil=%{x:.3f}<br>%{y}<extra></extra>",
                   showlegend=(r == 1)),
            row=r, col=1
        )
        fig.add_trace(
            go.Bar(x=in_scores, y=in_y, orientation="h",
                   name=f"C{c} inliers", legendgroup=f"C{c}",
                   marker_line_width=0, hovertemplate="sil=%{x:.3f}<br>%{y}<extra></extra>",
                   showlegend=(r == 1)),
            row=r, col=1
        )
        any_plotted = True
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.4)", row=r, col=1)

    title_txt = f"Silhouette (current GMM) — mean={sil_mean:.3f} | Showing top inliers & outliers per cluster"
    fig.update_layout(height=max(400, 160 * len(unique_clusters)),
                      title=title_txt, barmode="relative",
                      xaxis_title="Silhouette score",
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

    if any_plotted:
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        print(f"[plots] Saved silhouette plot to: {html_path}")
    else:
        print("[silhouette] No clusters had >=2 samples; silhouette plot was not generated.")

# ----------------------- NEW: Silhouette + Inertia vs k (dual-axis) -----------------------

def _kmeans_like_inertia(X, labels):
    """Within-cluster SSE using hard labels (k-means-style inertia) on X."""
    inertia = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if pts.size == 0:
            continue
        center = pts.mean(axis=0, keepdims=True)
        diffs = pts - center
        inertia += float(np.sum(diffs * diffs))
    return inertia

def silhouette_inertia_curve_plot(X_pca, cov_type, kmin, kmax, random_state, html_path):
    """
    For k in [kmin, kmax], fit GMM (cov_type), get hard labels.
    Compute mean silhouette (euclidean on X_pca) and k-means-style inertia (SSE).
    Plot both on dual y-axes: silhouette dashed, inertia solid.
    """
    k_vals = np.arange(kmin, kmax + 1)
    sil_vals = []
    inertia_vals = []

    for k in k_vals:
        if k < 2:
            sil_vals.append(np.nan)
            inertia_vals.append(np.nan)
            continue
        gm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=random_state).fit(X_pca)
        labels = gm.predict(X_pca)
        try:
            sil = silhouette_score(X_pca, labels, metric="euclidean")
        except Exception:
            sil = np.nan
        sil_vals.append(sil)
        inertia_vals.append(_kmeans_like_inertia(X_pca, labels))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=k_vals, y=sil_vals, mode="lines+markers", name="Mean Silhouette",
                   line=dict(dash="dash")),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=k_vals, y=inertia_vals, mode="lines+markers", name="Inertia (SSE)",
                   line=dict(dash="solid")),
        secondary_y=True
    )
    fig.update_layout(
        title=f"Silhouette (dashed) & Inertia (solid) vs k — GMM cov={cov_type}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(title_text="k (number of components)")
    fig.update_yaxes(title_text="Mean silhouette", range=[-1, 1], secondary_y=False)
    fig.update_yaxes(title_text="Inertia (SSE)", secondary_y=True)

    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    print(f"[plots] Saved Silhouette+Inertia plot to: {html_path}")

# ----------------------- /helper plots -----------------------

def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.csv)

    # Feature matrix (keep a copy BEFORE scaling for PC-weighted viz)
    X_df = df.drop(columns=args.drop_cols, errors="ignore")
    feature_names = list(X_df.columns)
    X_raw = X_df.values

    # Impute raw (for visualization) and also build standardized (for clustering)
    imputer = SimpleImputer(strategy="mean")
    X_raw_imp = imputer.fit_transform(X_raw)  # exact values with NaNs replaced by feature means

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw_imp)   # used for PCA/GMM/UMAP (clustering pipeline)

    # PCA for clustering
    pca_for_cluster = PCA(n_components=args.pca_dims, random_state=args.random_state)
    X_pca = pca_for_cluster.fit_transform(X_std)
    print(f"PCA (for clustering) shape: {X_pca.shape}")

    # PCA logs
    print_pca_logs(pca_for_cluster, feature_names)

    # Output paths base
    out_csv = args.out_csv or default_out_csv_path(args.csv)
    html_base = html_base_from_out(out_csv, args.csv)

    # --- PCA variance plot ---
    n_hit, var, cum = pca_variance_plot(
        X_std, target=args.pca_target_var,
        html_path=f"{html_base}_pca_variance.html"
    )
    print(f"[pca-variance] Components to reach ≥{args.pca_target_var:.0%}: n={n_hit}")

    # --- GMM model-selection plots (BIC/AIC vs k) ---
    gmm_model_selection_plots(
        X_pca, kmin=args.model_kmin, kmax=args.model_kmax,
        random_state=args.random_state, html_base=html_base
    )

    # --- NEW: Silhouette + Inertia vs k (dual-axis) ---
    silhouette_inertia_curve_plot(
        X_pca, cov_type=args.gmm_cov,
        kmin=args.model_kmin, kmax=args.model_kmax,
        random_state=args.random_state,
        html_path=f"{html_base}_silhouette_inertia.html"
    )

    # --- Fit the FINAL GMM for labeling ---
    if args.gmm_k == 0:
        best = {"bic": np.inf, "k": None, "cov": None, "model": None}
        for k in range(args.model_kmin, args.model_kmax + 1):
            for cov in ["diag", "full"]:
                gm = GaussianMixture(n_components=k, covariance_type=cov,
                                     random_state=args.random_state).fit(X_pca)
                bic = gm.bic(X_pca)
                if bic < best["bic"]:
                    best = {"bic": bic, "k": k, "cov": cov, "model": gm}
        gmm = best["model"]
        print(f"Auto-selected GMM: k={best['k']} cov={best['cov']} (BIC={best['bic']:.2f})")
    else:
        gmm = GaussianMixture(n_components=args.gmm_k, covariance_type=args.gmm_cov,
                              random_state=args.random_state).fit(X_pca)
        print(f"GMM fixed: k={args.gmm_k} cov={args.gmm_cov}")

    labels = gmm.predict(X_pca)
    print("Cluster sizes:", cluster_summary(labels))

    # Write CSV with labels
    df_out = df.copy()
    df_out["cluster"] = labels.astype(int)
    df_out.to_csv(out_csv, index=False)
    print("Wrote CSV with cluster labels to:", out_csv)

    # Prepare hover columns
    pid_col, frame_col = build_hover_columns(df, len(df))

    # Visualization (PC-weighted raw coordinates ONLY)
    if args.viz == "pca":
        if pca_for_cluster.n_components_ < args.plot_dims:
            pca_for_plot = PCA(n_components=args.plot_dims, random_state=args.random_state).fit(X_std)
            print(f"[PCA plot] recomputed PCA with {args.plot_dims} comps for PC-weighted coords")
        else:
            pca_for_plot = pca_for_cluster

        S = compute_pc_weight_scores(X_raw_imp, pca_for_plot, n_dims=args.plot_dims)
        axes_names = [f"PCw{i+1}" for i in range(args.plot_dims)]
        cluster_str = pd.Series(labels.astype(int), index=df.index).astype(str)

        if args.plot_dims == 2:
            plot_df = pd.DataFrame({
                axes_names[0]: S[:, 0],
                axes_names[1]: S[:, 1],
                "participant_id": pid_col,
                "frame": frame_col,
                "cluster_str": cluster_str
            })
            cluster_order = sorted(plot_df["cluster_str"].unique(), key=lambda s: int(s))
            fig = px.scatter(
                plot_df, x=axes_names[0], y=axes_names[1], color="cluster_str",
                category_orders={"cluster_str": cluster_order},
                hover_data=["participant_id", "frame", "cluster_str"],
            )
            fig = add_legend_toggles(fig, "PCA (2D) - GMM Clusters (coords = loading × raw value)")
            fig.write_html(f"{html_base}_pca2d_pcw.html", include_plotlyjs="cdn", full_html=True)
            fig.show()
        else:
            plot_df = pd.DataFrame({
                axes_names[0]: S[:, 0],
                axes_names[1]: S[:, 1],
                axes_names[2]: S[:, 2],
                "participant_id": pid_col,
                "frame": frame_col,
                "cluster_str": cluster_str
            })
            cluster_order = sorted(plot_df["cluster_str"].unique(), key=lambda s: int(s))
            fig = px.scatter_3d(
                plot_df, x=axes_names[0], y=axes_names[1], z=axes_names[2], color="cluster_str",
                category_orders={"cluster_str": cluster_order},
                hover_data=["participant_id", "frame", "cluster_str"],
            )
            fig = add_legend_toggles(fig, "PCA (3D) - GMM Clusters (coords = loading × raw value)")
            fig.write_html(f"{html_base}_pca3d_pcw.html", include_plotlyjs="cdn", full_html=True)
            fig.show()

    else:
        try:
            import umap
        except ImportError:
            print("Please install umap-learn: pip install umap-learn", file=sys.stderr)
            raise

        n_pcs_for_umap = pca_for_cluster.n_components_
        S_full = compute_pc_weight_scores(X_raw_imp, pca_for_cluster, n_dims=n_pcs_for_umap)
        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            n_components=args.plot_dims,
            random_state=args.random_state,
        )
        emb = reducer.fit_transform(S_full)

        cluster_str = pd.Series(labels.astype(int), index=df.index).astype(str)
        if args.plot_dims == 2:
            plot_df = pd.DataFrame({
                "UMAP1": emb[:, 0],
                "UMAP2": emb[:, 1],
                "participant_id": pid_col,
                "frame": frame_col,
                "cluster_str": cluster_str
            })
            cluster_order = sorted(plot_df["cluster_str"].unique(), key=lambda s: int(s))
            fig = px.scatter(
                plot_df, x="UMAP1", y="UMAP2", color="cluster_str",
                category_orders={"cluster_str": cluster_order},
                hover_data=["participant_id", "frame", "cluster_str"],
            )
            fig = add_legend_toggles(fig, "UMAP (2D) - GMM Clusters (input = PC-weighted raw scores)")
            fig.write_html(f"{html_base}_umap2d_pcw.html", include_plotlyjs="cdn", full_html=True)
            fig.show()
        else:
            plot_df = pd.DataFrame({
                "UMAP1": emb[:, 0],
                "UMAP2": emb[:, 1],
                "UMAP3": emb[:, 2],
                "participant_id": pid_col,
                "frame": frame_col,
                "cluster_str": cluster_str
            })
            cluster_order = sorted(plot_df["cluster_str"].unique(), key=lambda s: int(s))
            fig = px.scatter_3d(
                plot_df, x="UMAP1", y="UMAP2", z="UMAP3", color="cluster_str",
                category_orders={"cluster_str": cluster_order},
                hover_data=["participant_id", "frame", "cluster_str"],
            )
            fig = add_legend_toggles(fig, "UMAP (3D) - GMM Clusters (input = PC-weighted raw scores)")
            fig.write_html(f"{html_base}_umap3d_pcw.html", include_plotlyjs="cdn", full_html=True)
            fig.show()

    # --- Silhouette (current GMM) — top/bottom per cluster ---
    silhouette_topbot_plot(
        X_for_sil=X_pca, labels=labels,
        pid_col=pid_col, frame_col=frame_col,
        top_n=args.sil_top_n, bot_n=args.sil_bottom_n,
        html_path=f"{html_base}_silhouette_topbot.html"
    )

    # --- PC weights × exact values bar-chart visualization (per-row) ---
    if args.pc_weights:
        if not (0 <= args.pc_weights_row < X_raw_imp.shape[0]):
            print(f"[pc-weights] Row index {args.pc_weights_row} is out of bounds (0..{X_raw_imp.shape[0]-1}). Skipping.")
        else:
            row = X_raw_imp[args.pc_weights_row, :].copy()
            pid_val = df.iloc[args.pc_weights_row]["participant_id"] if "participant_id" in df.columns else "NA"
            frame_val = df.iloc[args.pc_weights_row]["frame"] if "frame" in df.columns else "NA"
            print(f"[pc-weights] Using row {args.pc_weights_row} (participant_id={pid_val}, frame={frame_val})")
            figs = make_pc_weights_figs(
                pca_for_cluster, feature_names, row,
                args.pc_weights_top, html_base, args.plot_dims
            )
            if len(figs) > 0:
                figs[0].show()
            print(f"[pc-weights] Saved HTML bar charts to {html_base}_pcweights_PC*.html "
                  f"(no z-scores; contributions = loading × raw value)")

if __name__ == "__main__":
    main()
