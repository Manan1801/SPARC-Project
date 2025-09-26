#!/usr/bin/env python3
"""
Generic cluster pipeline.

- Assumes first column is 'Participant ID' (labels only).
- All other columns (whatever they are) are treated as features.
- Normalizes features (Min–Max) → clustering_norm.csv
- Runs KMeans for k in [2..k_max], selects best k by silhouette score
- Saves:
    * Silhouette vs Inertia → PNG
    * PCA scatter 2D → HTML
    * PCA scatter 3D → HTML
    * Silhouette samples plot → HTML
- Report includes PCA loadings (PC1–PC3 when available) and explained variance.

Usage:
    python cluster_pipeline.py --csv /path/to/clustering.csv
"""

import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def parse_args():
    p = argparse.ArgumentParser(description="Clustering with PCA visualizations.")
    p.add_argument("--csv", default="clustering.csv", help="Path to input CSV")
    p.add_argument("--k-max", type=int, default=10, help="Max k to try (default: 10)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


def safe_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def minmax_normalize(df, id_col):
    feature_cols = [c for c in df.columns if c != id_col]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    norm_df = pd.concat(
        [df[[id_col]].reset_index(drop=True),
         pd.DataFrame(X, columns=feature_cols)], axis=1
    )
    return norm_df, feature_cols


def choose_k_by_silhouette(X, ks, random_state=42):
    inertias, sil_scores, models = [], [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        models[k] = (km, labels)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        sil_scores.append(sil)

    inertias = np.array(inertias, dtype=float)
    sil_scores = np.array(sil_scores, dtype=float)

    valid = np.where(~np.isnan(sil_scores))[0]
    if len(valid):
        k_final = ks[int(valid[np.argmax(sil_scores[valid])])]
    else:
        k_final = ks[int(np.argmin(inertias))]

    return k_final, inertias, sil_scores, models


def save_silhouette_vs_inertia_png(ks, inertias, sil_scores, out_path):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, inertias, marker="o", label="Inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax2 = ax1.twinx()
    ax2.plot(ks, sil_scores, marker="s", linestyle="--", color="orange", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score")
    plt.title("Silhouette Score (dashed) vs Inertia (solid)")
    fig.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def save_pca_scatter_2d_html(X, labels, pid, out_path):
    pca2 = PCA(n_components=2, random_state=0)
    X2 = pca2.fit_transform(X)
    df_plot = pd.DataFrame({
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
        "Cluster": labels,
        "Participant ID": pid
    })
    fig = px.scatter(
        df_plot, x="PC1", y="PC2", color=df_plot["Cluster"].astype(str),
        hover_data={"Participant ID": True, "Cluster": True},
        title="PCA(2D) Scatter", template="plotly_white"
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return pca2


def save_pca_scatter_3d_html(X, labels, pid, out_path):
    pca3 = PCA(n_components=3, random_state=0)
    X3 = pca3.fit_transform(X)
    df_plot = pd.DataFrame({
        "PC1": X3[:, 0],
        "PC2": X3[:, 1],
        "PC3": X3[:, 2],
        "Cluster": labels,
        "Participant ID": pid
    })
    fig = px.scatter_3d(
        df_plot, x="PC1", y="PC2", z="PC3",
        color=df_plot["Cluster"].astype(str),
        hover_data={"Participant ID": True, "Cluster": True},
        title="PCA(3D) Scatter", template="plotly_white"
    )
    fig.update_traces(marker=dict(size=4))
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return pca3


def save_silhouette_samples_html(X, labels, pid, out_path):
    s = silhouette_samples(X, labels)
    df = pd.DataFrame({"sil": s, "cluster": labels, "Participant ID": pid})
    df["rank_in_cluster"] = df.groupby("cluster")["sil"].rank(method="first")
    df_sorted = df.sort_values(by=["cluster", "sil"]).reset_index(drop=True)
    df_sorted["y"] = np.arange(len(df_sorted))

    fig = go.Figure()
    for cl in sorted(df_sorted["cluster"].unique()):
        sub = df_sorted[df_sorted["cluster"] == cl]
        fig.add_trace(go.Bar(
            x=sub["sil"], y=sub["y"], orientation="h", name=f"Cluster {cl}",
            hovertext=[f"PID: {pid} | sil: {v:.4f}" for pid, v in zip(sub["Participant ID"], sub["sil"])],
            hoverinfo="text", marker_line_width=0
        ))

    fig.update_layout(
        barmode="stack", title="Silhouette Samples",
        xaxis_title="Silhouette coefficient", yaxis_title="Samples", template="plotly_white"
    )
    fig.add_vline(x=float(np.mean(s)), line_dash="dash")
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def main():
    args = parse_args()
    in_path = Path(args.csv).expanduser().resolve()
    out_dir = in_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- Load ----------
    df = pd.read_csv(in_path)
    if df.columns[0] != "Participant ID":
        raise ValueError("First column must be 'Participant ID'.")

    id_col = "Participant ID"
    feat_cols = [c for c in df.columns if c != id_col]
    df = safe_numeric(df, feat_cols)
    df = df[df[feat_cols].notna().any(axis=1)].reset_index(drop=True)

    # ---------- Normalize ----------
    norm_df, used_feature_cols = minmax_normalize(df, id_col)
    norm_csv = out_dir / "clustering_norm.csv"
    norm_df.to_csv(norm_csv, index=False)

    # ---------- KMeans ----------
    X = norm_df[used_feature_cols].values
    pid = norm_df[id_col].values
    ks = list(range(2, min(args.k_max, X.shape[0]-1) + 1))
    k_final, inertias, sil_scores, models = choose_k_by_silhouette(X, ks, args.random_state)
    km_final, labels_final = models[k_final]

    # ---------- Plots ----------
    sil_inertia_png = out_dir / f"silhouette_vs_inertia_{ts}.png"
    save_silhouette_vs_inertia_png(ks, inertias, sil_scores, sil_inertia_png)

    pca2d_html = out_dir / f"pca_scatter_2d_k{k_final}_{ts}.html"
    pca2 = save_pca_scatter_2d_html(X, labels_final, pid, pca2d_html)

    pca3d_html = out_dir / f"pca_scatter_3d_k{k_final}_{ts}.html"
    pca3 = save_pca_scatter_3d_html(X, labels_final, pid, pca3d_html)

    sil_samples_html = out_dir / f"silhouette_samples_k{k_final}_{ts}.html"
    save_silhouette_samples_html(X, labels_final, pid, sil_samples_html)

    # ---------- Save assignments ----------
    cluster_csv = out_dir / f"cluster_assignments_k{k_final}_{ts}.csv"
    pd.DataFrame({id_col: pid, "cluster": labels_final}).to_csv(cluster_csv, index=False)

    # ---------- Report ----------
    report_lines = [
        "CLUSTERING REPORT", "-"*60,
        f"Input file: {in_path}",
        f"Timestamp: {ts}",
        f"Final k (by silhouette max): {k_final}",
        f"Inertias: {np.round(inertias,4).tolist()}",
        f"Silhouette scores: {np.round(sil_scores,4).tolist()}",
        ""
    ]

    explained2 = getattr(pca2, "explained_variance_ratio_", None)
    comps2 = getattr(pca2, "components_", None)
    if explained2 is not None and comps2 is not None:
        report_lines.append("PCA 2D loadings (PC1–PC2):")
        for i, (vec, var) in enumerate(zip(comps2, explained2), start=1):
            report_lines.append(f"  PC{i} (var={var:.4f}):")
            report_lines.extend([f"    - {f}: {val:.5f}" for f, val in zip(used_feature_cols, vec)])
        report_lines.append("")

    explained3 = getattr(pca3, "explained_variance_ratio_", None)
    comps3 = getattr(pca3, "components_", None)
    if explained3 is not None and comps3 is not None:
        report_lines.append("PCA 3D loadings (PC1–PC3):")
        for i, (vec, var) in enumerate(zip(comps3, explained3), start=1):
            report_lines.append(f"  PC{i} (var={var:.4f}):")
            report_lines.extend([f"    - {f}: {val:.5f}" for f, val in zip(used_feature_cols, vec)])
        report_lines.append("")

    report_lines += [
        "Artifacts:",
        f" - Normalized CSV           : {norm_csv.name}",
        f" - Silhouette vs Inertia PNG: {sil_inertia_png.name}",
        f" - PCA scatter 2D HTML      : {pca2d_html.name}",
        f" - PCA scatter 3D HTML      : {pca3d_html.name}",
        f" - Silhouette samples HTML  : {sil_samples_html.name}",
        f" - Cluster assignments CSV  : {cluster_csv.name}"
    ]

    report_txt = out_dir / f"clustering_report_{ts}.txt"
    with open(report_txt, "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print("\nDone.")


if __name__ == "__main__":
    main()
