# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# K-Means clustering on participant movement dataset with:
#   - Standardization
#   - Elbow Method (inertia vs k)
#   - 2D & 3D PCA visualizations
#   - Saves clustered CSV with labels

# Assumes the CSV has columns:
#   participant_id, frame, <11 feature columns...>
# """

# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# def run_elbow(X_scaled, max_k=10, random_state=42):
#     ks = list(range(2, max_k + 1))
#     inertias = []
#     for k in ks:
#         km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
#         km.fit(X_scaled)
#         inertias.append(km.inertia_)

#     plt.figure(figsize=(7, 5))
#     plt.plot(ks, inertias, marker="o")
#     plt.xlabel("k (number of clusters)")
#     plt.ylabel("Inertia (within-cluster SSE)")
#     plt.title("Elbow Method")
#     plt.grid(True, linestyle="--", alpha=0.4)
#     plt.show()

# def plot_pca_2d(X_scaled, clusters):
#     pca_2d = PCA(n_components=2, random_state=42)
#     Xp = pca_2d.fit_transform(X_scaled)

#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(Xp[:, 0], Xp[:, 1], c=clusters, cmap="tab10", alpha=0.75, s=18)
#     plt.xlabel("PCA 1")
#     plt.ylabel("PCA 2")
#     plt.title("2D PCA — KMeans Clusters")
#     handles, labels = sc.legend_elements(prop="colors", alpha=0.75)
#     plt.legend(handles, labels, title="Cluster", loc="best")
#     plt.tight_layout()
#     plt.show()

# def plot_pca_3d(X_scaled, clusters):
#     pca_3d = PCA(n_components=3, random_state=42)
#     Xp = pca_3d.fit_transform(X_scaled)

#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     sc = ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=clusters, cmap="tab10", alpha=0.8, s=18)
#     ax.set_xlabel("PCA 1")
#     ax.set_ylabel("PCA 2")
#     ax.set_zlabel("PCA 3")
#     ax.set_title("3D PCA — KMeans Clusters")
#     handles, labels = sc.legend_elements(prop="colors", alpha=0.8)
#     ax.legend(handles, labels, title="Cluster", loc="best")
#     plt.tight_layout()
#     plt.show()

# def main():
#     ap = argparse.ArgumentParser(description="KMeans clustering with elbow + PCA plots")
#     ap.add_argument("--csv", required=True, help="Path to input CSV")
#     ap.add_argument("--clusters", type=int, default=3, help="k for KMeans")
#     ap.add_argument("--max-k", type=int, default=10, help="Max k to test for elbow plot (>=2)")
#     ap.add_argument("--out", default="clustered_output.csv", help="Output CSV path")
#     ap.add_argument("--dropna", action="store_true",
#                     help="Drop rows with any NaNs in feature columns before clustering")
#     args = ap.parse_args()

#     # Load CSV
#     df = pd.read_csv(args.csv)

#     # Select features (exclude identifiers)
#     feature_cols = [c for c in df.columns if c not in ("participant_id", "frame")]
#     X = df[feature_cols].copy()

#     # Optional clean-up: drop rows with NaNs/Infs in features
#     if args.dropna:
#         X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
#         # keep df in sync with filtered X
#         df = df.loc[X.index].reset_index(drop=True)
#         X = X.reset_index(drop=True)
#     else:
#         # Replace inf with large finite values; fill remaining NaNs with column medians
#         X = X.replace([np.inf, -np.inf], np.nan)
#         X = X.fillna(X.median(numeric_only=True))

#     # Standardize
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # --- Elbow Method ---
#     if args.max_k >= 2:
#         run_elbow(X_scaled, max_k=args.max_k, random_state=42)

#     # --- KMeans ---
#     kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=42)
#     clusters = kmeans.fit_predict(X_scaled)
#     df["cluster"] = clusters

#     # Save
#     df.to_csv(args.out, index=False)
#     print(f"[OK] Clustering complete. Saved: {args.out}")

#     # --- Visualizations ---
#     plot_pca_2d(X_scaled, clusters)
#     plot_pca_3d(X_scaled, clusters)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-Means clustering on participant movement dataset with:
  - Standardization
  - Elbow Method (auto-picks optimal k via knee detection)
  - 2D & 3D PCA visualizations
  - Saves clustered CSV with labels

Assumes the CSV has columns:
  participant_id, frame, <11 feature columns...>
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --------------------------
# Utilities
# --------------------------
def kmeans_inertia_over_k(X_scaled, ks, random_state=42, n_init=10):
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return np.array(inertias, dtype=float)

def find_knee_via_max_distance(ks, inertias):
    """
    Finds elbow by the maximum perpendicular distance of points
    from the straight line joining (k_min, inertia_max) to (k_max, inertia_min).
    Returns: optimal_k (int), idx (index in ks), distances (np.ndarray)
    """
    x = np.array(ks, dtype=float)
    y = np.array(inertias, dtype=float)

    # Line endpoints
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # Vector from first to last point
    vec = np.array([x2 - x1, y2 - y1], dtype=float)
    # Normalize for distance calc
    norm = np.linalg.norm(vec)
    if norm == 0:
        # Degenerate (unlikely): fallback to smallest inertia
        idx = int(np.argmin(y))
        return int(x[idx]), idx, np.zeros_like(x)

    # Distances: area of parallelogram / base length
    distances = np.abs((vec[1] * (x - x1) - vec[0] * (y - y1))) / norm
    idx = int(np.argmax(distances))
    return int(x[idx]), idx, distances

def plot_elbow_with_knee(ks, inertias, optimal_k, idx, distances):
    plt.figure(figsize=(8, 6))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Method (auto-detected knee)")

    # Mark optimal k
    xk, yk = ks[idx], inertias[idx]
    plt.scatter([xk], [yk], s=80, zorder=5)
    plt.axvline(x=xk, linestyle="--", alpha=0.6)
    plt.text(xk, yk, f"  optimal k = {optimal_k}", va="bottom", fontsize=10)

    # (Optional) secondary info: show normalized distances on a twin axis for debugging
    # Comment out if you don’t want it
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(ks, distances / (distances.max() if distances.max() else 1.0),
             linestyle=":", alpha=0.4)
    ax2.set_ylabel("Normalized knee distance (aux)")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca_2d(X_scaled, clusters):
    pca_2d = PCA(n_components=2, random_state=42)
    Xp = pca_2d.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Xp[:, 0], Xp[:, 1], c=clusters, cmap="tab10", alpha=0.75, s=18)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("2D PCA — KMeans Clusters")
    handles, labels = sc.legend_elements(prop="colors", alpha=0.75)
    plt.legend(handles, labels, title="Cluster", loc="best")
    plt.tight_layout()
    plt.show()

def plot_pca_3d(X_scaled, clusters):
    pca_3d = PCA(n_components=3, random_state=42)
    Xp = pca_3d.fit_transform(X_scaled)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=clusters, cmap="tab10", alpha=0.8, s=18)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.set_title("3D PCA — KMeans Clusters")
    handles, labels = sc.legend_elements(prop="colors", alpha=0.8)
    ax.legend(handles, labels, title="Cluster", loc="best")
    plt.tight_layout()
    plt.show()

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="KMeans clustering with auto elbow + PCA plots")
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--k-min", type=int, default=2, help="Min k for Elbow search (>=2)")
    ap.add_argument("--k-max", type=int, default=10, help="Max k for Elbow search")
    ap.add_argument("--out", default="clustered_output.csv", help="Output CSV path")
    ap.add_argument("--dropna", action="store_true",
                    help="Drop rows with any NaNs in feature columns before clustering")
    ap.add_argument("--force-k", type=int, default=None,
                    help="Override: force a specific k instead of auto elbow")
    args = ap.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Select features (exclude identifiers)
    feature_cols = [c for c in df.columns if c not in ("participant_id", "frame")]
    X = df[feature_cols].copy()

    # Clean NaNs/Infs
    if args.dropna:
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        df = df.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
    else:
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow: compute inertias across ks and auto-pick knee ---
    k_min = max(args.k_min, 2)
    ks = list(range(k_min, max(k_min, args.k_max) + 1))
    inertias = kmeans_inertia_over_k(X_scaled, ks, random_state=42, n_init=10)
    optimal_k, idx, distances = find_knee_via_max_distance(ks, inertias)

    # Plot elbow with chosen k annotated
    plot_elbow_with_knee(ks, inertias, optimal_k, idx, distances)

    # Choose k to use
    k_to_use = args.force_k if args.force_k is not None else optimal_k
    print(f"[Elbow] Auto-selected optimal k = {optimal_k}")
    if args.force_k is not None:
        print(f"[Override] Forcing k = {k_to_use}")

    # --- Final KMeans with chosen k ---
    kmeans = KMeans(n_clusters=k_to_use, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    # Save labeled data
    df.to_csv(args.out, index=False)
    print(f"[OK] Clustering complete with k={k_to_use}. Saved: {args.out}")

    # --- Visualizations ---
    plot_pca_2d(X_scaled, clusters)
    plot_pca_3d(X_scaled, clusters)

if __name__ == "__main__":
    main()
