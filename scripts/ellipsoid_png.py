#!/usr/bin/env python3
"""
ellipsoid_png.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# ---------------- Tunables ----------------
FPS = 30
ELLIPSOID_SIGMA = 2.5
ELLIPSOID_ALPHA = 0.25
POINT_SIZE = 4
DPI = 300
ORTHO_VIEW = True
MARGIN_FACTOR = 1.3

ELLIPSOID_FACE_COLOR = "tab:blue"
ELLIPSOID_EDGE_COLOR = "k"
ELLIPSOID_EDGE_ALPHA = 0.30
OUTLIER_FADE_ALPHA = 0.1
OUTLIER_MARKER_SCALE = 0.5

COLORBAR_MAX_SECONDS = 1200
COLORBAR_STEP = 200

# ---- Participant sitting marker (absolute mm coordinates) ----
SHOW_SITTING_MARKER = True
SITTING_LABEL = "Participant's\nPosition"

# Absolute coordinates in mm
SITTING_START = np.array([250.0, 0.0, 750.0])   # where the arrow begins
SITTING_TIP   = np.array([250.0, 0.0, 700.0])  # where the arrow points

SITTING_LINE_COLOR = "k"
SITTING_LINE_WIDTH = 3.0
SITTING_TIP_COLOR  = "#1f1f1f"
SITTING_TIP_SIZE   = 180
SITTING_LABEL_FONTSIZE = 10

FIX_AXES_LIMITS = True   # set False for auto (ellipsoid-based)
X_RANGE = (0, 300)       # mm
Y_RANGE = (-350, 50)     # mm
Z_RANGE = (500, 800)     # mm (remember Z axis is flipped later)

# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="3D scatter + ellipsoid fit from landmark CSV")
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--landmarks", nargs="+", required=True,
                   help="Landmark IDs to include (e.g., 0 1 2 3 4)")
    p.add_argument("--hands", nargs="+", required=True, choices=["left", "right", "both"],
                   help="Which hands to include: left, right, or both")
    p.add_argument("--start", type=float, default=None,
                   help="Start time in seconds (based on 'frame' column @ 30 FPS)")
    p.add_argument("--stop", type=float, default=None,
                   help="Stop time in seconds (based on 'frame' column @ 30 FPS)")
    p.add_argument("--out", default=None, help="Output PNG path (default: cam2/plots/<csv_stem>_scatter_ellipsoid.png)")
    return p.parse_args()

def hands_to_letters(hands):
    if "both" in hands:
        return ["L", "R"]
    return ["L" if h.lower()=="left" else "R" for h in hands]

def ellipsoid_from_cov(mean: np.ndarray, cov: np.ndarray, k_sigma=2.0):
    evals, evecs = eigh(cov)
    evals = np.clip(evals, 0, None)
    radii = k_sigma * np.sqrt(evals)
    return mean, radii, evecs

def plot_ellipsoid(ax, center, radii, rotation, alpha=0.25, wire=False):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    XYZ = np.stack([x, y, z], axis=-1) @ rotation.T + center
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    if wire:
        return ax.plot_wireframe(X, Y, Z, rcount=20, ccount=12, linewidth=0.6)
    else:
        return ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0,
                               shade=True, alpha=alpha)

def add_sitting_marker(ax):
    """Draw bold arrow (line + tip dot) and label below the tip."""
    if not SHOW_SITTING_MARKER:
        return

    # Shaft
    ax.plot([SITTING_START[0], SITTING_TIP[0]],
            [SITTING_START[1], SITTING_TIP[1]],
            [SITTING_START[2], SITTING_TIP[2]],
            color=SITTING_LINE_COLOR, linewidth=SITTING_LINE_WIDTH, alpha=0.95)

    # Tip
    ax.scatter([SITTING_TIP[0]], [SITTING_TIP[1]], [SITTING_TIP[2]],
               s=SITTING_TIP_SIZE, c=SITTING_TIP_COLOR,
               edgecolors="white", linewidths=1.6, depthshade=False)

    # Label just below the tip in Z
    label_pos = SITTING_TIP.copy()
    label_pos[0] += 20  # move right in camera coords (+X right)
    label_pos[2] += 70  # move downward in camera coords (+Z down)
    ax.text(label_pos[0], label_pos[1], label_pos[2], SITTING_LABEL,
            fontsize=SITTING_LABEL_FONTSIZE, color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc="#3b3561", ec="none", alpha=0.9))

def main():
    args = parse_args()
    csv_path = Path(args.csv)

    if args.out:
        out_path = Path(args.out)
    else:
        cam2_dir = csv_path.parent.parent
        plots_dir = cam2_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f"{csv_path.stem}_scatter_ellipsoid.png"

    # Read CSV
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        raise ValueError("CSV must contain a 'frame' column.")

    start_f = int(args.start * FPS) if args.start is not None else None
    stop_f  = int(args.stop * FPS)  if args.stop  is not None else None

    mask = pd.Series(True, index=df.index)
    if start_f is not None: mask &= df["frame"] >= start_f
    if stop_f is not None:  mask &= df["frame"] <= stop_f
    df = df.loc[mask].copy()
    if df.empty: raise ValueError("No rows remain after applying start/stop filters.")

    lm_ids = [str(int(x)) for x in args.landmarks]
    hands_letters = hands_to_letters(args.hands)

    triplets = []
    for H in hands_letters:
        for lm in lm_ids:
            Xc, Yc, Zc = f"{H}_{lm}_X_mm", f"{H}_{lm}_Y_mm", f"{H}_{lm}_Z_mm"
            missing = [c for c in (Xc,Yc,Zc) if c not in df.columns]
            if missing: raise ValueError(f"Missing expected columns: {missing}")
            triplets.append((Xc, Yc, Zc))

    pts, frames = [], []
    for (Xc, Yc, Zc) in triplets:
        sub = df[["frame", Xc, Yc, Zc]].dropna()
        if not sub.empty:
            pts.append(sub[[Xc, Yc, Zc]].to_numpy(float))
            frames.append(sub["frame"].to_numpy(int))
    if not pts: raise ValueError("No valid 3D points found.")

    cloud = np.vstack(pts)
    frames_used = np.concatenate(frames)

    center = np.nanmean(cloud, axis=0)
    cov = np.cov(cloud.T, bias=False)
    center, radii, rotation = ellipsoid_from_cov(center, cov, k_sigma=ELLIPSOID_SIGMA)

    cov_inv = np.linalg.pinv(cov)
    diff = cloud - center
    mahal_sq = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
    inside_mask = np.sqrt(mahal_sq) <= ELLIPSOID_SIGMA

    cloud_in, cloud_out = cloud[inside_mask], cloud[~inside_mask]
    times_in, times_out = frames_used[inside_mask]/FPS, frames_used[~inside_mask]/FPS

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if ORTHO_VIEW:
        try: ax.set_proj_type('ortho')
        except: pass

    norm = plt.Normalize(vmin=0, vmax=COLORBAR_MAX_SECONDS)
    cmap = plt.cm.inferno

    ax.scatter(cloud_in[:,0], cloud_in[:,1], cloud_in[:,2],
               s=POINT_SIZE, c=cmap(norm(times_in)), alpha=0.9, depthshade=True)
    if cloud_out.size:
        ax.scatter(cloud_out[:,0], cloud_out[:,1], cloud_out[:,2],
                   s=POINT_SIZE*OUTLIER_MARKER_SCALE,
                   c=cmap(norm(times_out)), alpha=OUTLIER_FADE_ALPHA, depthshade=True)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Time (s)")
    cbar.set_ticks(np.arange(0, COLORBAR_MAX_SECONDS+1, COLORBAR_STEP))

    surf = plot_ellipsoid(ax, center, radii, rotation, alpha=ELLIPSOID_ALPHA, wire=False)
    try: surf.set_facecolor(ELLIPSOID_FACE_COLOR)
    except: ax.collections[-1].set_facecolor(ELLIPSOID_FACE_COLOR)

    wire = plot_ellipsoid(ax, center, radii, rotation, alpha=0.0, wire=True)
    try:
        wire.set_color(ELLIPSOID_EDGE_COLOR); wire.set_alpha(ELLIPSOID_EDGE_ALPHA)
    except: pass

    ax.set_title(f"Learner's Ellipsoid\nHands: {', '.join(hands_letters)} | Landmarks: {','.join(lm_ids)}")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")

    if FIX_AXES_LIMITS:
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Y_RANGE)
        ax.set_zlim(*Z_RANGE)
    else:
        ax.set_xlim(center[0]-MARGIN_FACTOR*radii[0], center[0]+MARGIN_FACTOR*radii[0])
        ax.set_ylim(center[1]-MARGIN_FACTOR*radii[1], center[1]+MARGIN_FACTOR*radii[1])
        ax.set_zlim(center[2]-MARGIN_FACTOR*radii[2], center[2]+MARGIN_FACTOR*radii[2])

    # Flips
    ax.set_zlim(ax.get_zlim()[1], ax.get_zlim()[0])
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    # ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])

    # Participant marker
    add_sitting_marker(ax)

    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)

if __name__ == "__main__":
    main()
