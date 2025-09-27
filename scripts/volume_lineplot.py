#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------- Config ----------
ALLOWED_KPS = ["L0", "R0", "L4", "R4"]
GROUP_ALIASES = {
    "p": "Learner",
    "n": "Non-Learner",
    "learner": "Learner",
    "non-learner": "Non-Learner",
    "non_learner": "Non-Learner",
    "nonlearner": "Non-Learner",
}

# Colors per keypoint (distinct)
KP_COLORS = {
    "L0": "#1f77b4",  # blue
    "R0": "#d62728",  # red
    "L4": "#2ca02c",  # green
    "R4": "#9467bd",  # purple
}

# Line style + marker per group
GROUP_STYLE = {
    "Learner": ("-", "o"),
    "Non-Learner": ("--", "s"),
}

# Matplotlib defaults (bolder lines/markers)
LINE_WIDTH = 2.5
MARKER_SIZE = 6

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Line plot of ellipsoid volumes across segments for selected keypoints and groups (clean dual legends)."
    )
    p.add_argument("--csv", required=True, help="Input CSV with participant_id, Learning Gain, and vol_* columns")
    p.add_argument("--keypoints", nargs="+", required=True, choices=ALLOWED_KPS,
                   help="One or more keypoints (e.g., L0 R0)")
    p.add_argument("--groups", nargs="+", required=True,
                   help='One or more groups: "Learner", "Non-Learner", or aliases P/N')
    p.add_argument("--agg", choices=["mean", "median"], default="mean",
                   help="Aggregation across participants per segment (default: mean)")
    p.add_argument("--out", default=None, help="Optional output image path (PNG, PDF, etc.)")
    p.add_argument("--title", default=None, help="Optional custom plot title")
    p.add_argument("--ylog", action="store_true", help="Use log scale on Y axis")
    return p.parse_args()

# ---------- Helpers ----------
def normalize_groups(groups_raw):
    norm = []
    for g in groups_raw:
        key = str(g).strip().lower()
        if key in GROUP_ALIASES:
            norm.append(GROUP_ALIASES[key])
        elif key in ("learner", "non-learner"):
            norm.append("Learner" if key == "learner" else "Non-Learner")
        else:
            raise ValueError(f"Unknown group: {g} (use Learner / Non-Learner / P / N)")
    # dedupe preserving order
    out, seen = [], set()
    for g in norm:
        if g not in seen:
            out.append(g); seen.add(g)
    return out

def autodetect_segments(columns, keypoints):
    segs = set()
    for kp in keypoints:
        pat = re.compile(rf"^vol_{re.escape(kp)}_seg(\d+)$", re.IGNORECASE)
        for c in columns:
            m = pat.match(str(c))
            if m:
                segs.add(int(m.group(1)))
    if not segs:
        raise RuntimeError("No vol_*_seg* columns found for the selected keypoints.")
    return sorted(segs)

def aggregate_series(df_group, kp, segments, agg="mean"):
    cols, segs_used = [], []
    for s in segments:
        col = f"vol_{kp}_seg{s}"
        if col in df_group.columns:
            cols.append(col); segs_used.append(s)
    if not cols:
        return np.array([]), np.array([])

    # Convert to numeric
    vals = df_group[cols].apply(pd.to_numeric, errors="coerce")

    # 🔴 New: drop participants with ANY NaN across selected segments
    vals = vals.dropna(axis=0, how="any")
    if vals.empty:
        return np.array(segs_used), np.full(len(segs_used), np.nan)

    if agg == "mean":
        y = vals.mean(axis=0).to_numpy()
    else:
        y = vals.median(axis=0).to_numpy()
    return np.array(segs_used), y


# ---------- Main ----------
def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    if "Learning Gain" not in df.columns:
        raise KeyError("Input CSV must contain 'Learning Gain' column")

    # Normalize Learning Gain labels to standard names
    df["Learning Gain"] = df["Learning Gain"].astype(str).str.strip()
    df["Learning Gain"] = df["Learning Gain"].map(lambda x: GROUP_ALIASES.get(x.lower(), x))
    df["Learning Gain"] = df["Learning Gain"].replace({"P": "Learner", "N": "Non-Learner"})

    groups = normalize_groups(args.groups)
    segments = autodetect_segments(df.columns, args.keypoints)

    plt.figure(figsize=(10, 6))

    plotted_any = False
    for group in groups:
        df_g = df[df["Learning Gain"] == group]
        if df_g.empty:
            print(f"⚠️ No rows for group: {group}")
            continue

        for kp in args.keypoints:
            xs, ys = aggregate_series(df_g, kp, segments, agg=args.agg)
            if xs.size == 0:
                print(f"⚠️ No valid columns for keypoint {kp} in group {group}")
                continue

            ls, mk = GROUP_STYLE.get(group, ("-", "o"))
            color = KP_COLORS.get(kp, None)
            plt.plot(
                xs, ys,
                linestyle=ls, marker=mk, color=color,
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                label=f"{kp} — {group}"  # not used in final legend; kept for debugging
            )
            plotted_any = True

    if not plotted_any:
        print("❌ Nothing to plot (check groups/keypoints/columns).")
        return

    # Axes/Title
    plt.xticks(segments)
    plt.xlabel("Segment")
    plt.ylabel("Ellipsoid Volume (mm³)")
    if args.ylog:
        plt.yscale("log")
    plt.title(args.title or "Ellipsoid Volumes across Segments")
    plt.grid(True, alpha=0.35)

    # ---- Dual legends: colors (keypoints) and linestyles (groups) ----
    # Legend A: keypoints (color meaning)
    kp_handles = [
        Line2D([0], [0], color=KP_COLORS[kp], lw=LINE_WIDTH, label=kp)
        for kp in args.keypoints
    ]
    leg_kp = plt.legend(handles=kp_handles, title="Keypoints", loc="upper left", frameon=True)
    plt.gca().add_artist(leg_kp)

    # Legend B: groups (linestyle/marker meaning) — only for groups actually requested
    grp_handles = []
    for grp in groups:
        ls, mk = GROUP_STYLE.get(grp, ("-", "o"))
        grp_handles.append(Line2D([0], [0], color="black", lw=LINE_WIDTH, linestyle=ls, marker=mk, markersize=MARKER_SIZE, label=grp))
    plt.legend(handles=grp_handles, title="Groups", loc="upper right", frameon=True)

    # Layout
    plt.tight_layout()

    # Output
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
