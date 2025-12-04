#!/usr/bin/env python3
"""
plot_LvNL_cumulative_plotly.py

Plot Learners vs. Non-Learners cumulative movement in ONE interactive Plotly figure,
with optional piecewise linear regression overlays — GREYSCALE with distinct class marker shapes.

Usage:
python plot_LvNL_cumulative_plotly.py --learner movement_L.csv --nonlearner movement_NL.csv \
    [--pidL ...] [--pidNL ...] [--keypoints ...] [--regression] [--regwin N] [--maxsegs N] [--penalty P] [--minseg N] [--save]
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# optional import for auto segmentation
try:
    import ruptures as rpt
except ImportError:
    rpt = None

COL_RE = re.compile(r'^(?P<hand>[LR])(?P<kp>[0-4])_p(?P<part>\d+)$')

# =========================
# Greyscale + marker scheme
# =========================
# Per-keypoint paired shades: Learners get slightly darker; Non-Learners get lighter.
L_GREYS  = ["#202020", "#303030", "#404040", "#505050", "#606060", "#707070", "#808080"]
NL_GREYS = ["#9A9A9A", "#A8A8A8", "#B6B6B6", "#C4C4C4", "#D2D2D2", "#E0E0E0", "#EAEAEA"]

# Distinct marker sets PER CLASS so they differ even for the same keypoint.
# L_MARKERS  = ["circle", "square", "diamond", "triangle-up", "pentagon", "hexagon", "star"]
# NL_MARKERS = ["x", "cross", "triangle-down", "triangle-left", "triangle-right", "hourglass", "bowtie"]
L_MARKERS  = ["circle", "square"]
NL_MARKERS = ["circle-open", "square-open"]

def parse_args():
    p = argparse.ArgumentParser(description="Learners vs. Non-Learners Plotly plot from cumulative movement CSVs.")
    p.add_argument("--learner", type=Path, required=True, help="Path to Learners cumulative CSV")
    p.add_argument("--nonlearner", type=Path, required=True, help="Path to Non-Learners cumulative CSV")
    p.add_argument("--pidL", nargs="+", default=None, help="Learner PIDs to include (default: all Learners)")
    p.add_argument("--pidNL", nargs="+", default=None, help="Non-Learner PIDs to include (default: all Non-Learners)")
    p.add_argument("--keypoints", nargs="+", default=None)
    p.add_argument("--save", type=Path, default=None)

    # regression options
    p.add_argument("--regression", action="store_true", help="Overlay piecewise linear regression lines")
    p.add_argument("--regwin", type=int, default=None, help="Fixed-window size (chunks)")
    p.add_argument("--maxsegs", type=int, default=8, help="Max segments for auto mode")
    p.add_argument("--penalty", type=float, default=None, help="Penalty for auto mode (PELT)")
    p.add_argument("--minseg", type=int, default=10, help="Min points per segment")
    return p.parse_args()

def extract_groups(columns):
    groups = {}
    for c in columns:
        m = COL_RE.match(c)
        if m:
            hand, kp, part = m.group("hand"), int(m.group("kp")), int(m.group("part"))
            groups.setdefault((hand, kp), []).append((c, part))
    for k in groups:
        groups[k].sort(key=lambda x: x[1])
        groups[k] = [name for name, _ in groups[k]]
    return groups

def pick_keypoints(groups, requested):
    all_labels = [f"{h}{k}" for (h, k) in sorted(groups.keys())]
    if not requested:
        return all_labels
    req = list(dict.fromkeys(requested))
    return [k for k in req if k in all_labels]

def class_mean_series(df, cols, pids=None):
    if pids is not None:
        df = df[df["pid"].isin(pids)]
        if df.empty:
            raise ValueError("After filtering by PIDs, no rows remain for this class.")
    return df[cols].astype(float).mean(axis=0, skipna=True).values

def compute_linfit(x, y):
    x = np.asarray(x); y = np.asarray(y)
    a, b = np.polyfit(x, y, 1)
    yfit = a * x + b
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, yfit, r2

# ---- Label laneing helper: staggers annotations per x bucket ----
class _LanePlacer:
    """
    Assigns each annotation to a 'lane' near its x so labels don't overlap.
    Uses small pixel x/y shifts that are invisible on the data scale but
    separate stacked labels cleanly.
    """
    def __init__(self, x_tolerance=22, y_step=28, x_step=10):
        # x_tolerance in axis units (seconds here) to group nearby midpoints
        self.x_tolerance = x_tolerance
        self.y_step = y_step
        self.x_step = x_step
        self._buckets = {}  # bucket_x -> next lane index

        # lane patterns: 0 is center, then up/down, then farther up/down, etc.
        self._y_mult = [0, +1, -1, +2, -2, +3, -3, +4, -4]
        self._x_mult = [0, +1, -1, -1, +1, +2, -2, -2, +2]

    def assign(self, x):
        # snap to an existing bucket if within tolerance
        bucket_x = None
        for bx in self._buckets.keys():
            if abs(bx - x) <= self.x_tolerance:
                bucket_x = bx
                break
        if bucket_x is None:
            bucket_x = x
            self._buckets[bucket_x] = 0

        lane = self._buckets[bucket_x]
        self._buckets[bucket_x] = (lane + 1) % len(self._y_mult)

        yshift = self._y_mult[lane] * self.y_step
        xshift = self._x_mult[lane] * self.x_step
        return x, xshift, yshift

# single global placer used by all annotations
_PLACER = _LanePlacer()

def _add_slope_annotation(fig, x_seg, a, b, y_context, label=None):
    """
    Place slope label without overlap:

    • Non-Learners  -> always treated as the UPPER line
                       → only bottom-right corner of box touches the line.
    • Learners      -> always treated as the LOWER line
                       → only top-left corner of box touches the line.
    """

    x0, x1 = float(x_seg[0]), float(x_seg[-1])
    xm = 0.5 * (x0 + x1)
    ym = a * xm + b

    label = label or ""

    if "Nlrn" in label:
        # Upper line (Non-Learners): bottom-right corner on the line
        # box extends up and left from (xm, ym)
        xanchor = "right"
        yanchor = "bottom"
        xshift = -10
        yshift = -10
    elif "Lrn" in label:
        # Lower line (Learners): top-left corner on the line
        # box extends down and right from (xm, ym)
        xanchor = "left"
        yanchor = "top"
        xshift = 10
        yshift = 10
    else:
        # Fallback (if something else ever appears)
        xanchor = "center"
        yanchor = "middle"
        xshift = 0
        yshift = 0

    fig.add_annotation(
        x=xm,
        y=ym,
        text=f"m={a:.3f} mm/s",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=28, family="Times New Roman", color="black"),  # bigger slope text
        align="center",
        xanchor=xanchor,
        yanchor=yanchor,
        xshift=xshift,
        yshift=yshift,
    )


def add_piecewise_fixed(fig, x_vals, y_vals, label, win_chunks, color):
    segcount = 0
    for start in range(0, len(x_vals), win_chunks):
        end = min(start + win_chunks, len(x_vals))
        if end - start < 2:
            continue
        a, b, yfit, r2 = compute_linfit(x_vals[start:end], y_vals[start:end])
        segcount += 1
        fig.add_trace(go.Scatter(
            x=x_vals[start:end], y=yfit, mode="lines",
            name=f"{label} fit {start+1}-{end}",
            line=dict(dash="dot", color=color),
            showlegend=False,
            hovertemplate=(f"Fit: {label}<br>"
                           "Time: %{{x}}s<br>"
                           "Fit value: %{{y:.2f}} mm<br>"
                           f"Slope: {a:.3f} mm/s<br>R²: {r2:.3f}")
        ))
        _add_slope_annotation(fig, x_vals[start:end], a, b, y_vals, label)
    return segcount

def add_piecewise_auto(fig, x_vals, y_vals, label, color, maxsegs=8, penalty=None, minseg=3):
    x_arr = np.asarray(x_vals)
    y_arr = np.asarray(y_vals)
    if len(x_arr) < max(2, minseg * 2):
        return 0
    if rpt is None:
        a, b, yfit, r2 = compute_linfit(x_arr, y_arr)
        fig.add_trace(go.Scatter(
            x=x_arr, y=yfit, mode="lines", name=f"{label} (fit)",
            showlegend=False,
            line=dict(dash="dot", color=color),
            hovertemplate=(f"Fit: {label}<br>Time: %{{x}}s<br>Fit: %{{y:.2f}} mm<br>Slope: {a:.3f} mm/s<br>R²: {r2:.3f}")
        ))
        _add_slope_annotation(fig, x_arr, a, b, y_arr)
        return 1

    signal = y_arr.reshape(-1, 1)
    model_name = "linear"
    def _fit_algo(algo_cls, **kwargs):
        try:
            return algo_cls(model=model_name, min_size=max(minseg, 2), **kwargs).fit(signal)
        except:
            return algo_cls(model="l2", min_size=max(minseg, 2), **kwargs).fit(signal)

    if penalty is not None:
        algo = _fit_algo(rpt.Pelt)
        bkps = algo.predict(pen=penalty)
    else:
        k_segments = max(1, min(maxsegs, len(x_arr) // max(minseg, 2)))
        algo = _fit_algo(rpt.Dynp)
        bkps = algo.predict(n_bkps=max(0, k_segments - 1))

    starts = [0] + bkps[:-1]
    ends = bkps
    segcount = 0
    for s, e in zip(starts, ends):
        if e - s < max(2, minseg):
            continue
        a, b, yfit, r2 = compute_linfit(x_arr[s:e], y_arr[s:e])
        fig.add_trace(go.Scatter(
            x=x_arr[s:e], y=yfit, mode="lines", name=f"{label} fit {s+1}-{e}",
            showlegend=False,
            line=dict(dash="dot", color=color),
            hovertemplate=(f"Fit: {label}<br>Time: %{{x}}s<br>Fit: %{{y:.2f}} mm<br>Slope: {a:.3f} mm/s<br>R²: {r2:.3f}")
        ))
        _add_slope_annotation(fig, x_arr[s:e], a, b, y_arr, label)
        segcount += 1
    return segcount


def main():
    args = parse_args()
    dfL = pd.read_csv(args.learner)
    dfNL = pd.read_csv(args.nonlearner)
    if "pid" not in dfL.columns or "pid" not in dfNL.columns:
        raise ValueError("Missing 'pid' column.")

    groupsL, groupsNL = extract_groups(dfL.columns), extract_groups(dfNL.columns)
    if set(groupsL.keys()) != set(groupsNL.keys()):
        raise ValueError("CSV structures mismatch between Learners and Non-Learners.")
    groups = groupsL
    kp_labels = pick_keypoints(groups, args.keypoints)
    if not kp_labels:
        raise ValueError("No valid keypoints found.")

    if args.pidL:
        for pid in args.pidL:
            if pid not in dfL["pid"].values:
                raise ValueError(f"Learner PID not found: {pid}")
    if args.pidNL:
        for pid in args.pidNL:
            if pid not in dfNL["pid"].values:
                raise ValueError(f"Non-Learner PID not found: {pid}")

    num_parts = len(next(iter(groups.values())))
    x_secs = [i * 10 for i in range(1, num_parts + 1)]
    fig = go.Figure()
    cache = []

    # Style index so each keypoint pair gets its own shades + markers
    style_idx = 0

    for kp_label in kp_labels:
        h, k = kp_label[0], int(kp_label[1])

        # Display label normalization from original code
        display_label = kp_label
        if kp_label[0] == 'L':
            display_label = 'R' + kp_label[1:]
        elif kp_label[0] == 'R':
            display_label = 'L' + kp_label[1:]

        # Choose paired colors + class-specific markers for this keypoint index
        lc = L_GREYS[style_idx % len(L_GREYS)]
        nlc = NL_GREYS[style_idx % len(NL_GREYS)]
        lmarker = L_MARKERS[style_idx % len(L_MARKERS)]
        nlmarker = NL_MARKERS[style_idx % len(NL_MARKERS)]
        style_idx += 1

        cols = groups[(h, k)]
        yL = class_mean_series(dfL, cols, pids=args.pidL)
        yNL = class_mean_series(dfNL, cols, pids=args.pidNL)

        # Learners — solid line, class-specific marker + darker grey
        fig.add_trace(go.Scatter(
            x=x_secs, y=yL, mode="lines+markers", name=f"{display_label} (Lrn)",
            line=dict(color=lc),  # solid
            marker=dict(symbol=lmarker, color=lc),
            customdata=[[c] for c in range(1, num_parts + 1)],
            hovertemplate="Group: Learners<br>Keypoint: " + display_label +
                          "<br>Chunk: %{customdata[0]}<br>Time: %{x}s<br>Mean: %{y:.2f} mm"
        ))

        # Non-Learners — solid line, different class marker + lighter grey
        fig.add_trace(go.Scatter(
            x=x_secs, y=yNL, mode="lines+markers", name=f"{display_label} (Nlrn)",
            line=dict(color=nlc),  # solid
            marker=dict(symbol=nlmarker, color=nlc),
            customdata=[[c] for c in range(1, num_parts + 1)],
            hovertemplate="Group: Non-Learners<br>Keypoint: " + display_label +
                          "<br>Chunk: %{customdata[0]}<br>Time: %{x}s<br>Mean: %{y:.2f} mm"
        ))

        # Cache for regression overlays (each class keeps its own shade)
        cache.append((f"{display_label} (Lrn)", x_secs, yL, lc))
        cache.append((f"{display_label} (Nlrn)", x_secs, yNL, nlc))

    # regression overlays — dotted, in same class shade so distinguishable
    if args.regression:
        for label, xv, yv, color in cache:
            if args.regwin:
                add_piecewise_fixed(fig, xv, yv, label, args.regwin, color)
            else:
                add_piecewise_auto(fig, xv, yv, label,
                                   color=color,
                                   maxsegs=args.maxsegs,
                                   penalty=args.penalty,
                                   minseg=args.minseg)

    title_bits = ["Learners vs. Non-Learners"]
    fig.update_layout(
        font=dict(size=20, family="Times New Roman", color="black"),  # base font size
        title=dict(
            text=" — ".join(title_bits),
            font=dict(size=32)    # plot title
        ),
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(size=24)),
            tickfont=dict(size=22)
        ),
        yaxis=dict(
            title=dict(text="Cumulative Movement (mm)", font=dict(size=24)),
            tickfont=dict(size=22),
            nticks=6,              # FEWER y-gridlines (default ~10 → now 6)
            gridcolor="rgba(0,0,0,0.1)",
            gridwidth=0.5,
        ),
        width=1000,
        height=800,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            title="Series",
            font=dict(size=24),
            title_font=dict(size=26),
            x=0.1, y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        )
    )


    if args.save:
        out_path = args.save
    else:
        l_stem = args.learner.stem
        nl_stem = args.nonlearner.stem
        pidL_part = "Lall" if not args.pidL else "L" + "_".join(args.pidL)
        pidNL_part = "NLall" if not args.pidNL else "NL" + "_".join(args.pidNL)
        kp_part = "all" if not args.keypoints else "_".join(kp_labels)
        suffix = f"_regw{args.regwin}" if args.regwin else f"_regauto"
        out_path = args.learner.with_name(f"{l_stem}_vs_{nl_stem}_{pidL_part}_{pidNL_part}_{kp_part}_LvNL{suffix if args.regression else ''}.html")

    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"[OK] Saved interactive plot to {out_path}")

if __name__ == "__main__":
    main()
