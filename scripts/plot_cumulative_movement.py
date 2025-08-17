#!/usr/bin/env python3
"""
plot_cumulative_movement_plotly.py

Interactive Plotly plots for cumulative movement CSVs.

- No --pid: average of ALL participants, one curve per selected keypoint.
- One --pid: that participant only, one curve per selected keypoint.
- Multiple --pid: average across listed PIDs (one curve per keypoint).

Regression overlays
- --regression with NO --regwin  => AUTO piecewise segmentation (PELT) + dotted fits
- --regression with --regwin N   => fixed-window piecewise (N chunks per segment) + dotted fits

File naming suffix:
- _pidavg when averaging across multiple PIDs
- _regautoK when auto mode is used (K segments detected)
- _regwN when fixed window mode is used
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
except Exception:
    rpt = None

COL_RE = re.compile(r'^(?P<hand>[LR])(?P<kp>[0-4])_p(?P<part>\d+)$')

def parse_args():
    p = argparse.ArgumentParser(description="Interactive Plotly curves from cumulative movement CSV.")
    p.add_argument("csv", type=Path, help="Path to cumulative movement CSV")
    p.add_argument("--pid", nargs="+", default=None,
                   help="One or more Participant IDs to plot; default: average of all participants")
    p.add_argument("--keypoints", nargs="+", default=None,
                   help="Keypoints to plot, e.g., L0 L2 R4. Default: all keypoints")
    p.add_argument("--save", type=Path, default=None,
                   help="Path to save HTML (overrides auto-naming)")

    # regression options
    p.add_argument("--regression", action="store_true",
                   help="Overlay dotted piecewise linear regression lines")
    p.add_argument("--regwin", type=int, default=None,
                   help="If set, use fixed windows of this many chunks for piecewise fits")
    p.add_argument("--maxsegs", type=int, default=8,
                   help="Auto mode: maximum number of segments (default 8)")
    p.add_argument("--penalty", type=float, default=None,
                   help="Auto mode: penalty strength for PELT; larger -> fewer segments")
    p.add_argument("--minseg", type=int, default=10,
                   help="Auto mode: minimum points per segment (default 10)")
    return p.parse_args()

def extract_groups(columns):
    groups = {}
    for c in columns:
        m = COL_RE.match(c)
        if m:
            hand = m.group("hand")
            kp = int(m.group("kp"))
            part = int(m.group("part"))
            groups.setdefault((hand, kp), []).append((c, part))
    for k in groups:
        groups[k].sort(key=lambda x: x[1])
        groups[k] = [name for name, _ in groups[k]]
    return groups

def pick_keypoints(groups, requested):
    all_labels = [f"{hand}{kp}" for (hand, kp) in sorted(groups.keys(), key=lambda x: (x[0], x[1]))]
    if not requested:
        return all_labels
    req = list(dict.fromkeys(requested))
    return [k for k in req if k in all_labels]

def compute_linfit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)           # y = a*x + b
    yfit = a * x + b
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, yfit, r2

def add_piecewise_fixed(fig, x_vals, y_vals, base_name, win_chunks):
    x_arr = np.asarray(x_vals, float)
    y_arr = np.asarray(y_vals, float)
    n = len(x_arr); segcount = 0
    for start in range(0, n, win_chunks):
        end = min(start + win_chunks, n)
        if end - start < 2:
            continue
        a, b, yfit, r2 = compute_linfit(x_arr[start:end], y_arr[start:end])
        segcount += 1
        fig.add_trace(go.Scatter(
            x=x_arr[start:end], y=yfit, mode="lines",
            name=f"{base_name} (fit {start+1}-{end})",
            line=dict(dash="dot", width=2),
            hovertemplate=(f"Series: {base_name} (fit {start+1}-{end})<br>"
                           "Time: %{x}s<br>"
                           "Fit value: %{y:.2f} mm<br>"
                           f"Slope: {a:.3f} mm/s<br>"
                           f"R²: {r2:.3f}")
        ))
    return segcount

def add_piecewise_auto(fig, x_vals, y_vals, base_name, maxsegs=8, penalty=None, minseg=3):
    x_arr = np.asarray(x_vals, float)
    y_arr = np.asarray(y_vals, float)
    n = len(x_arr)
    if n < max(2, minseg * 2):
        return 0

    if rpt is None:
        # Fallback: single dotted fit if ruptures isn't installed
        a, b, yfit, r2 = compute_linfit(x_arr, y_arr)
        fig.add_trace(go.Scatter(
            x=x_arr, y=yfit, mode="lines",
            name=f"{base_name} (fit 1-{n})",
            line=dict(dash="dot", width=2),
            hovertemplate=(f"Series: {base_name} (fit 1-{n})<br>"
                           "Time: %{x}s<br>"
                           "Fit value: %{y:.2f} mm<br>"
                           f"Slope: {a:.3f} mm/s<br>"
                           f"R²: {r2:.3f}")
        ))
        return 1

    signal = y_arr.reshape(-1, 1)
    model_name = "linear"

    # Try requested model, fallback to l2 if the installed ruptures lacks "linear"
    def _fit_algo(algo_cls, **kwargs):
        try:
            return algo_cls(model=model_name, min_size=max(minseg, 2), **kwargs).fit(signal)
        except Exception:
            return algo_cls(model="l2", min_size=max(minseg, 2), **kwargs).fit(signal)

    # Decide algorithm based on whether penalty is provided
    if penalty is not None:
        # PELT with penalty-driven segmentation
        algo = _fit_algo(rpt.Pelt)
        bkps = algo.predict(pen=penalty)  # returns end indices (1-based), includes n
    else:
        # Dynamic programming with explicit number of breakpoints
        # Cap segments by data length and minseg
        k_segments = max(1, min(maxsegs, n // max(minseg, 2)))
        algo = _fit_algo(rpt.Dynp)
        bkps = algo.predict(n_bkps=max(0, k_segments - 1))

    # Convert to segments [start:end) in 0-based indexing
    starts = [0] + bkps[:-1]
    ends = bkps

    segcount = 0
    for s, e in zip(starts, ends):
        if e - s < max(2, minseg):
            continue
        xs = x_arr[s:e]
        ys = y_arr[s:e]
        a, b, yfit, r2 = compute_linfit(xs, ys)
        segcount += 1
        fig.add_trace(go.Scatter(
            x=xs, y=yfit, mode="lines",
            name=f"{base_name} (fit {s+1}-{e})",
            line=dict(dash="dot", width=2),
            hovertemplate=(f"Series: {base_name} (fit {s+1}-{e})<br>"
                           "Time: %{x}s<br>"
                           "Fit value: %{y:.2f} mm<br>"
                           f"Slope: {a:.3f} mm/s<br>"
                           f"R²: {r2:.3f}")
        ))
    return segcount


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if "pid" not in df.columns:
        raise ValueError("Expected a 'pid' column in the CSV.")

    groups = extract_groups(df.columns)
    kp_labels = pick_keypoints(groups, args.keypoints)
    if not kp_labels:
        raise ValueError("No valid keypoints selected/found.")

    # pick participants
    if args.pid and len(args.pid) > 0:
        missing = [pid for pid in args.pid if pid not in df["pid"].values]
        if missing:
            raise ValueError(f"Participant IDs not found in {args.csv}: {', '.join(missing)}")
        pids = args.pid
    else:
        pids = None

    # x-axis: infer parts, convert to seconds (10s per chunk)
    (sample_hand, sample_kp) = next(iter(groups.keys()))
    num_parts = len(groups[(sample_hand, sample_kp)])
    chunks = list(range(1, num_parts + 1))
    x_secs = [i * 10 for i in chunks]

    fig = go.Figure()
    cache = []  # (name, x, y)

    if pids is None:
        avg_row = df.drop(columns=["pid"]).mean(numeric_only=True)
        title_text = "Cumulative Movement — Average of all participants"
        prefix = "Average"
        for kp_label in kp_labels:
            h, k = kp_label[0], int(kp_label[1])
            cols = groups[(h, k)]
            y = avg_row[cols].values.astype(float)
            name = f"{prefix} - {kp_label}"
            fig.add_trace(go.Scatter(
                x=x_secs, y=y, mode="lines+markers", name=name,
                hovertemplate=("Series: " + name + "<br>"
                               "Chunk: %{customdata[0]}<br>"
                               "Time: %{x}s<br>"
                               "Cumulative: %{y:.2f} mm"),
                customdata=[[c] for c in chunks]
            ))
            cache.append((name, x_secs, y))
        pid_part = "avg"; suffix = ""

    elif len(pids) == 1:
        pid = pids[0]
        row = df[df["pid"] == pid].iloc[0]
        title_text = f"Cumulative Movement — {pid}"
        prefix = pid
        for kp_label in kp_labels:
            h, k = kp_label[0], int(kp_label[1])
            cols = groups[(h, k)]
            y = row[cols].values.astype(float)
            name = f"{prefix} - {kp_label}"
            fig.add_trace(go.Scatter(
                x=x_secs, y=y, mode="lines+markers", name=name,
                hovertemplate=("Series: " + name + "<br>"
                               "Chunk: %{customdata[0]}<br>"
                               "Time: %{x}s<br>"
                               "Cumulative: %{y:.2f} mm"),
                customdata=[[c] for c in chunks]
            ))
            cache.append((name, x_secs, y))
        pid_part = pid; suffix = ""

    else:
        df_sel = df[df["pid"].isin(pids)].reset_index(drop=True)
        title_text = f"Cumulative Movement — mean across PIDs ({', '.join(pids)})"
        prefix = "PID-mean"
        for kp_label in kp_labels:
            h, k = kp_label[0], int(kp_label[1])
            cols = groups[(h, k)]
            y = df_sel[cols].astype(float).mean(axis=0, skipna=True).values
            name = f"{prefix} - {kp_label}"
            fig.add_trace(go.Scatter(
                x=x_secs, y=y, mode="lines+markers", name=name,
                hovertemplate=("Series: " + name + "<br>"
                               "PIDs: " + ", ".join(pids) + "<br>"
                               "Chunk: %{customdata[0]}<br>"
                               "Time: %{x}s<br>"
                               "Mean cumulative: %{y:.2f} mm"),
                customdata=[[c] for c in chunks]
            ))
            cache.append((name, x_secs, y))
        pid_part = "_".join(pids); suffix = "_pidavg"

    # Overlays
    seg_suffix = ""
    if args.regression:
        total_segments = 0
        if args.regwin is not None:
            # fixed-window piecewise
            for name, xv, yv in cache:
                total_segments += add_piecewise_fixed(fig, xv, yv, name, args.regwin)
            seg_suffix = f"_regw{args.regwin}"
        else:
            # AUTO segmentation (PELT)
            for name, xv, yv in cache:
                total_segments += add_piecewise_auto(
                    fig, xv, yv, name,
                    maxsegs=args.maxsegs, penalty=args.penalty, minseg=args.minseg
                )
            seg_suffix = f"_regauto{total_segments if total_segments>0 else ''}"

        suffix += seg_suffix

    fig.update_layout(
        title=title_text,
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Movement (mm)",
        hovermode="x unified",
        legend_title_text="Series (per keypoint)",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
    )

    # Naming
    if args.save:
        out_path = args.save
    else:
        kp_part = "all" if not args.keypoints else "_".join(kp_labels)
        out_path = args.csv.with_name(f"{args.csv.stem}_{pid_part}_{kp_part}{suffix}.html")

    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"[OK] Saved interactive plot to {out_path}")

if __name__ == "__main__":
    main()
