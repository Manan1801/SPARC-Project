#!/usr/bin/env python3
"""
plot_EnN_cumulative_plotly.py

Plot Experts vs. Novices cumulative movement in ONE interactive Plotly figure,
with optional piecewise linear regression overlays (from Script 1).

Usage:
python plot_EnN_cumulative_plotly.py --expert movement_E.csv --novice movement_N.csv \
    [--pidE ...] [--pidN ...] [--keypoints ...] [--regression] [--regwin N] [--maxsegs N] [--penalty P] [--minseg N]
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

def parse_args():
    p = argparse.ArgumentParser(description="Experts vs. Novices Plotly plot from cumulative movement CSVs.")
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--novice", type=Path, required=True)
    p.add_argument("--pidE", nargs="+", default=None)
    p.add_argument("--pidN", nargs="+", default=None)
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

def add_piecewise_fixed(fig, x_vals, y_vals, label, win_chunks):
    segcount = 0
    for start in range(0, len(x_vals), win_chunks):
        end = min(start + win_chunks, len(x_vals))
        if end - start < 2:
            continue
        a, b, yfit, r2 = compute_linfit(x_vals[start:end], y_vals[start:end])
        segcount += 1
        fig.add_trace(go.Scatter(
            x=x_vals[start:end], y=yfit, mode="lines",
            name=f"{label} fit {start+1}-{end}", line=dict(dash="dot"),
            hovertemplate=(f"Fit: {label}<br>"
                           "Time: %{x}s<br>"
                           "Fit value: %{y:.2f} mm<br>"
                           f"Slope: {a:.3f} mm/s<br>R²: {r2:.3f}")
        ))
    return segcount

def add_piecewise_auto(fig, x_vals, y_vals, label, maxsegs=8, penalty=None, minseg=3):
    x_arr = np.asarray(x_vals)
    y_arr = np.asarray(y_vals)
    if len(x_arr) < max(2, minseg * 2):
        return 0
    if rpt is None:
        a, b, yfit, r2 = compute_linfit(x_arr, y_arr)
        fig.add_trace(go.Scatter(
            x=x_arr, y=yfit, mode="lines", name=f"{label} (fit)",
            line=dict(dash="dot"),
            hovertemplate=(f"Fit: {label}<br>Time: %{{x}}s<br>Fit: %{{y:.2f}} mm<br>Slope: {a:.3f} mm/s<br>R²: {r2:.3f}")
        ))
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
        if e - s < max(2, minseg): continue
        a, b, yfit, r2 = compute_linfit(x_arr[s:e], y_arr[s:e])
        fig.add_trace(go.Scatter(
            x=x_arr[s:e], y=yfit, mode="lines", name=f"{label} fit {s+1}-{e}",
            line=dict(dash="dot"),
            hovertemplate=(f"Fit: {label}<br>Time: %{{x}}s<br>Fit: %{{y:.2f}} mm<br>Slope: {a:.3f} mm/s<br>R²: {r2:.3f}")
        ))
        segcount += 1
    return segcount

def main():
    args = parse_args()
    dfE = pd.read_csv(args.expert)
    dfN = pd.read_csv(args.novice)
    if "pid" not in dfE.columns or "pid" not in dfN.columns:
        raise ValueError("Missing 'pid' column.")

    groupsE, groupsN = extract_groups(dfE.columns), extract_groups(dfN.columns)
    if set(groupsE.keys()) != set(groupsN.keys()):
        raise ValueError("CSV structures mismatch between Experts and Novices.")
    groups = groupsE
    kp_labels = pick_keypoints(groups, args.keypoints)
    if not kp_labels:
        raise ValueError("No valid keypoints found.")

    if args.pidE:
        for pid in args.pidE:
            if pid not in dfE["pid"].values:
                raise ValueError(f"Expert PID not found: {pid}")
    if args.pidN:
        for pid in args.pidN:
            if pid not in dfN["pid"].values:
                raise ValueError(f"Novice PID not found: {pid}")

    num_parts = len(next(iter(groups.values())))
    x_secs = [i * 10 for i in range(1, num_parts + 1)]
    fig = go.Figure()
    cache = []

    for kp_label in kp_labels:
        h, k = kp_label[0], int(kp_label[1])
        cols = groups[(h, k)]
        yE = class_mean_series(dfE, cols, pids=args.pidE)
        yN = class_mean_series(dfN, cols, pids=args.pidN)

        fig.add_trace(go.Scatter(
            x=x_secs, y=yE, mode="lines+markers", name=f"Experts/Positive - {kp_label}",
            customdata=[[c] for c in range(1, num_parts + 1)],
            hovertemplate="Group: Exp/Pos<br>Keypoint: " + kp_label +
                          "<br>Chunk: %{customdata[0]}<br>Time: %{x}s<br>Mean: %{y:.2f} mm"
        ))
        fig.add_trace(go.Scatter(
            x=x_secs, y=yN, mode="lines+markers", name=f"Novices/Negative/Neutral - {kp_label}",
            customdata=[[c] for c in range(1, num_parts + 1)],
            hovertemplate="Group: Nov/Neg/Neu<br>Keypoint: " + kp_label +
                          "<br>Chunk: %{customdata[0]}<br>Time: %{x}s<br>Mean: %{y:.2f} mm"
        ))
        cache.append((f"Experts - {kp_label}", x_secs, yE))
        cache.append((f"Novices - {kp_label}", x_secs, yN))

    if args.regression:
        for label, xv, yv in cache:
            if args.regwin:
                add_piecewise_fixed(fig, xv, yv, label, args.regwin)
            else:
                add_piecewise_auto(fig, xv, yv, label,
                                   maxsegs=args.maxsegs,
                                   penalty=args.penalty,
                                   minseg=args.minseg)

    title_bits = ["Experts/Positive vs. Novices/Negative/Neutral"]
    title_bits.append("E/P: " + (", ".join(args.pidE) if args.pidE else "all"))
    title_bits.append("N: " + (", ".join(args.pidN) if args.pidN else "all"))
    title_bits.append("KP: " + ", ".join(kp_labels))
    fig.update_layout(
        title=" — ".join(title_bits),
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Movement (mm)",
        hovermode="x unified",
        legend_title_text="Group - Keypoint",
        template="plotly_white"
    )

    if args.save:
        out_path = args.save
    else:
        e_stem = args.expert.stem
        n_stem = args.novice.stem
        pidE_part = "Eall" if not args.pidE else "E" + "_".join(args.pidE)
        pidN_part = "Nall" if not args.pidN else "N" + "_".join(args.pidN)
        kp_part = "all" if not args.keypoints else "_".join(kp_labels)
        suffix = f"_regw{args.regwin}" if args.regwin else f"_regauto"
        out_path = args.expert.with_name(f"{e_stem}_vs_{n_stem}_{pidE_part}_{pidN_part}_{kp_part}_EvsN{suffix if args.regression else ''}.html")

    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"[OK] Saved interactive E-vs-N plot to {out_path}")

if __name__ == "__main__":
    main()
