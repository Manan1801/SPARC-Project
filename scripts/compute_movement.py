#!/usr/bin/env python3

'''
Compute 3D movement per keypoint for a participant.

Input: participant CSV file (e.g., .../<NN>/cam2/CSV/hand_landmark_xyz.csv)
Output: appends to movement_n_speed.csv (in current dir)
Also creates a debug log per participant if --debug is set.

Usage example:
    python scripts/compute_movement.py --csv_path ../data/01/cam2/CSV/hand_landmark_xyz.csv --start_time 10 --stop_time 70 --debug

Workflow:
  1) First-pass per-frame movement (strict Euclidean to last previous valid sample).
  2) Per-keypoint outlier detection via IQR (Q1, Q3; IQR=Q3-Q1; outliers < Q1-1.5*IQR or > Q3+1.5*IQR).
  3) Mark outlier frames as "missing" for THAT keypoint only.
  4) Recompute movement series/totals on the cleaned data.
  5) For each keypoint (h_k), compute speed using its own effective duration:
        effective_duration_{h_k} = raw_duration - (removed_frames_{h_k} / FPS)
     and speed_{h_k} = movement_{h_k} / effective_duration_{h_k}.
  6) Append totals and per-keypoint speeds to movement_n_speed.csv.

Requires: pandas, numpy

Assumes: input CSV has columns like L_0_X_mm, L_0_Y_mm, L_0_Z_mm, ..., R_4_X_mm, R_4_Y_mm, R_4_Z_mm
Keypoints: 0-4 for each hand (5 per hand, total 10)
Hands: 'L' (left) and 'R' (right)
FPS: 30 (frames per second)
Duration: stop_time - start_time (in seconds)

Output CSV columns:
Participant ID, Duration, L_0_movement, L_1_movement, L_2_movement, L_3_movement, L_4_movement,
R_0_movement, R_1_movement, R_2_movement, R_3_movement, R_4_movement,
L_0_norm_movement, L_1_norm_movement, L_2_norm_movement, L_3_norm_movement, L_4_norm_movement,
R_0_norm_movement, R_1_norm_movement, R_2_norm_movement, R_3_norm_movement, R_4_norm_movement
Where:
- movement = total 3D distance traveled by that keypoint during the time window (in mm)
- norm_movement = movement normalized by duration (in mm/s) 

Definition (strict):
- Per-frame distance for a keypoint = Euclidean distance (mm) between the current
  valid sample and the LAST PREVIOUS VALID sample of the SAME keypoint.
- Missing frames are skipped (no interpolation). The first ever valid sample gets 0.0.

- If --debug is passed:
    • Writes an END-OF-SESSION summary log to <NN>/cam2/logs/move_<NN>.log
    • Saves post-filter an INTERACTIVE per-frame movement plot to <NN>/cam2/plots/move_plot_<NN>.html
      (legend-click to show/hide keypoints)
    • Save pre-filter BOX & WHISKER plot (with outliers) to <NN>/cam2/plots/move_boxplot_<NN>.html
'''

import pandas as pd
import argparse
from pathlib import Path
import os
from typing import Optional, Tuple, Dict, List, Set
import math
import numpy as np  # for quantiles

KEYPOINTS = [0, 1, 2, 3, 4]
HANDS = ['L', 'R']
FPS = 30
OUT_CSV = 'movement_n_speed.csv'

def parse_args():
    p = argparse.ArgumentParser(description="Compute 3D movement per keypoint for a participant.")
    p.add_argument('--csv', required=True, help='Path to participant CSV (e.g., .../<NN>/cam2/CSV/hand_landmark_xyz.csv)')
    p.add_argument('--start', type=float, required=True, help='Start time in seconds')
    p.add_argument('--stop', type=float, required=True, help='Stop time in seconds')
    p.add_argument('--debug', action='store_true', help='If set, also write summary log & plots')
    return p.parse_args()

def find_numbered_folder(csv_path: Path) -> Optional[Path]:
    for p in [csv_path.parent, *csv_path.parents]:
        if p.name.isdigit():
            return p
    return None

def get_paths(csv_path: Path) -> Tuple[str, Path, Path]:
    numbered = find_numbered_folder(csv_path)
    if numbered is not None:
        pid = numbered.name
        base = numbered / 'cam2'
    else:
        pid = csv_path.stem
        base = csv_path.parent
    logs_dir = base / 'logs'
    plots_dir = base / 'plots'
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return pid, logs_dir, plots_dir

def euclidean_distance(p_xyz, q_xyz) -> float:
    """sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)"""
    dx = float(p_xyz[0]) - float(q_xyz[0])
    dy = float(p_xyz[1]) - float(q_xyz[1])
    dz = float(p_xyz[2]) - float(q_xyz[2])
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def compute_movement_series(df: pd.DataFrame, hand: str, kp: int,
                            treat_missing_idx: Optional[Set[int]] = None
                            ) -> Tuple[float, List[Optional[float]]]:
    """
    Compute total movement and per-frame series for a single keypoint.
    treat_missing_idx: row indices (within df) to be treated as missing for this keypoint (e.g., outliers).
    """
    cols = [f"{hand}_{kp}_X_mm", f"{hand}_{kp}_Y_mm", f"{hand}_{kp}_Z_mm"]
    coords = df[cols].copy()

    per_frame: List[Optional[float]] = [None] * len(coords)
    total = 0.0
    prev_valid_idx: Optional[int] = None

    tmiss = treat_missing_idx if treat_missing_idx is not None else set()

    for idx in range(len(coords)):
        if idx in tmiss:
            per_frame[idx] = None
            continue

        row = coords.iloc[idx]
        if row.isnull().any():
            per_frame[idx] = None
            continue

        if prev_valid_idx is None:
            per_frame[idx] = 0.0
        else:
            prev = coords.iloc[prev_valid_idx]
            dist = euclidean_distance(row.values, prev.values)
            per_frame[idx] = dist
            total += dist

        prev_valid_idx = idx

    return total, per_frame

def iqr_outliers(values: List[Optional[float]]) -> Set[int]:
    """
    Return indices (into 'values') that are outliers by IQR rule.
    None entries are ignored. Uses Q1, Q3 via numpy.percentile.
    """
    data = [v for v in values if v is not None]
    if len(data) == 0:
        return set()

    arr = np.array(data, dtype=float)
    # Use default linear interpolation method (NumPy 1.22+ uses 'method', older uses 'interpolation')
    try:
        q1 = np.percentile(arr, 25.0, method='linear')
        q3 = np.percentile(arr, 75.0, method='linear')
    except TypeError:
        q1 = np.percentile(arr, 25.0)
        q3 = np.percentile(arr, 75.0)

    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr

    out_idx = set()
    for i, v in enumerate(values):
        if v is None:
            continue
        if v < lo or v > hi:
            out_idx.add(i)
    return out_idx

def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()

    if args.stop <= args.start:
        raise ValueError("stop_time must be greater than start_time.")
    start_frame = int(args.start * FPS)
    stop_frame  = int(args.stop  * FPS)
    raw_duration = round(args.stop - args.start, 4)

    # Participant & dirs
    participant_id, logs_dir, plots_dir = get_paths(csv_path)
    debug_log_path = logs_dir / f"move_{participant_id}.log"
    debug_log = open(debug_log_path, 'w') if args.debug else None

    # Load CSV and slice using actual frame column
    df = pd.read_csv(csv_path)
    if 'frame' not in df.columns:
        raise ValueError("Input CSV must have a 'frame' column.")
    df = df[(df['frame'] >= start_frame) & (df['frame'] < stop_frame)].reset_index(drop=True)
    df.replace('', pd.NA, inplace=True)

    frame_numbers = df['frame'].tolist()

    # ---------- First pass: compute per-frame series (pre-filter) ----------
    per_kp_series_pre: Dict[str, List[Optional[float]]] = {}
    for hand in HANDS:
        for kp in KEYPOINTS:
            tag = f"{hand}_{kp}"
            _, series_pre = compute_movement_series(df, hand, kp, treat_missing_idx=None)
            per_kp_series_pre[tag] = series_pre

    # If --debug: save pre-filter box & whisker plot (shows outliers)
    if args.debug:
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception as e:
            print(f"[!] Plotly not available; skipping boxplot. ({e})")
            box_html_path = None
        else:
            fig_box = go.Figure()
            for hand in HANDS:
                for kp in KEYPOINTS:
                    tag = f"{hand}_{kp}"
                    vals = [v for v in per_kp_series_pre[tag] if v is not None]
                    fig_box.add_trace(go.Box(
                        y=vals if len(vals) > 0 else [None],
                        name=tag,
                        boxpoints='outliers',
                        jitter=0,
                        whiskerwidth=1.0
                    ))
            fig_box.update_layout(
                title=f"(Pre-filter) Per-frame Movement Distribution (mm) — Participant {participant_id}",
                xaxis_title="Keypoint",
                yaxis_title="Distance per frame (mm)",
                template='plotly_white',
                showlegend=False
            )
            box_html_path = plots_dir / f"move_boxplot_{participant_id}.html"
            pio.write_html(fig_box, file=str(box_html_path), auto_open=False, include_plotlyjs='cdn')
            print(f"[ℹ] Box & whisker plot saved to: {box_html_path}")
    else:
        box_html_path = None

    # ---------- Detect outliers per keypoint (IQR on pre-filter series) ----------
    outlier_idx_by_kp: Dict[str, Set[int]] = {}
    removed_frames_count_by_kp: Dict[str, int] = {}
    effective_duration_by_kp: Dict[str, float] = {}

    for hand in HANDS:
        for kp in KEYPOINTS:
            tag = f"{hand}_{kp}"
            idxs = iqr_outliers(per_kp_series_pre[tag])
            outlier_idx_by_kp[tag] = idxs
            removed_frames_count_by_kp[tag] = len(idxs)
            eff_dur = raw_duration - (len(idxs) / FPS)
            # Guardrail: never less than a tiny epsilon to avoid div-by-zero
            effective_duration_by_kp[tag] = max(eff_dur, 1e-9)

    # ---------- Second pass: recompute with outliers removed (treated as missing) ----------
    per_kp_total: Dict[str, float] = {}
    per_kp_series: Dict[str, List[Optional[float]]] = {}
    movement_values: List[float] = []  # order: [L_0..L_4, R_0..R_4]

    for hand in HANDS:
        for kp in KEYPOINTS:
            tag = f"{hand}_{kp}"
            total, series = compute_movement_series(df, hand, kp, treat_missing_idx=outlier_idx_by_kp[tag])
            per_kp_total[tag] = total
            per_kp_series[tag] = series
            movement_values.append(total)

    # Per-hand totals (movement only; speed shown against raw duration to avoid mixing durations)
    per_hand = {h: sum(per_kp_total[f"{h}_{k}"] for k in KEYPOINTS) for h in HANDS}
    grand_total = sum(movement_values)

    # ---------- Per-keypoint speeds using per-keypoint effective duration ----------
    per_kp_speed: Dict[str, float] = {}
    for hand in HANDS:
        for kp in KEYPOINTS:
            tag = f"{hand}_{kp}"
            per_kp_speed[tag] = round(per_kp_total[tag] / effective_duration_by_kp[tag], 4)

    # Build CSV row (speeds reflect per-kp effective durations)
    duration_for_csv = raw_duration  # Keep original duration column as raw window length
    output_row = [participant_id, duration_for_csv] \
                 + [round(per_kp_total[f"{h}_{k}"], 4) for h in HANDS for k in KEYPOINTS] \
                 + [per_kp_speed[f"{h}_{k}"] for h in HANDS for k in KEYPOINTS]

    header = ['Participant ID', 'Duration'] + \
             [f"{h}_{k}_move" for h in HANDS for k in KEYPOINTS] + \
             [f"{h}_{k}_speed" for h in HANDS for k in KEYPOINTS]

    file_exists = os.path.isfile(OUT_CSV)
    with open(OUT_CSV, 'a') as f:
        if not file_exists:
            f.write(','.join(header) + '\n')
        f.write(','.join(str(val) for val in output_row) + '\n')

    # ----- If --debug: save post-filter line plot & write outlier summary log -----
    line_html_path = None
    if args.debug:
        # Post-filter line plot (reflects cleaned data)
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception as e:
            print(f"[!] Plotly not available; skipping line plot. ({e})")
        else:
            fig_line = go.Figure()
            for hand in HANDS:
                for kp in KEYPOINTS:
                    tag = f"{hand}_{kp}"
                    y = per_kp_series[tag]
                    fig_line.add_trace(go.Scatter(
                        x=frame_numbers, y=y, mode='lines', name=tag,
                        hovertemplate="frame=%{x}<br>" + tag + "=%{y:.3f} mm<extra></extra>"
                    ))
            fig_line.update_layout(
                title=f"(Post-filter) Per-frame 3D Movement (mm) — Participant {participant_id}",
                xaxis_title="Frame",
                yaxis_title="Distance this frame (mm)",
                legend_title="Keypoints (click to toggle)",
                hovermode='x unified',
                template='plotly_white',
            )
            line_html_path = plots_dir / f"move_plot_{participant_id}.html"
            pio.write_html(fig_line, file=str(line_html_path), auto_open=False, include_plotlyjs='cdn')
            print(f"[ℹ] Line plot saved to: {line_html_path}")

        # Log end-of-session summary + outliers + per-kp effective durations
        debug_log.write(f"Participant: {participant_id}\n")
        debug_log.write(f"CSV: {csv_path}\n")
        debug_log.write(f"Window: start={args.start}s stop={args.stop}s raw_duration={raw_duration}s\n")
        debug_log.write(f"Frames: {start_frame}..{stop_frame-1} (FPS={FPS})\n\n")

        debug_log.write("===== OUTLIER REMOVAL (IQR rule) =====\n")
        total_removed = 0
        for hand in HANDS:
            for kp in KEYPOINTS:
                tag = f"{hand}_{kp}"
                idxs = sorted(outlier_idx_by_kp[tag])
                frames = [frame_numbers[i] for i in idxs]
                total_removed += len(frames)
                eff = effective_duration_by_kp[tag]
                debug_log.write(f"{tag}: removed {len(frames)} frame(s) | eff_dur={eff:.4f}s")
                if frames:
                    debug_log.write(" | frames=" + ",".join(str(fr) for fr in frames))
                debug_log.write("\n")
        debug_log.write(f"TOTAL removed (sum over keypoints): {total_removed}\n\n")

        debug_log.write("===== END OF SESSION SUMMARY =====\n")
        debug_log.write("Per-Hand Totals (movement mm | speed mm/s using raw_duration):\n")
        for h in HANDS:
            mv = per_hand[h]
            sp = mv / raw_duration if raw_duration > 0 else 0.0
            debug_log.write(f"  {h}: total={mv:.4f} | speed={sp:.4f}\n")
        debug_log.write(f"\nGrand Total Movement: {grand_total:.4f} mm | Avg Speed (raw_duration): {grand_total/raw_duration if raw_duration>0 else 0.0:.4f} mm/s\n\n")

        debug_log.write("Per-Keypoint Summary (movement mm | speed mm/s with per-kp eff. durations):\n")
        for h in HANDS:
            for k in KEYPOINTS:
                tag = f"{h}_{k}"
                mv = per_kp_total[tag]
                sp = per_kp_speed[tag]
                eff = effective_duration_by_kp[tag]
                debug_log.write(f"  {tag}: total={mv:.4f} | speed={sp:.4f} | eff_dur={eff:.4f}s\n")
        debug_log.write("===== SUMMARY END =====\n")
        debug_log.close()

    print(f"[✓] Movement data for {participant_id} appended to {OUT_CSV}.")
    if args.debug:
        print(f"[ℹ] Summary log saved to: {debug_log_path}")
        if box_html_path:
            print(f"[ℹ] Box & whisker plot saved to: {box_html_path}")
        if line_html_path:
            print(f"[ℹ] Line plot saved to: {line_html_path}")

if __name__ == "__main__":
    main()
