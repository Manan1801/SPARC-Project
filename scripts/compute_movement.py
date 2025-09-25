#!/usr/bin/env python3

'''
Compute 3D movement per keypoint for a participant.

Input: participant CSV file (e.g., .../<NN>/cam2/CSV/hand_landmark_xyz.csv)
Output: appends to movement_n_speed.csv (in current dir)
Also creates a debug log per participant if --debug is set.

Usage example:
    python scripts/compute_movement.py --csv_path ../data/01/cam2/CSV/hand_landmark_xyz.csv --start_time 10 --stop_time 70 --debug

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
    • Saves an INTERACTIVE per-frame movement plot to <NN>/cam2/plots/move_plot_<NN>.html
      (legend-click to show/hide keypoints)
'''

import pandas as pd
import argparse
from pathlib import Path
import os
from typing import Optional, Tuple, Dict, List
import math

KEYPOINTS = [0, 1, 2, 3, 4]
HANDS = ['L', 'R']
FPS = 30
OUT_CSV = 'movement_n_speed.csv'

def parse_args():
    parser = argparse.ArgumentParser(description="Compute 3D movement per keypoint for a participant.")
    parser.add_argument('--csv', required=True, help='Path to participant CSV file (e.g., .../<NN>/cam2/CSV/hand_landmark_xyz.csv)')
    parser.add_argument('--start', type=float, required=True, help='Start time in seconds')
    parser.add_argument('--stop', type=float, required=True, help='Stop time in seconds')
    parser.add_argument('--debug', action='store_true', help='If set, also write summary log and interactive plot')
    return parser.parse_args()

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
    dx = float(p_xyz[0]) - float(q_xyz[0])
    dy = float(p_xyz[1]) - float(q_xyz[1])
    dz = float(p_xyz[2]) - float(q_xyz[2])
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def compute_movement_series(df: pd.DataFrame, hand: str, kp: int) -> Tuple[float, List[Optional[float]]]:
    cols = [f"{hand}_{kp}_X_mm", f"{hand}_{kp}_Y_mm", f"{hand}_{kp}_Z_mm"]
    coords = df[cols].copy()

    per_frame: List[Optional[float]] = [None] * len(coords)
    total = 0.0
    prev_valid_idx: Optional[int] = None

    for idx in range(len(coords)):
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

def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()

    if args.stop <= args.start:
        raise ValueError("stop_time must be greater than start_time.")
    start_frame = int(args.start * FPS)
    stop_frame = int(args.stop * FPS)
    duration_sec = round(args.stop - args.start, 4)

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

    # Compute per-kp totals and per-frame series
    movement_values = []
    per_kp_total: Dict[str, float] = {}
    per_kp_series: Dict[str, List[Optional[float]]] = {}

    for hand in HANDS:
        for kp in KEYPOINTS:
            tag = f"{hand}_{kp}"
            total, series = compute_movement_series(df, hand, kp)
            per_kp_total[tag] = total
            per_kp_series[tag] = series
            movement_values.append(total)

    per_hand = {h: sum(per_kp_total[f"{h}_{k}"] for k in KEYPOINTS) for h in HANDS}
    grand_total = sum(movement_values)

    norm_movement_values = [round(m / duration_sec, 4) for m in movement_values]
    output_row = [participant_id, duration_sec] + [round(m, 4) for m in movement_values] + norm_movement_values

    header = ['Participant ID', 'Duration'] + \
             [f"{h}_{k}_move" for h in HANDS for k in KEYPOINTS] + \
             [f"{h}_{k}_speed" for h in HANDS for k in KEYPOINTS]

    file_exists = os.path.isfile(OUT_CSV)
    with open(OUT_CSV, 'a') as f:
        if not file_exists:
            f.write(','.join(header) + '\n')
        f.write(','.join(str(val) for val in output_row) + '\n')

    html_path = None
    if args.debug:
        debug_log.write(f"Participant: {participant_id}\n")
        debug_log.write(f"CSV: {csv_path}\n")
        debug_log.write(f"Window: start={args.start}s stop={args.stop}s duration={duration_sec}s\n")
        debug_log.write(f"Frames: {start_frame}..{stop_frame-1} (FPS={FPS})\n\n")

        debug_log.write("===== END OF SESSION SUMMARY =====\n")
        debug_log.write("Per-Hand Totals (movement mm | speed mm/s):\n")
        for h in HANDS:
            mv = per_hand[h]
            sp = mv / duration_sec
            debug_log.write(f"  {h}: total={mv:.4f} | speed={sp:.4f}\n")
        debug_log.write(f"\nGrand Total Movement: {grand_total:.4f} mm | Avg Speed: {grand_total/duration_sec:.4f} mm/s\n\n")

        debug_log.write("Per-Keypoint Summary (movement mm | speed mm/s):\n")
        for h in HANDS:
            for k in KEYPOINTS:
                tag = f"{h}_{k}"
                mv = per_kp_total[tag]
                sp = mv / duration_sec
                debug_log.write(f"  {tag}: total={mv:.4f} | speed={sp:.4f}\n")
        debug_log.write("===== SUMMARY END =====\n")
        debug_log.close()

        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception as e:
            print(f"[!] Plotly not available; skipping plot. ({e})")
        else:
            fig = go.Figure()
            for hand in HANDS:
                for kp in KEYPOINTS:
                    tag = f"{hand}_{kp}"
                    y = per_kp_series[tag]
                    fig.add_trace(go.Scatter(
                        x=frame_numbers, y=y, mode='lines', name=tag,
                        hovertemplate="frame=%{x}<br>" + tag + "=%{y:.3f} mm<extra></extra>"
                    ))
            fig.update_layout(
                title=f"Per-frame 3D Movement (mm) — Participant {participant_id}",
                xaxis_title="Frame",
                yaxis_title="Distance this frame (mm)",
                legend_title="Keypoints (click to toggle)",
                hovermode='x unified',
                template='plotly_white',
            )
            html_path = plots_dir / f"move_plot_{participant_id}.html"
            pio.write_html(fig, file=str(html_path), auto_open=False, include_plotlyjs='cdn')

    print(f"[✓] Movement data for {participant_id} appended to {OUT_CSV}.")
    if args.debug:
        print(f"[ℹ] Summary log saved to: {debug_log_path}")
        if html_path:
            print(f"[ℹ] Interactive plot saved to: {html_path}")

if __name__ == "__main__":
    main()
