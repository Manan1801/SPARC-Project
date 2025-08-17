#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os

KEYPOINTS = [0, 1, 2, 3, 4]
HANDS = ['L', 'R']
FPS = 30
SEGMENT_LABELS = ['Initial', 'Middle', 'End']

def parse_args():
    p = argparse.ArgumentParser(description="Compute 3D movement across percentage-based segments (Initial/Middle/End).")
    p.add_argument('--csv_path', required=True, help='Path to participant CSV file (e.g., AP_01_xyz.csv)')
    p.add_argument('--class', choices=['E', 'N'], required=True, help='Participant classification: E or N')
    p.add_argument('--start_time', type=float, required=True, help='Start time in seconds')
    p.add_argument('--stop_time', type=float, required=True, help='Stop time in seconds')
    p.add_argument('--percentages', required=True, help='Comma-separated segment percentages, e.g., 25,50,25')
    p.add_argument('--debug', action='store_true', help='Enable debug logging to file')
    return p.parse_args()

def extract_participant_id(csv_path: str) -> str:
    return Path(csv_path).stem.replace('_xyz', '')

def compute_movement(df: pd.DataFrame, hand: str, kp: int, debug_log=None, seg_label: str = None) -> float:
    cols = [f"{hand}_{kp}_X_mm", f"{hand}_{kp}_Y_mm", f"{hand}_{kp}_Z_mm"]
    coords = df[cols].copy()
    movement = 0.0
    prev_index = None
    for idx in coords.index:
        row = coords.loc[idx]
        if row.isnull().any():
            if debug_log:
                debug_log.write(f"[DEBUG] Frame {idx}: Missing data for {hand}_{kp}, skipping\n")
            continue
        if prev_index is not None:
            prev_row = coords.loc[prev_index]
            dist = np.linalg.norm(row.values - prev_row.values)
            movement += dist
            if debug_log:
                debug_log.write(f"[DEBUG] {seg_label} | {hand}_{kp} | Frame {prev_index}->{idx} | Δ={dist:.4f} mm\n")
        prev_index = idx
    if debug_log:
        debug_log.write(f"[DEBUG] Total movement for {hand}_{kp} in {seg_label}: {movement:.4f} mm\n\n")
    return movement

def main():
    args = parse_args()
    participant_id = extract_participant_id(args.csv_path)
    classification = args.__dict__['class']
    debug_enabled = args.debug

    # Parse & validate percentages
    try:
        percentages = [int(x.strip()) for x in args.percentages.split(',') if x.strip() != '']
    except ValueError:
        raise ValueError("Percentages must be integers, e.g., 25,50,25")
    if len(percentages) != 3 or sum(percentages) != 100:
        raise ValueError("Percentages must be exactly three integers that sum to 100, e.g., 25,50,25")

    # Output and debug paths
    output_file = 'movement_expert_segments.csv' if classification == 'E' else 'movement_novice_segments.csv'
    pct_tag = "_".join(str(p) for p in percentages)
    debug_log_path = Path(args.csv_path).parent / f"debug_movement_segments_{participant_id}_{pct_tag}.log"
    debug_log = open(debug_log_path, 'w') if debug_enabled else None

    # Load & slice by time
    df = pd.read_csv(args.csv_path)
    start_frame = int(args.start_time * FPS)
    stop_frame = int(args.stop_time * FPS)
    df = df.iloc[start_frame:stop_frame].reset_index(drop=True)

    total_frames = len(df)
    total_duration_sec = int(args.stop_time - args.start_time)

    # Segment durations (seconds) with any remainder to the LAST segment
    seg_secs = [int(total_duration_sec * p / 100) for p in percentages]
    remainder = total_duration_sec - sum(seg_secs)
    seg_secs[-1] += remainder

    if debug_log:
        debug_log.write(f"[DEBUG] Participant ID: {participant_id}\n")
        debug_log.write(f"[DEBUG] Start={args.start_time}s Stop={args.stop_time}s → Total={total_duration_sec}s ({total_frames} frames @ {FPS}fps)\n")
        debug_log.write(f"[DEBUG] Percentages: {percentages} → Segment seconds: {seg_secs}\n\n")

    # Build header (interleaved by segment: movements then norms per segment)
    header = ['Participant ID', 'Total Duration (s)']
    for label in SEGMENT_LABELS:
        header += [f"{label}_{h}_{k}_movement" for h in HANDS for k in KEYPOINTS]
        header += [f"{label}_{h}_{k}_norm_movement" for h in HANDS for k in KEYPOINTS]

    # Build row values interleaved by segment to match header
    row_values = [participant_id, total_duration_sec]

    start_idx = 0
    for seg_idx, label in enumerate(SEGMENT_LABELS):
        dur_sec = seg_secs[seg_idx]
        frames_in_seg = dur_sec * FPS
        end_idx = start_idx + frames_in_seg
        seg_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        if debug_log:
            debug_log.write(f"[DEBUG] {label}: {dur_sec}s → {frames_in_seg} frames (Frame {start_idx}..{end_idx-1})\n")

        # Movements for this segment (order: L_0..L_4, R_0..R_4)
        seg_movs = []
        for hand in HANDS:
            for kp in KEYPOINTS:
                mv = compute_movement(seg_df, hand, kp, debug_log, label)
                seg_movs.append(round(mv, 4))

        # Norms for this segment
        seg_norms = [round(m / dur_sec, 4) if dur_sec > 0 else 0.0 for m in seg_movs]

        # Append in same order as header: first all movements for this segment, then norms for this segment
        row_values.extend(seg_movs)
        row_values.extend(seg_norms)

        start_idx = end_idx

    # Write CSV
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a') as out_csv:
        if not file_exists:
            out_csv.write(','.join(header) + '\n')
        out_csv.write(','.join(str(v) for v in row_values) + '\n')

    if debug_log:
        debug_log.write("\n[DEBUG] Flattened (interleaved-by-segment) result written successfully.\n")
        debug_log.close()

    print(f"[✓] Movement data for {participant_id} segmented {percentages} saved to {output_file}. Debug: {debug_log_path.name}")

if __name__ == "__main__":
    main()
