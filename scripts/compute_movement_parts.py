#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os

KEYPOINTS = [0, 1, 2, 3, 4]
HANDS = ['L', 'R']
FPS = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Compute 3D movement across equal parts for a participant.")
    parser.add_argument('--csv_path', required=True, help='Path to participant CSV file (e.g., AP_01_xyz.csv)')
    parser.add_argument('--class', choices=['E', 'N'], required=True, help='Participant classification: E or N')
    parser.add_argument('--start_time', type=float, required=True, help='Start time in seconds')
    parser.add_argument('--stop_time', type=float, required=True, help='Stop time in seconds')
    parser.add_argument('--parts', type=int, required=True, help='Number of equal time parts to divide the data')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging to file')
    return parser.parse_args()

def extract_participant_id(csv_path):
    return Path(csv_path).stem.replace('_xyz', '')

def compute_movement(df, hand, kp, debug_log=None, part_num=None):
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
                debug_log.write(f"[DEBUG] Part {part_num} | {hand}_{kp} | Frame {prev_index}->{idx} | Δ={dist:.4f} mm\n")

        prev_index = idx

    if debug_log:
        debug_log.write(f"[DEBUG] Total movement for {hand}_{kp} in Part {part_num}: {movement:.4f} mm\n\n")

    return movement

def main():
    args = parse_args()
    participant_id = extract_participant_id(args.csv_path)
    classification = args.__dict__['class']
    num_parts = args.parts
    debug_enabled = args.debug

    output_file = f'movement_expert_parts_{num_parts}.csv' if classification == 'E' else f'movement_novice_parts_{num_parts}.csv'
    debug_log_path = Path(args.csv_path).parent / f"debug_movement_parts_{participant_id}_{num_parts}.log"
    debug_log = open(debug_log_path, 'w') if debug_enabled else None

    # Load and trim data by time
    df = pd.read_csv(args.csv_path)
    start_frame = int(args.start_time * FPS)
    stop_frame = int(args.stop_time * FPS)
    df = df.iloc[start_frame:stop_frame].reset_index(drop=True)

    total_frames = len(df)
    total_duration_sec = int(args.stop_time - args.start_time)

    if debug_log:
        debug_log.write(f"[DEBUG] Participant ID: {participant_id}\n")
        debug_log.write(f"[DEBUG] Start Time: {args.start_time}s | Stop Time: {args.stop_time}s | Total Duration: {total_duration_sec}s\n")
        debug_log.write(f"[DEBUG] Total Frames: {total_frames}, FPS: {FPS}\n")
        debug_log.write(f"[DEBUG] Dividing into {num_parts} parts (seconds-based, remainder to last)\n")

    # Compute part durations in WHOLE seconds (remainder seconds go to the last part)
    base_sec = total_duration_sec // num_parts
    remainder_sec = total_duration_sec % num_parts
    part_durations_sec = [base_sec] * (num_parts - 1) + [base_sec + remainder_sec]

    # Build header: interleave per part -> [Part1 movements, Part1 norms, Part2 movements, Part2 norms, ...]
    header = ['Participant ID', 'Total Duration (s)']
    for i in range(1, num_parts + 1):
        header += [f"Part{i}_{h}_{k}_movement" for h in HANDS for k in KEYPOINTS]
        header += [f"Part{i}_{h}_{k}_norm_movement" for h in HANDS for k in KEYPOINTS]

    # Build row values INTERLEAVED BY PART to match header
    row_values = [participant_id, total_duration_sec]

    start_idx = 0
    for part_num, part_duration_sec in enumerate(part_durations_sec, start=1):
        frames_in_part = part_duration_sec * FPS
        end_idx = start_idx + frames_in_part
        part_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        if debug_log:
            debug_log.write(f"\n[DEBUG] Part {part_num}: {part_duration_sec}s → {frames_in_part} frames (Frame {start_idx} to {end_idx - 1})\n")

        # Collect movement for this part (order: L_0..L_4, R_0..R_4)
        part_movements = []
        for hand in HANDS:
            for kp in KEYPOINTS:
                mv = compute_movement(part_df, hand, kp, debug_log, part_num)
                part_movements.append(round(mv, 4))

        # Norms for this part
        part_norms = [round(m / part_duration_sec, 4) if part_duration_sec > 0 else 0.0 for m in part_movements]

        # Append in the SAME order as header (movements for this part, then norms for this part)
        row_values.extend(part_movements)
        row_values.extend(part_norms)

        start_idx = end_idx

    # Write output
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a') as out_csv:
        if not file_exists:
            out_csv.write(','.join(header) + '\n')
        out_csv.write(','.join(str(v) for v in row_values) + '\n')

    if debug_log:
        debug_log.write(f"\n[DEBUG] Flattened (interleaved-by-part) result written to {output_file}\n")
        debug_log.close()

    print(f"[✓] Movement data for {participant_id} (flattened over {num_parts} parts) added to {output_file}.")

if __name__ == "__main__":
    main()
