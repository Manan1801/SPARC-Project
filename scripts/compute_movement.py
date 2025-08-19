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
    parser = argparse.ArgumentParser(description="Compute 3D movement per keypoint for a participant.")
    parser.add_argument('--csv_path', required=True, help='Path to participant CSV file (e.g., AP_01_xyz.csv)')
    parser.add_argument('--start_time', type=float, required=True, help='Start time in seconds')
    parser.add_argument('--stop_time', type=float, required=True, help='Stop time in seconds')
    parser.add_argument('--class', choices=['E', 'N'], required=True, help='Participant classification: E or N')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging to file')
    return parser.parse_args()

def extract_participant_id(csv_path):
    return Path(csv_path).stem.replace('_xyz', '')

def compute_movement(df, hand, kp, debug_log=None):
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
                debug_log.write(f"[DEBUG] {hand}_{kp} | Frame {prev_index} -> {idx} | Distance: {dist:.4f} mm\n")

        prev_index = idx

    if debug_log:
        debug_log.write(f"[DEBUG] Total movement for {hand}_{kp}: {movement:.4f} mm\n\n")

    return movement

def main():
    args = parse_args()
    csv_path = args.csv_path
    start_frame = int(args.start_time * FPS)
    stop_frame = int(args.stop_time * FPS)
    duration_sec = round(args.stop_time - args.start_time, 4)
    classification = args.__dict__['class']
    debug_enabled = args.debug

    participant_id = extract_participant_id(csv_path)
    output_file = 'movement_expert.csv' if classification == 'E' else 'movement_novice.csv'
    debug_log_path = Path(csv_path).parent / f"debug_movement_{participant_id}.log"
    debug_log = open(debug_log_path, 'w') if debug_enabled else None

    if debug_log:
        debug_log.write(f"[DEBUG] Participant ID: {participant_id}\n")
        debug_log.write(f"[DEBUG] Task Start Time: {args.start_time}s | Stop Time: {args.stop_time}s\n")
        debug_log.write(f"[DEBUG] Frame Range: {start_frame} - {stop_frame}\n\n")

    # Load and clean data
    df = pd.read_csv(csv_path)
    df = df.iloc[start_frame:stop_frame].reset_index(drop=True)
    df.replace('', np.nan, inplace=True)

    movement_values = []
    for hand in HANDS:
        for kp in KEYPOINTS:
            movement = compute_movement(df, hand, kp, debug_log)
            movement_values.append(movement)

    norm_movement_values = [round(m / duration_sec, 4) for m in movement_values]
    output_row = [participant_id, duration_sec] + [round(m, 4) for m in movement_values] + norm_movement_values

    header = ['Participant ID', 'Duration'] + \
             [f"{h}_{k}_movement" for h in HANDS for k in KEYPOINTS] + \
             [f"{h}_{k}_norm_movement" for h in HANDS for k in KEYPOINTS]

    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a') as f:
        if not file_exists:
            f.write(','.join(header) + '\n')
        f.write(','.join(str(val) for val in output_row) + '\n')

    if debug_log:
        debug_log.write(f"[DEBUG] Appended results to {output_file}\n")
        debug_log.close()

    print(f"[✓] Movement data for {participant_id} added to {output_file}.")

if __name__ == "__main__":
    main()
