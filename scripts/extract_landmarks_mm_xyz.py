#!/usr/bin/env python3
"""
extract_landmarks_mm_xyz_flat.py

Batch process multiple 'cam2' folders under AP_* folders inside a given parent directory.
For each valid 'cam2', convert 2D pixel landmarks to 3D coordinates using RealSense depth.
Skips frames where 'multiple hands detected' is logged in hand_detection_warnings.log.
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
from collections import defaultdict, OrderedDict
import re

# === DEPTH CAMERA INTRINSICS (D435 - cam2) ===
fx = 384.4244079589844
fy = 384.4244079589844
cx = 320.60400390625
cy = 241.220947265625
depth_units = 0.0010000000474974513

# === MANUAL ADJUSTMENT VALUES ===
SCALE_X = 0.65
SCALE_Y = 0.65
OFFSET_X = 105
OFFSET_Y = 85

landmark_ids = [0, 1, 2, 3, 4]
hand_labels = ['L', 'R']


def setup_logger(log_path, verbose=False):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(message)s'
    )


def find_latest_realsense_folder(ap_path):
    folders = [f for f in ap_path.glob("realsense_recording_*") if f.is_dir()]
    if not folders:
        return None
    if len(folders) == 1:
        return folders[0]
    try:
        folders.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    except:
        folders.sort()
    return folders[0]


def process_single_cam2_folder(cam2_path, args):
    csv_path = cam2_path / "hand_landmarks_pixel_handedness_cleaned.csv"
    color_path = cam2_path / "color"
    depth_path = cam2_path / "depth"
    warning_log_path = cam2_path / "hand_detection_warnings.log"

    if not csv_path.exists() or not color_path.exists() or not depth_path.exists():
        print(f"Skipping {cam2_path.parents[1].name}: Missing hand_landmarks_pixel_handedness_cleaned.csv, color/, or depth/")
        return

    # === Step 1: Parse hand_detection_warnings.log ===
    skip_frames = set()
    if warning_log_path.exists():
        with open(warning_log_path, 'r') as f:
            for line in f:
                if "❌ Multiple" in line:
                    match = re.search(r'\[Frame (\d+)\]', line)
                    if match:
                        frame_id = int(match.group(1))
                        skip_frames.add(frame_id)
    else:
        print(f"Warning: hand_detection_warnings.log not found in {cam2_path.parents[1].name}/cam2. Proceeding without excluding frames.")

    # === Set up logger if needed ===
    log_path = cam2_path / f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    if args.debug or args.debug_verbose:
        setup_logger(log_path, verbose=args.debug_verbose)
        logging.info(f"[Log Created] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Input CSV: {csv_path}")

    # === Read CSV ===
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Skipping {cam2_path.parents[1].name}: Failed to read CSV: {e}")
        return

    df = df[df['frame_id'].apply(lambda x: str(x).strip().isdigit())].copy()
    df['frame_id'] = df['frame_id'].astype(int)

    output_csv = csv_path.with_name(csv_path.stem + "_xyz_flat.csv")
    output_dict = defaultdict(lambda: OrderedDict())
    total_skipped = 0
    total_processed = 0
    visualize_enabled = args.visualize

    for idx, row in df.iterrows():
        try:
            frame_num = int(float(row['frame_id']))
        except Exception:
            continue

        if frame_num in skip_frames:
            if args.debug or args.debug_verbose:
                logging.info(f"Skipping Frame {frame_num:06d}: Multiple hands detected in warning log")
            continue

        if 'frame' not in output_dict[frame_num]:
            output_dict[frame_num]['frame'] = frame_num

        missing_cols = []
        out_of_bounds = []
        invalid_depth = []
        skipped = 0
        processed = 0

        depth_img_path = None
        for ext in ['.tiff', '.tif']:
            test_path = depth_path / f"frame_{frame_num:06d}{ext}"
            if test_path.exists():
                depth_img_path = test_path
                break

        if depth_img_path is None:
            if args.debug:
                logging.info("[Missing Data] Depth image not found.")
            total_skipped += 1
            continue

        depth_img = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            if args.debug:
                logging.info("[Missing Data] Failed to read depth image.")
            total_skipped += 1
            continue

        if visualize_enabled:
            depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            color_img_path = color_path / f"frame_{frame_num:06d}.png"
            color_img = cv2.imread(str(color_img_path)) if color_img_path.exists() else None
            if color_img is None:
                color_img = np.zeros_like(depth_vis)

        for hand in hand_labels:
            hand_full = "Left" if hand == "L" else "Right"
            for lm_id in landmark_ids:
                x_col = f'pixel_x_{hand_full}_{lm_id}'
                y_col = f'pixel_y_{hand_full}_{lm_id}'

                if x_col not in row or y_col not in row:
                    missing_cols.append((hand, lm_id))
                    skipped += 1
                    continue

                try:
                    x_color = float(row[x_col])
                    y_color = float(row[y_col])
                    x = int(x_color * SCALE_X + OFFSET_X)
                    y = int(y_color * SCALE_Y + OFFSET_Y)
                except ValueError:
                    out_of_bounds.append((hand, lm_id, "non-numeric"))
                    skipped += 1
                    continue

                if not (0 <= y < depth_img.shape[0] and 0 <= x < depth_img.shape[1]):
                    out_of_bounds.append((hand, lm_id, "xy"))
                    skipped += 1
                    continue

                depth_raw = int(depth_img[y, x])
                if depth_raw <= 0:
                    invalid_depth.append((hand, lm_id))
                    skipped += 1
                    continue

                z_mm = depth_raw * depth_units * 1000
                X_mm = (x - cx) * z_mm / fx
                Y_mm = (y - cy) * z_mm / fy
                Z_mm = z_mm

                base = f"{hand}_{lm_id}"
                output_dict[frame_num][f"{base}_X_px"] = int(x_color)
                output_dict[frame_num][f"{base}_Y_px"] = int(y_color)
                output_dict[frame_num][f"{base}_X_mm"] = round(X_mm, 2)
                output_dict[frame_num][f"{base}_Y_mm"] = round(Y_mm, 2)
                output_dict[frame_num][f"{base}_Z_mm"] = round(Z_mm, 2)

                processed += 1
                total_processed += 1

                if args.debug_verbose:
                    logging.debug(f"✔ Processed {hand}_{lm_id}: ({int(x_color)}, {int(y_color)}) -> ({round(X_mm,2)}, {round(Y_mm,2)}, {round(Z_mm,2)}) mm")

                if visualize_enabled:
                    cv2.circle(depth_vis, (x, y), 3, (0, 255, 0), -1)
                    cv2.circle(color_img, (int(x_color), int(y_color)), 3, (0, 0, 255), -1)

        if args.debug:
            if missing_cols:
                logging.info(f"[Missing Data] {len(missing_cols)} missing keypoints")
            if out_of_bounds:
                logging.info(f"[Out of Bounds] {len(out_of_bounds)} keypoints skipped")
            if invalid_depth:
                logging.info(f"[Invalid Depth] {len(invalid_depth)} zero-depth keypoints")
            logging.info(f"Summary: Processed {processed}, Skipped {skipped} keypoints")

        if visualize_enabled:
            combined = cv2.hconcat([color_img, depth_vis])
            cv2.imshow("Color (L) + Depth (R)", combined)
            key = cv2.waitKey(100)
            if key == 27:
                visualize_enabled = False
                cv2.destroyAllWindows()
                if args.debug:
                    logging.info("Visualization disabled by ESC")

    if output_dict:
        ordered_cols = ['frame']
        for hand in hand_labels:
            for lm_id in landmark_ids:
                ordered_cols.append(f"{hand}_{lm_id}_X_px")
                ordered_cols.append(f"{hand}_{lm_id}_Y_px")
        for hand in hand_labels:
            for lm_id in landmark_ids:
                ordered_cols.append(f"{hand}_{lm_id}_X_mm")
                ordered_cols.append(f"{hand}_{lm_id}_Y_mm")
                ordered_cols.append(f"{hand}_{lm_id}_Z_mm")

        flat_rows = []
        for frame_id, data in sorted(output_dict.items()):
            row = {col: data.get(col, "") for col in ordered_cols}
            flat_rows.append(row)

        pd.DataFrame(flat_rows)[ordered_cols].to_csv(output_csv, index=False)
        if args.debug:
            logging.info(f"[OUTPUT] Saved: {output_csv}")
            logging.info(f"Total valid 3D landmarks: {total_processed}")
    else:
        if args.debug:
            logging.warning("No valid data to write. Output CSV not created.")


def main():
    parser = argparse.ArgumentParser(description="Batch convert 2D hand landmarks to 3D in mm across multiple folders.")
    parser.add_argument("main_folder", type=str, help="Path to the April_Pilot folder containing AP_* subfolders.")
    parser.add_argument("--visualize", action="store_true", help="Visualize color and depth frames.")
    parser.add_argument("--debug", action="store_true", help="Enable structured debug logging.")
    parser.add_argument("--debug-verbose", action="store_true", help="Enable full verbose debug logging.")
    args = parser.parse_args()

    main_path = Path(args.main_folder)
    if not main_path.exists():
        print(f"[ERROR] Main folder not found: {main_path}")
        return

    ap_folders = sorted([f for f in main_path.glob("AP_*") if f.is_dir()])
    if not ap_folders:
        print(f"[ERROR] No AP_* folders found inside {main_path}")
        return

    print(f"[INFO] Found {len(ap_folders)} folders. Processing...")

    for ap_folder in ap_folders:
        latest_rs_folder = find_latest_realsense_folder(ap_folder)
        if latest_rs_folder is None:
            print(f"Skipping {ap_folder.name}: No realsense_recording_* folder found.")
            continue

        cam2_path = latest_rs_folder / "cam2"
        if not cam2_path.exists():
            print(f"Skipping {ap_folder.name}: cam2 folder not found.")
            continue

        process_single_cam2_folder(cam2_path, args)

    print("[INFO] Batch processing completed.")


if __name__ == "__main__":
    main()
