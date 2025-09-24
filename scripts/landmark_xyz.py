#!/usr/bin/env python3
"""
landmark_xyz.py

Convert 2D (pixel) hand landmarks to 3D (mm) for cam2 with depth aligned to color.
- Use COLOR intrinsics (fx, fy, cx, cy) from cam2/camera_intrinsics_*.json.
- --debug flag: writes a concise END-OF-RUN SUMMARY (no anomalies section).
- If the provided root contains numbered subfolders (01, 02, ...), the script
       will automatically iterate through each and process their cam2/ folder.
- --skip lets you skip specific numbered folders (e.g., --skip 03 07 or --skip 03,07).

Expected layout inside each numbered session folder:
  <PARENT_ROOT>/<NN>/
    cam2/
      color/frame_XXXXXX.png
      depth/frame_XXXXXX.tiff
      CSV/hand_landmark.csv
      camera_intrinsics_<serial>.json
      logs/

Input CSV headers:
  frame_id,
  x_0_L_px,y_0_L_px, ... , x_20_L_px,y_20_L_px,
  x_0_R_px,y_0_R_px, ... , x_20_R_px,y_20_R_px

Output (per session):
  <SESSION_ROOT>/cam2/CSV/hand_landmark_xyz.csv
  (Flat schema: frame, then *_X_px/*_Y_px, then *_X_mm/*_Y_mm/*_Z_mm)
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
from collections import defaultdict, OrderedDict
import json
import re

# ====== DEFAULTS / FALLBACKS (used only if JSON intrinsics are missing) ======
FALLBACK_FX = 604.7083740234375
FALLBACK_FY = 604.9390258789062
FALLBACK_CX = 313.78240966796875
FALLBACK_CY = 252.73080444335938
# RealSense typical depth scale ~ 0.001 m per unit:
DEPTH_UNITS = 0.0010000000474974513  # meters per unit

# LANDMARK_IDS = list(range(21))  # All 21 landmarks
LANDMARK_IDS = [0, 1, 2, 3, 4]    # Thumb and Wrist only
HANDS = ['L', 'R']


def setup_logger(log_path: Path):
    logging.basicConfig(
        filename=str(log_path),
        filemode='w',
        level=logging.INFO,              # summary-only
        format='%(message)s'
    )


def find_cam2_paths(session_root: Path):
    cam2 = session_root / "cam2"
    color = cam2 / "color"
    depth = cam2 / "depth"
    csv_dir = cam2 / "CSV"
    logs_dir = cam2 / "logs"
    return cam2, color, depth, csv_dir, logs_dir


def load_color_intrinsics(cam2_dir: Path):
    """
    Load COLOR intrinsics from cam2/camera_intrinsics_*.json produced by the recorder.
    Pick the first key that starts with 'color_' and return its fx, fy, cx, cy.
    """
    fx = FALLBACK_FX
    fy = FALLBACK_FY
    cx = FALLBACK_CX
    cy = FALLBACK_CY
    found = False
    intr_file = None

    try:
        for jf in cam2_dir.glob("camera_intrinsics_*.json"):
            with open(jf, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if isinstance(v, dict) and k.startswith("color_"):
                    fx = float(v.get("fx", fx))
                    fy = float(v.get("fy", fy))
                    cx = float(v.get("cx", cx))
                    cy = float(v.get("cy", cy))
                    found = True
                    intr_file = jf.name
                    break
            if found:
                break
    except Exception as e:
        print(f"[WARN] Could not load JSON intrinsics ({e}). Using fallbacks.")

    if found:
        print(f"[INFO] Using COLOR intrinsics from JSON: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    else:
        print(f"[WARN] No camera_intrinsics_*.json with color_* found. Using fallback intrinsics.")

    return fx, fy, cx, cy, intr_file


def process_cam2(session_root: Path, visualize: bool, debug: bool) -> bool:
    """
    Process a single session root that contains cam2/.
    Returns True if processed (even partially), False if skipped due to missing structure.
    """
    cam2, color_path, depth_path, csv_dir, logs_dir = find_cam2_paths(session_root)

    csv_path = csv_dir / "hand_landmark.csv"
    if not (cam2.exists() and color_path.exists() and depth_path.exists() and csv_path.exists()):
        print(f"[SKIP] {session_root.name}: Missing cam2/color, cam2/depth, or CSV/hand_landmark.csv")
        return False

    # Load COLOR intrinsics (depth is aligned to color)
    fx, fy, cx, cy, intr_file = load_color_intrinsics(cam2)

    # Logger (same naming pattern retained)
    log_path = None
    if debug:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"debug_xyz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logger(log_path)
        logging.info(f"[Log Created] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Input CSV: {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] {session_root.name}: Failed to read CSV: {e}")
        return True  # counted as processed attempt

    # Ensure numeric frame_id
    df = df[df['frame_id'].apply(lambda x: str(x).strip().isdigit())].copy()
    if df.empty:
        print(f"[WARN] {session_root.name}: No numeric frame_id rows found.")
    df['frame_id'] = df['frame_id'].astype(int)

    # Output CSV
    output_csv = csv_dir / "hand_landmark_xyz.csv"

    output_dict = defaultdict(lambda: OrderedDict())
    visualize_enabled = visualize

    # --------- Summary counters ---------
    frames_scanned = 0
    frames_processed = 0
    frames_missing_depth = 0
    frames_depth_read_fail = 0
    frames_zero_valid = 0

    total_attempted_kp = 0  # finite (x,y) present in CSV
    total_valid_kp = 0      # successfully projected to 3D
    total_oob_kp = 0        # out-of-bounds
    total_invalid_depth_kp = 0
    total_missing_cols_kp = 0

    per_hand_valid = {h: 0 for h in HANDS}
    per_lm_valid = {h: {lm: 0 for lm in LANDMARK_IDS} for h in HANDS}

    first_depth_shape = None
    first_color_shape = None

    # ------------------------------------
    for _, row in df.iterrows():
        frame_num = int(row['frame_id'])
        frames_scanned += 1

        if 'frame' not in output_dict[frame_num]:
            output_dict[frame_num]['frame'] = frame_num

        # Depth image path (tiff/tif)
        depth_img_path = None
        for ext in ('.tiff', '.tif'):
            p = depth_path / f"frame_{frame_num:06d}{ext}"
            if p.exists():
                depth_img_path = p
                break
        if depth_img_path is None:
            frames_missing_depth += 1
            continue

        depth_img = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            frames_depth_read_fail += 1
            continue

        if first_depth_shape is None:
            first_depth_shape = depth_img.shape[:2]

        # Visualization buffers
        if visualize_enabled:
            depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            color_img_path = color_path / f"frame_{frame_num:06d}.png"
            color_img = cv2.imread(str(color_img_path)) if color_img_path.exists() else None
            if color_img is None or color_img.shape[:2] != depth_vis.shape[:2]:
                color_img = np.zeros_like(depth_vis)
            if first_color_shape is None:
                first_color_shape = color_img.shape[:2]

        attempted_kp = 0
        valid_kp = 0
        oob_kp = 0
        invdepth_kp = 0
        missing_kp = 0

        h, w = depth_img.shape[:2]

        for hand in HANDS:
            for lm_id in LANDMARK_IDS:
                x_col = f"x_{lm_id}_{hand}_px"
                y_col = f"y_{lm_id}_{hand}_px"

                if x_col not in row or y_col not in row:
                    missing_kp += 1
                    continue

                try:
                    x_px = float(row[x_col])
                    y_px = float(row[y_col])
                except Exception:
                    missing_kp += 1
                    continue

                if not np.isfinite(x_px) or not np.isfinite(y_px):
                    missing_kp += 1
                    continue

                attempted_kp += 1

                # IDENTITY mapping because depth is aligned to COLOR
                x = int(round(x_px))
                y = int(round(y_px))

                if not (0 <= x < w and 0 <= y < h):
                    oob_kp += 1
                    continue

                depth_raw = int(depth_img[y, x])
                if depth_raw <= 0:
                    invdepth_kp += 1
                    continue

                # Convert to mm
                z_mm = depth_raw * DEPTH_UNITS * 1000.0

                # Back-project using COLOR intrinsics
                X_mm = (x - cx) * z_mm / fx
                Y_mm = (y - cy) * z_mm / fy
                Z_mm = z_mm

                base = f"{hand}_{lm_id}"
                output_dict[frame_num][f"{base}_X_px"] = int(round(x_px))
                output_dict[frame_num][f"{base}_Y_px"] = int(round(y_px))
                output_dict[frame_num][f"{base}_X_mm"] = round(X_mm, 2)
                output_dict[frame_num][f"{base}_Y_mm"] = round(Y_mm, 2)
                output_dict[frame_num][f"{base}_Z_mm"] = round(Z_mm, 2)

                valid_kp += 1
                per_hand_valid[hand] += 1
                per_lm_valid[hand][lm_id] += 1

                if visualize_enabled:
                    cv2.circle(depth_vis, (x, y), 3, (0, 255, 0), -1)
                    cv2.circle(color_img, (x, y), 3, (0, 0, 255), -1)

        total_attempted_kp += attempted_kp
        total_valid_kp     += valid_kp
        total_oob_kp       += oob_kp
        total_invalid_depth_kp += invdepth_kp
        total_missing_cols_kp  += missing_kp

        if attempted_kp > 0:
            frames_processed += 1
            if valid_kp == 0:
                frames_zero_valid += 1

        if visualize_enabled:
            combined = cv2.hconcat([color_img, depth_vis])
            cv2.imshow("Color (L) + Depth-aligned (R)", combined)
            key = cv2.waitKey(1)
            if key == 27:
                visualize_enabled = False
                cv2.destroyAllWindows()

    # Emit flat CSV
    if output_dict:
        ordered_cols = ['frame']
        for hand in HANDS:
            for lm_id in LANDMARK_IDS:
                ordered_cols.append(f"{hand}_{lm_id}_X_px")
                ordered_cols.append(f"{hand}_{lm_id}_Y_px")
        for hand in HANDS:
            for lm_id in LANDMARK_IDS:
                ordered_cols.append(f"{hand}_{lm_id}_X_mm")
                ordered_cols.append(f"{hand}_{lm_id}_Y_mm")
                ordered_cols.append(f"{hand}_{lm_id}_Z_mm")

        rows = []
        for frame_id, data in sorted(output_dict.items()):
            rows.append({col: data.get(col, "") for col in ordered_cols})

        pd.DataFrame(rows, columns=ordered_cols).to_csv(output_csv, index=False)
        print(f"[OK] {session_root.name}: wrote {output_csv}  (total 3D points: {total_valid_kp})")
    else:
        print(f"[WARN] {session_root.name}: No valid data to write.")

    # --------- Compact END-OF-RUN SUMMARY (no anomalies) ---------
    if debug:
        lines = []
        lines.append("=== XYZ Extraction Summary ===")
        lines.append(f"Session Root      : {session_root}")
        lines.append(f"Input CSV         : {csv_path.name}  (rows: {len(df)})")
        lines.append(f"Output CSV        : {output_csv.name}")
        if intr_file:
            lines.append(f"Intrinsics Source : {intr_file}")
        else:
            lines.append("Intrinsics Source : FALLBACKS (no JSON color_* found)")
        lines.append(f"Intrinsics (color): fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        if first_color_shape or first_depth_shape:
            lines.append(f"First color shape : {first_color_shape if first_color_shape else 'n/a'}")
            lines.append(f"First depth shape : {first_depth_shape if first_depth_shape else 'n/a'}")
        lines.append("")
        lines.append(f"Frames scanned    : {frames_scanned}")
        lines.append(f"Frames processed  : {frames_processed}")
        lines.append(f"Frames no depth   : {frames_missing_depth}")
        lines.append(f"Frames depth read : {frames_depth_read_fail} (failed to decode)")
        lines.append(f"Frames zero-valid : {frames_zero_valid}")
        lines.append("")
        lines.append(f"Keypoints attempted : {total_attempted_kp}")
        lines.append(f"  ├─ valid           : {total_valid_kp}")
        lines.append(f"  ├─ out-of-bounds   : {total_oob_kp}")
        lines.append(f"  ├─ invalid depth   : {total_invalid_depth_kp}")
        lines.append(f"  └─ missing/NaN     : {total_missing_cols_kp}")
        lines.append("")
        lines.append(f"Valid KP by hand  : L={per_hand_valid['L']}, R={per_hand_valid['R']}")
        # Landmark coverage (show only low-coverage)
        def pct(n):
            return (100.0 * n / max(frames_processed, 1))
        low_cov = []
        for hand in HANDS:
            for lm_id in LANDMARK_IDS:
                cov = per_lm_valid[hand][lm_id]
                if cov < max(5, 0.25 * frames_processed):  # flag poor coverage
                    low_cov.append(f"{hand}:{lm_id}={cov} ({pct(cov):.1f}%)")
        if low_cov:
            lines.append("Low-coverage landmarks (valid per frame):")
            for s in low_cov[:20]:
                lines.append(f"  - {s}")
            if len(low_cov) > 20:
                lines.append(f"  ... (+{len(low_cov)-20} more)")
        else:
            lines.append("Landmark coverage : all selected landmarks look reasonably covered.")

        logging.info("\n".join(lines))
        print(f"[INFO] {session_root.name}: summary log written → {log_path}")

    return True


def _parse_skip_list(skip_args) -> set:
    """
    Accepts values like: ['03', '07'] or ['03,07'] or mixed.
    Returns a set of folder names exactly as they appear (e.g., '03', '7', etc.).
    """
    out = set()
    for item in skip_args or []:
        parts = re.split(r'[,\s]+', item.strip())
        for p in parts:
            if p:
                out.add(p)
    return out


def main():
    p = argparse.ArgumentParser(description="Batch convert 2D (pixel) hand landmarks to 3D (mm) for cam2 with depth aligned to color.")
    p.add_argument("root", type=str,
                   help="Path to a single session (containing cam2/) OR a parent folder containing numbered session folders (e.g., /media/.../HRI_26_sept25/)")
    p.add_argument("--visualize", action="store_true", help="Show color/depth with keypoints (per session).")
    p.add_argument("--debug", action="store_true", help="Write a concise summary log to cam2/logs/ at end of each session.")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Numbered subfolders to skip under the parent (e.g., --skip 03 07 or --skip 03,07).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return

    # Case 1: root itself is a single session (has cam2/)
    if (root / "cam2").exists():
        process_cam2(root, args.visualize, args.debug)
        return

    # Case 2: root is a parent; iterate numbered subfolders with cam2/
    skip_set = _parse_skip_list(args.skip)
    children = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)

    processed_any = False
    for session_dir in children:
        name = session_dir.name
        # only consider folders that look like numbers (e.g., 01, 1, 12, etc.)
        if not re.fullmatch(r'\d+', name):
            continue
        if name in skip_set:
            print(f"[SKIP] {name}: requested via --skip")
            continue
        if not (session_dir / "cam2").exists():
            print(f"[SKIP] {name}: no cam2/ folder")
            continue

        print(f"[INFO] Processing session: {name}")
        processed = process_cam2(session_dir, args.visualize, args.debug)
        processed_any = processed_any or processed

    if not processed_any:
        print("[WARN] No sessions were processed. Check folder names and --skip filters.")


if __name__ == "__main__":
    main()
