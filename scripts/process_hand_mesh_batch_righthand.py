#!/usr/bin/env python3
"""
process_hand_mesh_right_only_onehand.py

- Filter frames to those with hand_type == 'right' using InterHand2.6M_*_joint_3d.json
- Sample exactly SAMPLE_SIZE images (random unless fewer exist)
- Run MediaPipe Hands (1 hand) per image
- Output ONE CSV row per image with:
    capture_folder, gesture_class, cam_id, image_name, frame_idx
    mp_world_x_mm_[lm], mp_world_y_mm_[lm]
    gt_world_x_mm_[lm], gt_world_y_mm_[lm]
    dist_mp_xy_mm_[lm]   = ||(mp_x_mm, mp_y_mm)||
    dist_gt_xy_mm_[lm]   = ||(gt_x_mm, gt_y_mm)||
    diff_xy_mm_[lm]      = ||(mp_x_mm - gt_x_mm, mp_y_mm - gt_y_mm)||
    rms_diff_xy_mm       = RMS over valid landmarks
"""

import json
import random
import math
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# ------------------ USER SETTINGS ------------------
BASE_LOCATION    = Path("/media/robotics/One Touch/InterHand2.6M_Extracted/InterHand2.6M_5fps_batch1/images/train")
GT_JSON_PATH     = Path("/media/robotics/One Touch/InterHand2.6M_Extracted/annotations/train/InterHand2.6M_train_joint_3d.json")
OUTPUT_BASE      = BASE_LOCATION / "annotated_right_onehand"   # images+csv will go here
SAMPLE_SIZE      = 1000
RANDOM_SEED      = 40
SAVE_ANNOTATIONS = True       # set True if you want annotated images
NUM_LM           = 21
# ----------------------------------------------------

mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
HAND_COLOR = (0, 0, 255)  # BGR red

def parse_frame_idx(fname: str) -> int:
    stem = Path(fname).stem
    try:
        return int(stem)
    except ValueError:
        digits = ''.join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else -1

def collect_images(base: Path):
    out = []
    for cap_dir in base.iterdir():
        if not cap_dir.is_dir() or cap_dir.name.startswith("annotated_"):
            continue
        cap = cap_dir.name
        for gest_dir in cap_dir.iterdir():
            if not gest_dir.is_dir():
                continue
            gest = gest_dir.name
            for cam_dir in gest_dir.iterdir():
                if not cam_dir.is_dir() or not cam_dir.name.startswith("cam"):
                    continue
                cam = cam_dir.name
                for img_path in cam_dir.glob("*.jpg"):
                    idx = parse_frame_idx(img_path.name)
                    out.append((img_path, cap, gest, cam, idx))
    return out

def load_gt(json_path: Path):
    with open(json_path, 'r') as f:
        return json.load(f)

def cap_to_key(capture_name: str):
    return capture_name[len("Capture"):] if capture_name.lower().startswith("capture") else capture_name

def is_right(gt_dict, cap_key: str, frame_idx: int):
    try:
        return gt_dict[cap_key][str(frame_idx)]['hand_type'] == 'right'
    except KeyError:
        return False

def get_gt_xy_mm(gt_dict, cap_key: str, frame_idx: int, num_lm: int):
    """
    Robustly extract GT XY (mm) and validity mask for up to num_lm landmarks.
    Handles cases where 'joint_valid' rows are shorter or malformed.
    """
    entry = gt_dict[cap_key][str(frame_idx)]
    world = entry.get('world_coord', [])
    valid = entry.get('joint_valid', [])

    xy   = []
    mask = []

    J = min(num_lm, len(world), len(valid))  # safe usable count

    for i in range(J):
        w = world[i] if isinstance(world[i], (list, tuple)) else []
        v = valid[i] if isinstance(valid[i], (list, tuple)) else []

        gx = w[0] if len(w) > 0 else None
        gy = w[1] if len(w) > 1 else None
        xy.append((gx, gy))

        okx = (len(v) > 0 and v[0] == 1)
        oky = (len(v) > 1 and v[1] == 1)
        mask.append(okx and oky)

    # pad if needed
    while len(xy)   < num_lm: xy.append((None, None))
    while len(mask) < num_lm: mask.append(False)

    return xy, mask

def rms(vals):
    return math.sqrt(sum(v*v for v in vals) / len(vals)) if vals else None

def main():
    random.seed(RANDOM_SEED)

    gt = load_gt(GT_JSON_PATH)
    all_imgs = collect_images(BASE_LOCATION)

    # Filter to frames with right hand in GT
    filtered = []
    for img_path, cap, gest, cam, frame_idx in all_imgs:
        cap_key = cap_to_key(cap)
        if is_right(gt, cap_key, frame_idx):
            filtered.append((img_path, cap, gest, cam, frame_idx, cap_key))

    # Sample
    if len(filtered) < SAMPLE_SIZE:
        print(f"[WARN] Only {len(filtered)} right-hand frames found. Using all.")
        sampled = filtered
    else:
        sampled = random.sample(filtered, SAMPLE_SIZE)

    if SAVE_ANNOTATIONS:
        for _, cap, gest, cam, *_ in sampled:
            (OUTPUT_BASE / cap / gest / cam).mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    records = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for img_path, cap, gest, cam, frame_idx, cap_key in tqdm(sampled, desc="Processing"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if SAVE_ANNOTATIONS:
                annotated = img.copy()

            # Ground truth (robust)
            try:
                gt_xy_mm, gt_valid = get_gt_xy_mm(gt, cap_key, frame_idx, NUM_LM)
            except Exception as e:
                print(f"[GT ERROR] cap={cap_key} frame={frame_idx}: {e}")
                gt_xy_mm = [(None, None)] * NUM_LM
                gt_valid = [False] * NUM_LM

            row = {
                'capture_folder': cap,
                'gesture_class': gest,
                'cam_id': cam,
                'image_name': img_path.name,
                'frame_idx': frame_idx
            }

            # Init columns
            for lm in range(NUM_LM):
                row[f'mp_world_x_mm_{lm}'] = None
                row[f'mp_world_y_mm_{lm}'] = None
                gx, gy = gt_xy_mm[lm]
                row[f'gt_world_x_mm_{lm}'] = gx
                row[f'gt_world_y_mm_{lm}'] = gy
                row[f'dist_mp_xy_mm_{lm}'] = None
                row[f'dist_gt_xy_mm_{lm}'] = None
                row[f'diff_xy_mm_{lm}']    = None

            diffs = []

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                world_ok = result.multi_hand_world_landmarks and len(result.multi_hand_world_landmarks) > 0

                if SAVE_ANNOTATIONS:
                    mp_draw.draw_landmarks(
                        annotated, lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=HAND_COLOR, thickness=2)
                    )

                for i in range(NUM_LM):
                    if world_ok:
                        wl = result.multi_hand_world_landmarks[0].landmark[i]
                        mp_x_mm = wl.x * 1000.0
                        mp_y_mm = wl.y * 1000.0
                        row[f'mp_world_x_mm_{i}'] = mp_x_mm
                        row[f'mp_world_y_mm_{i}'] = mp_y_mm
                        row[f'dist_mp_xy_mm_{i}'] = math.hypot(mp_x_mm, mp_y_mm)
                    else:
                        mp_x_mm = mp_y_mm = None

                    gx = row[f'gt_world_x_mm_{i}']
                    gy = row[f'gt_world_y_mm_{i}']
                    if gx is not None and gy is not None:
                        row[f'dist_gt_xy_mm_{i}'] = math.hypot(gx, gy)

                    if (mp_x_mm is not None and mp_y_mm is not None and
                        gx is not None and gy is not None and
                        i < len(gt_valid) and gt_valid[i]):

                        d = math.hypot(mp_x_mm - gx, mp_y_mm - gy)
                        row[f'diff_xy_mm_{i}'] = d
                        diffs.append(d)

            row['rms_diff_xy_mm']      = rms(diffs)
            row['num_valid_landmarks'] = len(diffs)

            if SAVE_ANNOTATIONS:
                out_path = OUTPUT_BASE / cap / gest / cam / img_path.name
                cv2.imwrite(str(out_path), annotated)

            records.append(row)

    df = pd.DataFrame(records)
    csv_path = OUTPUT_BASE / "hand_mesh_results_right_onehand.csv"
    df.to_csv(csv_path, index=False)
    print(f"Done! CSV: {csv_path}")
    if SAVE_ANNOTATIONS:
        print(f"Annotated images: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
