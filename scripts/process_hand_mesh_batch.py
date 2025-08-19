#!/usr/bin/env python3
"""
process_hand_mesh_batch.py

- Samples exactly 1000 images from a base dataset of InterHand2.6M images.
- Runs MediaPipe HandMesh to detect up to 2 hands per image.
- Saves annotated images (with distinct colors per hand) preserving folder structure under "annotated_images".
- Records one CSV row per image, including capture folder, gesture class, cam_id, filename, and X-Y pixel & world coordinates for each of 21 landmarks of up to 2 hands.
"""
import os
import random
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# ----- User Settings -----
BASE_LOCATION = Path("/media/robotics/One Touch/InterHand2.6M_Extracted/InterHand2.6M_5fps_batch1/images/train")
ANNOTATED_BASE = BASE_LOCATION / "annotated_images"
SAMPLE_SIZE = 1000
RANDOM_SEED = 40

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define colors for up to two hands (BGR)
HAND_COLORS = [(0, 0, 255), (0, 255, 0)]  # Hand 1: Red, Hand 2: Green

# 1. Collect all image file paths with metadata
all_images = []  # list of tuples: (Path, capture_folder, gesture, cam_id)
for capture_dir in BASE_LOCATION.iterdir():
    if not capture_dir.is_dir():
        continue
    capture_name = capture_dir.name  # e.g., "Capture0"
    for gesture_dir in capture_dir.iterdir():
        if not gesture_dir.is_dir():
            continue
        gesture_name = gesture_dir.name
        for cam_dir in gesture_dir.iterdir():
            if not cam_dir.is_dir() or not cam_dir.name.startswith("cam"):
                continue
            cam_id = cam_dir.name  # e.g., "cam0"
            for img_path in cam_dir.glob("*.jpg"):
                all_images.append((img_path, capture_name, gesture_name, cam_id))

# 2. Sample exactly 1000 images
random.seed(RANDOM_SEED)
random.shuffle(all_images)
sampled = all_images[:SAMPLE_SIZE]

# 3. Ensure output directories exist
for img_path, cap, gesture, cam in sampled:
    out_dir = ANNOTATED_BASE / cap / gesture / cam
    out_dir.mkdir(parents=True, exist_ok=True)

# 4. Prepare CSV records
records = []

# 5. Process images
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    for img_path, cap, gesture, cam in tqdm(sampled, desc="Processing images"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            # Skip unreadable images
            continue
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(rgb)

        # Annotate image copy
        annotated = img.copy()
        # Initialize row dict
        row = {
            'capture_folder': cap,
            'gesture_class': gesture,
            'cam_id': cam,
            'image_name': img_path.name
        }

        # Default empty fields for 2 hands x 21 landmarks
        for hand_idx in range(2):
            for lm_idx in range(21):
                row[f'pixel_x_{hand_idx}_{lm_idx}'] = None
                row[f'pixel_y_{hand_idx}_{lm_idx}'] = None
                row[f'world_x_{hand_idx}_{lm_idx}'] = None
                row[f'world_y_{hand_idx}_{lm_idx}'] = None

        # If hands detected, draw and record
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:
                    break
                # Draw landmarks & connections
                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=HAND_COLORS[hand_idx], thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=HAND_COLORS[hand_idx], thickness=2))

                # Corresponding world landmarks
                world_landmarks = results.multi_hand_world_landmarks[hand_idx]

                # Record coordinates
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    wx = world_landmarks.landmark[lm_idx].x
                    wy = world_landmarks.landmark[lm_idx].y

                    row[f'pixel_x_{hand_idx}_{lm_idx}'] = px
                    row[f'pixel_y_{hand_idx}_{lm_idx}'] = py
                    row[f'world_x_{hand_idx}_{lm_idx}'] = wx
                    row[f'world_y_{hand_idx}_{lm_idx}'] = wy

        # Save annotated image
        out_path = ANNOTATED_BASE / cap / gesture / cam / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # Append row
        records.append(row)

# 6. Save CSV
df = pd.DataFrame(records)
csv_path = ANNOTATED_BASE / "hand_mesh_results.csv"
df.to_csv(csv_path, index=False)

print(f"Done! Annotated images and CSV saved under {ANNOTATED_BASE}")
