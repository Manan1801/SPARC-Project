#!/usr/bin/env python3
"""
process_hand_mesh_frames_with_images_handedness_logged.py

- Uses MediaPipe handedness directly
- Filters out low-confidence Right hands
- Logs:
  - Frames with multiple Left hands
  - Frames with no hands
  - Frames with only one hand (logs which one is missing)
  - Handedness scores for every detection
"""

import cv2
import mediapipe as mp
import pandas as pd
import argparse
import os
import glob

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# BGR colors
HAND_COLORS = {'Left': (0, 255, 0), 'Right': (0, 0, 255)}  # Green = Left, Red = Right

def process_color_frames(color_dir):
    color_dir = os.path.abspath(color_dir)
    parent_dir = os.path.dirname(color_dir)
    output_csv = os.path.join(parent_dir, "hand_landmarks_pixel_handedness_cleaned.csv")
    output_img_dir = os.path.join(parent_dir, "color_mp_handedness_cleaned")
    warning_log_path = os.path.join(parent_dir, "hand_detection_warnings.log")

    os.makedirs(output_img_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(color_dir, "frame_*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No 'frame_*.png' files found in {color_dir}")

    records = []
    with open(warning_log_path, 'w') as log_file:  # overwrite on every run
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

            for path in frame_paths:
                frame = cv2.imread(path)
                if frame is None:
                    continue

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                annotated = frame.copy()
                frame_id = int(os.path.basename(path).split('_')[1].split('.')[0])
                row = {'frame_id': frame_id}

                # Init all coordinates
                for label in ['Left', 'Right']:
                    for lm_idx in range(21):
                        row[f'pixel_x_{label}_{lm_idx}'] = None
                        row[f'pixel_y_{label}_{lm_idx}'] = None

                hands_data = []
                if results.multi_hand_landmarks and results.multi_handedness:
                    for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        score = handedness.classification[0].score

                        # Filter out weak right hands
                        if label == 'Right' and score < 0.8:
                            continue
                        hands_data.append((label, landmarks, score))

                    labels = [h[0] for h in hands_data]
                    scores = [f"{h[0]}={h[2]:.2f}" for h in hands_data]

                    if labels.count('Left') > 1:
                        log_file.write(f"[Frame {frame_id}] ❌ Multiple Left hands detected | Scores: {', '.join(scores)}\n")
                    elif len(labels) == 1:
                        missing = 'Right' if labels[0] == 'Left' else 'Left'
                        log_file.write(f"[Frame {frame_id}] ⚠️ Only one hand detected: {missing} missing | Score: {scores[0]}\n")

                else:
                    log_file.write(f"[Frame {frame_id}] ⚠️ No hands detected\n")

                # Draw and record hands
                for label, hand_landmarks, _ in hands_data:
                    color = HAND_COLORS[label]
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=color, thickness=1)
                    )
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        px = int(lm.x * width)
                        py = int(lm.y * height)
                        row[f'pixel_x_{label}_{lm_idx}'] = px
                        row[f'pixel_y_{label}_{lm_idx}'] = py

                # Save annotated image
                save_path = os.path.join(output_img_dir, os.path.basename(path))
                cv2.imwrite(save_path, annotated)
                records.append(row)

    df = pd.DataFrame(records).sort_values("frame_id")
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Saved annotated images to: {output_img_dir}")
    print(f"✅ Saved hand landmarks CSV to: {output_csv}")
    print(f"📄 Logged detection issues to: {warning_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_dir", required=True)
    args = parser.parse_args()
    process_color_frames(args.color_dir)
