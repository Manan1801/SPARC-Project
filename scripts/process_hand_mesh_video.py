#!/usr/bin/env python3
"""
process_hand_mesh_video.py

- Reads an input video file.
- Runs MediaPipe HandMesh on each frame, detecting up to 2 hands.
- Draws each hand's mesh in a distinct color (Hand 1: red, Hand 2: green).
- Writes out an annotated video named <original_name>_hand_mesh<original_ext> in the same folder.
- Builds a CSV with one row per frame, flattening all landmark coordinates for up to 2 hands:
    * frame index
    * pixel_x_{hand_idx}_{landmark_idx}, pixel_y_{hand_idx}_{landmark_idx}
    * world_x_{hand_idx}_{landmark_idx}, world_y_{hand_idx}_{landmark_idx}
"""
import cv2
import mediapipe as mp
import pandas as pd
import argparse
import os

# MediaPipe and drawing setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Colors and mapping
HAND_COLORS = [(0, 0, 255), (0, 255, 0)]  # BGR: red, green
COLOR_NAMES = ["red", "green"]


def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base, ext = os.path.splitext(input_path)
    output_video = f"{base}_hand_mesh{ext}"
    output_csv = f"{base}_hand_mesh.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    records = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()
            row = {'frame': frame_idx}
            # Initialize flattened columns for 2 hands x 21 landmarks
            for hand_idx in range(2):
                for lm_idx in range(21):
                    row[f'pixel_x_{hand_idx}_{lm_idx}'] = None
                    row[f'pixel_y_{hand_idx}_{lm_idx}'] = None
                    # row[f'world_x_{hand_idx}_{lm_idx}'] = None
                    # row[f'world_y_{hand_idx}_{lm_idx}'] = None

            # Process landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= 2:
                        break
                    # Draw on frame
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=HAND_COLORS[hand_idx], thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=HAND_COLORS[hand_idx], thickness=1)
                    )
                    world_landmarks = results.multi_hand_world_landmarks[hand_idx]
                    # Fill row
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        px = int(lm.x * width)
                        py = int(lm.y * height)
                        # wx = world_landmarks.landmark[lm_idx].x
                        # wy = world_landmarks.landmark[lm_idx].y
                        row[f'pixel_x_{hand_idx}_{lm_idx}'] = px
                        row[f'pixel_y_{hand_idx}_{lm_idx}'] = py
                        # row[f'world_x_{hand_idx}_{lm_idx}'] = wx
                        # row[f'world_y_{hand_idx}_{lm_idx}'] = wy

            out_vid.write(annotated)
            records.append(row)
            frame_idx += 1

    cap.release()
    out_vid.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"Output video saved to: {output_video}")
    print(f"Landmarks CSV saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with MediaPipe HandMesh.")
    parser.add_argument("--input_video", help="Path to the input video file.")
    args = parser.parse_args()

    process_video(args.input_video)
