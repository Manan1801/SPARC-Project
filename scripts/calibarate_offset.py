#!/usr/bin/env python3
"""
calibrate_color_to_depth_offset_from_frame.py

Usage:
    python calibrate_color_to_depth_offset_from_frame.py --parent_dir <folder> --frame_id <int>

Requirements:
    - parent_dir should contain:
        * hand_landmarks_pixel_handedness_cleaned.csv
        * color/frame_<id>.png
        * depth/frame_<id>.tiff

Instructions:
    1. Left-click on a point in the left (color) image.
    2. Left-click on the same point in the right (depth) image.
    3. Press 'q' to finish and save the average offset.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

color_points = []
depth_points = []

def mouse_callback(event, x, y, flags, param):
    global color_points, depth_points
    img_width = param["color"].shape[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        if x < img_width:
            print(f"[Color] Clicked: ({x}, {y})")
            color_points.append((x, y))
        else:
            x_adj = x - img_width
            print(f"[Depth] Clicked: ({x_adj}, {y})")
            depth_points.append((x_adj, y))

def compute_offset():
    if len(color_points) != len(depth_points) or len(color_points) == 0:
        print("[ERROR] Mismatched or no point pairs selected.")
        return None

    diffs = [(c[0] - d[0], c[1] - d[1]) for c, d in zip(color_points, depth_points)]
    dx = int(np.mean([d[0] for d in diffs]))
    dy = int(np.mean([d[1] for d in diffs]))
    return dx, dy

def draw_csv_keypoints(csv_path, frame_id, image):
    df = pd.read_csv(csv_path)
    df = df[df["frame_id"] == frame_id]

    if df.empty:
        print(f"[WARNING] No data for frame_id {frame_id}")
        return image

    for _, row in df.iterrows():
        try:
            x, y = int(float(row["pixel_x"])), int(float(row["pixel_y"]))
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        except:
            continue
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", required=True, help="Parent folder containing CSV, color/, and depth/")
    parser.add_argument("--frame_id", required=True, type=int, help="Frame ID to process")
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    csv_path = parent / "hand_landmarks_pixel_handedness_cleaned.csv"
    color_path = parent / "color" / f"frame_{args.frame_id:06d}.png"
    depth_path = parent / "depth" / f"frame_{args.frame_id:06d}.tiff"

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    if not color_path.exists() or not depth_path.exists():
        print("[ERROR] Frame not found.")
        return

    color_img = cv2.imread(str(color_path))
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if color_img is None or depth_raw is None:
        print("[ERROR] Could not load one of the images.")
        return

    # Draw keypoints from CSV
    color_img = draw_csv_keypoints(csv_path, args.frame_id, color_img)

    # Prepare depth visualization
    if depth_raw.ndim == 2:
        depth_vis = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
    else:
        depth_vis = depth_raw

    combined = np.hstack((color_img, depth_vis))

    cv2.namedWindow("Match points: Color (L) vs Depth (R)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Match points: Color (L) vs Depth (R)", mouse_callback, param={"color": color_img})

    print("[INFO] Left-click matching points. Press 'q' to compute offset.")

    while True:
        overlay = combined.copy()
        for (cx, cy) in color_points:
            cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)
        for (dx, dy) in depth_points:
            cv2.circle(overlay, (dx + color_img.shape[1], dy), 5, (0, 255, 0), -1)

        cv2.imshow("Match points: Color (L) vs Depth (R)", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    offset = compute_offset()
    if offset:
        dx, dy = offset
        with open("color_to_depth_offset.txt", "w") as f:
            f.write(f"{dx},{dy}\n")
        print(f"[✅] Saved average offset: dx={dx}, dy={dy} → color_to_depth_offset.txt")

if __name__ == "__main__":
    main()
