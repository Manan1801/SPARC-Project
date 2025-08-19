#!/usr/bin/env python3
"""
frame_scale.py

Displays keypoints from a CSV file on a single pair of color and depth images,
accounting for manual scale and offset to align color keypoints with the depth map.

Usage:
    python frame_scale.py --parent_dir /path/to/parent --frame_id 000100
"""

import argparse
import os
import cv2
import pandas as pd
import numpy as np

def visualize_frame(parent_dir, frame_id, scale_x, scale_y, offset_x, offset_y):
    color_path = os.path.join(parent_dir, "color", f"frame_{frame_id}.png")
    depth_path = os.path.join(parent_dir, "depth", f"frame_{frame_id}.tiff")

    # Locate CSV file
    csv_path = None
    for f in os.listdir(parent_dir):
        if f.endswith(".csv"):
            csv_path = os.path.join(parent_dir, f)
            break

    if not csv_path:
        print("[ERROR] No CSV file found in parent directory.")
        return

    # Load images
    color_img = cv2.imread(color_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if color_img is None or depth_img is None:
        print("[ERROR] Could not load one or both images.")
        return

    # Convert depth to heatmap
    if depth_img.ndim == 2:
        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_BONE)
    else:
        depth_vis = depth_img

    # Load keypoints from CSV
    df = pd.read_csv(csv_path)
    df_frame = df[df['frame_id'] == int(frame_id)]

    if df_frame.empty:
        print(f"[WARNING] No keypoints found for frame {frame_id}")
        return

    row = df_frame.iloc[0]

    for hand_label, color_kp, depth_kp in [('Left', (0, 0, 255), (0, 255, 0)),  # Red, Green
                                           ('Right', (255, 0, 0), (0, 255, 255))]:  # Blue, Yellow

        for i in range(21):
            x_col = f"pixel_x_{hand_label}_{i}"
            y_col = f"pixel_y_{hand_label}_{i}"

            if x_col not in row or y_col not in row or pd.isna(row[x_col]) or pd.isna(row[y_col]):
                continue

            x_color = int(float(row[x_col]))
            y_color = int(float(row[y_col]))

            # Draw on color image
            cv2.circle(color_img, (x_color, y_color), 4, color_kp, -1)

            # Transform to depth image scale and offset
            x_depth = int(x_color * scale_x + offset_x)
            y_depth = int(y_color * scale_y + offset_y)

            if 0 <= x_depth < depth_vis.shape[1] and 0 <= y_depth < depth_vis.shape[0]:
                cv2.circle(depth_vis, (x_depth, y_depth), 4, depth_kp, -1)
            else:
                print(f"[WARNING] {hand_label} point {i} out of bounds: ({x_depth}, {y_depth})")

    # Combine and show
    combined = np.hstack((color_img, depth_vis))
    window_name = "Color (L) + Depth (R)"
    cv2.imshow(window_name, combined)
    print("[INFO] Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir', required=True, help='Directory containing color/, depth/, and the keypoints CSV')
    parser.add_argument('--frame_id', required=True, help='Frame ID (e.g. 000100)')
    args = parser.parse_args()

    # Adjust these values manually if needed
    SCALE_X = 0.65
    SCALE_Y = 0.65
    OFFSET_X = 105
    OFFSET_Y = 85

    visualize_frame(args.parent_dir, args.frame_id, SCALE_X, SCALE_Y, OFFSET_X, OFFSET_Y)

if __name__ == "__main__":
    main()
