import cv2
import numpy as np
import argparse
from pathlib import Path
import os

def normalize_depth(depth_img, depth_unit=0.001):
    """Normalize 16-bit depth image to 8-bit heatmap (optional scaling)."""
    depth_in_mm = depth_img.astype(np.float32) * depth_unit * 1000  # Convert to mm
    clipped = np.clip(depth_in_mm, 300, 1500)  # Focused range for hand (~30cm to ~150cm)
    norm = ((clipped - 300) / (1500 - 300) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

def draw_overlay(frame, text):
    """Draws overlay text on the top of the image."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return overlay

def main(depth_dir, out_dir=None):
    depth_dir = Path(depth_dir)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(depth_dir.glob("frame_*.tiff")) + list(depth_dir.glob("frame_*.tif")))
    if not image_files:
        print(f"[ERROR] No .tiff or .tif files found in: {depth_dir}")
        return

    index = 0
    total = len(image_files)

    print(f"[INFO] Loaded {total} depth frames. Use ← and → to navigate, S to save, ESC to quit.")

    while True:
        img_path = image_files[index]
        depth = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if depth is None:
            print(f"[WARNING] Could not read: {img_path}")
            index = (index + 1) % total
            continue

        heatmap = normalize_depth(depth)
        display = draw_overlay(heatmap, f"{img_path.name}  ({index+1}/{total})")
        cv2.imshow("Depth Heatmap Viewer", display)

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):
            if out_dir:
                save_path = out_dir / f"{img_path.stem}_heatmap.png"
                cv2.imwrite(str(save_path), display)
                print(f"[✔] Saved: {save_path}")
        elif key == 81:  # ←
            index = (index - 1) % total
        elif key == 83:  # →
            index = (index + 1) % total

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RealSense depth .tiff frames as heatmaps.")
    parser.add_argument("--depth_dir", type=str, required=True, help="Folder containing .tiff depth frames")
    parser.add_argument("--out_dir", type=str, help="Optional output folder to save heatmaps (press S)")
    args = parser.parse_args()

    main(args.depth_dir, args.out_dir)
