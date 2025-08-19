#!/usr/bin/env python3
"""
Script: hand_bbox_depth_skin.py

Use case (steps to run):
    1. Install dependencies:
         pip install opencv-python numpy
    2. Save this script anywhere on your machine.
    3. From a terminal, invoke:
         python hand_bbox_depth_skin.py \
           --color_dir /path/to/your/color \
           --depth_dir /path/to/your/depth \
           --out_dir   /path/to/save/annotated
    4. The script will:
         • Read every frames_XXXXXX.png in --color_dir
         • Match each to frames_XXXXXX.tiff in --depth_dir
         • Compute per-hand bounding boxes, 2D COM, and average depth
         • Draw boxes & centers on the color images
         • Save annotated PNGs into --out_dir

Description:
    - Parses command-line args for your color, depth, and output folders.
    - For each frame pair:
        • Thresholds depth to a 0.3–0.8 m band.
        • Segments skin in YCrCb.
        • ANDs the masks to isolate hand contours.
        • Finds the two largest contours (hands).
        • Computes axis-aligned bbox (±10 px pad), center (cx,cy), and mean depth.
        • Draws and saves the result.
"""

import os
import cv2
import numpy as np
import argparse

# === OPTIONAL YOLOv8 SETUP ===
# Uncomment the following lines if you want to enable YOLOv8 hand detection
# from ultralytics import YOLO
# YOLO_MODEL = YOLO('yolov8n-hand.pt')

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(
    description="Compute per-hand bounding boxes + COM from color+depth frames"
)
parser.add_argument(
    "--color_dir", required=True,
    help="Folder of color PNGs (e.g. frames_000000.png)"
)
parser.add_argument(
    "--depth_dir", required=True,
    help="Folder of depth TIFFs (e.g. frames_000000.tiff)"
)
parser.add_argument(
    "--out_dir", default="out",
    help="Where to save annotated images"
)
args = parser.parse_args()

COLOR_DIR = args.color_dir
DEPTH_DIR = args.depth_dir
OUT_DIR   = args.out_dir

# === PROCESSING PARAMETERS ===
MIN_DIST_MM, MAX_DIST_MM = 300, 800   # keep hands between 0.3 m and 0.8 m
SKIN_YCRCB_LOW  = np.array([0, 133,  77], dtype=np.uint8)
SKIN_YCRCB_HIGH = np.array([255,173, 127], dtype=np.uint8)
PAD             = 10                 # pad each bbox by 10 pixels

os.makedirs(OUT_DIR, exist_ok=True)


def depth_skin_boxes(color, depth):
    """Return list of hand boxes + centers + avg depth, and the combined mask."""
    h, w = color.shape[:2]

    # 1) Depth mask
    mask_d = cv2.inRange(depth, MIN_DIST_MM, MAX_DIST_MM)

    # 2) Skin-color mask
    ycrcb = cv2.cvtColor(color, cv2.COLOR_BGR2YCrCb)
    mask_s = cv2.inRange(ycrcb, SKIN_YCRCB_LOW, SKIN_YCRCB_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN,  kernel)
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_CLOSE, kernel)

    # 3) Combine & contour
    mask = cv2.bitwise_and(mask_d, mask_s)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    boxes = []
    for cnt in contours:
        x, y, w_, h_ = cv2.boundingRect(cnt)
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(w, x + w_ + PAD)
        y2 = min(h, y + h_ + PAD)

        # 2D COM
        cx2d, cy2d = (x1 + x2) // 2, (y1 + y2) // 2

        # Avg depth (m)
        mask_roi = mask[y1:y2, x1:x2] > 0
        coords = np.argwhere(mask_roi)
        coords[:,0] += y1; coords[:,1] += x1
        zs = depth[coords[:,0], coords[:,1]].astype(np.float32) / 1000.0
        avg_depth_m = float(zs.mean()) if zs.size else None

        boxes.append({
            "bbox":        (x1, y1, x2, y2),
            "com2d":       (cx2d, cy2d),
            "avg_depth_m": avg_depth_m
        })

    return boxes, mask


def draw_and_save(color, boxes, prefix):
    """Annotate the color image with boxes + COM, then save to disk."""
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        cx, cy        = b["com2d"]
        cv2.rectangle(color, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(   color, (cx, cy), 4, (0,255,0), -1)
        cv2.putText(color,
                    f"{cx},{cy}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1)

    out_path = os.path.join(OUT_DIR, f"{prefix}.png")
    cv2.imwrite(out_path, color)


if __name__ == "__main__":
    color_files = sorted(f for f in os.listdir(COLOR_DIR) if f.endswith(".png"))
    for cf in color_files:
        idx    = cf.split("_")[1].split(".")[0]
        cpath  = os.path.join(COLOR_DIR, cf)
        dpath  = os.path.join(DEPTH_DIR, f"frame_{idx}.tiff")
        prefix = f"frame_{idx}_annot"

        color = cv2.imread(cpath)
        depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)

        boxes, mask = depth_skin_boxes(color.copy(), depth)
        print(f"[{idx}] Detected hands:", boxes)

        draw_and_save(color, boxes, prefix)

        # === OPTIONAL YOLOv8 PROCESSING ===
        # Uncomment below to enable YOLOv8 on the same frames
        # results = YOLO_MODEL.predict(color, conf=0.5, max_det=2)
        # for r in results:
        #     for box in r.boxes:
        #         x1,y1,x2,y2 = map(int, box.xyxy[0])
        #         cx, cy      = (x1+x2)//2, (y1+y2)//2
        #         cv2.rectangle(color, (x1,y1), (x2,y2), (255,0,0), 2)
        #         cv2.circle(   color, (cx, cy), 4, (255,0,0), -1)
        #         print(f"[{idx}] YOLO → BBox={(x1,y1,x2,y2)}, COM={(cx,cy)}, conf={float(box.conf[0]):.2f}")
        # cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_yolo.png"), color)
