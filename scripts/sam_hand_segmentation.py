#===============================================================================
# File: sam_hand_segmentation.py
#===============================================================================
#!/usr/bin/env python3
"""
Script: sam_hand_segmentation.py

Use case (steps to run):
    1. Install dependencies:
         pip install opencv-python numpy torch torchvision \
                     git+https://github.com/facebookresearch/segment-anything.git
    2. Save this script anywhere in your project.
    3. Download a SAM checkpoint (e.g. sam_vit_b.ckpt) and place in ./models/.
    4. From your project root, invoke:
         python sam_hand_segmentation.py \
           --color_dir /path/to/color_frames \
           --out_dir   /path/to/annotated \
           --checkpoint ./models/sam_vit_b.ckpt \
           [--topk 2]
    5. The script will:
         • Load SAM and build an automatic mask generator.
         • Read each frames_XXXXXX.png.
         • Generate many masks, pick the top-k largest by area.
         • Compute bounding box & centroid for each mask.
         • Draw mask overlay, boxes & centers.
         • Save annotated images to --out_dir.
"""

import os
import cv2
import numpy as np
import argparse
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

parser = argparse.ArgumentParser(
    description="Hand segmentation + bbox + COM using SAM (Segment Anything)"
)
parser.add_argument(
    "--color_dir", required=True,
    help="Folder of color PNGs (e.g. frame_000000.png)"
)
parser.add_argument(
    "--out_dir", default="out",
    help="Directory to save annotated images"
)
parser.add_argument(
    "--checkpoint", required=True,
    help="Path to SAM checkpoint (e.g. sam_vit_b.ckpt)"
)
parser.add_argument(
    "--model_type", default="vit_b",
    choices=["vit_b","vit_l","vit_h"],
    help="Which SAM backbone to use (default: vit_b)"
)
parser.add_argument(
    "--topk", type=int, default=2,
    help="Number of largest masks to keep per image (default: 2)"
)
args = parser.parse_args()

COLOR_DIR    = args.color_dir
OUT_DIR      = args.out_dir
CKPT_PATH    = args.checkpoint
MODEL_TYPE   = args.model_type
TOPK         = args.topk

os.makedirs(OUT_DIR, exist_ok=True)

# load SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[f"vit_{MODEL_TYPE}"](checkpoint=CKPT_PATH).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

def process_frame(path, prefix):
    img = cv2.imread(path)
    if img is None:
        print(f"[Error] Cannot load {path}")
        return

    masks = mask_generator.generate(img)  # list of dicts with 'segmentation' & 'area'
    # sort by area and keep topk
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:TOPK]

    for i, m in enumerate(masks):
        mask = m["segmentation"].astype(bool)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        y1, x1, y2, x2 = m["bbox"].astype(int)

        # overlay mask (blue)
        overlay = img.copy()
        overlay[mask] = (255, 0, 0)
        img = cv2.addWeighted(img, 1.0, overlay, 0.3, 0)

        # draw bbox, center
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(img,    (cx,cy), 4, (0,255,0), -1)
        print(f"{prefix}#{i}: BBox=({x1},{y1},{x2},{y2}), COM=({cx},{cy}), area={m['area']}")

    out_path = os.path.join(OUT_DIR, f"{prefix}.png")
    cv2.imwrite(out_path, img)

if __name__ == "__main__":
    if not os.path.isdir(COLOR_DIR):
        raise NotADirectoryError(f"Color directory not found: {COLOR_DIR}")

    files = sorted(f for f in os.listdir(COLOR_DIR) if f.endswith(".png"))
    for f in files:
        idx      = f.split("_")[1].split(".")[0]
        img_path = os.path.join(COLOR_DIR, f)
        prefix   = f"frame_{idx}_sam"
        process_frame(img_path, prefix)
