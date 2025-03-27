#!/usr/bin/env python3
"""
Script: process_hands.py
Description:
  1) Recursively searches a directory for .jpg images.
  2) For each .jpg, runs:
     - MediaPipe => 21 keypoints in full-image pixel coords
     - HaMeR => 21 keypoints in full-image pixel coords (via 'pred_keypoints_2d' + un-cropping),
       using bounding boxes from the .json in the same folder.
  3) Writes:
     - mediapipe_landmarks.csv
     - hamer_landmarks.csv
     - ground_truth.csv
     - rms_errors.csv (with per-keypoint normalized distances)

Usage:
  python process_hands.py /path/to/images --output_dir /path/to/results [--hamer_ckpt /path/to/hamer.ckpt]
"""

import os
import sys
import csv
import glob
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

########################################
# MediaPipe Setup
########################################
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except ImportError:
    print("ERROR: Please install MediaPipe (pip install mediapipe).")
    sys.exit(1)

########################################
# HaMeR Setup
########################################
import torch

HAMER_REPO_PATH = "/home/hpm_mv_2/Desktop/hamer"
DEFAULT_CKPT_PATH = "/home/hpm_mv_2/Desktop/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
sys.path.append(HAMER_REPO_PATH)

try:
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import download_models, load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.renderer import Renderer
    from hamer.utils import recursive_to
except ImportError as e:
    print("ERROR: Could not import 'hamer' modules from:", HAMER_REPO_PATH)
    print(e)
    sys.exit(1)

# Optionally import the VitPose model if used
try:
    from vitpose_model import ViTPoseModel
except ImportError:
    print("WARNING: Could not import 'vitpose_model.py'. If not needed, ignore.")
    ViTPoseModel = None

########################################
# CSV Headers
########################################
def build_csv_header():
    """
    => [image_name, hand_idx, lm_0_x, lm_0_y, ..., lm_20_x, lm_20_y]
    """
    header = ["image_name", "hand_idx"]
    for i in range(21):
        header.append(f"lm_{i}_x")
        header.append(f"lm_{i}_y")
    return header


########################################
# MediaPipe: 21 2D Landmarks
########################################
def run_mediapipe_on_image(image_bgr):
    """
    Returns a list of (21,2) for each detected hand in pixel coords.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_bgr.shape

    results = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5
    ) as hands:
        output = hands.process(image_rgb)
        if not output.multi_hand_landmarks:
            return results

        for hand_lms in output.multi_hand_landmarks:
            pts_2d = []
            for lm in hand_lms.landmark:
                x_px = lm.x * w
                y_px = lm.y * h
                pts_2d.append((x_px, y_px))
            results.append(np.array(pts_2d))
    return results


########################################
# Inverse Crop => Full Pixel
########################################
def uncrop_pred_2d(pred_2d, box_center, box_size, model_res):
    """
    Convert 2D points from model's cropped coords => full-image pixel coords.
    """
    x_crop = pred_2d[:, 0]
    y_crop = pred_2d[:, 1]

    cx = float(box_center[0])
    cy = float(box_center[1])

    # box_size may be scalar or array
    if isinstance(box_size, np.ndarray):
        if box_size.ndim == 0:
            box_size_val = float(box_size.item())
        elif box_size.ndim == 1:
            if box_size.size == 1:
                box_size_val = float(box_size.item())
            else:
                box_size_val = float(box_size[0])
        else:
            box_size_val = float(box_size.flatten()[0])
    else:
        box_size_val = float(box_size)

    x_full = (x_crop / model_res)*box_size_val + (cx - box_size_val/2.0)
    y_full = (y_crop / model_res)*box_size_val + (cy - box_size_val/2.0)

    return np.stack([x_full, y_full], axis=-1)


########################################
# HaMeR: 21 2D => using pred_keypoints_2d
########################################
class HamerInference:
    def __init__(self, checkpoint=DEFAULT_CKPT_PATH, rescale_factor=2.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        download_models(CACHE_DIR_HAMER)

        self.model_cfg = None
        self.hamer_model = None
        self.renderer   = None
        self.load_model(checkpoint)

        self.cpm = None
        if ViTPoseModel is not None:
            self.cpm = ViTPoseModel(self.device)

        self.rescale_factor = rescale_factor

    def load_model(self, checkpoint_path):
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Cannot find HaMeR checkpoint => {ckpt}")
        print(f"[INFO] Loading HaMeR checkpoint => {ckpt}")
        self.hamer_model, self.model_cfg = load_hamer(str(ckpt))
        self.hamer_model.to(self.device)
        self.hamer_model.eval()

        self.renderer = Renderer(self.model_cfg, faces=self.hamer_model.mano.faces)

    def run_on_image(self, image_bgr, bboxes):
        """
        1) Crop each bounding box => model input
        2) Run HaMeR => pred_keypoints_2d (cropped coords)
        3) Convert to full-image pixel coords => store in all_joints2d
        """
        if len(bboxes) == 0:
            return []

        is_right = np.zeros(len(bboxes), dtype=np.float32)
        dataset = ViTDetDataset(
            self.model_cfg,
            image_bgr,
            bboxes,
            is_right,
            rescale_factor=self.rescale_factor
        )

        all_joints2d = []
        for i in range(len(dataset)):
            sample = dataset[i]

            # Convert sample["img"] to torch if needed
            if isinstance(sample["img"], np.ndarray):
                arr = sample["img"]
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                sample["img"] = torch.from_numpy(arr).float().unsqueeze(0)

            # Possibly convert other fields to torch 
            for k in ["box_center", "box_size", "img_size"]:
                val = sample[k]
                if isinstance(val, np.ndarray):
                    val_t = torch.from_numpy(val).float()
                    if val_t.ndim == 0:
                        val_t = val_t.unsqueeze(0)
                    sample[k] = val_t.unsqueeze(0)

            if isinstance(sample["right"], (int, float, np.float32)):
                sample["right"] = torch.tensor(sample["right"], dtype=torch.float32).unsqueeze(0)

            sample = recursive_to(sample, self.device)

            with torch.no_grad():
                out = self.hamer_model(sample)

            if "pred_keypoints_2d" not in out:
                continue

            pred_2d_crop = out["pred_keypoints_2d"][0].cpu().numpy()  # (21,2)

            bc_tensor = sample["box_center"]
            bs_tensor = sample["box_size"]
            model_res = self.model_cfg.MODEL.IMAGE_SIZE

            if torch.is_tensor(bc_tensor):
                bc_tensor = bc_tensor.cpu().numpy()
            if torch.is_tensor(bs_tensor):
                bs_tensor = bs_tensor.cpu().numpy()

            box_center = bc_tensor.flatten()
            box_size   = bs_tensor.flatten()

            box_size_val = float(box_size[0])

            pred_2d_full = uncrop_pred_2d(pred_2d_crop, box_center, box_size_val, model_res)
            all_joints2d.append(pred_2d_full)

        return all_joints2d


########################################
# RMS Utilities
########################################
def compute_rms_error(pred_pts, gt_pts):
    """
    Legacy function: Returns RMS across all 21 keypoints.
    (You may or may not need it, but kept here for reference.)
    """
    if pred_pts.shape != (21,2) or gt_pts.shape != (21,2):
        raise ValueError("Expected shape (21,2) for both pred & gt.")
    diff = pred_pts - gt_pts
    sq_diff = (diff**2).sum(axis=1)
    mean_sq = np.mean(sq_diff)
    return float(np.sqrt(mean_sq))


def compute_normalized_errors(pred_pts, gt_pts, eps=1e-8):
    """
    For each keypoint i:
      distance_i = Euclidian distance (pred[i] vs. gt[i])
      origin_dist_i = Euclidian distance (gt[i] vs. origin)
    => ratio_i = distance_i / (origin_dist_i + eps)

    Returns a list of 21 ratios (one per keypoint).
    """
    if pred_pts.shape != (21,2) or gt_pts.shape != (21,2):
        raise ValueError("Expected shape (21,2) for both pred & gt.")
    ratios = []
    for i in range(21):
        px, py = pred_pts[i]
        gx, gy = gt_pts[i]
        dist_pred_gt = np.sqrt((px - gx)**2 + (py - gy)**2)
        dist_gt_origin = np.sqrt((gx**2) + (gy**2))
        ratio_i = dist_pred_gt / (dist_gt_origin + eps)
        ratios.append(ratio_i)
    return ratios


########################################
# Main
########################################
def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe + HaMeR (pred_keypoints_2d => uncropped) w/ ground-truth + per-keypoint normalized error (CSV-only)."
    )
    parser.add_argument("root_dir", type=str, help="Root directory of .jpg & .json.")
    parser.add_argument("--hamer_ckpt", type=str, default=DEFAULT_CKPT_PATH,
                        help="Path to HaMeR checkpoint (.ckpt).")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save CSVs.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV paths
    mediapipe_csv_path    = out_dir / "mediapipe_landmarks.csv"
    hamer_csv_path        = out_dir / "hamer_landmarks.csv"
    ground_truth_csv_path = out_dir / "ground_truth.csv"
    rms_csv_path          = out_dir / "rms_errors.csv"

    # Build the standard (x,y) landmark CSV headers
    header = build_csv_header()

    # Overwrite the landmarks CSVs
    with open(mediapipe_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    with open(hamer_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    with open(ground_truth_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    # Build new RMS CSV header => image_name + 21 MP columns + 21 HaMeR columns
    rms_header = ["image_name"]
    for i in range(21):
        rms_header.append(f"mediapipe_kp{i}")
    for i in range(21):
        rms_header.append(f"hamer_kp{i}")

    with open(rms_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(rms_header)

    # Initialize HaMeR
    hamer_infer = HamerInference(checkpoint=args.hamer_ckpt, rescale_factor=2.0)

    # Find all .jpg
    all_jpgs = list(root_dir.rglob("*.jpg"))
    all_jpgs.sort()

    for jpg_path in all_jpgs:
        image_name = jpg_path.name
        print(f"\n[INFO] Processing => {jpg_path}")

        image_bgr = cv2.imread(str(jpg_path))
        if image_bgr is None:
            print(f"WARNING: Could not read {jpg_path}")
            continue

        # 1) MediaPipe
        mp_results = run_mediapipe_on_image(image_bgr)
        with open(mediapipe_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for hand_idx, coords_21 in enumerate(mp_results):
                row = [image_name, hand_idx]
                for (x_px, y_px) in coords_21:
                    row.append(f"{x_px:.2f}")
                    row.append(f"{y_px:.2f}")
                writer.writerow(row)

        # 2) JSON => bounding boxes + keypoints
        base_stem = jpg_path.stem
        json_candidates = list(jpg_path.parent.glob(f"{base_stem}*.json"))
        if not json_candidates:
            print("  -> No JSON => skip HaMeR + RMS.")
            continue

        gt_json = json_candidates[0]
        with open(gt_json, "r") as jf:
            data_list = json.load(jf)
        if not data_list or not isinstance(data_list, list):
            print(f"  -> JSON not in list format => {gt_json}")
            continue

        data_dict = data_list[0]
        if "bbox" not in data_dict or "keypoints" not in data_dict:
            print(f"  -> JSON missing 'bbox' or 'keypoints': {gt_json}")
            continue

        bboxes = np.array(data_dict["bbox"], dtype=np.float32)
        h, w = image_bgr.shape[:2]
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i]
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))
            bboxes[i] = [x1, y1, x2, y2]

        # 3) HaMeR => un-cropped pred_keypoints_2d
        hamer_results = hamer_infer.run_on_image(image_bgr, bboxes)
        with open(hamer_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for hand_idx, coords_21 in enumerate(hamer_results):
                row = [image_name, hand_idx]
                for (x_px, y_px) in coords_21:
                    row.append(f"{x_px:.2f}")
                    row.append(f"{y_px:.2f}")
                writer.writerow(row)

        # 4) Ground Truth => store
        gt_keypoints = np.array(data_dict["keypoints"], dtype=np.float32)
        if gt_keypoints.shape != (21,2):
            print(f"WARNING: GT keypoints => shape {gt_keypoints.shape}, expected (21,2).")
            continue

        with open(ground_truth_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [image_name, 0]
            for (gx, gy) in gt_keypoints:
                row.append(f"{gx:.2f}")
                row.append(f"{gy:.2f}")
            writer.writerow(row)

        # 5) Compute per-keypoint normalized error => pred vs GT, / GT vs origin
        #    We'll only use the first predicted hand from MP & HaMeR if they exist
        mp_ratios = [None]*21
        hamer_ratios = [None]*21

        if len(mp_results) > 0:
            mp_ratios = compute_normalized_errors(mp_results[0], gt_keypoints)
        if len(hamer_results) > 0:
            hamer_ratios = compute_normalized_errors(hamer_results[0], gt_keypoints)

        # 6) Write row => image_name + 21 MP ratio columns + 21 HaMeR ratio columns
        row = [image_name]
        for val in mp_ratios:
            row.append(f"{val:.4f}" if val is not None else "None")
        for val in hamer_ratios:
            row.append(f"{val:.4f}" if val is not None else "None")

        with open(rms_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    print("\n[INFO] Processing complete.")
    print(f"[INFO] MediaPipe CSV   => {mediapipe_csv_path}")
    print(f"[INFO] HaMeR CSV       => {hamer_csv_path}")
    print(f"[INFO] Ground Truth CSV=> {ground_truth_csv_path}")
    print(f"[INFO] RMS CSV         => {rms_csv_path}")
    # Plot references removed; no HTML or Plotly calls.

if __name__ == "__main__":
    main()
