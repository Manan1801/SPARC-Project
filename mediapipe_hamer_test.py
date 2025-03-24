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
     - rms_errors.csv
  4) Generates an interactive Plotly bar chart => 'rms_plot.html'.
  
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
# Plotly Setup (for RMS)
########################################
try:
    import pandas as pd
    import plotly.express as px
except ImportError:
    print("ERROR: Please install 'pandas' & 'plotly': pip install pandas plotly")
    sys.exit(1)

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
    Assuming top-left origin => [0, model_res], square crop.

    x_full = (x_crop / model_res)*box_size_val + (cx - box_size_val/2)
    y_full = (y_crop / model_res)*box_size_val + (cy - box_size_val/2)
    """
    x_crop = pred_2d[:, 0]
    y_crop = pred_2d[:, 1]

    cx = float(box_center[0])
    cy = float(box_center[1])

    # box_size may be scalar or array
    if isinstance(box_size, np.ndarray):
        # shape can be 0D => scalar, or (1,), (2,) etc.
        if box_size.ndim == 0:
            # scalar => shape ()
            box_size_val = float(box_size.item())
        elif box_size.ndim == 1:
            # e.g. shape(1,) or shape(2,).
            if box_size.size == 1:
                box_size_val = float(box_size.item())
            else:
                # shape(2,) => pick one
                box_size_val = float(box_size[0])
        else:
            # fallback
            box_size_val = float(box_size.flatten()[0])
    else:
        # if it's just a float
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
                    # shape can be scalar (0D) or small vector
                    # We do *not* call [0] on it; just wrap it => shape(1,...)
                    val_t = torch.from_numpy(val).float()
                    if val_t.ndim == 0:
                        # shape ()
                        val_t = val_t.unsqueeze(0)  # shape(1,)
                    sample[k] = val_t.unsqueeze(0)  # => shape(1,1) or shape(1,2), etc.

            if isinstance(sample["right"], (int, float, np.float32)):
                sample["right"] = torch.tensor(sample["right"], dtype=torch.float32).unsqueeze(0)

            sample = recursive_to(sample, self.device)

            with torch.no_grad():
                out = self.hamer_model(sample)

            if "pred_keypoints_2d" not in out:
                continue

            # shape => (B,21,2), typically B=1
            pred_2d_crop = out["pred_keypoints_2d"][0].cpu().numpy()  # (21,2)

            # Inverse the crop
            # sample["box_center"] => shape(1, D). We'll flatten it
            bc_tensor = sample["box_center"]
            bs_tensor = sample["box_size"]
            model_res = self.model_cfg.MODEL.IMAGE_SIZE

            # Convert them to CPU => might be float or tensor
            if torch.is_tensor(bc_tensor):
                bc_tensor = bc_tensor.cpu().numpy()  # shape(N, D)
            if torch.is_tensor(bs_tensor):
                bs_tensor = bs_tensor.cpu().numpy()

            # box_center => if shape(1,2), flatten to (2,)
            box_center = bc_tensor.flatten()
            # box_size => might be shape(1,) or shape(1,2), etc.
            box_size = bs_tensor.flatten()  # shape(...)

            # We'll pick the first element
            box_size_val = float(box_size[0])  # e.g. 
            # If you prefer a safer approach, check if len(box_size)>1

            pred_2d_full = uncrop_pred_2d(pred_2d_crop, box_center, box_size_val, model_res)
            all_joints2d.append(pred_2d_full)

        return all_joints2d

########################################
# RMS Utility
########################################
def compute_rms_error(pred_pts, gt_pts):
    if pred_pts.shape != (21,2) or gt_pts.shape != (21,2):
        raise ValueError("Expected shape (21,2) for both pred & gt.")
    diff = pred_pts - gt_pts
    sq_diff = (diff**2).sum(axis=1)
    mean_sq = np.mean(sq_diff)
    return float(np.sqrt(mean_sq))

########################################
# Plotting
########################################
def generate_rms_plot(rms_csv_path, out_dir):
    df = pd.read_csv(rms_csv_path)
    df["mp_rms"]    = pd.to_numeric(df["mp_rms"], errors="coerce")
    df["hamer_rms"] = pd.to_numeric(df["hamer_rms"], errors="coerce")

    df_melt = df.melt(
        id_vars="image_name",
        value_vars=["mp_rms", "hamer_rms"],
        var_name="method",
        value_name="rms"
    )

    fig = px.bar(
        df_melt,
        x="image_name",
        y="rms",
        color="method",
        barmode="group",
        title="RMS Error Comparison (MediaPipe vs. HaMeR)",
        labels={"image_name":"Image","rms":"RMS Error"}
    )
    fig.update_layout(
        xaxis={"type":"category","categoryorder":"category ascending"}
    )

    out_html = out_dir / "rms_plot.html"
    fig.write_html(str(out_html))
    print(f"[INFO] RMS plot saved => {out_html}")

########################################
# Main
########################################
def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe + HaMeR (pred_keypoints_2d => uncropped) w/ ground-truth + RMS + Plot."
    )
    parser.add_argument("root_dir", type=str, help="Root directory of .jpg & .json.")
    parser.add_argument("--hamer_ckpt", type=str, default=DEFAULT_CKPT_PATH,
                        help="Path to HaMeR checkpoint (.ckpt).")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save CSVs + plot.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV paths
    mediapipe_csv_path    = out_dir / "mediapipe_landmarks.csv"
    hamer_csv_path        = out_dir / "hamer_landmarks.csv"
    ground_truth_csv_path = out_dir / "ground_truth.csv"
    rms_csv_path          = out_dir / "rms_errors.csv"

    header = build_csv_header()

    # Overwrite CSVs
    with open(mediapipe_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    with open(hamer_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    with open(ground_truth_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    with open(rms_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["image_name", "mp_rms", "hamer_rms"])

    # Initialize HaMeR
    hamer_infer = HamerInference(checkpoint=args.hamer_ckpt, rescale_factor=2.0)

    # Find .jpg
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
        # clamp bounding box if needed
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

        # 5) RMS => compare first hand from MP & HaMeR to GT
        mp_rms_value = None
        if len(mp_results) > 0:
            mp_rms_value = compute_rms_error(mp_results[0], gt_keypoints)

        hamer_rms_value = None
        if len(hamer_results) > 0:
            hamer_rms_value = compute_rms_error(hamer_results[0], gt_keypoints)

        print(f"  -> Found GT => {gt_json}")
        print(f"  -> MediaPipe RMS => {mp_rms_value if mp_rms_value is not None else 'None'}")
        print(f"  -> HaMeR RMS     => {hamer_rms_value if hamer_rms_value is not None else 'None'}")

        with open(rms_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                image_name,
                f"{mp_rms_value:.4f}" if mp_rms_value is not None else "None",
                f"{hamer_rms_value:.4f}" if hamer_rms_value is not None else "None"
            ])

    # 6) Plot
    generate_rms_plot(rms_csv_path, out_dir)

    print("\n[INFO] Processing complete.")
    print(f"[INFO] MediaPipe CSV   => {mediapipe_csv_path}")
    print(f"[INFO] HaMeR CSV       => {hamer_csv_path}")
    print(f"[INFO] Ground Truth CSV=> {ground_truth_csv_path}")
    print(f"[INFO] RMS CSV         => {rms_csv_path}")
    print(f"[INFO] Plot            => {out_dir / 'rms_plot.html'}")

if __name__ == "__main__":
    main()
