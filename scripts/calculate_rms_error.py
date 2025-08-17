#!/usr/bin/env python3
"""
calculate_rms_error.py

Calculates per-keypoint RMS error between MediaPipe predicted X-Y world coordinates stored in a CSV
and ground-truth annotated 3D joint coordinates (world_coord) stored in an InterHand2.6M JSON file.
Includes detailed debug logging of predictions vs ground-truth values per sample.

Input CSV columns expected:
  capture_folder, gesture_class, cam_id, image_name,
  pixel_x_h_kp, pixel_y_h_kp, world_x_h_kp, world_y_h_kp for h∈{0,1}, kp∈{0..20}

Output CSV columns:
  capture_folder, gesture_class, cam_id, image_name,
  rms_error_k0, ..., rms_error_k41

Usage:
    python calculate_rms_error.py \
        --csv_file path/to/predictions.csv \
        --json_file path/to/InterHand2.6M_train_joint_3d.json
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import re


def load_groundtruth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt = {}
    for cap_id, frames in data.items():
        for frame_idx, entry in frames.items():
            coords3d = entry.get('world_coord')
            valid3d = entry.get('joint_valid')
            if coords3d is None or valid3d is None:
                continue
            coords_xy = []
            valid_xy = []
            for kp, v in zip(coords3d, valid3d):
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    coords_xy.append([kp[0], kp[1]])
                else:
                    coords_xy.append([np.nan, np.nan])
                # correctly handle scalar or list valid flags
                if isinstance(v, (list, tuple)):
                    # list case: take first element
                    valid_xy.append(bool(v[0]) if len(v) >= 1 else False)
                else:
                    # scalar case: 0 or 1
                    valid_xy.append(bool(v))
            # normalize IDs: strip non-digits and leading zeros: strip non-digits and leading zeros
            norm_cap = str(int(re.sub(r'\D+', '', str(cap_id))))
            norm_frame = str(int(re.sub(r'\D+', '', str(frame_idx))))
            gt[(norm_cap, norm_frame)] = {'coords': coords_xy, 'valid': valid_xy}
    return gt


def calculate_errors(df, groundtruth):
    # metadata
    meta_cols = ['capture_folder', 'gesture_class', 'cam_id', 'image_name']
    for c in meta_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    meta = df[meta_cols].copy().reset_index(drop=True)

    # identify world columns
    world_pattern = re.compile(r'world_[xy]_(\d+)_(\d+)')
    world_cols = [col for col in df.columns if world_pattern.fullmatch(col)]
    if not world_cols:
        raise ValueError("No world_x/world_y columns found.")
    world_df = df[world_cols].reset_index(drop=True)

    # drop rows with no hand0 coords
    hand0_cols = [c for c in world_cols if c.startswith('world_x_0_') or c.startswith('world_y_0_')]
    mask = world_df[hand0_cols].notna().any(axis=1)
    dropped = (~mask).sum()
    if dropped:
        print(f"Dropping {dropped} rows with no hand0 world coordinates.")
    meta = meta[mask].reset_index(drop=True)
    world_vals = world_df[mask].reset_index(drop=True).astype(float) * 1000.0

    # prepare
    pred_n = world_vals.shape[1] // 2
    errors = []
    total_j = None

    for i in range(len(world_vals)):
        # debug sample info
        cap_raw = meta.at[i, 'capture_folder']
        img_raw = meta.at[i, 'image_name']
        cap_id = str(int(re.sub(r'\D+', '', cap_raw)))
        frame_base = os.path.splitext(img_raw)[0]
        frame_idx = str(int(re.sub(r'\D+', '', frame_base)))
        key = (cap_id, frame_idx)
        print(f"\n--- Sample {i}: {key} ---")
        # show predicted values
        vals = world_vals.iloc[i].to_numpy()
        px, py = vals[0::2], vals[1::2]
        print(f"Pred px[:5]={px[:5]}")
        print(f"Pred py[:5]={py[:5]}")
        if key not in groundtruth:
            print("Available GT keys sample:", list(groundtruth.keys())[:5])
            raise KeyError(f"GT missing for {key}")
        entry = groundtruth[key]
        gt_xy = np.array(entry['coords'], dtype=float)
        valid_xy = np.array(entry['valid'], dtype=bool)
        print(f"GT coords[:5]={gt_xy[:5]}")
        print(f"GT valid[:5]={valid_xy[:5]}")

        if total_j is None:
            total_j = gt_xy.shape[0]
        if gt_xy.shape[0] != total_j:
            raise ValueError("Inconsistent GT joint count.")

        # compute errors
        err_vec = np.full(total_j, np.nan)
        if pred_n == total_j:
            diff2 = (px - gt_xy[:,0])**2 + (py - gt_xy[:,1])**2
            err = np.sqrt(diff2/2)
            # err[~valid_xy] = np.nan
            err_vec = err
        elif total_j == 2*pred_n:
            v1 = valid_xy[:pred_n].sum(); v2 = valid_xy[pred_n:].sum()
            if v1==pred_n and v2<pred_n:
                seg=0
            elif v2==pred_n and v1<pred_n:
                seg=pred_n
            else:
                e1 = np.sqrt(((px-gt_xy[:pred_n,0])**2+(py-gt_xy[:pred_n,1])**2)/2)
                e2 = np.sqrt(((px-gt_xy[pred_n:,0])**2+(py-gt_xy[pred_n:,1])**2)/2)
                seg = 0 if np.nanmean(e1)<=np.nanmean(e2) else pred_n
            diff2 = (px - gt_xy[seg:seg+pred_n,0])**2 + (py - gt_xy[seg:seg+pred_n,1])**2
            err = np.sqrt(diff2/2)
            vs = valid_xy[seg:seg+pred_n]
            err[~vs] = np.nan
            err_vec[seg:seg+pred_n] = err
        else:
            raise ValueError(f"Keypoint mismatch: pred={pred_n}, gt={total_j}")
        errors.append(err_vec)

    # assemble
    rms_cols = [f"rms_error_k{j}" for j in range(total_j)]
    df_err = pd.DataFrame(errors, columns=rms_cols)
    return pd.concat([meta, df_err], axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_file', required=True)
    p.add_argument('--json_file', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_file)
    gt = load_groundtruth(args.json_file)
    out = calculate_errors(df, gt)
    print(f"\nCalculated RMS for {len(out)} samples.")

    base = os.path.splitext(os.path.basename(args.csv_file))[0]
    out_path = f"{base}_rms.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved RMS to {out_path}")

if __name__ == '__main__':
    main()
