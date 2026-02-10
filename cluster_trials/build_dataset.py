#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 (uploaded wide schema + task window):
Build pooled movement dataset from per-participant CSVs listed in manifest.txt.

MANIFEST FORMAT (one line per participant):
<csv_path><sep><start_time_sec><sep><end_time_sec>
- <sep> can be comma(s) or whitespace. Example lines:
    /mnt/data/AP_04_xyz.csv,180,600
    /mnt/data/AP_25_xyz.csv  45   510

ASSUMPTIONS (matches uploaded files like /mnt/data/AP_04_xyz.csv, AP_25_xyz.csv):
- Wide layout columns include:
    frame,
    L_0_X_mm, L_0_Y_mm, L_0_Z_mm, ..., R_4_X_mm, R_4_Y_mm, R_4_Z_mm   (USED)
    (Pixel columns like *_px may exist but are ignored)
- Known labeling issue: input 'L_*' columns are ACTUAL RIGHT, and input 'R_*' are ACTUAL LEFT.
  We canonicalize so that:
      canonical L_* (left)  = input R_*  (swap)
      canonical R_* (right) = input L_*  (swap)
- Δ movement computed only when BOTH hands and ALL 10 keypoints (0..4 each) are present
  at frame t and t-1 (in mm space).
- Only frames within the task window are considered, where:
    start_frame = ceil(start_time_sec * fps), end_frame = floor(end_time_sec * fps)

Outputs:
  outdir/pooled_movement_raw.csv        -> participant_id, frame, [L0..L4]_move, [R0..R4]_move
  outdir/pooled_features_raw.csv        -> same + frame_norm ∈ [0,1] (per participant, after Δ filtering)
  outdir/per_participant/{PID}_movement_raw.csv (optional if --save-per-participant)

Run:
  python step2_build_dataset_uploaded_wide_with_window.py \
      --manifest ingest/manifest.txt \
      --outdir build \
      --fps 30 \
      --save-per-participant
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

HANDS = ["L", "R"]          # Canonical after swap: L = left, R = right
KP_RANGE = range(0, 5)      # keypoints 0..4 used for clustering

# ---------- manifest parsing ----------

_manifest_line_re = re.compile(r"""^\s*
    (?P<path>.+?)            # path (greedy but minimal)
    [,\s]+
    (?P<start>[-+]?\d*\.?\d+)
    [,\s]+
    (?P<end>[-+]?\d*\.?\d+)
    \s*$""", re.VERBOSE)

def read_manifest_with_times(manifest_path: Path) -> list[tuple[Path, float, float]]:
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found at {manifest_path}")
    items: list[tuple[Path,float,float]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = _manifest_line_re.match(s)
            if not m:
                sys.exit(f"ERROR: manifest line {ln} not parsable (expect 'path,start,end'): {line!r}")
            pth = Path(m.group("path")).expanduser()
            try:
                start = float(m.group("start"))
                end = float(m.group("end"))
            except Exception:
                sys.exit(f"ERROR: manifest line {ln} has non-numeric times: {line!r}")
            items.append((pth, start, end))
    if not items:
        sys.exit("ERROR: manifest contains no entries.")
    return items

def derive_participant_id(path: Path) -> str:
    m = re.search(r"(?P<pid>AP_\d+)", str(path), flags=re.IGNORECASE)
    return (m.group("pid") if m else path.stem).upper()

# ---------- schema helpers ----------

def required_mm_columns() -> list[str]:
    cols = []
    for h in HANDS:
        for k in KP_RANGE:
            for comp in ["X", "Y", "Z"]:
                cols.append(f"{h}_{k}_{comp}_mm")
    return cols

def swap_hands_mm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Swap input L_*_mm <-> R_*_mm to canonical.
    After this, df['L_*_mm'] means canonical LEFT; df['R_*_mm'] means canonical RIGHT.
    """
    df = df.copy()
    mm_cols = [c for c in df.columns if c.endswith("_mm") and re.match(r"^[LR]_\d+_[XYZ]_mm$", c)]
    # snapshot original L_*_mm
    left_cols = [c for c in mm_cols if c.startswith("L_")]
    tmp_left = df[left_cols].copy() if left_cols else pd.DataFrame(index=df.index)

    # replace L_* with R_* from input
    for c in left_cols:
        rcol = "R_" + c[2:]
        df[c] = df[rcol] if rcol in df.columns else np.nan

    # replace R_* with saved original L_* from input
    right_cols = [c for c in mm_cols if c.startswith("R_")]
    for c in right_cols:
        lcol = "L_" + c[2:]
        df[c] = tmp_left[lcol] if lcol in tmp_left.columns else (df[c] if c in df.columns else np.nan)

    return df

# ---------- core computations ----------

def clip_to_task_window(df: pd.DataFrame, start_sec: float, end_sec: float, fps: float) -> pd.DataFrame:
    if "frame" not in df.columns:
        raise ValueError("Missing 'frame' column.")
    if end_sec < start_sec:
        raise ValueError(f"Task window end<{end_sec}> earlier than start<{start_sec}> seconds.")
    df = df.copy()
    df = df.dropna(subset=["frame"])
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["frame"]).copy()
    df["frame"] = df["frame"].astype(int)

    start_f = int(np.ceil(start_sec * fps))
    end_f = int(np.floor(end_sec * fps))
    if end_f < start_f:
        # if times are super tight (e.g., <1/fps), expand conservatively to 1 frame
        end_f = start_f

    # clip to available range
    min_f, max_f = int(df["frame"].min()), int(df["frame"].max())
    start_f = max(start_f, min_f)
    end_f = min(end_f, max_f)

    return df[(df["frame"] >= start_f) & (df["frame"] <= end_f)].copy()

def compute_movements_from_mm(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: wide, contains 'frame' and canonical mm columns (already swapped), within task window.
    Returns: DataFrame with columns: frame, [L0..L4]_move, [R0..R4]_move
    Keeps only frames t where all 10 keypoints are present at t and t-1.
    """
    req = required_mm_columns()
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required mm columns after swap: {missing[:6]}{'...' if len(missing)>6 else ''}")

    df = df.sort_values("frame").drop_duplicates(subset=["frame"], keep="first").reset_index(drop=True)

    # presence masks
    present_t = df[req].notna().all(axis=1)
    present_tm1 = present_t.shift(1, fill_value=False)
    valid_delta = present_t & present_tm1

    out = pd.DataFrame({"frame": df["frame"]})
    for h in HANDS:
        for k in KP_RANGE:
            dx = df[f"{h}_{k}_X_mm"] - df[f"{h}_{k}_X_mm"].shift(1)
            dy = df[f"{h}_{k}_Y_mm"] - df[f"{h}_{k}_Y_mm"].shift(1)
            dz = df[f"{h}_{k}_Z_mm"] - df[f"{h}_{k}_Z_mm"].shift(1)
            out[f"{h}{k}_move"] = np.sqrt(dx*dx + dy*dy + dz*dz)

    out = out[valid_delta.values].reset_index(drop=True)
    return out

def add_frame_norm_per_participant(mv: pd.DataFrame) -> pd.DataFrame:
    if mv.empty:
        mv["frame_norm"] = []
        return mv
    mv = mv.sort_values("frame").reset_index(drop=True)
    n = len(mv)
    mv["frame_norm"] = (np.arange(n, dtype=float) / float(max(n - 1, 1))).astype(float)
    return mv

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Step 2: Build pooled movement dataset (uploaded wide schema + task windows).")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest.txt (path,start_sec,end_sec per line)")
    ap.add_argument("--outdir", type=Path, default=Path("build"), help="Output folder (default: ./build)")
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second for time→frame mapping (default: 30)")
    ap.add_argument("--save-per-participant", action="store_true", help="Also save per-participant movement CSVs")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    per_part_dir = args.outdir / "per_participant"
    if args.save_per_participant:
        per_part_dir.mkdir(parents=True, exist_ok=True)

    entries = read_manifest_with_times(args.manifest)

    pooled = []
    total_in_frames = 0
    total_kept_frames = 0

    for path, start_sec, end_sec in entries:
        if not path.exists():
            print(f"[WARN] Missing file: {path}", file=sys.stderr)
            continue

        pid = derive_participant_id(path)

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}", file=sys.stderr)
            continue

        if "frame" not in df.columns:
            print(f"[ERROR] {path.name}: no 'frame' column found.", file=sys.stderr)
            continue

        total_in_frames += len(df)

        # 1) Swap hand labels on mm columns to canonical L/R
        df_swapped = swap_hands_mm(df)

        # 2) Clip to task window in frames based on provided seconds & fps
        try:
            df_clip = clip_to_task_window(df_swapped, start_sec, end_sec, args.fps)
        except Exception as e:
            print(f"[ERROR] {pid}: invalid task window: {e}", file=sys.stderr)
            continue
        if df_clip.empty:
            print(f"[WARN] {pid}: task window yields 0 frames after clipping.", file=sys.stderr)
            continue

        # 3) Compute Δ movements on clipped data
        try:
            mv = compute_movements_from_mm(df_clip)
        except Exception as e:
            print(f"[ERROR] {pid}: movement computation failed: {e}", file=sys.stderr)
            continue
        if mv.empty:
            print(f"[WARN] {pid}: no valid Δ frames inside task window (needs consecutive valid frames).", file=sys.stderr)
            continue

        mv.insert(0, "participant_id", pid)
        total_kept_frames += len(mv)

        # 4) Add per-participant frame_norm AFTER Δ filtering
        mv = add_frame_norm_per_participant(mv)

        if args.save_per_participant:
            cols = ["participant_id", "frame"] + [f"{h}{k}_move" for h in HANDS for k in KP_RANGE]
            mv[cols].to_csv(per_part_dir / f"{pid}_movement_raw.csv", index=False)

        pooled.append(mv)
        print(f"[OK] {pid}: window=({start_sec:.3f}s→{end_sec:.3f}s @ {args.fps:g}fps), "
              f"frames_in={len(df):6d}, frames_in_window={len(df_clip):6d}, frames_kept_for_delta={len(mv):6d}")

    if not pooled:
        sys.exit("ERROR: No participants processed successfully.")

    pooled_df = pd.concat(pooled, ignore_index=True)

    move_cols = [f"{h}{k}_move" for h in HANDS for k in KP_RANGE]

    # pooled_movement_raw.csv (without frame_norm)
    pooled_movement = pooled_df[["participant_id", "frame"] + move_cols].copy()
    pooled_movement.to_csv(args.outdir / "pooled_movement_raw.csv", index=False)

    # pooled_features_raw.csv (with frame_norm)
    pooled_features = pooled_df[["participant_id", "frame", "frame_norm"] + move_cols].copy()
    pooled_features.to_csv(args.outdir / "pooled_features_raw.csv", index=False)

    print("\n[SUMMARY]")
    print(f"  participants: {len(pooled)}")
    print(f"  total_frames_in: {total_in_frames}")
    print(f"  total_frames_kept_for_delta: {total_kept_frames}")
    print(f"[WROTE] {args.outdir/'pooled_movement_raw.csv'}")
    print(f"[WROTE] {args.outdir/'pooled_features_raw.csv'}")

if __name__ == "__main__":
    main()
