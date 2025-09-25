#!/usr/bin/env python3
"""

Given a CSV with columns:
  frame,
  L_0_X_px, L_0_Y_px, ... L_4_X_px, L_4_Y_px,
  R_0_X_px, R_0_Y_px, ... R_4_X_px, R_4_Y_px,
  L_0_X_mm, L_0_Y_mm, L_0_Z_mm, ... L_4_X_mm, L_4_Y_mm, L_4_Z_mm,
  R_0_X_mm, R_0_Y_mm, R_0_Z_mm, ... R_4_X_mm, R_4_Y_mm, R_4_Z_mm

…it flips the hand label convention (L ↔ R) for all rows with frame >= START_FRAME.

Usage:
  python scripts/labels_flip.py --csv /path/to/input.csv --start-frame 123 \
      [--out /path/to/output.csv]
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

def build_columns():
    idxs = range(5) # 5 fingertips: 0..4
    px_axes = ["X_px", "Y_px"]
    mm_axes = ["X_mm", "Y_mm", "Z_mm"]

    L_px = [f"L_{i}_{ax}" for i in idxs for ax in px_axes]
    R_px = [f"R_{i}_{ax}" for i in idxs for ax in px_axes]
    L_mm = [f"L_{i}_{ax}" for i in idxs for ax in mm_axes]
    R_mm = [f"R_{i}_{ax}" for i in idxs for ax in mm_axes]
    return L_px, R_px, L_mm, R_mm

def validate_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("[ERROR] Missing expected columns:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

def flip_pairs(df, mask, left_cols, right_cols):
    # Swap values under mask: L <-> R for aligned column pairs.
    for Lc, Rc in zip(left_cols, right_cols):
        left_vals  = df.loc[mask, Lc].values
        right_vals = df.loc[mask, Rc].values
        df.loc[mask, Lc] = right_vals
        df.loc[mask, Rc] = left_vals

def main():
    ap = argparse.ArgumentParser(description="Flip L/R hand labels from a given frame (inclusive).")
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--start-frame", type=int, required=True, help="Frame number from which to flip (inclusive)")
    ap.add_argument("--out", help="Path to output CSV (default: <input>_flipped.csv)")
    args = ap.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        print(f"[ERROR] File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_flipped.csv")

    df = pd.read_csv(in_path)

    if "frame" not in df.columns:
        print("[ERROR] 'frame' column is required.", file=sys.stderr)
        sys.exit(1)

    # Ensure 'frame' is numeric for correct comparison
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    if df["frame"].isna().any():
        print("[ERROR] Non-numeric entries found in 'frame' column.", file=sys.stderr)
        sys.exit(1)

    L_px, R_px, L_mm, R_mm = build_columns()
    validate_columns(df, ["frame"] + L_px + R_px + L_mm + R_mm)

    mask = df["frame"] >= args.start_frame
    affected = int(mask.sum())

    if affected == 0:
        print(f"[INFO] No rows with frame >= {args.start_frame}. Nothing to flip.")
    else:
        flip_pairs(df, mask, L_px, R_px)
        flip_pairs(df, mask, L_mm, R_mm)
        print(f"[OK] Flipped L/R labels on {affected} row(s) (frame >= {args.start_frame}).")

    df.to_csv(out_path, index=False)
    print(f"[DONE] Wrote: {out_path}")

if __name__ == "__main__":
    main()
