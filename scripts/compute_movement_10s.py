#!/usr/bin/env python3
"""
compute_movement_parts.py

Compute summed 3D movement (mm) for landmarks [0..4] of both hands,
chunked into 10-second parts, and flatten to one row per participant.

Usage
-----
python compute_movement_parts.py --class {E,N} --start 0 --stop 607 --csv /path/to/AP_27_xyz.csv --debug

Changes in this version:
- Debug log saved in same folder as input CSV.
- Output CSV excludes 'cls', 'dur_s', and 'parts' columns.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

FPS = 30
CHUNK_SEC = 10
HANDS = ("L", "R")
LMS = [0, 1, 2, 3, 4]  # landmark ids
MM_COLS_TEMPLATE = "{hand}_{lm}_{axis}_mm"  # e.g., L_0_X_mm

def parse_args():
    p = argparse.ArgumentParser(description="Compute 10s-part landmark movement and append to class CSV.")
    p.add_argument("--class", dest="cls", choices=["E", "N", "p", "n"], required=True, help="Expert (E) or Novice (N) or Positive (P) class or Neutral/Negative (N)")
    p.add_argument("--start", type=int, required=True, help="Start time in seconds")
    p.add_argument("--stop", type=int, required=True, help="Stop time in seconds (exclusive)")
    p.add_argument("--csv", dest="csv_path", required=True, help="Input participant CSV path")
    p.add_argument("--debug", action="store_true", help="Write a debug log")
    return p.parse_args()

def extract_pid(csv_path: Path) -> str:
    stem = csv_path.stem
    if "_xyz" in stem:
        return stem.split("_xyz")[0]
    return stem.split("_")[0]

def log_path_short(pid: str, parts: int, cls: str, csv_dir: Path) -> Path:
    return csv_dir / f"debug_movement_{pid}_{cls}_p{parts}.log"

def safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def segment_bounds(start_s: int, stop_s: int):
    total_s = stop_s - start_s
    full_parts = total_s // CHUNK_SEC
    rem = total_s % CHUNK_SEC
    parts = full_parts + (1 if rem > 0 else 0)
    for i in range(parts):
        seg_s_len = CHUNK_SEC if i < full_parts else (rem if rem > 0 else CHUNK_SEC)
        s = (start_s + i * CHUNK_SEC) * FPS
        e = s + seg_s_len * FPS
        yield (i + 1, int(s), int(e), int(seg_s_len))

def path_length_3d(x, y, z):
    arr = np.stack([x, y, z], axis=1)
    valid = ~np.isnan(arr).any(axis=1)
    arr = arr[valid]
    if len(arr) < 2:
        return 0.0
    diffs = arr[1:] - arr[:-1]
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

def compute_for_window(df: pd.DataFrame, f0: int, f1: int, debug_lines) -> dict:
    out = {}
    max_idx = len(df)
    f0 = max(0, min(f0, max_idx))
    f1 = max(0, min(f1, max_idx))
    if f1 <= f0:
        for hand in HANDS:
            for lm in LMS:
                out[(hand, lm)] = 0.0
        debug_lines.append(f"    Window empty after clipping: [{f0},{f1}) -> zeros")
        return out

    sl = slice(f0, f1)
    for hand in HANDS:
        for lm in LMS:
            cols = [MM_COLS_TEMPLATE.format(hand=hand, lm=lm, axis=a) for a in ("X", "Y", "Z")]
            if not all(c in df.columns for c in cols):
                debug_lines.append(f"    Missing columns for {hand}{lm}: {cols}")
                out[(hand, lm)] = np.nan
                continue
            x = df.loc[sl, cols[0]].to_numpy(dtype=float, copy=False)
            y = df.loc[sl, cols[1]].to_numpy(dtype=float, copy=False)
            z = df.loc[sl, cols[2]].to_numpy(dtype=float, copy=False)
            mv = path_length_3d(x, y, z)
            out[(hand, lm)] = mv
            debug_lines.append(f"    {hand}{lm}: movement={mv:.3f} mm")
    return out

def ensure_union_columns(existing: pd.DataFrame, new_cols: list) -> pd.DataFrame:
    union = list(existing.columns)
    for c in new_cols:
        if c not in union:
            union.append(c)
    return existing.reindex(columns=union)

def main():
    args = parse_args()
    in_path = Path(args.csv_path).expanduser().resolve()
    if not in_path.exists():
        print(f"[ERROR] Input CSV not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    pid = extract_pid(in_path)
    cls = args.cls
    start_s, stop_s = args.start, args.stop
    csv_dir = in_path.parent

    df = pd.read_csv(in_path)

    mm_needed = [MM_COLS_TEMPLATE.format(hand=hand, lm=lm, axis=ax)
                 for hand in HANDS for lm in LMS for ax in ("X", "Y", "Z")]
    df = safe_numeric(df, mm_needed)

    parts_info = list(segment_bounds(start_s, stop_s))
    n_parts = len(parts_info)

    debug_lines = [
        f"Input: {in_path}",
        f"PID: {pid} | Class: {cls}",
        f"Start_s={start_s}, Stop_s={stop_s}, FPS={FPS}",
        f"Parts (10s chunks): {n_parts}"
    ]
    for pidx, f0, f1, slen in parts_info:
        debug_lines.append(f"  Part {pidx:02d}: frames [{f0},{f1}) ~ {slen}s")

    # Output row without class/duration/parts
    row = {"pid": pid}

    for pidx, f0, f1, _slen in parts_info:
        debug_lines.append(f"  Computing part {pidx:02d} …")
        part_res = compute_for_window(df, f0, f1, debug_lines)
        for hand in HANDS:
            for lm in LMS:
                col = f"{hand}{lm}_p{pidx:02d}"
                row[col] = part_res.get((hand, lm), np.nan)

    out_csv = Path.cwd() / f"movement_{cls}.csv"

    if out_csv.exists():
        exist_df = pd.read_csv(out_csv)
        new_cols = list(row.keys())
        exist_df = ensure_union_columns(exist_df, new_cols)
        for c in exist_df.columns:
            if c not in row:
                row[c] = np.nan
        new_row_df = pd.DataFrame([row], columns=exist_df.columns)
        final_df = pd.concat([exist_df, new_row_df], ignore_index=True)
    else:
        final_df = pd.DataFrame([row])

    final_df.to_csv(out_csv, index=False)

    if args.debug:
        log_path = log_path_short(pid, n_parts, cls, csv_dir)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(debug_lines))
            f.write("\n")
            f.write(f"Appended to: {out_csv}\n")
            f.write(f"Final columns ({len(final_df.columns)}): {list(final_df.columns)}\n")
        print(f"[DEBUG] Log written: {log_path}")

    print(f"[OK] Appended {pid} → {out_csv} (parts={n_parts})")

if __name__ == "__main__":
    main()
