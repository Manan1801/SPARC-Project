#!/usr/bin/env python3
"""
cumulative_movement_csv.py  — dynamic header growth (append-safe) + selectable keypoints

- Computes cumulative movement per keypoint from RAW 3D landmark CSV(s) with:
    * Strict per-frame Euclidean distance to last previous valid sample (no interpolation)
    * Per-keypoint IQR outlier removal (on pre-filter series), then recompute
    * Chunking by --chunksize seconds → per-chunk sums → cumulative
- Writes/updates ONE output CSV that your Plotly script expects:
    pid, class, L0_p1..L0_pM, L1_p1.., ..., R4_p1..R4_pM

Dynamic header growth:
- If the output file exists with N chunks and a new run needs M > N chunks:
    * The script widens the header to M chunks.
    * Pads all existing rows per keypoint by repeating their last cumulative value.
- If M < N (new row shorter), it pads the new row to N chunks by repeating its last value.

Usage:
  python generate_cumulative_movement_csv.py \
    --csv /data/01/cam2/CSV/hand_landmark_xyz.csv \
    --csv /data/02/cam2/CSV/hand_landmark_xyz.csv \
    --start 10 --stop 70 \
    --class Learner \
    --keypoints 0 1 4    → compute/append only those keypoint numbers for BOTH hands (L*, R*).
                        (e.g., --keypoints 0 → L0 and R0) \
    --out cumulative_movement.csv
"""

from pathlib import Path
import argparse
import math
import re
import os
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ----------------------- Constants -----------------------
HANDS = ['L', 'R']
KEYPOINTS_ALL = [0, 1, 2, 3, 4]
PID_RE = re.compile(r"(?:^|/)(\d{2})(?:/|$)")
COL_PART_RE = re.compile(r'^(?P<lab>[LR][0-4])_p(?P<p>\d+)$')

# ----------------------- Args ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/append cumulative movement CSV (dynamic header growth).")
    p.add_argument("--csv", type=Path, action="append", required=True,
                   help="Path(s) to participant landmark CSV(s). Pass multiple times.")
    p.add_argument("--start", type=float, required=True, help="Start time in seconds.")
    p.add_argument("--stop",  type=float, required=True, help="Stop time in seconds (must be > start).")
    p.add_argument("--class", dest="klass", type=str, required=True,
                   choices=["Learner", "Non-Learner"],
                   help="Participant class label written for these rows.")
    p.add_argument("--out", type=Path, default=Path("cumulative_movement.csv"),
                   help="Output CSV path (single file). Default: cumulative_movement.csv")
    p.add_argument("--chunksize", type=int, default=10,
                   help="Chunk size in seconds (default: 10).")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second (default: 30).")
    p.add_argument("--keypoints", type=int, nargs="+", choices=KEYPOINTS_ALL, default=[0,4],
                   help="Keypoint numbers to compute (0–4). Applies to BOTH hands (L*, R*). "
                        "Default: 0 and 4.")
    return p.parse_args()

# -------------------- Helpers / Math ---------------------
def infer_pid(csv_path: Path) -> str:
    m = PID_RE.search(str(csv_path))
    return m.group(1) if m else csv_path.stem

def euclid3(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))

def perframe_series_strict(df: pd.DataFrame, hand: str, kp: int,
                           treat_missing_idx: Optional[Set[int]] = None) -> Tuple[float, List[Optional[float]]]:
    cols = [f"{hand}_{kp}_X_mm", f"{hand}_{kp}_Y_mm", f"{hand}_{kp}_Z_mm"]
    coords = df[cols].copy()
    series: List[Optional[float]] = [None] * len(coords)
    total = 0.0
    prev_valid: Optional[int] = None
    tmiss = treat_missing_idx or set()

    for i in range(len(coords)):
        if i in tmiss:
            series[i] = None
            continue
        row = coords.iloc[i]
        if row.isnull().any():
            series[i] = None
            continue
        if prev_valid is None:
            series[i] = 0.0
        else:
            cur = row.values.astype(float)
            prv = coords.iloc[prev_valid].values.astype(float)
            dist = euclid3(cur, prv)
            series[i] = dist
            total += dist
        prev_valid = i
    return total, series

def iqr_outlier_indices(values: List[Optional[float]]) -> Set[int]:
    data = [v for v in values if v is not None]
    if not data:
        return set()
    arr = np.asarray(data, float)
    try:
        q1 = np.percentile(arr, 25.0, method="linear")
        q3 = np.percentile(arr, 75.0, method="linear")
    except TypeError:
        q1 = np.percentile(arr, 25.0)
        q3 = np.percentile(arr, 75.0)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    out = set()
    for i, v in enumerate(values):
        if v is None:
            continue
        if v < lo or v > hi:
            out.add(i)
    return out

# ---------------- Lab selection / Columns -----------------
def labs_from_keypoints(knums: List[int]) -> List[str]:
    """Build labs (e.g., L0,R0,L2,R2) in stable order L0..L4,R0..R4 filtered by knums."""
    order = [f"{h}{k}" for h in HANDS for k in KEYPOINTS_ALL]  # L0..L4,R0..R4
    want = set()
    for k in knums:
        for h in HANDS:
            want.add(f"{h}{k}")
    return [lab for lab in order if lab in want]

def extract_existing_labs(columns: List[str]) -> List[str]:
    labs = []
    seen = set()
    for c in columns:
        m = COL_PART_RE.match(c)
        if m:
            lab = m.group("lab")
            if lab not in seen:
                labs.append(lab); seen.add(lab)
    # keep in canonical order
    order = [f"{h}{k}" for h in HANDS for k in KEYPOINTS_ALL]
    labs = [lab for lab in order if lab in set(labs)]
    return labs

def build_columns(n_chunks: int, labs: List[str]) -> List[str]:
    cols = ["pid", "class"]
    for lab in labs:
        for p in range(1, n_chunks + 1):
            cols.append(f"{lab}_p{p}")
    return cols

def existing_chunk_count(columns: List[str]) -> int:
    max_p = 0
    for c in columns:
        m = COL_PART_RE.match(c)
        if m:
            p = int(m.group("p"))
            if p > max_p:
                max_p = p
    return max_p

# ----------- Optimized header widening (no fragmentation) -----------
def ensure_dataframe_has_chunks_and_labs(df: pd.DataFrame, n_chunks_target: int, labs_target: List[str]) -> pd.DataFrame:
    """
    Ensure df has all target labs and up to target chunks; pad by repeating last values.
    Uses batch concat to avoid DataFrame fragmentation warnings and improve performance.
    """
    def _existing_chunk_count(cols: List[str]) -> int:
        max_p = 0
        for c in cols:
            m = COL_PART_RE.match(c)
            if m:
                p = int(m.group("p"))
                if p > max_p:
                    max_p = p
        return max_p

    current_chunks = _existing_chunk_count(list(df.columns))

    # 1) Ensure missing lab columns up to current_chunks exist (if any rows present)
    to_add_1 = {}
    if len(df) > 0 and current_chunks > 0:
        for lab in labs_target:
            for p in range(1, current_chunks + 1):
                col = f"{lab}_p{p}"
                if col not in df.columns:
                    to_add_1[col] = np.zeros(len(df), dtype=float)
    if to_add_1:
        df = pd.concat([df, pd.DataFrame(to_add_1, index=df.index)], axis=1, copy=False)

    # 2) Widen chunks from current_chunks → n_chunks_target by repeating last available values
    if n_chunks_target > current_chunks and len(df) > 0:
        to_add_2 = {}
        for lab in labs_target:
            if current_chunks == 0:
                last_series = np.zeros(len(df), dtype=float)
            else:
                last_col = f"{lab}_p{current_chunks}"
                last_series = df[last_col].to_numpy(dtype=float) if last_col in df.columns else np.zeros(len(df), dtype=float)
            reps = n_chunks_target - current_chunks
            if reps > 0:
                block = np.tile(last_series.reshape(-1, 1), (1, reps))
                for off, p in enumerate(range(current_chunks + 1, n_chunks_target + 1)):
                    to_add_2[f"{lab}_p{p}"] = block[:, off]
        if to_add_2:
            df = pd.concat([df, pd.DataFrame(to_add_2, index=df.index)], axis=1, copy=False)

    # 3) Reorder columns to standard order (pid, class, labs_target×p1..pN)
    ordered = build_columns(n_chunks_target, labs_target)
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras] if extras else df[ordered]

def pad_row_values(values_per_lab: Dict[str, List[float]], n_chunks_target: int, labs_target: List[str]) -> List[str]:
    """Flatten per-lab cumulative lists into ordered p1..pN strings for target labs, padding by last value."""
    out: List[str] = []
    for lab in labs_target:
        series = values_per_lab.get(lab, None)
        if series is None:
            # This lab wasn't computed this run → write blanks for all chunks
            out += ["" for _ in range(n_chunks_target)]
            continue
        if len(series) < n_chunks_target:
            if series:
                series = series + [series[-1]] * (n_chunks_target - len(series))
            else:
                series = [0.0] * n_chunks_target
        elif len(series) > n_chunks_target:
            series = series[:n_chunks_target]
        out += [f"{v:.6f}" for v in series]
    return out

# ---------------- Core cumulative computation --------------
def compute_cumulative_for_csv(csv_path: Path, start_s: float, stop_s: float,
                               fps: int, chunksize: int,
                               labs_for_run: List[str]) -> Tuple[str, Dict[str, List[float]]]:
    pid = infer_pid(csv_path)
    df = pd.read_csv(csv_path)

    if "frame" not in df.columns:
        raise ValueError(f"{csv_path} missing 'frame' column.")

    # Sanity columns exist for labs we plan to compute
    for lab in labs_for_run:
        h, k = lab[0], int(lab[1])
        for ax in ("X", "Y", "Z"):
            col = f"{h}_{k}_{ax}_mm"
            if col not in df.columns:
                raise ValueError(f"{csv_path} missing column: {col}")

    if stop_s <= start_s:
        raise ValueError("--stop must be greater than --start.")

    start_f = int(start_s * fps)
    stop_f  = int(stop_s  * fps)
    win = df[(df["frame"] >= start_f) & (df["frame"] < stop_f)].reset_index(drop=True)
    win.replace("", pd.NA, inplace=True)

    # Pre-filter series (only for requested labs)
    series_pre: Dict[str, List[Optional[float]]] = {}
    for lab in labs_for_run:
        h, k = lab[0], int(lab[1])
        _, s = perframe_series_strict(win, h, k, treat_missing_idx=None)
        series_pre[lab] = s

    # IQR outliers
    out_idx: Dict[str, Set[int]] = {lab: iqr_outlier_indices(series_pre[lab]) for lab in labs_for_run}

    # Post-filter series
    series_post: Dict[str, List[Optional[float]]] = {}
    for lab in labs_for_run:
        h, k = lab[0], int(lab[1])
        _, s = perframe_series_strict(win, h, k, treat_missing_idx=out_idx[lab])
        series_post[lab] = s

    # Chunking → cumulative
    n_frames = len(win)
    frames_per_chunk = max(chunksize * fps, 1)
    n_chunks = math.ceil(n_frames / frames_per_chunk) if n_frames > 0 else 0

    cumulative_by_lab: Dict[str, List[float]] = {lab: [] for lab in labs_for_run}
    if n_chunks == 0:
        return pid, cumulative_by_lab  # empty lists

    for lab in labs_for_run:
        pf = series_post[lab]
        per_chunk = []
        for i in range(n_chunks):
            s = i * frames_per_chunk
            e = min((i + 1) * frames_per_chunk, n_frames)
            vals = [v for v in pf[s:e] if v is not None]
            per_chunk.append(float(np.sum(vals)) if vals else 0.0)
        cumulative_by_lab[lab] = np.cumsum(per_chunk).tolist()

    return pid, cumulative_by_lab

# ------------------------------- Main -------------------------------
def main():
    args = parse_args()
    if args.stop <= args.start:
        raise ValueError("--stop must be greater than --start.")

    # Determine which keypoint numbers to compute this run
    knums = args.keypoints if args.keypoints is not None else KEYPOINTS_ALL
    labs_for_run = labs_from_keypoints(knums)          # labs computed now (e.g., L0,R0,L2,R2)
    canonical_order = [f"{h}{k}" for h in HANDS for k in KEYPOINTS_ALL]

    # Compute all rows for this run and find max chunks needed
    run_rows: List[Tuple[str, Dict[str, List[float]]]] = []
    run_max_chunks = 0
    for csv_path in args.csv:
        pid, cum_by_lab = compute_cumulative_for_csv(
            csv_path=csv_path,
            start_s=args.start,
            stop_s=args.stop,
            fps=args.fps,
            chunksize=args.chunksize,
            labs_for_run=labs_for_run
        )
        chunks_this = len(next(iter(cum_by_lab.values()), [])) if cum_by_lab else 0
        if chunks_this > run_max_chunks:
            run_max_chunks = chunks_this
        run_rows.append((pid, cum_by_lab))

    if run_max_chunks == 0:
        print("[!] No frames in the specified window for provided CSV(s). Nothing appended.")
        return

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(out_path)

    if not file_exists:
        # Fresh file: labs present will be only those computed this run
        target_labs = labs_for_run[:]  # keep current selection
        cols = build_columns(run_max_chunks, target_labs)
        with open(out_path, "w") as f:
            f.write(",".join(cols) + "\n")
        existing_chunks = run_max_chunks
        df_existing = pd.DataFrame(columns=cols)
    else:
        # Load and merge labs
        df_existing = pd.read_csv(out_path)
        existing_chunks = existing_chunk_count(list(df_existing.columns))
        existing_labs = extract_existing_labs(list(df_existing.columns))
        # target labs = union(existing, current run), in canonical order
        union_set = set(existing_labs) | set(labs_for_run)
        target_labs = [lab for lab in canonical_order if lab in union_set]

        if existing_chunks == 0:
            # Recover from malformed file: reset with this run's header
            cols = build_columns(run_max_chunks, target_labs)
            df_existing = pd.DataFrame(columns=cols)
            existing_chunks = run_max_chunks

    # Determine target chunk width (max of existing vs this run)
    target_chunks = max(existing_chunks, run_max_chunks)

    # Ensure existing dataframe has all target labs and chunks (optimized, no fragmentation)
    if len(df_existing) > 0:
        df_existing = ensure_dataframe_has_chunks_and_labs(df_existing, target_chunks, target_labs)
    else:
        df_existing = pd.DataFrame(columns=build_columns(target_chunks, target_labs))

    # Build new rows padded/truncated to target_chunks, across target_labs
    new_rows_data: List[List[str]] = []
    for pid, cum_by_lab in run_rows:
        row = [pid, args.klass] + pad_row_values(cum_by_lab, target_chunks, target_labs)
        new_rows_data.append(row)

    # Write back (keep consistent ordering)
    cols_final = build_columns(target_chunks, target_labs)
    df_new = pd.DataFrame(new_rows_data, columns=cols_final)
    df_out = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
    df_out = df_out[cols_final]
    df_out.to_csv(out_path, index=False)

    print(f"[OK] Wrote {len(run_rows)} row(s) → {out_path}")
    print(f"    Header chunks: {target_chunks}  (each chunk = {args.chunksize}s)")
    print(f"    Labs in file: {', '.join(target_labs)}")

if __name__ == "__main__":
    main()
