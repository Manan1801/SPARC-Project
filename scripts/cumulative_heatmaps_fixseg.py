#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import math
import re
import sys

# ================== Fixed master header (always these columns) ==================
MASTER_KP_LABELS = ["L0", "R0", "L4", "R4"]
N_SEGMENTS = 6
FPS = 30
MAX_DURATION = 1213.0

# ELLIPSOID_NSTD = 2.5  # 90% coverage
ELLIPSOID_NSTD = 2.795  # 95% coverage

def fit_cov_ellipsoid(points, n_std=ELLIPSOID_NSTD):
    if points is None or points.shape[0] < 4:
        return None
    center = np.mean(points, axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    radii = n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    return center, radii, eigvecs, eigvals

def extract_participant_id(csv_path: Path) -> str:
    p = csv_path.resolve()
    for parent in p.parents:
        if parent.name.lower() == 'cam2':
            return parent.parent.name
    for parent in p.parents:
        if re.fullmatch(r'\d+', parent.name):
            return parent.name
    return p.stem

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_kp_token(tok: str):
    m = re.fullmatch(r'([LRlr])\s*(\d{1,2})', tok.strip())
    if not m:
        raise ValueError(f"Bad keypoint token: {tok} (use like L0, R4)")
    return m.group(1).upper(), int(m.group(2))

def load_points_for_keypoint(df, hand_prefix: str, lm: int):
    cols = [f'{hand_prefix}_{lm}_X_mm', f'{hand_prefix}_{lm}_Y_mm', f'{hand_prefix}_{lm}_Z_mm']
    if not all(c in df.columns for c in cols):
        return None
    sub = df[cols].dropna()
    if sub.empty:
        return None
    return sub.to_numpy()

# ================== CSV append (ALWAYS master schema over 6 segments) ==================
def expected_header():
    cols = ['participant_id', 'Learning Gain']
    for s in range(1, N_SEGMENTS + 1):
        for label in MASTER_KP_LABELS:
            cols.append(f'vol_{label}_seg{s}')
    return cols

def append_volumes(out_csv: Path, participant_id: str, learning_gain: str, seg_to_vols: dict):
    cols = expected_header()
    write_header = not out_csv.exists()

    if out_csv.exists():
        try:
            with open(out_csv, 'r') as f:
                existing = [c.strip() for c in f.readline().strip().split(',')]
            if existing and existing != cols:
                print("⚠️ Existing header differs from expected master schema (6 segments).")
                print("   Existing:", existing)
                print("   Expected:", cols)
        except Exception as e:
            print(f"⚠️ Could not validate existing header: {e}")

    row = [participant_id, learning_gain]
    for s in range(1, N_SEGMENTS + 1):
        kp_map = seg_to_vols.get(s, {}) or {}
        for label in MASTER_KP_LABELS:
            v = kp_map.get(label, "")
            row.append("" if v in (None, "None", "") else str(v))

    assert len(row) == len(cols), f"Row length {len(row)} != header length {len(cols)}"

    with open(out_csv, 'a' if out_csv.exists() else 'w') as f:
        if write_header:
            f.write(','.join(cols) + '\n')
        f.write(','.join(row) + '\n')

# ================== Segmentation helpers ==================
def seg_bounds_seconds():
    seg_len = MAX_DURATION / N_SEGMENTS
    return [(seg_len * (j - 1), seg_len * j) for j in range(1, N_SEGMENTS + 1)]

def time_from_frame(frame_idx: np.ndarray) -> np.ndarray:
    return frame_idx.astype(float) / FPS

# ================== CLI / Main ==================
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute-only: append per-segment (dynamic) 3D ellipsoid volumes (L0,R0,L4,R4) to a master CSV. No plots."
    )
    p.add_argument('--csv', required=True, help="Path to input CSV")
    p.add_argument('--out-csv', required=True, help="CSV to append volumes (fixed master schema over 6 segments)")
    p.add_argument('--keypoints', nargs='+', default=['L0','R0','L4','R4'])
    p.add_argument('--start', type=float, default=0.0, help="Start time in seconds (inclusive)")
    p.add_argument('--stop', type=float, default=None, help="Stop time in seconds (inclusive). If omitted, uses data extent.")
    p.add_argument('--class', dest='learning_gain', required=True, choices=['N','P'],
                   help="Learning Gain class label (N or P) to store in 'Learning Gain' column")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    ensure_parent(out_csv)

    try:
        kp_defs = [parse_kp_token(k) for k in args.keypoints]
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if 'frame' not in df.columns:
        df = df.reset_index(drop=True)
        df['frame'] = np.arange(len(df), dtype=int)
    else:
        df = df.sort_values('frame').reset_index(drop=True)

    df['_t'] = time_from_frame(df['frame'].to_numpy())

    t_start = max(0.0, float(args.start))
    if args.stop is None:
        data_tmax = float(df['_t'].iloc[-1]) if len(df) else 0.0
        t_stop = min(data_tmax, MAX_DURATION)
    else:
        t_stop = min(float(args.stop), MAX_DURATION)

    if t_stop < t_start:
        print("❌ stop < start after capping to MAX_DURATION; nothing to do.")
        return

    participant_id = extract_participant_id(csv_path)

    seg_bounds = seg_bounds_seconds()
    seg_to_vols = {}

    for j, (seg_start, seg_end) in enumerate(seg_bounds, start=1):
        if seg_start >= t_stop:
            seg_to_vols[j] = {kp: "" for kp in MASTER_KP_LABELS}
            continue

        eff_end = min(t_stop, seg_end)
        if eff_end <= t_start:
            seg_to_vols[j] = {kp: "" for kp in MASTER_KP_LABELS}
            continue

        dfc = df[(df['_t'] >= t_start) & (df['_t'] <= eff_end)]
        if dfc.empty:
            seg_to_vols[j] = {kp: "" for kp in MASTER_KP_LABELS}
            continue

        vols_master = {kp: "" for kp in MASTER_KP_LABELS}
        for (hand_prefix, lm) in kp_defs:
            tag = f"{hand_prefix}{lm}"
            pts = load_points_for_keypoint(dfc, hand_prefix, lm)
            if pts is not None and pts.shape[0] >= 4:
                res = fit_cov_ellipsoid(pts, n_std=ELLIPSOID_NSTD)
                if res is not None:
                    _, radii, _, _ = res
                    vol = (4.0/3.0) * math.pi * (radii[0] * radii[1] * radii[2])
                    vols_master[tag] = f"{vol:.6f}"
        seg_to_vols[j] = vols_master

    append_volumes(out_csv, participant_id, args.learning_gain, seg_to_vols)
    print(f"📝 Appended volumes for {participant_id} with class {args.learning_gain} to: {out_csv}")

if __name__ == "__main__":
    main()
