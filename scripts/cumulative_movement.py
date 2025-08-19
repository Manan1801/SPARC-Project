#!/usr/bin/env python3
"""
cumulative_movement.py

Convert per-chunk movement CSV (e.g., movement_E.csv, movement_N.csv)
into cumulative movement over time for each keypoint and hand.

- Keeps the same columns and ordering (pid first, then the original columns).
- Treats NaN as 0 *only for summation* so cumulative values continue across gaps.
- Writes <input_stem>_cumulative.csv by default next to the input file.

Usage:
    python cumulative_movement.py /path/to/movement_E.csv
    python cumulative_movement.py /path/to/movement_N.csv -o /path/to/out.csv
"""

import re
import argparse
from pathlib import Path
import pandas as pd


COL_RE = re.compile(r'^(?P<hand>[LR])(?P<kp>[0-4])_p(?P<part>\d+)$')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert movement CSV to cumulative per-keypoint values.")
    p.add_argument("csv", type=Path, help="Input CSV path (e.g., movement_E.csv)")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output CSV path (default: <input_stem>_cumulative.csv beside the input)")
    return p.parse_args()

def group_columns(columns):
    """
    Return:
      groups: dict[(hand, kp)] -> list of (col_name, part_number_int)
    """
    groups = {}
    for c in columns:
        m = COL_RE.match(c)
        if not m:
            # skip non-movement columns (e.g., 'pid')
            continue
        hand = m.group("hand")
        kp = int(m.group("kp"))
        part = int(m.group("part"))
        groups.setdefault((hand, kp), []).append((c, part))
    # sort each group's columns by part number ascending
    for k in groups:
        groups[k].sort(key=lambda x: x[1])
    return groups

def make_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve original order
    original_cols = df.columns.tolist()

    # Identify movement columns grouped by (hand, kp)
    groups = group_columns(original_cols)

    # Work on a copy so we don’t mutate the input df
    out = df.copy()

    # For each keypoint group, compute row-wise cumulative sums over parts
    # NaN treated as 0 for summation (only for the cumulative computation)
    for (hand, kp), items in groups.items():
        cols_sorted = [name for name, _ in items]
        block = out[cols_sorted]                   # shape: (n_rows, n_parts)
        cum_block = block.fillna(0).cumsum(axis=1) # cumulative per row across parts
        out[cols_sorted] = cum_block

    # Ensure column order unchanged
    out = out[original_cols]
    return out

def main():
    args = parse_args()
    inp: Path = args.csv
    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    df = pd.read_csv(inp)
    if "pid" not in df.columns:
        raise ValueError("Expected a 'pid' column as the first column.")

    out_df = make_cumulative(df)

    out_path = args.out if args.out is not None else inp.with_name(f"{inp.stem}_cumulative.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote cumulative CSV: {out_path}")

if __name__ == "__main__":
    main()
