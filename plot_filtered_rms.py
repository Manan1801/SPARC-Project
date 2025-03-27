#!/usr/bin/env python3
"""
Script: plot_filtered_rms.py

Description:
  - Reads an "rms_errors.csv" file produced by your pipeline (with columns like):
      image_name,
      mediapipe_kp0, mediapipe_kp1, ..., mediapipe_kp20,
      hamer_kp0,     hamer_kp1,     ..., hamer_kp20
  - Applies various filters:
       1) images_to_include or images_to_exclude
       2) random sampling
       3) keypoint indices to include for MediaPipe & HaMeR
       4) minimum/maximum error thresholds (optional)
  - Sorts the data by specified columns and order.
  - Produces an interactive Plotly chart of the filtered data,
    with each keypoint in a different color, and including
    the keypoint number in the hover tooltips.
  - Saves the chart as an HTML file.

Usage (example):
  python plot_filtered_rms.py \
    --csv rms_errors.csv \
    --output rms_subset_plot.html \
    --include_images frame0001.jpg frame0002.jpg \
    --exclude_images frame9999.jpg \
    --random_sample 10 \
    --mp_kps 0 5 9 \
    --hmr_kps 0 5 20 \
    --min_error 0.05 \
    --max_error 1.5 \
    --sort_by image_name \
    --sort_ascending \
    --plot_type line

Explanation of Key CLI Arguments:
  --csv                 Path to your "rms_errors.csv" file.
  --output              Where to store the resulting HTML plot.
  --include_images      List of exact image names to include (others are dropped).
  --exclude_images      List of exact image names to exclude from the dataset.
  --random_sample       If set, randomly sample this many rows from the filtered set.
  --mp_kps              MediaPipe keypoints to include, e.g. --mp_kps 0 5 9. If empty, uses None.
  --hmr_kps             HaMeR keypoints to include, e.g. --hmr_kps 0 5 20. If empty, uses None.
  --min_error           Filter out rows if all keypoints are below this threshold.
  --max_error           Filter out rows if all keypoints are above this threshold.
  --sort_by             Which column to sort by, e.g. 'image_name', 'mediapipe_kp0'.
  --sort_ascending      Sort ascending (if not specified, defaults to descending).
  --plot_type           'bar', 'box', or 'line' for different visualization styles.
  --random_seed         Random seed used for sampling.
"""

import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter, sort, and plot RMS error data from 'rms_errors.csv'."
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to the 'rms_errors.csv' file."
    )
    parser.add_argument(
        "--output", type=str, default="rms_filtered_plot.html",
        help="Where to save the output HTML plot."
    )
    parser.add_argument(
        "--include_images", nargs='*', default=[],
        help="List of image names to include (exact match). If empty, include all."
    )
    parser.add_argument(
        "--exclude_images", nargs='*', default=[],
        help="List of image names to exclude from the dataset."
    )
    parser.add_argument(
        "--random_sample", type=int, default=0,
        help="If > 0, randomly sample this many rows from the filtered set."
    )
    parser.add_argument(
        "--mp_kps", nargs='*', type=int, default=[],
        help="MediaPipe keypoints to include. If empty => none are used."
    )
    parser.add_argument(
        "--hmr_kps", nargs='*', type=int, default=[],
        help="HaMeR keypoints to include. If empty => none are used."
    )
    parser.add_argument(
        "--min_error", type=float, default=None,
        help="Min error threshold; discard rows if *all* keypoints are below this."
    )
    parser.add_argument(
        "--max_error", type=float, default=None,
        help="Max error threshold; discard rows if *all* keypoints are above this."
    )
    parser.add_argument(
        "--sort_by", type=str, default=None,
        help="Column to sort by, e.g. 'image_name' or 'mediapipe_kp0'."
    )
    parser.add_argument(
        "--sort_ascending", action='store_true',
        help="Sort in ascending order if specified, otherwise descending."
    )
    parser.add_argument(
        "--plot_type", type=str, default="bar",
        choices=["bar", "box", "line"],
        help="Type of plot to create: 'bar', 'box', or 'line'."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed used for sampling."
    )
    return parser.parse_args()


def parse_method_and_kp(col_name):
    """
    Utility: given a column name like 'mediapipe_kp10' => ('MediaPipe', 10).
             for 'hamer_kp0' => ('HaMeR', 0).
             returns (None, None) if it doesn't match either pattern.
    """
    if col_name.startswith("mediapipe_kp"):
        idx_str = col_name.replace("mediapipe_kp", "")
        return ("MediaPipe", int(idx_str))
    elif col_name.startswith("hamer_kp"):
        idx_str = col_name.replace("hamer_kp", "")
        return ("HaMeR", int(idx_str))
    else:
        return (None, None)


def filter_by_min_max_error(df, min_val=None, max_val=None):
    """
    Filter out rows that don't meet min/max error criteria.
    - If `min_val` is set, keep row if it has *at least one* keypoint >= min_val
    - If `max_val` is set, keep row if it has *at least one* keypoint <= max_val
    If a row is entirely below `min_val` or entirely above `max_val`, it's discarded.
    """
    method_cols = [c for c in df.columns if c.startswith("mediapipe_kp") or c.startswith("hamer_kp")]
    df_num = df[method_cols].apply(pd.to_numeric, errors="coerce")

    keep_mask = pd.Series([True]*len(df), index=df.index)

    if min_val is not None:
        row_has_kp_above_min = (df_num >= min_val).any(axis=1)
        keep_mask &= row_has_kp_above_min

    if max_val is not None:
        row_has_kp_below_max = (df_num <= max_val).any(axis=1)
        keep_mask &= row_has_kp_below_max

    return df[keep_mask].copy()


def build_long_format(df):
    """
    Convert wide columns (mediapipe_kp0..20, hamer_kp0..20) to "long" DataFrame
    => columns: [image_name, method, keypoint_idx, error_value].
    """
    melted_rows = []
    for _, row in df.iterrows():
        img_name = row["image_name"]
        for col_name in df.columns:
            if col_name == "image_name":
                continue

            method, kp_idx = parse_method_and_kp(col_name)
            if method is None:
                continue  # Not a recognized method/kp column

            val_str = row[col_name]
            try:
                val = float(val_str)
            except:
                val = None

            melted_rows.append({
                "image_name": img_name,
                "method": method,
                "keypoint_idx": kp_idx,
                "error_value": val
            })
    return pd.DataFrame(melted_rows)


def main():
    args = parse_args()

    # 1) Read CSV
    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found => {args.csv}")
        return

    # Read the file
    df = pd.read_csv(args.csv)

    # ---- STRIP TRAILING SPACES in column names ----
    df.columns = df.columns.str.strip()

    # Attempt to rename if there's a trailing-space or different naming:
    if "image_name " in df.columns:
        df.rename(columns={"image_name ": "image_name"}, inplace=True)
    if "filename" in df.columns and "image_name" not in df.columns:
        df.rename(columns={"filename": "image_name"}, inplace=True)

    if "image_name" not in df.columns:
        print("ERROR: 'rms_errors.csv' is missing a usable 'image_name' column (even after stripping/renaming).")
        print("Columns present are:", df.columns.tolist())
        return

    # ---- STRIP TRAILING SPACES from the image_name column values, if they're strings ----
    if pd.api.types.is_string_dtype(df["image_name"]):
        df["image_name"] = df["image_name"].str.strip()

    # 2) (Optional) Filter by images_to_include
    if len(args.include_images) > 0:
        images_incl_stripped = [x.strip() for x in args.include_images]
        df = df[df["image_name"].isin(images_incl_stripped)]
        if df.empty:
            print("No rows left after include_images filter; check your image names.")
            return

    # 3) (Optional) Filter by images_to_exclude
    if len(args.exclude_images) > 0:
        images_excl_stripped = [x.strip() for x in args.exclude_images]
        df = df[~df["image_name"].isin(images_excl_stripped)]
        if df.empty:
            print("No rows left after exclude_images filter.")
            return

    # 4) Filter by min/max error (optional)
    if args.min_error is not None or args.max_error is not None:
        df = filter_by_min_max_error(df, min_val=args.min_error, max_val=args.max_error)
        if df.empty:
            print("No rows left after min/max error filtering.")
            return

    # 5) If no keypoints specified => none used
    mp_keypoints_to_include = args.mp_kps
    hmr_keypoints_to_include = args.hmr_kps

    # 6) Build a list of columns to keep
    keep_cols = ["image_name"]
    for i in mp_keypoints_to_include:
        col = f"mediapipe_kp{i}"
        if col in df.columns:
            keep_cols.append(col)
    for i in hmr_keypoints_to_include:
        col = f"hamer_kp{i}"
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols]

    # 7) (Optional) Random sampling
    if args.random_sample > 0:
        df = df.sample(n=min(args.random_sample, len(df)), random_state=args.random_seed).reset_index(drop=True)

    # 8) (Optional) Sorting
    if args.sort_by is not None:
        if args.sort_by in df.columns:
            df = df.sort_values(by=args.sort_by, ascending=args.sort_ascending).reset_index(drop=True)
        else:
            print(f"Warning: sort_by='{args.sort_by}' not found in columns. Skipping sort.")

    # 9) Convert to "long" format
    df_long = build_long_format(df)
    if df_long.empty:
        print("No data after building long format—possibly all data was filtered out.")
        return

    # 10) Generate a plot
    # We'll unify the approach so that each keypoint has a different color.
    # For bar/box, we separate the methods into different facets.
    # For line, we color by keypoint_idx and vary line style by method.

    hover_data = ["image_name", "method", "keypoint_idx", "error_value"]

    if args.plot_type == "bar":
        # Keypoints => color
        # Methods => facet_col => each method gets its own subplot
        fig = px.bar(
            df_long,
            x="image_name",
            y="error_value",
            color="keypoint_idx",
            facet_col="method",
            facet_col_wrap=2,  # in case we have more than 2 methods
            barmode="group",
            title="Filtered RMS Errors (Bar Chart): facet=method, color=keypoint_idx",
            labels={
                "image_name": "Image",
                "error_value": "Normalized Error",
                "keypoint_idx": "Keypoint #"
            },
            hover_data=hover_data
        )
        fig.update_layout(
            xaxis={"type": "category", "categoryorder": "category ascending"}
        )

    elif args.plot_type == "box":
        # Keypoints => color
        # Methods => facet_col
        # We'll place each method in a different subplot, x-axis => keypoint_idx
        fig = px.box(
            df_long,
            x="keypoint_idx",
            y="error_value",
            color="keypoint_idx",
            facet_col="method",
            facet_col_wrap=2,
            title="Filtered RMS Errors (Box Plot): facet=method, color=keypoint_idx by x=keypoint_idx",
            labels={
                "keypoint_idx": "Keypoint #",
                "error_value": "Normalized Error"
            },
            hover_data=hover_data
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40)
        )

    elif args.plot_type == "line":
        # Keypoints => color
        # Methods => line dash
        # We'll keep them all in a single subplot, but different lines
        fig = px.line(
            df_long,
            x="image_name",
            y="error_value",
            color="keypoint_idx",
            line_dash="method",
            markers=True,
            title="Filtered RMS Errors (Line Plot): color=keypoint_idx, dash=method",
            labels={
                "image_name": "Image",
                "error_value": "Normalized Error",
                "keypoint_idx": "Keypoint #"
            },
            hover_data=hover_data
        )
        fig.update_layout(
            xaxis={"type": "category", "categoryorder": "category ascending"}
        )
    else:
        print(f"Unknown plot type => {args.plot_type}")
        return

    # 11) Save the figure
    fig.write_html(args.output)
    print(f"[INFO] Plot saved => {args.output}")


if __name__ == "__main__":
    main()
