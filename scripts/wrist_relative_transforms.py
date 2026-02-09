#!/usr/bin/env python3
"""
wrist_relative_transforms.py

Transform 3D hand keypoint coordinates from camera/world frame to a
per-hand, per-frame local frame:

- Origin: wrist of that hand (keypoint id 0)
- Rotation: 180 degrees around the Y-axis, i.e. diag(-1, 1, -1)
- Applied independently for each frame

MODES
-----
1) Single CSV mode:
   input_path = path/to/.../hand_landmark_xyz.csv

2) Batch mode:
   input_path = root directory containing numbered folders:
       <root>/
         01/
           cam2/CSV/hand_landmark_xyz.csv
         02/
           cam2/CSV/hand_landmark_xyz.csv
         ...

In batch mode, each numbered folder is processed independently.

INPUT CSV (per file):
    Must contain at least:
        - frame
        - Some subset of columns matching:
          {hand}_{kpID}_{axis}_mm
          where hand ∈ {L, R}, axis ∈ {X, Y, Z}, kpID is integer (e.g., 0..20)

    Other columns (e.g., t_ns, *_px, *_move_mm) are ignored.

OUTPUT CSV:
    Fresh CSV with:
        - frame
        - Transformed coordinates as:
          {hand}_{kpID}_{axis}_lcl_mm
          e.g., L_0_X_lcl_mm, L_0_Y_lcl_mm, L_0_Z_lcl_mm, ...

LOG:
    For each processed CSV, a log is written to:
        <numbered-folder>/cam2/logs/transform_localframe_<timestamp>.log

    Log lines DO NOT contain timestamps/date; only level + message.
"""

import argparse
import logging
from pathlib import Path
import re
from datetime import datetime

import numpy as np  # kept to avoid unnecessary edits if you later extend
import pandas as pd


# Regex to match columns like: L_0_X_mm, R_14_Z_mm, etc.
KP_COL_RE = re.compile(r'^(?P<hand>[LR])_(?P<kp>\d+)_(?P<axis>[XYZ])_mm$')


def setup_logging(
    input_csv: Path,
    explicit_log_dir: Path | None = None,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger that writes per-file logs into <numbered-folder>/cam2/logs/.

    - No timestamps/date in any log line.
    - enable_console=False → used for batch mode to keep terminal clean.
    """

    if explicit_log_dir is not None:
        logs_dir = explicit_log_dir
    else:
        csv_dir = input_csv.parent
        cam2_dir = csv_dir.parent  # expect .../<numbered-folder>/cam2/CSV -> cam2
        logs_dir = cam2_dir / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"transform_localframe_{timestamp}.log"

    logger = logging.getLogger(f"localframe_transform_{input_csv}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler — no timestamps in lines
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("%(levelname)s %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler only if allowed (single-file mode)
    if enable_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    logger.info("Log file created at: %s", log_path)
    return logger


def parse_keypoint_columns(columns: list[str], logger: logging.Logger):
    """
    Scan column names and group them by hand and keypoint id.

    Returns:
        coords: dict[hand][kp_id] = {'X': colname, 'Y': colname, 'Z': colname}
    """
    coords: dict[str, dict[str, dict[str, str]]] = {"L": {}, "R": {}}

    for col in columns:
        m = KP_COL_RE.match(col)
        if not m:
            continue
        hand = m.group("hand")  # 'L' or 'R'
        kp = m.group("kp")      # string id, e.g. '0', '1', ...
        axis = m.group("axis")  # 'X', 'Y', 'Z'

        coords.setdefault(hand, {})
        coords[hand].setdefault(kp, {})
        coords[hand][kp][axis] = col

    # Log what we found
    for hand in ["L", "R"]:
        if hand in coords and coords[hand]:
            kp_ids = sorted(coords[hand].keys(), key=lambda x: int(x))
            logger.info("Detected %s-hand keypoints: %s", hand, kp_ids)
        else:
            logger.info("No %s-hand keypoints found in columns.", hand)

    return coords


def transform_hand(
    df: pd.DataFrame,
    hand: str,
    hand_coords: dict[str, dict[str, str]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Perform per-frame, per-hand transform:

        1) Translate so wrist kp=0 is origin.
        2) Rotate 180° about Y-axis: diag(-1, 1, -1).

    Returns:
        new_df: DataFrame with transformed columns for this hand only.
    """
    if "0" not in hand_coords:
        logger.warning("Hand %s: no wrist keypoint (kp=0) found. Skipping this hand.", hand)
        return pd.DataFrame(index=df.index)

    wrist = hand_coords["0"]
    if not all(ax in wrist for ax in ("X", "Y", "Z")):
        logger.warning("Hand %s: wrist keypoint does not have complete X/Y/Z. Skipping this hand.", hand)
        return pd.DataFrame(index=df.index)

    wrist_x = df[wrist["X"]]
    wrist_y = df[wrist["Y"]]
    wrist_z = df[wrist["Z"]]

    # Track frames where wrist is invalid
    wrist_invalid_mask = wrist_x.isna() | wrist_y.isna() | wrist_z.isna()
    num_invalid_wrist_frames = int(wrist_invalid_mask.sum())
    logger.info(
        "Hand %s: %d/%d frames have invalid wrist (NaN).",
        hand,
        num_invalid_wrist_frames,
        len(df),
    )

    new_cols: dict[str, pd.Series] = {}

    for kp_id, axes_map in hand_coords.items():
        # Ensure full XYZ for this keypoint
        if not all(ax in axes_map for ax in ("X", "Y", "Z")):
            logger.warning(
                "Hand %s, kp %s: missing one of X/Y/Z. Skipping this kp.",
                hand,
                kp_id,
            )
            continue

        src_x = df[axes_map["X"]]
        src_y = df[axes_map["Y"]]
        src_z = df[axes_map["Z"]]

        # Translate relative to wrist
        rel_x = src_x - wrist_x
        rel_y = src_y - wrist_y
        rel_z = src_z - wrist_z

        # Rotate 180° around Y-axis:
        # R = diag(-1, 1, -1) → (X, Y, Z) -> (-X, Y, -Z)
        x_local = -rel_x
        y_local = rel_y
        z_local = -rel_z

        # Output column names
        base = f"{hand}_{kp_id}"
        new_cols[f"{base}_X_lcl_mm"] = x_local
        new_cols[f"{base}_Y_lcl_mm"] = y_local
        new_cols[f"{base}_Z_lcl_mm"] = z_local

    logger.info("Hand %s: transformed %d keypoints.", hand, len(new_cols) // 3)

    return pd.DataFrame(new_cols, index=df.index)


def process_single_csv(
    input_csv: Path,
    output_suffix: str,
    batch_mode: bool = False,
) -> None:
    """
    Process a single CSV file:
    - Derive output CSV path with the given suffix.
    - Set up per-file logging in <numbered-folder>/cam2/logs/.
    - Run the transform and save output.

    batch_mode:
        - True  → logger does NOT print to console (only file logs).
        - False → logger prints to console (single-file mode).
    """
    # Determine output CSV path:
    # e.g. hand_landmark_xyz.csv -> hand_landmark_xyz_localframe.csv
    stem = input_csv.stem
    suffix = input_csv.suffix or ".csv"
    new_name = f"{stem}{output_suffix}{suffix}"
    output_csv = input_csv.with_name(new_name)

    # Setup logging (per-file)
    logger = setup_logging(input_csv, explicit_log_dir=None, enable_console=not batch_mode)

    logger.info("=== Local-frame transform session started ===")
    logger.info("Input CSV: %s", input_csv)
    logger.info("Output CSV: %s", output_csv)
    logger.info("Rotation: 180 degrees around Y-axis (diag(-1, 1, -1))")
    logger.info("Translation: per-hand wrist (kp=0) used as origin, per frame")

    # Load data
    df = pd.read_csv(input_csv)
    logger.info("Loaded CSV with %d rows and %d columns.", len(df), len(df.columns))

    if "frame" not in df.columns:
        logger.error("Input CSV does not contain 'frame' column. Aborting.")
        logger.handlers.clear()
        raise KeyError(f"Missing 'frame' column in input CSV: {input_csv}")

    # Parse keypoint columns
    coords = parse_keypoint_columns(df.columns.tolist(), logger)

    # Base of output DataFrame: only frame
    out_df = pd.DataFrame()
    out_df["frame"] = df["frame"]

    # Transform each hand separately
    for hand in ["L", "R"]:
        hand_coords = coords.get(hand, {})
        if not hand_coords:
            continue
        transformed = transform_hand(df, hand, hand_coords, logger)
        # Align index and concat
        out_df = pd.concat([out_df, transformed], axis=1)

    # Save result
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    local_cols = [c for c in out_df.columns if c.endswith("_lcl_mm")]
    out_df[local_cols] = out_df[local_cols].round(2)  # round to 2 decimal places
    out_df.to_csv(output_csv, index=False)
    logger.info("Saved transformed coordinates to: %s", output_csv)

    # Summary
    logger.info("=== Summary ===")
    logger.info("Total frames: %d", len(df))
    logger.info("Output columns: %d", len(out_df.columns))
    logger.info("Session complete.")

    logger.handlers.clear()


def main():
    parser = argparse.ArgumentParser(description=("Transform 3D hand keypoint coordinates to wrist-centered local frame with 180-degree rotation around Y-axis.\n\n"
            "If input_path is a CSV file, only that file is processed.\n"
            "If input_path is a directory, all <numbered-folder>/cam2/CSV/hand_landmark_xyz.csv\n"
            "files under it are processed in batch."
        )
    )
    parser.add_argument("input_path", type=str, help=("Path to a single CSV file OR to a root directory containing numbered folders with cam2/CSV/hand_landmark_xyz.csv inside."))
    parser.add_argument("--output-suffix", type=str, default="_localframe", help=("Suffix to append to each output CSV stem (default: '_localframe', so 'hand_landmark_xyz.csv' -> 'hand_landmark_xyz_localframe.csv')."))
    parser.add_argument("--skip", nargs="*", default=[], help=("List of numbered folders to skip in batch mode. Values like '8' or '08' refer to the same folder."))
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()

    # Normalise skip folders: "8", "08", "008" → "8"; "0", "00" → "0"
    skip_normalised: set[str] = set()
    for raw in args.skip:
        s = str(raw).strip()
        if not s:
            continue
        norm = s.lstrip("0")
        if norm == "":
            norm = "0"
        skip_normalised.add(norm)

    if input_path.is_file():
        # Single CSV mode
        print(f"[INFO] Single-file mode: {input_path}")
        process_single_csv(input_path, output_suffix=args.output_suffix, batch_mode=False)

    elif input_path.is_dir():
        # Batch mode
        root_dir = input_path
        print(f"[INFO] Batch mode. Root directory: {root_dir}")
        if skip_normalised:
            print(f"[INFO] Normalised skip folders: {sorted(skip_normalised)}")
        else:
            print(f"[INFO] Normalised skip folders: None")

        processed = 0
        skipped_missing = 0
        skipped_by_user = 0

        for child in sorted(root_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue

            folder_name = child.name
            if not folder_name.isdigit():
                continue  # only process numeric folders

            # Normalise folder name similarly to skip list
            norm_id = folder_name.lstrip("0")
            if norm_id == "":
                norm_id = "0"

            if norm_id in skip_normalised:
                print(
                    f"[INFO] Skipping folder {folder_name} (normalised: {norm_id}) due to --skip."
                )
                skipped_by_user += 1
                continue

            csv_path = child / "cam2" / "CSV" / "hand_landmark_xyz.csv"
            if not csv_path.exists():
                print(f"[WARN] Expected CSV not found: {csv_path}. Skipping this folder.")
                skipped_missing += 1
                continue

            print(f"[PROCESS] Processing folder {folder_name} → {csv_path}")
            process_single_csv(csv_path, output_suffix=args.output_suffix, batch_mode=True)
            print(f"[FINISHED] Folder {folder_name}")
            processed += 1

        print("\n[INFO] Batch processing complete.")
        print(f"[INFO] Folders processed: {processed}")
        print(f"[INFO] Folders skipped (missing CSV): {skipped_missing}")
        print(f"[INFO] Folders skipped (user request): {skipped_by_user}")

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()