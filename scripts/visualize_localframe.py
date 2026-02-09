#!/usr/bin/env python3
"""
visualize_localframe.py

Visualise transformed local-frame 3D hand keypoints (from wrist-centered, rotated
coordinates) as an interactive 3D Plotly UI:

- Black background
- Clear X/Y/Z axes (in mm)
- Each finger in a different colour
- Both hands (L and/or R) shown if present
- Slider to scrub through frames
- Frame ID clearly displayed
- Buttons to toggle: Both Hands / Left Only / Right Only (when both hands present)

MODES
-----
1) Single CSV mode:
   input_path = path/to/.../hand_landmark_xyz_localframe.csv

2) Batch mode:
   input_path = root directory containing numbered folders:
       <root>/
         01/
           cam2/CSV/hand_landmark_xyz_localframe.csv
         02/
           cam2/CSV/hand_landmark_xyz_localframe.csv
         ...

In batch mode, each numbered folder is processed independently.

INPUT CSV (per file):
    Must contain at least:
        - frame
        - Columns like:
          L_0_X_lcl_mm, L_0_Y_lcl_mm, L_0_Z_lcl_mm,
          ...
          R_0_X_lcl_mm, R_0_Y_lcl_mm, R_0_Z_lcl_mm, ...

OUTPUT HTML:
    Saved into:
        <numbered-folder>/cam2/plots/<name>_3d.html

In batch mode, all HTMLs are saved but NOT opened in a browser.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# Match columns like: L_0_X_lcl_mm, R_14_Z_lcl_mm, etc.
LOCAL_COL_RE = re.compile(
    r'^(?P<hand>[LR])_(?P<kp>\d+)_(?P<axis>[XYZ])_lcl_mm$'
)

# MediaPipe-style hand indexing (21 keypoints)
FINGER_GROUPS = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# Colours per finger group (works on black bg)
FINGER_COLORS = {
    "wrist": "white",
    "thumb": "red",
    "index": "orange",
    "middle": "lime",
    "ring": "cyan",
    "pinky": "magenta",
}


def parse_local_columns(columns):
    """
    Parse column names and group them by hand and keypoint id.

    Returns:
        coords: dict[hand][kp_id] = {'X': colname, 'Y': colname, 'Z': colname}
    """
    coords: dict[str, dict[int, dict[str, str]]] = {"L": {}, "R": {}}

    for col in columns:
        m = LOCAL_COL_RE.match(col)
        if not m:
            continue
        hand = m.group("hand")  # 'L' or 'R'
        kp = int(m.group("kp"))
        axis = m.group("axis")  # 'X', 'Y', 'Z'

        coords.setdefault(hand, {})
        coords[hand].setdefault(kp, {})
        coords[hand][kp][axis] = col

    return coords


def gather_frame_points(df, coords, framex, hands=("L", "R")):
    """
    For a given frame index, return a dict:
        points[hand][kp_idx] = (x, y, z) or (np.nan, np.nan, np.nan)
    """
    row = df.iloc[framex]
    points = {}

    for hand in hands:
        if hand not in coords or not coords[hand]:
            continue
        points[hand] = {}
        for kp_idx, axes_map in coords[hand].items():
            if not all(ax in axes_map for ax in ("X", "Y", "Z")):
                # Missing one axis, mark as NaN
                points[hand][kp_idx] = (np.nan, np.nan, np.nan)
                continue
            x = row[axes_map["X"]]
            y = row[axes_map["Y"]]
            z = row[axes_map["Z"]]
            points[hand][kp_idx] = (x, y, z)
    return points


def build_initial_traces(frame_points, coords, show_hands=("L", "R")):
    """
    Build initial Plotly Scatter3d traces for each finger group & hand.

    One trace per (hand, finger_group).

    IMPORTANT:
    - We do NOT draw a separate 'wrist' trace anymore.
    - Each finger trace starts from wrist (kp 0) and then its own joints.
    """
    traces = []
    trace_meta = []  # to know mapping: (hand, finger_group)

    for hand in show_hands:
        if hand not in frame_points:
            continue

        for finger_name, kp_list in FINGER_GROUPS.items():
            # Skip separate wrist-only trace
            if finger_name == "wrist":
                continue

            xs, ys, zs = [], [], []

            # Connection from wrist (0) to each finger joint
            combined_kps = [0] + kp_list

            for kp_idx in combined_kps:
                if kp_idx not in frame_points[hand]:
                    continue
                x, y, z = frame_points[hand][kp_idx]
                xs.append(x)
                ys.append(y)
                zs.append(z)

            if not xs:
                # No points for this finger in this hand
                continue

            color = FINGER_COLORS.get(finger_name, "white")
            name = f"{hand}-{finger_name}"

            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+lines",
                    name=name,
                    marker=dict(size=5, color=color),
                    line=dict(color=color, width=3),
                    hovertemplate=(
                        f"Hand: {hand}<br>"
                        f"Finger: {finger_name}<br>"
                        "X: %{x:.1f} mm<br>"
                        "Y: %{y:.1f} mm<br>"
                        "Z: %{z:.1f} mm<br>"
                        "<extra></extra>"
                    ),
                )
            )
            trace_meta.append((hand, finger_name))

    return traces, trace_meta


def build_frames(df, coords, trace_meta, frame_indices, show_hands=("L", "R")):
    """
    Build Plotly animation frames.

    Each frame updates all traces (one per (hand, finger_group)).

    IMPORTANT:
    - For each finger, we again use [0] + finger_kps so wrist is connected.
    """
    frames = []

    for idx in frame_indices:
        frame_points = gather_frame_points(df, coords, idx, hands=show_hands)
        data = []

        # Rebuild x,y,z per trace in the same order as trace_meta
        for (hand, finger_name) in trace_meta:
            xs, ys, zs = [], [], []
            if hand not in frame_points:
                # if hand missing, set empty (still mark as scatter3d!)
                data.append(dict(type="scatter3d", x=[], y=[], z=[]))
                continue

            # For this finger, look up its kp list and prepend wrist 0
            kp_list = FINGER_GROUPS[finger_name]
            combined_kps = [0] + kp_list

            for kp_idx in combined_kps:
                if kp_idx not in frame_points[hand]:
                    continue
                x, y, z = frame_points[hand][kp_idx]
                xs.append(x)
                ys.append(y)
                zs.append(z)

            data.append(dict(type="scatter3d", x=xs, y=ys, z=zs))

        frame = df["frame"].iloc[idx]
        frames.append(go.Frame(data=data, name=str(frame)))

    return frames


def make_figure_for_hands(
    df: pd.DataFrame,
    coords: dict,
    csv_path: Path,
    show_hands: list[str],
    step: int,
    output_name: str | None,
    filename_suffix: str,
    open_browser: bool = True,
):
    """
    Build and save a Plotly figure for one or more hands.
    This reuses your existing logic, just parametrised.
    """

    # Determine frame indices to use
    total_frames = len(df)
    frame_indices = list(range(0, total_frames, max(1, step)))
    if not frame_indices:
        raise RuntimeError("No frames selected. Check --step and CSV length.")

    # Initial frame (first in the sequence)
    initial_idx = frame_indices[0]
    initial_points = gather_frame_points(df, coords, initial_idx, hands=show_hands)
    traces, trace_meta = build_initial_traces(initial_points, coords, show_hands=show_hands)

    # Build animation frames
    frames = build_frames(df, coords, trace_meta, frame_indices, show_hands=show_hands)

    initial_frame = df["frame"].iloc[initial_idx]

    # Scene layout with black background and visible axes
    scene = dict(
        xaxis=dict(
            title="X_local_mm",
            showgrid=True,
            zeroline=True,
            showline=True,
            color="white",
            gridcolor="gray",
            zerolinecolor="white",
            linecolor="white",
        ),
        yaxis=dict(
            title="Y_local_mm",
            showgrid=True,
            zeroline=True,
            showline=True,
            color="white",
            gridcolor="gray",
            zerolinecolor="white",
            linecolor="white",
        ),
        zaxis=dict(
            title="Z_local_mm",
            showgrid=True,
            zeroline=True,
            showline=True,
            color="white",
            gridcolor="gray",
            zerolinecolor="white",
            linecolor="white",
        ),
        aspectmode="data",
    )

    # Slider steps — one per frame
    slider_steps = []
    for fr in frames:
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [fr.name],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=fr.name,
            )
        )

    sliders = [
        dict(
            active=0,
            pad=dict(t=50),
            currentvalue=dict(
                visible=True,
                prefix="frame: ",
                xanchor="right",
                font=dict(color="white", size=14),
            ),
            steps=slider_steps,
        )
    ]

    # --- Play/Pause buttons (first updatemenu) ---
    n_traces = len(traces)
    play_pause_menu = dict(
        type="buttons",
        showactive=False,
        x=0.1,
        y=0,
        xanchor="right",
        yanchor="top",
        direction="left",
        pad=dict(t=40, r=10),
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=40, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    ),
                ],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
            ),
        ],
    )

    updatemenus = [play_pause_menu]

    # --- Hand toggle buttons (only if *both* hands are present) ---
    if "L" in show_hands and "R" in show_hands:
        visible_both = [True] * n_traces
        visible_left_only = [tm[0] == "L" for tm in trace_meta]
        visible_right_only = [tm[0] == "R" for tm in trace_meta]

        hand_toggle_menu = dict(
            type="buttons",
            showactive=True,
            x=0.99,
            y=1.07,
            xanchor="right",
            yanchor="top",
            direction="right",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Both Hands",
                    method="update",
                    args=[
                        {"visible": visible_both},
                        {},
                    ],
                ),
                dict(
                    label="Left Only",
                    method="update",
                    args=[
                        {"visible": visible_left_only},
                        {},
                    ],
                ),
                dict(
                    label="Right Only",
                    method="update",
                    args=[
                        {"visible": visible_right_only},
                        {},
                    ],
                ),
            ],
        )
        updatemenus.append(hand_toggle_menu)

    title_text = f"Hand Local-Frame 3D Visualization — frame: {initial_frame}"

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title_text, font=dict(color="white")),
            scene=scene,
            showlegend=True,
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0.4)"
            ),
            sliders=sliders,
            updatemenus=updatemenus,
        ),
        frames=frames,
    )

    # ---------- Save to <numbered-folder>/cam2/plots/ ----------
    # Expect structure: <numbered-folder>/cam2/CSV/input.csv
    csv_dir = csv_path.parent       # .../cam2/CSV
    cam2_dir = csv_dir.parent       # .../cam2
    plots_dir = cam2_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        filename = Path(output_name).name  # only take the name, ignore any path
    else:
        filename = csv_path.stem + filename_suffix

    out_path = plots_dir / filename
    fig.write_html(out_path)
    print(f"[INFO] Saved interactive HTML to: {out_path}")

    if open_browser:
        fig.show()


def run_visualization_for_csv(
    csv_path: Path,
    step: int,
    hand_mode: str,
    split_hands: bool,
    output: str | None,
    batch_mode: bool,
):
    """
    Wrapper that loads a CSV and dispatches to make_figure_for_hands based on
    hand_mode and split_hands.

    batch_mode:
        - True  → do NOT open browser automatically.
        - False → open browser (single-file mode).
    """
    print(f"[INFO] Visualising: {csv_path}")
    df = pd.read_csv(csv_path)

    if "frame" not in df.columns:
        raise KeyError(f"Input CSV must contain 'frame' column: {csv_path}")

    coords = parse_local_columns(df.columns.tolist())
    hands_present = [h for h in ("L", "R") if h in coords and coords[h]]
    if not hands_present:
        raise RuntimeError(f"No L_*/R_* local-frame columns found in: {csv_path}")

    open_browser = not batch_mode

    # --- Routing based on hand_mode / split_hands ---
    if hand_mode in ("L", "R"):
        # Single-hand mode
        if hand_mode not in hands_present:
            raise RuntimeError(f"Requested hand {hand_mode} not present in CSV: {csv_path}")
        make_figure_for_hands(
            df=df,
            coords=coords,
            csv_path=csv_path,
            show_hands=[hand_mode],
            step=step,
            output_name=output if not batch_mode else None,
            filename_suffix=f"_{hand_mode}_3d.html",
            open_browser=open_browser,
        )
    else:
        # hand_mode == "both"
        if split_hands:
            # Generate separate L and R files
            if "L" in hands_present:
                make_figure_for_hands(
                    df=df,
                    coords=coords,
                    csv_path=csv_path,
                    show_hands=["L"],
                    step=step,
                    output_name=None,
                    filename_suffix="_L_3d.html",
                    open_browser=False if batch_mode else open_browser,
                )
            if "R" in hands_present:
                make_figure_for_hands(
                    df=df,
                    coords=coords,
                    csv_path=csv_path,
                    show_hands=["R"],
                    step=step,
                    output_name=None,
                    filename_suffix="_R_3d.html",
                    open_browser=open_browser,
                )
        else:
            # Combined figure (original behaviour, with toggles, if both hands present)
            make_figure_for_hands(
                df=df,
                coords=coords,
                csv_path=csv_path,
                show_hands=hands_present,
                step=step,
                output_name=output if not batch_mode else None,
                filename_suffix="_3d.html",
                open_browser=open_browser,
            )


def main():
    parser = argparse.ArgumentParser(description=("Interactive 3D visualisation of wrist-centered local-frame hand keypoints.\n\n"
            "If input_path is a CSV file, only that file is visualised.\n"
            "If input_path is a directory, all <numbered-folder>/cam2/CSV/*_localframe.csv\n"
            "files under it are visualised in batch (no auto browser open)."
        )
    )
    parser.add_argument("input_path", type=str, help=("Path to a single local-frame CSV file OR to a root directory containing numbered folders with cam2/CSV/hand_landmark_xyz_localframe.csv inside."))
    parser.add_argument("--step", type=int, default=10, help="Use every N-th frame (default: 1 = use all).")
    parser.add_argument("--output", type=str, default=None, help="Optional HTML file name (single-file mode only). Saved into <numbered-folder>/cam2/plots/.")
    parser.add_argument("--hand", type=str, choices=["L", "R", "both"], default="both", help="Which hand(s) to include in the visualisation (default: both).")
    parser.add_argument("--split-hands", action="store_true", help="If set and --hand=both, generate separate HTML files for L and R instead of a combined plot.")
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
        run_visualization_for_csv(
            csv_path=input_path,
            step=args.step,
            hand_mode=args.hand,
            split_hands=args.split_hands,
            output=args.output,
            batch_mode=False,
        )

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

            csv_dir = child / "cam2" / "CSV"
            primary_csv = csv_dir / "hand_landmark_xyz_localframe.csv"

            if primary_csv.exists():
                csv_path = primary_csv
            else:
                # Fallback: any *_localframe.csv (if exactly one)
                candidates = list(csv_dir.glob("*_localframe.csv"))
                if len(candidates) == 1:
                    csv_path = candidates[0]
                elif len(candidates) == 0:
                    print(
                        f"[WARN] No *_localframe.csv found in: {csv_dir}. Skipping folder {folder_name}."
                    )
                    skipped_missing += 1
                    continue
                else:
                    print(
                        f"[WARN] Multiple *_localframe.csv files found in: {csv_dir}. "
                        f"Skipping folder {folder_name} to avoid ambiguity."
                    )
                    skipped_missing += 1
                    continue

            print(f"[PROCESS] Processing folder {folder_name} → {csv_path}")
            run_visualization_for_csv(
                csv_path=csv_path,
                step=args.step,
                hand_mode=args.hand,
                split_hands=args.split_hands,
                output=None,         # per-folder automatic names
                batch_mode=True,     # don't open browser in batch
            )
            print(f"[FINISHED] Folder {folder_name}")
            processed += 1

        print("\n[INFO] Batch processing complete.")
        print(f"[INFO] Folders processed: {processed}")
        print(f"[INFO] Folders skipped (missing/ambiguous CSV): {skipped_missing}")
        print(f"[INFO] Folders skipped (user request): {skipped_by_user}")

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()