#!/usr/bin/env python3
"""
boxplot_chunk.py

Make box-and-whisker plots of *per-frame* 3D movement (mm) in 10s parts.
Each data point is the 3D distance between two **consecutive valid frames**,
where a frame is valid only if *all required* mm coords are present for the
selected landmarks/hands (see --lid).

Keypoint/hand selection (--lid)
-------------------------------
- --lid L0   => only Left-hand landmark 0
- --lid R4   => only Right-hand landmark 4
- --lid L    => all Left-hand landmarks (0..4)
- --lid R    => all Right-hand landmarks (0..4)
- (omit)     => all landmarks on both hands
- You may pass multiple, e.g. --lid L R  or  --lid L0 R2

Gappy data handling
-------------------
If some frames are missing (or have any NaNs on the selected landmarks),
they are skipped entirely. Distances bridge over gaps to the **next valid
frame**: e.g., valid frames ... 3,4,8,9 ... produce steps (3→4), (4→8), (8→9).

Chunking
--------
- Chunks are defined by absolute frame numbers at FPS (default 30).
- Chunk length = --chunk seconds (default 10) ⇒ chunk_f = chunk*FPS frames.
- Step (a→b) is counted in the chunk of destination frame b.
- start/stop (seconds) clip which steps are included.
- If a 'frame' column exists, it is respected (sorted, gaps allowed);
  otherwise, row index is treated as frame index starting at 0.

Ranges manifest (--ranges)
--------------------------
Optional CSV with rows keyed by absolute file path or by PID (from filename):
  file,start,stop
  pid,start,stop
- 'stop' may be blank ⇒ until end.
- If both file and pid rules match, file rule wins.
- Class (E/N) is not stored here; it comes from -E/-N inputs.

Usage
-----
python boxplot_chunk.py \
  --experts "/path/to/experts/*.csv" \
  --novices "/path/to/novices/*.csv" \
  --fps 30 --chunk 10 \
  --lid L0 \
  --ranges /path/to/ranges.csv \
  --out box_parts_EN.html \
  [--png] [--debug]

Input columns required (others ignored)
---------------------------------------
Selected subset of:
  {L|R}_{0..4}_{X|Y|Z}_mm  (30 columns total if all are selected)
'frame' column optional (recommended).

Output
------
- Interactive HTML box plot at --out.
- Optional PNG with --png (needs `pip install kaleido`).
- Optional debug summary in CWD with --debug.
"""

import argparse
from pathlib import Path
import glob
import sys
from typing import Dict, Optional, Tuple, List, Set

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # for mean/median overlays

HANDS = ("L", "R")
ALL_LMS = [0, 1, 2, 3, 4]
MM_COLS_TEMPLATE = "{hand}_{lm}_{axis}_mm"  # e.g., L_0_X_mm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Framewise movement box plot by 10s parts (grouped E vs N), gap-bridged, with optional per-file ranges."
    )
    p.add_argument("-E", "--experts", nargs="+", default=[], help="Expert CSVs (globs or paths)")
    p.add_argument("-N", "--novices", nargs="+", default=[], help="Novice CSVs (globs or paths)")
    p.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    p.add_argument("--chunk", type=int, default=10, help="Part length in seconds (default: 10)")
    p.add_argument("--start", type=int, default=0, help="Default start time (s) for files without overrides")
    p.add_argument("--stop", type=int, default=None, help="Default stop time (s, exclusive) for files without overrides")
    p.add_argument("--ranges", type=str, default=None, help="CSV with per-file or per-pid start/stop overrides")
    p.add_argument("--lid", nargs="+", default=None,
                   help="Landmark/hand selector(s): L, R, L0..L4, R0..R4. "
                        "Omit to use all landmarks on both hands. "
                        "May be provided multiple times, e.g., --lid L R or --lid L0 R2")
    p.add_argument("--out", type=str, default=None, help="Output HTML path (default based on classes)")
    p.add_argument("--png", action="store_true", help="Also export PNG (needs kaleido)")
    p.add_argument("--debug", action="store_true", help="Write debug summary in CWD")
    return p.parse_args()


def expand_inputs(patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # dedupe & sort
    return sorted(set(str(Path(f).expanduser().resolve()) for f in files))


def parse_lid_list(lid_args: Optional[List[str]]) -> Tuple[List[Tuple[str, int]], str]:
    """
    Parse --lid selections into a list of (hand, landmark) pairs and a short label.
    Accepts tokens: 'L', 'R', 'L0'..'L4', 'R0'..'R4' (case-insensitive).
    Multiple tokens are unioned. If None/empty => all (L/R × 0..4).
    """
    if not lid_args:
        pairs = [(h, lm) for h in HANDS for lm in ALL_LMS]
        return pairs, "All"

    pairs_set: Set[Tuple[str, int]] = set()
    toks_norm: List[str] = []

    for tok in lid_args:
        if tok is None:
            continue
        t = tok.strip().upper()
        if not t:
            continue
        toks_norm.append(t)
        if t in ("L", "R"):
            hand = t
            for lm in ALL_LMS:
                pairs_set.add((hand, lm))
        else:
            hand = t[0]
            if hand not in HANDS:
                continue
            try:
                lm = int(t[1:])
            except Exception:
                continue
            if lm in ALL_LMS:
                pairs_set.add((hand, lm))

    if not pairs_set:
        # fallback to all if nothing valid was parsed
        pairs = [(h, lm) for h in HANDS for lm in ALL_LMS]
        return pairs, "All"

    pairs = sorted(pairs_set, key=lambda x: (x[0], x[1]))
    label = " ".join(sorted(set(toks_norm)))
    return pairs, label


def needed_cols_for(pairs: List[Tuple[str, int]]) -> List[str]:
    cols = []
    for hand, lm in pairs:
        for ax in ("X", "Y", "Z"):
            cols.append(MM_COLS_TEMPLATE.format(hand=hand, lm=lm, axis=ax))
    return cols


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_frame_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == "frame":
            return c
    return None


def extract_pid(csv_path: Path) -> str:
    """AP_27_xyz.csv -> AP_27 ; else use prefix before first underscore."""
    stem = csv_path.stem
    if "_xyz" in stem:
        return stem.split("_xyz")[0]
    return stem.split("_")[0]


def compute_gap_bridged_steps(
    df: pd.DataFrame,
    pairs: List[Tuple[str, int]],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Returns (to_frames, dists, min_frame, max_frame_inclusive)

    - Valid frames: all *selected* mm columns present (non-NaN).
    - Distances are computed between consecutive valid frames (gap-bridged),
      summing Euclidean distances across the selected landmarks.
    - 'to_frames' holds the destination frame number 'b' for each distance.
    """
    needed = needed_cols_for(pairs)
    frame_col = get_frame_col(df)

    if frame_col is not None:
        df = df.sort_values(by=frame_col).reset_index(drop=True)
        frames = pd.to_numeric(df[frame_col], errors="coerce").to_numpy()
    else:
        frames = np.arange(len(df), dtype=int)

    # Column existence check
    for c in needed:
        if c not in df.columns:
            # Missing required columns => nothing to compute
            min_all = int(np.nanmin(frames)) if frames.size else 0
            max_all = int(np.nanmax(frames)) if frames.size else -1
            return np.array([], dtype=int), np.array([], dtype=float), min_all, max_all

    # Build wide coords for selected pairs
    coords = df[needed].to_numpy(dtype=float, copy=False)  # shape (n, 3*K) where K=len(pairs)
    # Mask valid rows: all selected coords present AND frame present
    has_frame_idx = ~np.isnan(frames)
    all_ok = ~np.isnan(coords).any(axis=1)
    valid = has_frame_idx & all_ok

    min_all = int(np.nanmin(frames)) if frames.size else 0
    max_all = int(np.nanmax(frames)) if frames.size else -1

    if valid.sum() < 2:
        # Not enough valid frames to form steps
        return np.array([], dtype=int), np.array([], dtype=float), min_all, max_all

    K = len(pairs)
    # reshape to (m, K, 3)
    V = coords[valid].reshape(-1, K, 3)  # consecutive valid frames only
    F = frames[valid].astype(int)

    diffs = V[1:] - V[:-1]                        # (m-1, K, 3)
    dists = np.linalg.norm(diffs, axis=2).sum(axis=1)  # sum over K landmarks -> (m-1,)
    to_frames = F[1:]                              # destination frame

    return to_frames, dists.astype(float), min_all, max_all


def choose_range(
    csv_path: Path,
    default_start: int,
    default_stop: Optional[int],
    path_map: Dict[str, Tuple[int, Optional[int]]],
    pid_map: Dict[str, Tuple[int, Optional[int]]],
) -> Tuple[int, Optional[int], str]:
    """Pick (start_s, stop_s, source_tag) for a specific CSV."""
    key = str(csv_path.resolve())
    if key in path_map:
        s, t = path_map[key]
        return s, t, "path"
    pid = extract_pid(csv_path)
    if pid in pid_map:
        s, t = pid_map[pid]
        return s, t, f"pid({pid})"
    return default_start, default_stop, "default"


def load_ranges(path: Path) -> Tuple[Dict[str, Tuple[int, Optional[int]]], Dict[str, Tuple[int, Optional[int]]]]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    path_map: Dict[str, Tuple[int, Optional[int]]] = {}
    pid_map: Dict[str, Tuple[int, Optional[int]]] = {}

    def parse_stop(v) -> Optional[int]:
        if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
            return None
        return int(v)

    if "file" in df.columns and "start" in df.columns:
        for _, r in df.iterrows():
            f = str(r.get("file", "")).strip()
            if not f:
                continue
            s = int(r["start"])
            t = parse_stop(r.get("stop", None))
            try:
                key = str(Path(f).expanduser().resolve())
            except Exception:
                key = f
            path_map[key] = (s, t)

    if "pid" in df.columns and "start" in df.columns:
        for _, r in df.iterrows():
            pid = str(r.get("pid", "")).strip()
            if not pid:
                continue
            s = int(r["start"])
            t = parse_stop(r.get("stop", None))
            pid_map[pid] = (s, t)

    return path_map, pid_map


def gather_class(
    label: str,
    csvs: List[str],
    fps: int,
    chunk_s: int,
    default_start: int,
    default_stop: Optional[int],
    path_map: Dict[str, Tuple[int, Optional[int]]],
    pid_map: Dict[str, Tuple[int, Optional[int]]],
    pairs: List[Tuple[str, int]],
    debug_lines: List[str],
):
    """
    Build per-part distance pools for one class (E or N).
    Returns (per_part_values, max_parts_observed).
    """
    per_part_values: Dict[int, List[float]] = {}
    max_parts = 0
    chunk_f = fps * chunk_s

    needed = needed_cols_for(pairs)

    for path in csvs:
        p = Path(path)
        if not p.exists():
            debug_lines.append(f"[{label}] MISSING: {path}")
            continue

        start_s, stop_s, src = choose_range(p, default_start, default_stop, path_map, pid_map)

        df = pd.read_csv(p)
        df = safe_numeric(df, needed)

        to_frames, dists, min_frame, max_frame = compute_gap_bridged_steps(df, pairs)
        if to_frames.size == 0:
            debug_lines.append(f"[{label}] {p.name}: no valid step distances; range={start_s}->{stop_s if stop_s is not None else 'end'} ({src})")
            continue

        # Absolute frame bounds for chunking
        start_f = min_frame + start_s * fps
        stop_f  = (min_frame + stop_s * fps) if (stop_s is not None) else (max_frame + 1)

        # Steps whose destination frame is within [start_f, stop_f)
        mask = (to_frames >= start_f) & (to_frames < stop_f)
        tf = to_frames[mask]
        dv = dists[mask]

        if tf.size == 0:
            debug_lines.append(f"[{label}] {p.name}: no steps within time range; frames=[{min_frame},{max_frame}], use {start_s}->{stop_s or 'end'}s")
            continue

        # Part index: floor((to_frame - start_f)/chunk_f) + 1
        parts_idx = ((tf - start_f) // chunk_f).astype(int) + 1
        max_parts = max(max_parts, int(parts_idx.max()))

        for k, val in zip(parts_idx.tolist(), dv.tolist()):
            per_part_values.setdefault(k, []).append(float(val))

        debug_lines.append(
            f"[{label}] {p.name}: steps_kept={tf.size}, "
            f"range={start_s}->{stop_s if stop_s is not None else 'end'}s ({src}), "
            f"parts_seen≤{int(parts_idx.max())}"
        )

    return per_part_values, max_parts


def main():
    args = parse_args()
    E_files = expand_inputs(args.experts)
    N_files = expand_inputs(args.novices)
    pairs, lid_label = parse_lid_list(args.lid)

    if not E_files and not N_files:
        print("[ERROR] No input CSVs provided. Use -E and/or -N.", file=sys.stderr)
        sys.exit(1)

    # Load ranges (if provided)
    path_map: Dict[str, Tuple[int, Optional[int]]] = {}
    pid_map: Dict[str, Tuple[int, Optional[int]]] = {}
    if args.ranges:
        ranges_csv = Path(args.ranges).expanduser().resolve()
        if not ranges_csv.exists():
            print(f"[ERROR] Ranges file not found: {ranges_csv}", file=sys.stderr)
            sys.exit(1)
        path_map, pid_map = load_ranges(ranges_csv)

    debug_lines: List[str] = []
    all_parts = set()

    E_vals, E_parts = ({}, 0)
    N_vals, N_parts = ({}, 0)

    if E_files:
        E_vals, E_parts = gather_class(
            "E", E_files, args.fps, args.chunk, args.start, args.stop, path_map, pid_map, pairs, debug_lines
        )
        all_parts.update(E_vals.keys())
    if N_files:
        N_vals, N_parts = gather_class(
            "N", N_files, args.fps, args.chunk, args.start, args.stop, path_map, pid_map, pairs, debug_lines
        )
        all_parts.update(N_vals.keys())

    max_parts = max([E_parts, N_parts, (max(all_parts) if all_parts else 0)])

    # Long-form dataframe: ['part', 'dist', 'cls']
    records: List[Tuple[str, float, str]] = []
    order = [str(i) for i in range(1, max_parts + 1)]
    if E_files:
        for i in range(1, max_parts + 1):
            for v in E_vals.get(i, []):
                records.append((str(i), float(v), "E"))
    if N_files:
        for i in range(1, max_parts + 1):
            for v in N_vals.get(i, []):
                records.append((str(i), float(v), "N"))

    if not records:
        print("[ERROR] No valid movement values found after processing.", file=sys.stderr)
        sys.exit(1)

    df_long = pd.DataFrame(records, columns=["part", "dist", "cls"])

    # Title and output path
    tag = "EN" if (E_files and N_files) else ("E" if E_files else "N")
    sel_tag = lid_label if lid_label else "All"
    out_html = Path(args.out).expanduser().resolve() if args.out else (Path.cwd() / f"box_parts_{tag}_{sel_tag.replace(' ','_')}.html")
    title = f"Per-frame 3D movement by {args.chunk}s parts (class={tag}, fps={args.fps}, sel={sel_tag}, gap-bridged)"

    # --- Box plot
    fig = px.box(
        df_long,
        x="part",
        y="dist",
        color="cls" if (E_files and N_files) else None,
        category_orders={"part": order},
        points=False,
        title=title,
        template="plotly_white",
        color_discrete_map={
            "E": "#aad4f5",  # light blue box
            "N": "#ffd8a6",  # light orange box
        },
    )
    fig.update_traces(marker=dict(opacity=0.7), line=dict(width=1.2))  # light box outlines

    fig.update_layout(
        xaxis_title=f"{args.chunk}s part index",
        yaxis_title="Per-frame movement (mm)",
        boxmode="group",
        legend_title_text="Class" if (E_files and N_files) else "",
        margin=dict(l=60, r=20, t=60, b=60),
    )

    # --- Mean & Median overlays (per class, connected across parts)
    stats = (
        df_long.groupby(["part", "cls"])["dist"]
        .agg(mean="mean", median="median")
        .reset_index()
    )

    def sort_parts(s: pd.Series) -> np.ndarray:
        try:
            return s.astype(int)
        except Exception:
            return s

    for cls_label in (["E", "N"] if (E_files and N_files) else df_long["cls"].unique()):
        scls = stats[stats["cls"] == cls_label].sort_values("part", key=sort_parts)

        # Choose colors
        if cls_label == "E":
            mean_color = "#0057b7"   # dark blue
        else:
            mean_color = "#cc6c00"   # dark orange

        # Mean line (solid)
        fig.add_trace(
            go.Scatter(
                x=scls["part"],
                y=scls["mean"],
                mode="lines+markers",
                name=f"{cls_label} mean",
                hovertemplate="Part %{x}<br>Mean: %{y:.3f}<extra></extra>",
                line=dict(width=2, color=mean_color),
                marker=dict(size=6, color=mean_color),
            )
        )
        # Median line (dashed grey)
        fig.add_trace(
            go.Scatter(
                x=scls["part"],
                y=scls["median"],
                mode="lines+markers",
                name=f"{cls_label} median",
                hovertemplate="Part %{x}<br>Median: %{y:.3f}<extra></extra>",
                line=dict(width=2, dash="dash", color="#666666"),
                marker=dict(size=6, color="#666666"),
            )
        )

    # --- Write outputs
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[OK] Wrote HTML: {out_html}")

    if args.png:
        try:
            import plotly.io as pio
            out_png = out_html.with_suffix(".png")
            pio.write_image(fig, str(out_png), scale=2)
            print(f"[OK] Wrote PNG:  {out_png}")
        except Exception as e:
            print(f"[WARN] PNG export failed (install kaleido?): {e}", file=sys.stderr)

    if args.debug:
        lines = []
        lines.append(f"Classes tag={tag}, E_files={len(E_files)}, N_files={len(N_files)}")
        lines.append(
            f"FPS={args.fps}, CHUNK={args.chunk}s, DEFAULT_START={args.start}s, "
            f"DEFAULT_STOP={args.stop if args.stop is not None else 'full'}"
        )
        lines.append(f"Selection (--lid): {sel_tag} -> {pairs}")
        if args.ranges:
            lines.append(f"Ranges file: {args.ranges}")
        lines.append(f"Max parts observed: {max_parts}")
        lines.append("")
        # Count values per part per class
        for i in range(1, max_parts + 1):
            e_cnt = len(E_vals.get(i, [])) if E_files else 0
            n_cnt = len(N_vals.get(i, [])) if N_files else 0
            lines.append(f"Part {i:02d}: E={e_cnt} values, N={n_cnt} values")
        dbg = Path.cwd() / f"debug_box_{tag}_{sel_tag.replace(' ','_')}_p{max_parts}.log"
        dbg.write_text("\n".join(lines), encoding="utf-8")
        print(f"[DEBUG] Summary: {dbg}")


if __name__ == "__main__":
    main()
