#!/usr/bin/env python3
import pandas as pd
import plotly.graph_objects as go
import argparse
from pathlib import Path
import numpy as np
import math
import re
import sys

# ================== Fixed master header (always these columns) ==================
MASTER_KP_LABELS = ["L0", "R0", "L4", "R4"]   # schema is fixed to these

# ================== Ellipsoid settings (EXACT logic you provided) ==================
# ELLIPSOID_NSTD = 2.795  # 95% Point Coverage
ELLIPSOID_NSTD = 2.5  # 90% Point Coverage
ELLIPSOID_OPACITY = 0.20
ELLIPSOID_SAMPLES = 40

# ================== Ellipsoid helpers (EXACT logic you provided) ==================
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

def ellipsoid_mesh(center, radii, rotation, samples=ELLIPSOID_SAMPLES):
    if center is None:
        return None
    u = np.linspace(0.0, 2.0*np.pi, samples)
    v = np.linspace(0.0, np.pi, samples)
    uu, vv = np.meshgrid(u, v)
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)
    S = np.stack([xs, ys, zs], axis=-1)        # (...,3)
    A = rotation @ np.diag(radii)              # 3x3
    E = S @ A.T                                # (...,3)
    E += center[None, None, :]
    return E[...,0], E[...,1], E[...,2]

def count_points_enclosed(points, center, rotation, eigvals, n_std=ELLIPSOID_NSTD):
    if points is None or points.size == 0:
        return 0
    R = rotation
    d = (points - center) @ R
    inv_lam = 1.0 / np.maximum(eigvals, 1e-12)
    mahal2 = (d**2 * inv_lam).sum(axis=1)
    return int((mahal2 <= (n_std**2)).sum())

def add_ellipsoid_to_fig(fig, name, X, Y, Z, center, radii, enclosed_count,
                         opacity=ELLIPSOID_OPACITY, color='rgba(0,0,0,1)'):
    if X is None:
        return
    volume = (4.0/3.0) * math.pi * (radii[0] * radii[1] * radii[2])  # mm^3
    cdata = np.tile(np.array([[center[0], center[1], center[2],
                               radii[0], radii[1], radii[2],
                               enclosed_count, volume]]),
                    (X.shape[0], X.shape[1], 1))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        name=name,
        showscale=False,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        customdata=cdata,
        hovertemplate=(
            "<b>%{meta}</b><br>"
            "Center (mm): (%{customdata[0]:.2f}, %{customdata[1]:.2f}, %{customdata[2]:.2f})<br>"
            "Axes (mm): a=%{customdata[3]:.2f}, b=%{customdata[4]:.2f}, c=%{customdata[5]:.2f}<br>"
            "Volume (mm³): %{customdata[7]:.2f}<br>"
            "# Points enclosed: %{customdata[6]}<extra></extra>"
        ),
        meta=name
    ))

# ================== IO / naming helpers ==================
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

def segmented_indices(n_rows: int, segments: int):
    ends = []
    for i in range(1, segments + 1):
        end_i = int(np.ceil(i * n_rows / segments) - 1)
        end_i = min(max(end_i, 0), n_rows - 1)
        ends.append(end_i)
    return ends

def parse_kp_token(tok: str):
    """
    'L0' -> ('L', 0) ; 'R4' -> ('R', 4)
    """
    m = re.fullmatch(r'([LRlr])\s*(\d{1,2})', tok.strip())
    if not m:
        raise ValueError(f"Bad keypoint token: {tok} (use like L0, R4)")
    hand = m.group(1).upper()
    idx = int(m.group(2))
    return hand, idx

def load_points_for_keypoint(df, hand_prefix: str, lm: int):
    cols = [f'{hand_prefix}_{lm}_X_mm', f'{hand_prefix}_{lm}_Y_mm', f'{hand_prefix}_{lm}_Z_mm']
    if not all(c in df.columns for c in cols):
        return None, None
    sub = df[['frame'] + cols].dropna()
    if sub.empty:
        return None, None
    pts = sub[cols].to_numpy()
    frames = sub['frame'].to_numpy()
    return pts, frames

# ================== CSV append (ALWAYS master schema) ==================
def expected_header(segments: int):
    cols = ['participant_id']
    for s in range(1, segments + 1):
        for label in MASTER_KP_LABELS:
            cols.append(f'vol_{label}_seg{s}')
    return cols

def append_volumes(out_csv: Path, participant_id: str, seg_to_vols: dict, segments: int):
    """
    seg_to_vols: { seg_index : { 'L0': vol, 'R0': vol, 'L4': vol, 'R4': vol } }
    Only selected keypoints have numbers; others stay "".
    """
    cols = expected_header(segments)
    write_header = not out_csv.exists()

    # Validate existing header if file exists
    if out_csv.exists():
        try:
            with open(out_csv, 'r') as f:
                existing = [c.strip() for c in f.readline().strip().split(',')]
            if existing and existing != cols:
                print("⚠️ Existing header differs from expected master schema for current --segments.")
                print("   Existing:", existing)
                print("   Expected:", cols)
        except Exception as e:
            print(f"⚠️ Could not validate existing header: {e}")

    # Build row strictly in header order
    row = [participant_id]
    for s in range(1, segments + 1):
        kp_map = seg_to_vols.get(s, {}) or {}
        for label in MASTER_KP_LABELS:
            v = kp_map.get(label, "")
            row.append("" if v in (None, "None") else str(v))

    # Hard safety: lengths must match
    assert len(row) == len(cols), f"Row length {len(row)} != header length {len(cols)}"

    with open(out_csv, 'a' if out_csv.exists() else 'w') as f:
        if write_header:
            f.write(','.join(cols) + '\n')
        f.write(','.join(row) + '\n')

# ================== Per-segment figure ==================
def build_segment_figure(df_seg, kp_defs, colors_for_kp, participant_id, seg_idx, segments):
    """
    For each selected keypoint:
      - scatter3d colored by frame (time) with ONE colorbar placed left
      - ellipsoid at 2.795σ (exact logic)
    Returns (fig, volumes_dict) where dict keys are the selected labels only.
    """
    fig = go.Figure()
    volumes = {}
    first_cb_trace_idx = None  # track the single colorbar trace

    for i, (hand_prefix, lm) in enumerate(kp_defs):
        tag = f"{hand_prefix}{lm}"

        pts, frames = load_points_for_keypoint(df_seg, hand_prefix, lm)
        # scatter with time gradient
        if pts is not None:
            showscale = (first_cb_trace_idx is None)  # only the first scatter shows colorbar
            fig.add_trace(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=frames if frames is not None else np.arange(len(pts)),
                    colorscale='Turbo',
                    showscale=showscale,
                    colorbar=dict(title='Frame') if showscale else None
                ),
                name=f'{tag} points',
                legendgroup=tag,
                showlegend=True
            ))
            if showscale:
                first_cb_trace_idx = len(fig.data) - 1

        vol_val = ""
        if pts is not None and pts.shape[0] >= 4:
            res = fit_cov_ellipsoid(pts, n_std=ELLIPSOID_NSTD)
            if res is not None:
                center, radii, rot, eigvals = res
                X, Y, Z = ellipsoid_mesh(center, radii, rot, samples=ELLIPSOID_SAMPLES)
                enclosed = count_points_enclosed(pts, center, rot, eigvals, n_std=ELLIPSOID_NSTD)
                vol = (4.0/3.0) * math.pi * (radii[0] * radii[1] * radii[2])
                vol_val = f"{vol:.6f}"
                add_ellipsoid_to_fig(
                    fig, f'{tag} Ellipsoid', X, Y, Z, center, radii, enclosed,
                    opacity=ELLIPSOID_OPACITY, color=colors_for_kp[tag]
                )

        volumes[tag] = vol_val  # only for selected KPs

    kp_list_pretty = ','.join([f'{h}{l}' for h,l in kp_defs])
    fig.update_layout(
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        title=f"{participant_id} — Seg {seg_idx}/{segments} (cumulative) — 3D {kp_list_pretty} + {ELLIPSOID_NSTD:.3g}σ ellipsoids",
        height=900,
        legend=dict(
            x=1.02,  # keep legend on the right, outside plot area
            y=1.0
        ),
        margin=dict(l=80, r=120, t=60, b=60)  # a little extra right margin for legend
    )

    # Move the single colorbar to the left side so it doesn't overlap the legend
    if first_cb_trace_idx is not None:
        try:
            # Shift colorbar to the left margin and center vertically
            fig.data[first_cb_trace_idx].marker.colorbar.x = -0.15
            fig.data[first_cb_trace_idx].marker.colorbar.y = 0.5
        except Exception:
            pass

    return fig, volumes

# ================== CLI / Main ==================
def parse_args():
    p = argparse.ArgumentParser(
        description="Cumulative segmented 3D scatter with per-keypoint ellipsoid fits (exact logic), time-colored points. Appends volumes using a fixed master header."
    )
    p.add_argument('--csv', required=True, help="Path to input CSV")
    p.add_argument('--segments', type=int, default=6, help="Number of equal segments (default: 6)")
    p.add_argument('--out-csv', required=True, help="CSV to append volumes per segment (fixed master schema)")
    p.add_argument('--out-dir', required=True, help="Directory to save HTML plots")
    p.add_argument('--keypoints', nargs='+', default=['L0','R0','L4','R4'],
                   help="Keypoints like L0 R0 L4 R4 (case-insensitive). Default: all four")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse keypoints (subset used for fitting/plots)
    try:
        kp_defs = [parse_kp_token(k) for k in args.keypoints]
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    kp_labels = [f"{h}{l}" for h,l in kp_defs]

    # color palette per selected keypoint (ellipsoid surface color)
    base_colors = [
        'rgba(31,119,180,1)',   # blue
        'rgba(214,39,40,1)',    # red
        'rgba(44,160,44,1)',    # green
        'rgba(148,103,189,1)',  # purple
        'rgba(255,127,14,1)',   # orange
        'rgba(23,190,207,1)',   # teal
        'rgba(140,86,75,1)',    # brown
        'rgba(227,119,194,1)',  # pink
    ]
    colors_for_kp = {}
    for i, lab in enumerate(kp_labels):
        colors_for_kp[lab] = base_colors[i % len(base_colors)]

    # load CSV
    df = pd.read_csv(csv_path)
    if 'frame' in df.columns:
        df = df.sort_values('frame').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df['frame'] = np.arange(len(df))

    n = len(df)
    if n == 0:
        print("❌ Empty CSV.")
        return

    seg_ends = segmented_indices(n, args.segments)
    participant_id = extract_participant_id(csv_path)

    seg_to_vols = {}
    segment_figs = []

    # per-segment cumulative plots
    for idx, end_i in enumerate(seg_ends, start=1):
        df_seg = df.iloc[:end_i + 1].copy()
        fig_seg, vols_selected = build_segment_figure(df_seg, kp_defs, colors_for_kp, participant_id, idx, args.segments)

        # map selected to master (strictly fill only selected KPs; keep others blank)
        vols_master = {kp: "" for kp in MASTER_KP_LABELS}
        for lab, val in vols_selected.items():
            if lab in vols_master:
                vols_master[lab] = val
        seg_to_vols[idx] = vols_master

        # Save per-segment HTML
        kp_str = "_".join(kp_labels)
        html_name = f"{participant_id}__trajectory3d__kp-{kp_str}__seg-{idx}.html"
        out_file = out_dir / html_name
        ensure_parent(out_file)
        fig_seg.write_html(out_file)
        print(f"✅ Saved: {out_file}")

        segment_figs.append(fig_seg)

    # Append volumes to CSV with master header
    ensure_parent(out_csv)
    append_volumes(out_csv, participant_id, seg_to_vols, args.segments)
    print(f"📝 Appended volumes to: {out_csv}")

    # Build combined interactive with slider/dropdown
    combined = go.Figure()
    vis_masks = []
    start_idx = 0
    for fig_seg in segment_figs:
        for tr in fig_seg.data:
            combined.add_trace(tr)
        seg_count = len(fig_seg.data)
        mask = [False] * len(combined.data)
        for j in range(start_idx, start_idx + seg_count):
            mask[j] = True
        vis_masks.append(mask)
        start_idx += seg_count

    steps = []
    for i, mask in enumerate(vis_masks, start=1):
        steps.append(dict(
            method="update",
            args=[{"visible": mask},
                  {"title": f"{participant_id} — Seg {i}/{args.segments} (cumulative) — 3D {','.join(kp_labels)} + {ELLIPSOID_NSTD:.3g}σ ellipsoids"}],
            label=f"{i}"
        ))

    initial = vis_masks[0] if vis_masks else []
    for k in range(len(combined.data)):
        combined.data[k].visible = initial[k] if k < len(initial) else False

    combined.update_layout(
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        title=f"{participant_id} — Seg 1/{args.segments} (cumulative) — 3D {','.join(kp_labels)} + {ELLIPSOID_NSTD:.3g}σ ellipsoids",
        height=900,
        sliders=[dict(active=0, currentvalue={"prefix": "Segment: "}, steps=steps, pad={"t": 30})],
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=[dict(method="update", label=f"Segment {i}",
                          args=[{"visible": vis_masks[i-1]},
                                {"title": f"{participant_id} — Seg {i}/{args.segments} (cumulative) — 3D {','.join(kp_labels)} + {ELLIPSOID_NSTD:.3g}σ ellipsoids"}])
                     for i in range(1, args.segments+1)],
            x=0.02, y=1.12
        )],
        legend=dict(x=1.02, y=1.0),
        margin=dict(l=80, r=120, t=60, b=60)
    )

    # Move the (one) visible colorbar of the active segment to the left.
    # Only traces from the active segment are visible, so this is sufficient.
    # Find the first visible Scatter3d with a colorbar and shift it.
    for tr in combined.data:
        if isinstance(tr, go.Scatter3d) and getattr(tr.marker, "showscale", False):
            try:
                tr.marker.colorbar.x = -0.15
                tr.marker.colorbar.y = 0.5
            except Exception:
                pass
            break

    kp_str = "_".join(kp_labels)
    combined_name = f"{participant_id}__trajectory3d_cumulative_ellipsoids__kp-{kp_str}__S{args.segments}.html"
    combined_file = out_dir / combined_name
    combined.write_html(combined_file)
    print(f"🌐 Combined all segments into: {combined_file}")

if __name__ == "__main__":
    main()
