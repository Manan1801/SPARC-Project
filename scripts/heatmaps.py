import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse
from pathlib import Path
import numpy as np
import math
import re

# ================== Hand-prefix mapping (NORMAL convention) ==================
CSV_HAND_PREFIX = {'left': 'L', 'right': 'R'}

def prefix_for(hand: str) -> str:
    return CSV_HAND_PREFIX[hand.lower()]

# ==================================== Args ====================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D/3D hand motion plots from 3D landmark CSV")
    parser.add_argument('--csv', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--landmarks', type=int, nargs='+', required=True, help="Landmark indices to include (0–4)")
    parser.add_argument('--axes', type=str, nargs='+', choices=['X', 'Y', 'Z', 'x', 'y', 'z'], required=True,
                        help="Two or three axes to plot")
    parser.add_argument('--hands', type=str, nargs='+', choices=['left', 'right'], required=True,
                        help="Select hand(s): left, right, or both")
    parser.add_argument('--plots', type=str, choices=['trajectory', 'density', 'both',
                                                     'trajectory3d', 'density3d', 'both3d'],
                        required=True, help="Which plot(s) to generate")
    parser.add_argument('--start', type=float, default=5.0,
                        help="Start time in seconds (frames before this are ignored). Default: 5")
    parser.add_argument('--stop', type=float, default=None,
                        help="Stop time in seconds (limits frames plotted)")

    # Ellipsoids control:
    parser.add_argument(
        '--ellipsoids',
        nargs='*',
        choices=['left', 'right', 'both'],
        help="Overlay covariance ellipsoids/ellipses. "
             "No values: left+right. Values: choose among left, right, both."
    )
    return parser.parse_args()

# ========================== Collectors (2D / 3D) ==========================

def collect_points_3d(df, landmarks, hand):
    prefix = prefix_for(hand)
    pts_list = []
    for lm in landmarks:
        cols = [f'{prefix}_{lm}_X_mm', f'{prefix}_{lm}_Y_mm', f'{prefix}_{lm}_Z_mm']
        if all(c in df.columns for c in cols):
            arr = df[cols].dropna().to_numpy()
            if arr.size:
                pts_list.append(arr)
    return np.vstack(pts_list) if pts_list else None

def collect_points_2d(df, landmarks, hand, axes):
    prefix = prefix_for(hand)
    pts_list = []
    for lm in landmarks:
        cols = [f'{prefix}_{lm}_{axes[0]}_mm', f'{prefix}_{lm}_{axes[1]}_mm']
        if all(c in df.columns for c in cols):
            arr = df[cols].dropna().to_numpy()
            if arr.size:
                pts_list.append(arr)
    return np.vstack(pts_list) if pts_list else None

# ======================= Ellipsoid / Ellipse helpers =======================

# ELLIPSOID_NSTD = 2.5    # 90% confidence interval
ELLIPSOID_NSTD = 2.795  # 95% confidence interval
ELLIPSOID_OPACITY = 0.20
ELLIPSOID_SAMPLES = 40

# --- 3D ---
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

# --- 2D ---
def fit_cov_ellipse_2d(points, n_std=ELLIPSOID_NSTD):
    if points is None or points.shape[0] < 3:
        return None
    c = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    radii = n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    return c, radii, eigvecs, eigvals

def ellipse_poly_2d(center, radii, rotation, n=200):
    t = np.linspace(0, 2*np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)], axis=0)  # 2 x n
    E = rotation @ (radii[:, None] * circ)          # 2 x n
    E = (E.T + center[None, :])                     # n x 2
    return E[:,0], E[:,1]

def count_points_enclosed_2d(points, center, rotation, eigvals, n_std=ELLIPSOID_NSTD):
    if points is None or points.size == 0:
        return 0
    R = rotation
    d = (points - center) @ R
    inv_lam = 1.0 / np.maximum(eigvals, 1e-12)
    mahal2 = (d**2 * inv_lam).sum(axis=1)
    return int((mahal2 <= (n_std**2)).sum())

def add_ellipse2d_to_fig(fig, name, x, y, center, radii, enclosed, color='rgba(0,0,0,1)', opacity=0.18):
    custom = np.column_stack([
        np.full_like(x, center[0], dtype=float),
        np.full_like(x, center[1], dtype=float),
        np.full_like(x, radii[0], dtype=float),
        np.full_like(x, radii[1], dtype=float),
        np.full_like(x, enclosed, dtype=float),
    ])
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name=name,
        fill='toself',
        fillcolor=color.replace('1)', f'{opacity})') if color.endswith('1)') else color,
        line=dict(width=1),
        customdata=custom,
        hovertemplate=(
            "<b>%{meta}</b><br>"
            "Center (mm): (%{customdata[0]:.2f}, %{customdata[1]:.2f})<br>"
            "Axes (mm): a=%{customdata[2]:.2f}, b=%{customdata[3]:.2f}<br>"
            "# Points enclosed: %{customdata[4]:.0f}<extra></extra>"
        ),
        meta=name
    ))

# ============================ Plotting functions ============================

def generate_trajectory_plot(df, selected_landmarks, axes, hands, frames, csv_path, overlay_modes=None):
    fig = go.Figure()
    color = frames

    for hand in hands:
        prefix = prefix_for(hand)
        for lm in selected_landmarks:
            x_col = f'{prefix}_{lm}_{axes[0]}_mm'
            y_col = f'{prefix}_{lm}_{axes[1]}_mm'
            if x_col not in df.columns or y_col not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(color=color, colorscale='Turbo', size=3, showscale=True, colorbar=dict(title='Frame')),
                name=f'{prefix}_{lm}'
            ))

    fig.update_layout(
        title=f"2D Trajectory Plot: {axes[0]} vs {axes[1]}",
        xaxis_title=f"{axes[0]} (mm)",
        yaxis_title=f"{axes[1]} (mm)",
        legend_title="Hand-Landmark",
        height=600
    )
    fig.update_xaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed')

    if len(axes) == 2 and overlay_modes is not None:
        modes = overlay_modes if len(overlay_modes) > 0 else ['left', 'right']
        left2  = collect_points_2d(df, selected_landmarks, 'left', axes)  if 'left'  in hands and 'left'  in modes else None
        right2 = collect_points_2d(df, selected_landmarks, 'right', axes) if 'right' in hands and 'right' in modes else None

        colors = {'Left':'rgba(31,119,180,1)','Right':'rgba(214,39,40,1)','Both':'rgba(44,160,44,1)'}
        for label, pts in [('Left', left2), ('Right', right2)]:
            if pts is None:
                continue
            res = fit_cov_ellipse_2d(pts, n_std=ELLIPSOID_NSTD)
            if res is None:
                continue
            c, r, R, lam = res
            enclosed = count_points_enclosed_2d(pts, c, R, lam, n_std=ELLIPSOID_NSTD)
            ex, ey = ellipse_poly_2d(c, r, R)
            add_ellipse2d_to_fig(fig, f'{label} Ellipse', ex, ey, c, r, enclosed,
                                 color=colors[label], opacity=ELLIPSOID_OPACITY)

        if 'both' in modes and (left2 is not None or right2 is not None):
            both2 = np.vstack([p for p in [left2, right2] if p is not None])
            res = fit_cov_ellipse_2d(both2, n_std=ELLIPSOID_NSTD)
            if res is not None:
                c, r, R, lam = res
                enclosed = count_points_enclosed_2d(both2, c, R, lam, n_std=ELLIPSOID_NSTD)
                ex, ey = ellipse_poly_2d(c, r, R)
                add_ellipse2d_to_fig(fig, 'Both Ellipse', ex, ey, c, r, enclosed,
                                     color=colors['Both'], opacity=ELLIPSOID_OPACITY)

    save_plot(fig, csv_path, selected_landmarks, hands, axes, 'trajectory')

def generate_density_heatmap(df, selected_landmarks, axes, hands, csv_path):
    combined = []
    for hand in hands:
        prefix = prefix_for(hand)
        for lm in selected_landmarks:
            x_col = f'{prefix}_{lm}_{axes[0]}_mm'
            y_col = f'{prefix}_{lm}_{axes[1]}_mm'
            if x_col not in df.columns or y_col not in df.columns:
                continue
            temp = df[[x_col, y_col]].copy()
            temp.columns = ['X', 'Y']
            temp['hand'] = f'{prefix}_{lm}'
            combined.append(temp)
    if not combined:
        print("❌ No 2D points found for density heatmap.")
        return

    full_df = pd.concat(combined)
    fig = px.density_heatmap(
        full_df,
        x='X',
        y='Y',
        facet_col='hand' if len(selected_landmarks) > 1 else None,
        nbinsx=50,
        nbinsy=50,
        color_continuous_scale='Viridis',
        title=f"2D Density Heatmap: {axes[0]} vs {axes[1]}"
    )
    fig.update_layout(xaxis_title=f"{axes[0]} (mm)", yaxis_title=f"{axes[1]} (mm)", height=600)
    fig.update_xaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed')
    save_plot(fig, csv_path, selected_landmarks, hands, axes, 'density')

def generate_trajectory_3d(df, selected_landmarks, hands, frames, csv_path, overlay_modes=None):
    fig = go.Figure()
    color = frames

    for hand in hands:
        prefix = prefix_for(hand)
        for lm in selected_landmarks:
            x_col, y_col, z_col = f'{prefix}_{lm}_X_mm', f'{prefix}_{lm}_Y_mm', f'{prefix}_{lm}_Z_mm'
            if not all(c in df.columns for c in [x_col, y_col, z_col]):
                continue
            fig.add_trace(go.Scatter3d(
                x=df[x_col], y=df[y_col], z=df[z_col],
                mode='markers',
                marker=dict(color=color, colorscale='Turbo', size=2, showscale=True, colorbar=dict(title='Frame')),
                name=f'{prefix}_{lm}'
            ))

    if overlay_modes is not None:
        modes = overlay_modes if len(overlay_modes) > 0 else ['left', 'right']
        left_pts  = collect_points_3d(df, selected_landmarks, 'left')  if 'left'  in hands and 'left'  in modes else None
        right_pts = collect_points_3d(df, selected_landmarks, 'right') if 'right' in hands and 'right' in modes else None

        colors = {'Left':'rgba(31,119,180,1)','Right':'rgba(214,39,40,1)','Both':'rgba(44,160,44,1)'}
        for label, pts in [('Left', left_pts), ('Right', right_pts)]:
            if pts is None:
                continue
            res = fit_cov_ellipsoid(pts, n_std=ELLIPSOID_NSTD)
            if res is None:
                continue
            center, radii, rot, eigvals = res
            enclosed = count_points_enclosed(pts, center, rot, eigvals, n_std=ELLIPSOID_NSTD)
            X, Y, Z = ellipsoid_mesh(center, radii, rot, samples=ELLIPSOID_SAMPLES)
            add_ellipsoid_to_fig(fig, f'{label} Ellipsoid', X, Y, Z, center, radii, enclosed,
                                 opacity=ELLIPSOID_OPACITY, color=colors[label])

        if 'both' in modes and (left_pts is not None or right_pts is not None):
            both_pts = np.vstack([p for p in [left_pts, right_pts] if p is not None])
            res = fit_cov_ellipsoid(both_pts, n_std=ELLIPSOID_NSTD)
            if res is not None:
                center, radii, rot, eigvals = res
                enclosed = count_points_enclosed(both_pts, center, rot, eigvals, n_std=ELLIPSOID_NSTD)
                X, Y, Z = ellipsoid_mesh(center, radii, rot, samples=ELLIPSOID_SAMPLES)
                add_ellipsoid_to_fig(fig, 'Both Ellipsoid', X, Y, Z, center, radii, enclosed,
                                     opacity=ELLIPSOID_OPACITY, color=colors['Both'])

    fig.update_layout(
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        title="3D Trajectory Plot",
        height=1000
    )
    save_plot(fig, csv_path, selected_landmarks, hands, ['X', 'Y', 'Z'], 'trajectory3d')

def generate_density_3d(df, selected_landmarks, hands, csv_path):
    all_points = []
    for hand in hands:
        prefix = prefix_for(hand)
        for lm in selected_landmarks:
            x_col, y_col, z_col = f'{prefix}_{lm}_X_mm', f'{prefix}_{lm}_Y_mm', f'{prefix}_{lm}_Z_mm'
            if not all(c in df.columns for c in [x_col, y_col, z_col]):
                continue
            pts = df[[x_col, y_col, z_col]].dropna().to_numpy()
            if pts.size:
                all_points.append(pts)
    if not all_points:
        print("❌ No 3D points found.")
        return

    all_points = np.vstack(all_points)
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    bins = 30
    hist, edges = np.histogramdd((x, y, z), bins=bins)
    x_c, y_c, z_c = [0.5 * (e[1:] + e[:-1]) for e in edges]

    X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing='ij')
    values = hist.flatten()
    coords = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    fig = go.Figure(data=go.Volume(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        value=values, isomin=1, isomax=values.max(),
        opacity=0.1, surface_count=15, colorscale='Viridis',
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        title="3D Density Volume",
        height=1000
    )
    save_plot(fig, csv_path, selected_landmarks, hands, ['X', 'Y', 'Z'], 'density3d')

# ================================ Utilities ================================

def get_cam2_plots_dir(csv_path: Path) -> Path:
    p = csv_path.resolve()
    if p.parent.name.upper() == 'CSV' and p.parent.parent.name.lower() == 'cam2':
        cam2_dir = p.parent.parent
    else:
        cam2_dir = None
        for parent in p.parents:
            if parent.name.lower() == 'cam2':
                cam2_dir = parent
                break
        if cam2_dir is None:
            cam2_dir = p.parent
    plots_dir = cam2_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def extract_participant_id(csv_path: Path) -> str:
    """
    Expect .../<participant_id>/cam2/CSV/file.csv → returns <participant_id>.
    Falls back to the directory above 'cam2' if present, else csv stem.
    """
    p = csv_path.resolve()
    # Find cam2 and take its parent as participant id folder
    for parent in p.parents:
        if parent.name.lower() == 'cam2':
            pid = parent.parent.name  # folder above cam2
            return pid
    # Fallback: try first numeric-looking parent
    for parent in p.parents:
        if re.fullmatch(r'\d+', parent.name):
            return parent.name
    return p.stem

def save_plot(fig, csv_path, landmarks, hands, axes, plot_type):
    out_dir = get_cam2_plots_dir(Path(csv_path))
    landmarks_str = "_".join(str(lm) for lm in landmarks)
    hand_str = ''.join(['L' if h == 'left' else 'R' for h in hands])

    if '3d' in plot_type:  # special naming for 3D plots
        participant_id = extract_participant_id(Path(csv_path))
        filename = f"{participant_id}__{plot_type}__{hand_str}__lm-{landmarks_str}.html"
    else:  # 2D plots: keep csvbase + axes
        csv_base = Path(csv_path).stem
        axes_str = ''.join(axes).upper()
        filename = f"{csv_base}__{plot_type}__{axes_str}__{hand_str}__lm-{landmarks_str}.html"

    out_file = out_dir / filename
    fig.write_html(out_file)
    print(f"✅ Saved {plot_type} plot to {out_file}")

# ================================== Main ===================================

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    # Time window trimming (assume 30 FPS)
    start_frame = int(args.start * 30)
    if 'frame' in df.columns:
        df = df[df['frame'] >= start_frame]
    else:
        print("⚠️ 'frame' column not found; --start ignored.")

    if args.stop is not None:
        stop_frame = int(args.stop * 30)
        if 'frame' in df.columns:
            df = df[df['frame'] <= stop_frame]
        else:
            print("⚠️ 'frame' column not found; --stop ignored.")

    frames = df['frame'] if 'frame' in df.columns else pd.Series(np.arange(len(df)))
    axes = [a.upper() for a in args.axes]
    hands = [h.lower() for h in args.hands]
    landmarks = args.landmarks

    overlay_modes = args.ellipsoids  # None -> off; [] -> left+right; subset -> as chosen

    if args.plots in ['trajectory', 'both'] and len(axes) == 2:
        generate_trajectory_plot(df, landmarks, axes, hands, frames, csv_path, overlay_modes=overlay_modes)
    elif args.ellipsoids is not None and len(axes) == 2:
        pass

    if args.plots in ['density', 'both'] and len(axes) == 2:
        generate_density_heatmap(df, landmarks, axes, hands, csv_path)

    if args.plots in ['trajectory3d', 'both3d']:
        generate_trajectory_3d(df, landmarks, hands, frames, csv_path, overlay_modes=overlay_modes)
    elif args.ellipsoids is not None:
        print("ℹ️ --ellipsoids overlays apply to 2D/3D trajectory plots (not density).")

    if args.plots in ['density3d', 'both3d']:
        generate_density_3d(df, landmarks, hands, csv_path)

if __name__ == '__main__':
    main()
