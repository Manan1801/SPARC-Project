#!/usr/bin/env python3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D/3D hand motion plots from 3D landmark CSV")
    parser.add_argument('--csv', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--landmarks', type=int, nargs='+', required=True, help="Landmark indices to include (0–4)")
    parser.add_argument('--axes', type=str, nargs='+', choices=['X', 'Y', 'Z', 'x', 'y', 'z'], required=True, help="Two or three axes to plot")
    parser.add_argument('--hands', type=str, nargs='+', choices=['left', 'right'], required=True, help="Select hand(s): left, right, or both")
    parser.add_argument('--plots', type=str, choices=['trajectory', 'density', 'both', 'trajectory3d', 'density3d', 'both3d'], required=True, help="Which plot(s) to generate")
    parser.add_argument('--stop_time', type=float, default=None, help="Stop time in seconds (limits frames plotted)")
    return parser.parse_args()

def generate_trajectory_plot(df, selected_landmarks, axes, hands, frames, out_path):
    fig = go.Figure()
    color = frames

    for hand in hands:
        prefix = 'L' if hand == 'left' else 'R'
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
    # Reverse vertical direction
    fig.update_xaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed')

    save_plot(fig, out_path, selected_landmarks, hands, axes, 'trajectory')

def generate_density_heatmap(df, selected_landmarks, axes, hands, out_path):
    combined = []

    for hand in hands:
        prefix = 'L' if hand == 'left' else 'R'
        for lm in selected_landmarks:
            x_col = f'{prefix}_{lm}_{axes[0]}_mm'
            y_col = f'{prefix}_{lm}_{axes[1]}_mm'
            if x_col not in df.columns or y_col not in df.columns:
                continue
            temp = df[[x_col, y_col]].copy()
            temp.columns = ['X', 'Y']
            temp['hand'] = f'{prefix}_{lm}'
            combined.append(temp)

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

    fig.update_layout(
        xaxis_title=f"{axes[0]} (mm)",
        yaxis_title=f"{axes[1]} (mm)",
        height=600
    )
    # Reverse vertical direction (applies to all facets)
    fig.update_xaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed')

    save_plot(fig, out_path, selected_landmarks, hands, axes, 'density')

def generate_trajectory_3d(df, selected_landmarks, hands, frames, out_path):
    fig = go.Figure()
    color = frames

    for hand in hands:
        prefix = 'L' if hand == 'left' else 'R'
        for lm in selected_landmarks:
            x_col, y_col, z_col = f'{prefix}_{lm}_X_mm', f'{prefix}_{lm}_Y_mm', f'{prefix}_{lm}_Z_mm'
            if not all(c in df.columns for c in [x_col, y_col, z_col]):
                continue
            fig.add_trace(go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode='markers',
                marker=dict(color=color, colorscale='Turbo', size=2, showscale=True, colorbar=dict(title='Frame')),
                name=f'{prefix}_{lm}'
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)'
        ),
        title="3D Trajectory Plot",
        height=700
    )
    
    save_plot(fig, out_path, selected_landmarks, hands, ['X', 'Y', 'Z'], 'trajectory3d')

def generate_density_3d(df, selected_landmarks, hands, out_path):
    all_points = []

    for hand in hands:
        prefix = 'L' if hand == 'left' else 'R'
        for lm in selected_landmarks:
            x_col, y_col, z_col = f'{prefix}_{lm}_X_mm', f'{prefix}_{lm}_Y_mm', f'{prefix}_{lm}_Z_mm'
            if not all(c in df.columns for c in [x_col, y_col, z_col]):
                continue
            pts = df[[x_col, y_col, z_col]].dropna().to_numpy()
            all_points.append(pts)

    if not all_points:
        print("❌ No 3D points found.")
        return

    all_points = np.vstack(all_points)
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]

    # Convert to voxel grid
    bins = 30
    hist, edges = np.histogramdd((x, y, z), bins=bins)
    x_c, y_c, z_c = [0.5 * (e[1:] + e[:-1]) for e in edges]

    X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing='ij')
    values = hist.flatten()
    coords = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    fig = go.Figure(data=go.Volume(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        value=values,
        isomin=1,
        isomax=values.max(),
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis',
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)'
        ),
        title="3D Density Volume",
        height=700
    )
    
    save_plot(fig, out_path, selected_landmarks, hands, ['X', 'Y', 'Z'], 'density3d')

def save_plot(fig, out_path, landmarks, hands, axes, plot_type):
    landmark_str = "lm" + "_".join(str(lm) for lm in landmarks)
    hand_str = ''.join(['L' if h == 'left' else 'R' for h in hands])
    axes_str = ''.join(axes)
    filename = f"{out_path.stem}_{landmark_str}_{hand_str}_{axes_str}_{plot_type}.html"
    fig.write_html(out_path.with_name(filename))
    print(f"✅ Saved {plot_type} plot to {out_path.with_name(filename)}")

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    # Filter DataFrame based on stop_time if provided
    if args.stop_time is not None:
        stop_frame = int(args.stop_time * 30)  # assuming 30 FPS
        df = df[df['frame'] <= stop_frame]

    frames = df['frame']
    axes = [a.upper() for a in args.axes]
    hands = [h.lower() for h in args.hands]
    landmarks = args.landmarks

    if args.plots in ['trajectory', 'both'] and len(axes) == 2:
        generate_trajectory_plot(df, landmarks, axes, hands, frames, csv_path)
    if args.plots in ['density', 'both'] and len(axes) == 2:
        generate_density_heatmap(df, landmarks, axes, hands, csv_path)
    if args.plots in ['trajectory3d', 'both3d']:
        generate_trajectory_3d(df, landmarks, hands, frames, csv_path)
    if args.plots in ['density3d', 'both3d']:
        generate_density_3d(df, landmarks, hands, csv_path)

if __name__ == '__main__':
    main()
