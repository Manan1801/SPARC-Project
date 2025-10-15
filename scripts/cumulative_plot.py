#!/usr/bin/env python3
"""
cumulative_plot.py

Generate a cumulative movement line plot (interactive HTML + clean PNG)
with piecewise linear regression overlays.

INPUT CSV COLUMNS (wide format):
    pid, class, L{k_id}_p{chunk}, R{k_id}_p{chunk}, ...
where {k_id} is the keypoint id and {chunk} is the 1..M chunk index
(10-second groups). Columns for a given keypoint are contiguous.

WHAT GETS PLOTTED:
- One time-series per "entity" (either a specific pid, or the class-average
  over pids) by summing the selected keypoints across both hands (L* and R*).
  If --keypoints is not provided, all keypoints present are included.
- If --pid is provided: plot those pid(s) as separate lines.
- If --class is provided: plot the average across pids in that class.
- If BOTH are provided: only those pid(s) that belong to the class are plotted.
- If NEITHER --pid nor --class is provided: plot BOTH class-averages.

PIECEWISE REGRESSION:
- Default 6 segments (i.e., 5 breakpoints). Dotted regression lines.
- Uses 'ruptures' if available for optimal breakpoints; otherwise falls
  back to equal-sized segments.

LEGEND & LABELS:
- HTML (Plotly): short legend (actual series only). Regression shows hover with R².
- PNG (Matplotlib): legend only for actual series; each regression segment shows
  a non-overlapping slope label (mm/s) with a small series-colored marker.

KEYPOINT SELECTION:
    --keypoints can include a mix of:
        - hand-specific tokens: L0 R0 R4 ...
        - id-only tokens: 0 1 2 ...   (applies to BOTH hands)
    Examples:
        --keypoints L0 R4 2    → selects L0 + R4 + L2 & R2
        --keypoints 0 1        → selects L0,R0 and L1,R1
    If omitted, include ALL detected keypoints for BOTH hands.

UNEQUAL DURATIONS:
- Use --truncate-at-max (default) to truncate each series at the time it
  reaches its own maximum cumulative movement. The min–max shaded band is
  computed only over timepoints where at least two series are present.

USAGE EXAMPLES:
    python cumulative_plot.py --csv /path/to/cumulative.csv
    python cumulative_plot.py --csv data.csv --class Learner
    python cumulative_plot.py --csv data.csv --pid AP_03 AP_05 --class Learner
    python cumulative_plot.py --csv data.csv --keypoints 0 1 2 --segments 5 --no-regression

DEPENDENCIES:
    pip install pandas numpy plotly matplotlib
    (optional) pip install ruptures
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Try to import ruptures; if unavailable, we’ll fall back.
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False

# ---------- Regexes ----------
_KP_COL_RE = re.compile(r'^([LR])(\d+)_p(\d+)$')
_HAND_TOKEN_RE = re.compile(r'^[LR](\d+)$')
_ID_TOKEN_RE = re.compile(r'^\d+$')


# ---------- Parse CSV structure ----------
def parse_kp_columns(df: pd.DataFrame):
    mapping = {'L': {}, 'R': {}}
    keypoints = set()
    chunks = set()
    for col in df.columns:
        m = _KP_COL_RE.match(col)
        if not m:
            continue
        hand, k_id_s, chunk_s = m.groups()
        k_id = int(k_id_s)
        chunk = int(chunk_s)
        keypoints.add(k_id)
        chunks.add(chunk)
        mapping[hand].setdefault(k_id, []).append(col)
    for hand in ('L', 'R'):
        for k in mapping.get(hand, {}):
            mapping[hand][k].sort(key=lambda c: int(_KP_COL_RE.match(c).group(3)))
    return sorted(keypoints), sorted(chunks), mapping


# ---------- Parse --keypoints tokens ----------
def parse_keypoint_tokens(tokens, all_kps, mapping):
    sel = {'L': set(), 'R': set()}
    if not tokens:
        for h in ('L', 'R'):
            sel[h] = set(mapping[h].keys())
        return sel
    for t in tokens:
        t = str(t).strip()
        m_hand = _HAND_TOKEN_RE.match(t)
        m_id = _ID_TOKEN_RE.match(t)
        if m_hand:
            k = int(m_hand.group(1))
            hand = 'L' if t[0].upper() == 'L' else 'R'
            if k in mapping.get(hand, {}):
                sel[hand].add(k)
        elif m_id:
            k = int(m_id.group(0))
            for hand in ('L', 'R'):
                if k in mapping.get(hand, {}):
                    sel[hand].add(k)
        else:
            pass
    if not sel['L'] and not sel['R']:
        for h in ('L', 'R'):
            sel[h] = set(mapping[h].keys())
    return sel


# ---------- Build a single aggregated series (no tail filling) ----------
def build_series_for_rows(rows_df: pd.DataFrame,
                          mapping, selected_map: dict) -> pd.Series:
    # Determine all chunk indices present
    all_chunks = set()
    for hand in ('L', 'R'):
        for k, cols in mapping.get(hand, {}).items():
            for c in cols:
                m = _KP_COL_RE.match(c)
                if m:
                    all_chunks.add(int(m.group(3)))
    chunks_sorted = sorted(all_chunks)

    # Gather relevant columns
    cols_to_sum = []
    for hand in ('L', 'R'):
        for k in sorted(selected_map.get(hand, [])):
            cols_to_sum.extend(mapping.get(hand, {}).get(k, []))
    if not cols_to_sum:
        raise ValueError("No columns found for selected --keypoints selection.")

    per_chunk_vals = []
    for chunk in chunks_sorted:
        chunk_cols = [c for c in cols_to_sum if int(_KP_COL_RE.match(c).group(3)) == chunk]
        if not chunk_cols:
            per_chunk_vals.append(np.nan)
        else:
            summed = rows_df[chunk_cols].sum(axis=1, skipna=True)
            per_chunk_vals.append(summed.mean(skipna=True))

    # Do NOT forward-fill the tail; keep NaNs beyond last valid
    s = pd.Series(per_chunk_vals, index=chunks_sorted, dtype=float)
    # Fill ONLY internal gaps to keep regression stable: ff inside [first_valid, last_valid]
    if s.notna().any():
        first_idx = s.first_valid_index()
        last_idx = s.last_valid_index()
        s.loc[first_idx:last_idx] = s.loc[first_idx:last_idx].ffill()
    return s


# ---------- Segmentation & fitting ----------
def segment_breakpoints(y: np.ndarray, n_segments: int) -> list:
    n = len(y)
    n_segments = max(1, min(n_segments, n))
    if n_segments == 1:
        return [n]
    if HAS_RUPTURES and n > n_segments:
        algo = rpt.KernelCPD(kernel="linear").fit(y.reshape(-1, 1))
        try:
            bkps = algo.predict(n_bkps=n_segments - 1)
            if bkps[-1] != n:
                bkps[-1] = n
            return bkps
        except Exception:
            pass
    edges = np.linspace(0, n, num=n_segments + 1, dtype=int)
    return list(edges[1:])


def fit_segments(x: np.ndarray, y: np.ndarray, n_segments: int):
    """
    Return list of segments: {x0, x1, slope, intercept, r2}
    """
    n = len(y)
    if n < 2:
        return []
    bkps = segment_breakpoints(y, n_segments)
    segs = []
    start = 0
    for end in bkps:
        xs = x[start:end]
        ys = y[start:end]
        if len(xs) >= 2 and np.any(np.isfinite(ys)):
            m, b = np.polyfit(xs, ys, 1)
            y_pred = m * xs + b
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            m, b, r2 = np.nan, np.nan, np.nan
        segs.append({'x0': start, 'x1': end,
                     'slope': m, 'intercept': b, 'r2': r2})
        start = end
    return segs


# ---------- Label placement helper (non-overlapping, cleans up trials) ----------
def place_text(ax, placed, xm, ym, txt, max_tries=20):
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    dy_step = 0.02 * yspan
    y_try = ym
    for k in range(max_tries):
        art = ax.text(xm, y_try, txt, ha='center', va='bottom',
                      fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.7,
                                edgecolor='none', pad=1.5))
        plt.draw()
        bb = art.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        inv = ax.transData.inverted()
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        w = abs(x1 - x0); h = abs(y1 - y0)

        ok = True
        for (px, py, pw, ph) in placed:
            if (abs(xm - px) < (w + pw) * 0.55) and (abs(y_try - py) < (h + ph) * 0.75):
                ok = False
                break
        if ok:
            placed.append((xm, y_try, w, h))
            return art
        else:
            art.remove()
            y_try = ym + ((k // 2 + 1) * dy_step) * (1 if k % 2 else -1)
    placed.append((xm, y_try, w, h))
    return art


# ---------- Plotting (Plotly) ----------
def plot_plotly_html(series_dict, segs_dict, out_html: Path, title: str,
                     show_regression: bool, band):
    fig = go.Figure()

    # Min–Max shaded band
    if band is not None:
        xb, y_min, y_max = band
        fig.add_trace(go.Scatter(
            x=xb, y=y_min, mode="lines",
            line=dict(width=0),
            name="min", showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=xb, y=y_max, mode="lines",
            line=dict(width=0),
            fill='tonexty',
            fillcolor="rgba(128,128,128,0.18)",
            name="Range", showlegend=False, hoverinfo="skip"
        ))

    # Actual lines (solid)
    for label, (x, y) in series_dict.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name=label,
            line=dict(dash="solid"),
            hovertemplate="t=%{x:.0f}s<br>Cumulative=%{y:.1f} mm<extra>" + label + "</extra>"
        ))

    # Regression segments (dotted, densified, hover with R²)
    if show_regression:
        for label, segs in segs_dict.items():
            x_full, _y_full = series_dict[label]
            for seg in segs:
                x0 = float(x_full[seg['x0']])
                x1 = float(x_full[seg['x1'] - 1])
                xs = np.linspace(x0, x1, 50, dtype=float)
                ys = seg['slope'] * xs + seg['intercept']
                dt = x1 - x0
                dy = float(ys[-1] - ys[0])
                slope = float(seg['slope'])
                r2 = float(seg['r2'])
                segment_meta = np.array([x0, x1, dt, dy, slope, r2], dtype=float)
                custom = np.tile(segment_meta, (len(xs), 1))

                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(dash="dot"),
                    showlegend=False,
                    hovertemplate=(
                        "Regression segment<br>"
                        "t: %{customdata[0]:.0f}–%{customdata[1]:.0f} s (Δt=%{customdata[2]:.0f} s)<br>"
                        "Δmovement: %{customdata[3]:.1f} mm<br>"
                        "Slope: %{customdata[4]:.3f} mm/s<br>"
                        "R²: %{customdata[5]:.3f}"
                        "<extra></extra>"
                    ),
                    customdata=custom
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Movement (mm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    fig.write_html(str(out_html))


# ---------- Plotting (Matplotlib) ----------
def plot_matplotlib_png(series_dict, segs_dict, out_png: Path, title: str,
                        show_regression: bool, band):
    plt.figure(figsize=(12, 6), dpi=160)
    ax = plt.gca()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Min–Max shaded band
    if band is not None:
        xb, y_min, y_max = band
        ax.fill_between(xb, y_min, y_max, color="0.5", alpha=0.18, linewidth=0, zorder=0)

    # Actual lines
    handles, labels = [], []
    for i, (label, (x, y)) in enumerate(series_dict.items()):
        line, = ax.plot(x, y, linewidth=2.2, label=label,
                        color=color_cycle[i % len(color_cycle)])
        handles.append(line); labels.append(label)

    # Non-overlapping labels state
    placed = []  # (x,y,w,h) in data units

    # Regression (optional) + slope label & colored marker
    if show_regression:
        for i, (label, segs) in enumerate(segs_dict.items()):
            color = color_cycle[i % len(color_cycle)]
            x_full, _y_full = series_dict[label]
            for seg in segs:
                x0 = x_full[seg['x0']]
                x1 = x_full[seg['x1'] - 1]
                xs = np.array([x0, x1])
                ys = seg['slope'] * xs + seg['intercept']
                ax.plot(xs, ys, linestyle=':', linewidth=1.8, color=color)

                # slope label at segment mid in data space + colored marker
                xm = xs.mean()
                ym = np.polyval([seg['slope'], seg['intercept']], xm)
                txt = f"{seg['slope']:.3f} mm/s"
                text_artist = place_text(ax, placed, xm, ym, txt)

                xt, yt = text_artist.get_position()
                ax.scatter([xt], [yt], s=28, c=[color], edgecolors='white',
                           linewidths=0.8, zorder=text_artist.get_zorder()+1)

                # nudge text ~8pt to the right of the marker
                def _offset_in_data(ax, dx_pts=8, dy_pts=0):
                    inv = ax.transData.inverted()
                    x0p, y0p = inv.transform((0, 0))
                    x1p, y1p = inv.transform((dx_pts, dy_pts))
                    return (x1p - x0p), (y1p - y0p)
                dx_data, _ = _offset_in_data(ax, 8, 0)
                text_artist.set_position((xt + dx_data, yt))

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cumulative Movement (mm)")
    ax.legend(handles, labels, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Cumulative movement plot (HTML + PNG) with optional piecewise regression")
    ap.add_argument("--csv", required=True, help="Path to cumulative-movement CSV")
    ap.add_argument("--outdir", default=".", help="Output directory for plots")

    # Multiple PIDs supported
    ap.add_argument("--pid", nargs="+",
                    help="One or more PIDs to plot (space-separated). If used with --class, only PIDs in that class are plotted.")
    ap.add_argument("--class", dest="klass", choices=["Learner", "Non-Learner"],
                    help="If set, plot only the average over this class (unless --pid is given, then only those PIDs in the class).")

    ap.add_argument("--keypoints", nargs="*", help="Mixed tokens e.g. L0 R4 2 (ids apply to BOTH hands)")
    ap.add_argument("--segments", type=int, default=6, help="Number of piecewise segments (default: 6)")
    ap.add_argument("--prefix", default="cumulative", help="Filename prefix for outputs")

    # Toggles
    ap.add_argument("--no-regression", dest="show_regression", action="store_false",
                    help="Disable piecewise regression overlays")
    ap.add_argument("--shade-band", dest="shade_band", action="store_true",
                    help="Show min–max shaded band across plotted series (default)")
    ap.add_argument("--no-shade-band", dest="shade_band", action="store_false",
                    help="Hide min–max shaded band")
    ap.add_argument("--truncate-at-max", dest="truncate_at_max", action="store_true",
                    help="Truncate each series at its own max cumulative value (default)")
    ap.add_argument("--no-truncate-at-max", dest="truncate_at_max", action="store_false",
                    help="Do not truncate; plot entire available span")
    ap.set_defaults(show_regression=True, shade_band=True, truncate_at_max=True)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if 'pid' not in df.columns or 'class' not in df.columns:
        raise ValueError("CSV must include 'pid' and 'class' columns.")

    # Discover structure
    all_kps, all_chunks, mapping = parse_kp_columns(df)

    # Resolve keypoint selection
    sel_map = parse_keypoint_tokens(args.keypoints, all_kps, mapping)

    # Filter rows based on pid/class
    filt = pd.Series(True, index=df.index)
    title_parts = []

    pid_list = [str(p) for p in args.pid] if args.pid else None
    if pid_list:
        filt &= df['pid'].astype(str).isin(pid_list)
        title_parts.append(f"PID={','.join(pid_list)}")
    if args.klass:
        filt &= (df['class'] == args.klass)
        title_parts.append(f"Class={args.klass}")

    filtered = df.loc[filt].copy()
    if filtered.empty:
        raise ValueError("No rows left after applying --pid/--class filters.")

    # Base x grid in seconds (common across chunks)
    base_x = np.array(all_chunks, dtype=float) * 10.0

    # Build series (truncate per-series if requested)
    series_dict, segs_dict = {}, {}
    y_on_base_for_band = {}  # label -> full-length array (NaN beyond series length)

    def _truncate_to_valid_and_max(x_arr, y_arr):
        valid = np.isfinite(y_arr)
        if not valid.any():
            return np.array([]), np.array([])
        i0 = np.argmax(valid)  # first True
        i1 = len(y_arr) - 1 - np.argmax(valid[::-1])  # last True
        x2 = x_arr[i0:i1+1]
        y2 = y_arr[i0:i1+1]
        if args.truncate_at_max and len(y2) > 0 and np.isfinite(y2).any():
            imax = int(np.nanargmax(y2))
            x2 = x2[:imax+1]
            y2 = y2[:imax+1]
        return x2, y2

    def _add_series(label, subdf):
        y = build_series_for_rows(subdf, mapping, sel_map).values
        x2, y2 = _truncate_to_valid_and_max(base_x, y)
        if len(x2) < 2:
            return
        series_dict[label] = (x2, y2)
        segs_dict[label] = fit_segments(x2, y2, args.segments)
        # stash on base grid for band
        buf = np.full_like(base_x, np.nan, dtype=float)
        buf[:len(y2)] = y2  # because x2 is always a prefix of base_x
        y_on_base_for_band[label] = buf

    if pid_list and not args.klass:
        for pid in pid_list:
            sub = filtered[filtered['pid'].astype(str) == pid]
            if sub.empty:
                continue
            _add_series(pid, sub)

    elif args.klass and pid_list:
        for pid in pid_list:
            sub = filtered[filtered['pid'].astype(str) == pid]
            if sub.empty:
                continue
            _add_series(pid, sub)

    elif args.klass and not pid_list:
        label = args.klass
        _add_series(label, filtered)

    else:
        for klass in ["Learner", "Non-Learner"]:
            sub = df[df['class'] == klass]
            if sub.empty:
                continue
            _add_series(klass, sub)

    if not series_dict:
        raise ValueError("No series to plot after truncation/filters. Check inputs.")

    # Min–Max band across all plotted series (only where ≥2 have data)
    band = None
    if args.shade_band and len(series_dict) >= 2:
        Y = np.column_stack([y_on_base_for_band[lbl] for lbl in series_dict.keys()])
        count = np.sum(np.isfinite(Y), axis=1)
        y_min = np.nanmin(Y, axis=1)
        y_max = np.nanmax(Y, axis=1)
        y_min[count < 2] = np.nan
        y_max[count < 2] = np.nan
        band = (base_x, y_min, y_max)

    # Filenames
    def kp_tag_from_sel(sel):
        allL = set(mapping['L'].keys()); allR = set(mapping['R'].keys())
        if sel['L'] == allL and sel['R'] == allR:
            return "ALL"
        Ltag = ("L" + "-".join(map(str, sorted(sel['L'])))) if sel['L'] else "Lnone"
        Rtag = ("R" + "-".join(map(str, sorted(sel['R'])))) if sel['R'] else "Rnone"
        return f"{Ltag}_{Rtag}"

    ctx_tag = "__".join(title_parts) if title_parts else "ClassAverages"
    base = f"{args.prefix}__{ctx_tag}__{kp_tag_from_sel(sel_map)}__seg{args.segments}"
    out_html = outdir / f"{base}.html"
    out_png  = outdir / f"{base}.png"

    # Title
    pretty_sel = []
    if sel_map['L']: pretty_sel.append("L{" + ",".join(map(str, sorted(sel_map['L']))) + "}")
    if sel_map['R']: pretty_sel.append("R{" + ",".join(map(str, sorted(sel_map['R']))) + "}")
    sel_txt = " | ".join(pretty_sel) if pretty_sel else "ALL"
    title = "Cumulative Movement"
    if title_parts: title += " | " + " & ".join(title_parts)
    title += f" | keypoints: {sel_txt} | segments={args.segments}"

    # Save plots
    plot_plotly_html(series_dict, segs_dict, out_html, title, args.show_regression, band)
    plot_matplotlib_png(series_dict, segs_dict, out_png, title, args.show_regression, band)

    print(f"[OK] Saved interactive HTML → {out_html}")
    print(f"[OK] Saved PNG            → {out_png}")
    if not HAS_RUPTURES:
        print("[NOTE] 'ruptures' not found; used equal-sized segments fallback.")
        print("       Install for optimal changepoints: pip install ruptures")


if __name__ == "__main__":
    main()
