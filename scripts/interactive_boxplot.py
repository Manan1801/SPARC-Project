#!/usr/bin/env python3
"""
interactive_boxplot.py
----------------------
Create an attractive interactive box-and-whisker plot from a CSV file.

Usage
-----
python interactive_boxplot.py path/to/data.csv [--cols col1 col2 ...]

If --cols is omitted, every numeric column is plotted.

Enhancements
------------
* Uses Plotly's "plotly_white" template with nicer fonts & spacing.
* Displays only the outlier points (no full scatter cloud).
* Draws dashed red line for medians and solid blue line for means,
  both connecting across all boxes.
* Hover shows exact statistics for each box plus mean/median traces.
* Cleans non-numeric junk (e.g. '#DIV/0!') and reports what was coerced.

Dependencies
------------
pip install pandas plotly
"""

import argparse
from pathlib import Path
import numpy as np

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as po


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an interactive box-and-whisker plot from a CSV."
    )
    p.add_argument("csv", type=Path, help="Path to the CSV file")
    p.add_argument(
        "--cols",
        nargs="+",
        metavar="COLUMN",
        help="Specific column names to plot (default: all numeric columns)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---------- Load ----------
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        raise SystemExit(f"❌ Failed to read CSV: {e}")

    # ---------- Column selection ----------
    if args.cols:
        missing = [c for c in args.cols if c not in df.columns]
        if missing:
            raise SystemExit(f"❌ Column(s) not found: {', '.join(missing)}")
        numeric_cols = args.cols
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise SystemExit("❌ No numeric columns found to plot.")

    # ---------- Clean & coerce ----------
    # Replace common junk tokens with NaN, then force numeric
    junk_tokens = ["#DIV/0!", "inf", "Infinity", "-inf", ""]
    df.replace(junk_tokens, np.nan, inplace=True)

    clean_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Debug: show which values were coerced
    bad_mask = clean_df.isna() & df[numeric_cols].notna()
    if bad_mask.any().any():
        print("⚠️ Non-numeric entries found and coerced to NaN:")
        for c in numeric_cols:
            bad_vals = df.loc[bad_mask[c], c].unique()
            if len(bad_vals):
                preview = ", ".join(map(str, bad_vals[:8]))
                more = " ..." if len(bad_vals) > 8 else ""
                count = bad_mask[c].sum()
                print(f"  - {c}: {count} value(s) → {preview}{more}")

    # Drop columns that became entirely NaN
    all_nan_cols = [c for c in clean_df.columns if clean_df[c].isna().all()]
    if all_nan_cols:
        print(f"⚠️ Dropping columns with all NaN after coercion: {', '.join(all_nan_cols)}")
        clean_df = clean_df.drop(columns=all_nan_cols)

    if clean_df.empty:
        raise SystemExit("❌ After cleaning, no valid numeric data remained to plot.")

    # Recompute column order after drops
    final_cols = clean_df.columns.tolist()

    # ---------- Prepare data for plotting ----------
    long_df = clean_df.melt(var_name="Variable", value_name="Value")

    # Compute per-column stats for lines
    stats = clean_df.agg(["median", "mean"]).T.reset_index()
    stats.columns = ["Variable", "Median", "Mean"]

    # ---------- Build figure ----------
    fig = px.box(
        long_df,
        x="Variable",
        y="Value",
        color="Variable",  # one colour per box
        points="outliers",
        template="plotly_white",
        title=f"Interactive Box-and-Whisker Plot • {args.csv.name}",
        color_discrete_sequence=px.colors.qualitative.Set2  # optional palette
    )

    # Hide each box’s legend entry (avoid clutter)
    fig.update_traces(showlegend=False, selector=dict(type="box"))

    # Add median and mean polylines
    fig.add_trace(
        go.Scatter(
            x=stats["Variable"],
            y=stats["Median"],
            mode="lines+markers",
            name="Median",
            line=dict(color="firebrick", width=2, dash="dash"),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="Median %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["Variable"],
            y=stats["Mean"],
            mode="lines+markers",
            name="Mean",
            line=dict(color="royalblue", width=2),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="Mean %{y}<extra></extra>",
            connectgaps=True,  # connect means across boxes
        )
    )

    # ---------- Layout polish ----------
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Value",
        font=dict(family="Helvetica Neue, Arial, sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(tickangle=-45)

    # ---------- Display ----------
    html_file = args.csv.with_suffix(".boxplot.html")
    po.plot(fig, filename=str(html_file), auto_open=True)
    print(f"✅ Plot saved to {html_file} and opened in your browser.")


if __name__ == "__main__":
    main()
