#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path

def min_max_normalize(input_csv, output_csv):
    # Read CSV
    df = pd.read_csv(input_csv)
    
    # Keep participant_id separately
    participant_col = df["participant_id"]
    
    # Normalize frame separately
    frame = df["frame"]
    frame_norm = (frame - frame.min()) / (frame.max() - frame.min())
    
    # Normalize all other features except participant_id and frame
    feature_cols = df.drop(columns=["participant_id", "frame"])
    normalized_features = (feature_cols - feature_cols.min()) / (feature_cols.max() - feature_cols.min())
    
    # Combine all back
    normalized_df = pd.concat(
        [participant_col, frame, frame_norm.rename("frame_norm"), normalized_features],
        axis=1
    )
    
    # Save result
    normalized_df.to_csv(output_csv, index=False)
    print(f"✅ Normalized CSV saved at: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Min-Max normalize features in a CSV")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=Path, required=True, help="Path to output normalized CSV")
    args = parser.parse_args()

    min_max_normalize(args.input, args.output)
