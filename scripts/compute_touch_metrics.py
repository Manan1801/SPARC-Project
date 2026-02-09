import csv
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


# --------------------------------------------------
# Parse untouched intervals
# --------------------------------------------------
def parse_intervals(
    log_text: str,
    objects: List[str]
) -> Dict[str, List[Tuple[int, int]]]:

    data = {obj: [] for obj in objects}

    for obj in objects:
        pattern = rf"{obj}:\s*(.*)"
        match = re.search(pattern, log_text, flags=re.IGNORECASE)

        if match:
            intervals = re.findall(r"\[(\d+),\s*(\d+)\]", match.group(1))
            data[obj] = [(int(s), int(e)) for s, e in intervals]

    return data


# --------------------------------------------------
# Compute metrics (clip first, then compute)
# --------------------------------------------------
def compute_metrics(
    log_file: str,
    objects: List[str],
    fps: int,
    start_sec: float | None,
    end_sec: float | None
) -> Dict[str, Dict[str, float]]:

    log_text = Path(log_file).read_text()
    intervals = parse_intervals(log_text, objects)

    start_frame = int(start_sec * fps) if start_sec is not None else 0
    end_frame = int(end_sec * fps) if end_sec is not None else None

    results = {}

    for obj, ivals in intervals.items():
        clipped = []

        for s, e in ivals:
            cs = max(s, start_frame)
            ce = e if end_frame is None else min(e, end_frame)

            if ce > cs:
                clipped.append((cs, ce))

        if not clipped:
            results[obj] = {
                "first_touch_sec": None,
                "untouched_sec": 0.0
            }
            continue

        first_touch_sec = clipped[0][1] / fps
        untouched_frames = sum(e - s for s, e in clipped)

        results[obj] = {
            "first_touch_sec": round(first_touch_sec, 3),
            "untouched_sec": round(untouched_frames / fps, 3)
        }

    return results


# --------------------------------------------------
# Extract participant ID from path
# ../<participant>/cam2/...
# --------------------------------------------------
def extract_participant_id(log_file: str) -> str:
    """
    Extract participant ID from path.
    Expected layout:
      .../<numbered_folder>/cam2/objects_untouched/untouched_intervals_*.txt

    We return <numbered_folder>.
    """
    path = Path(log_file).resolve()
    parts = path.parts

    # Find ".../<participant>/cam2/..." and return the folder right before "cam2"
    try:
        cam2_idx = next(i for i, p in enumerate(parts) if p.lower() == "cam2")
        if cam2_idx == 0:
            raise ValueError
        return parts[cam2_idx - 1]
    except (StopIteration, ValueError):
        raise ValueError(f"Cannot extract participant ID from path: {log_file}")


# --------------------------------------------------
# Append SINGLE ROW per participant
# --------------------------------------------------
def append_participant_to_csv(
    log_file: str,
    objects: List[str],
    output_csv: str,
    fps: int,
    start_sec: float | None,
    end_sec: float | None
):

    participant_id = extract_participant_id(log_file)

    metrics = compute_metrics(
        log_file=log_file,
        objects=objects,
        fps=fps,
        start_sec=start_sec,
        end_sec=end_sec
    )

    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ["participant_id"]
            for obj in objects:
                header.extend([
                    f"{obj}_first_touch_sec",
                    f"{obj}_untouched_sec"
                ])
            writer.writerow(header)

        row = [participant_id]
        for obj in objects:
            row.append(metrics[obj]["first_touch_sec"])
            row.append(metrics[obj]["untouched_sec"])

        writer.writerow(row)

    return file_exists, participant_id


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute object touch metrics from untouched-interval logs"
    )

    parser.add_argument("--log", required=True,
                        help="Path to untouched_intervals log file")
    parser.add_argument("--objects", required=True, nargs="+",
                        help="Object names")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--start", type=float, default=None,
                        help="Start time in seconds (optional)")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds (optional)")
    parser.add_argument("--output", default="touch_metrics.csv",
                        help="Output CSV file")

    args = parser.parse_args()

    file_existed, participant_id = append_participant_to_csv(
        log_file=args.log,
        objects=args.objects,
        output_csv=args.output,
        fps=args.fps,
        start_sec=args.start,
        end_sec=args.end
    )

    if file_existed:
        print(f"✔ Participant {participant_id}: results appended to CSV")
    else:
        print(f"✔ Participant {participant_id}: CSV created and results written")


if __name__ == "__main__":
    main()
