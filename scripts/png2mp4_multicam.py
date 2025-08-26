#!/usr/bin/env python3
"""
png2mp4_multicam.py  
Batch-convert RealSense “color” frames (frame_XXXXX.png) to .mp4 videos.

Folder layout expected
----------------------
session_root/
├─ audio/              (ignored)
├─ cam1/
│  ├─ color/  <------ processed
│  └─ depth/  (ignored)
├─ cam2/
│  ├─ color/  <------ processed
│  └─ depth/
└─ cam3/
   ├─ color/  <------ processed
   └─ depth/


Output
------
For each camX/color/ folder a video named `<cam_name>.mp4` is written right there:

session_root/cam1/cam1.mp4
session_root/cam2/cam2.mp4
session_root/cam3/cam3.mp4


CLI
---
python png2mp4_multicam.py  /path/to/session_root  [--fps 30] [--codec avc1] \
                            [--cams cam1 cam3] [--overwrite]

Usage
------
# One-liner to build cam1.mp4, cam2.mp4, cam3.mp4 (30 fps, H.264)
python png2mp4_multicam.py  ~/Desktop/realsense_recording_20250522_160432

# Re-encode only cam2 at 60 fps MJPEG, overwriting if it exists
python png2mp4_multicam.py  ~/Desktop/realsense_recording_20250522_160432 \
        --cams cam2 --fps 60 --codec MJPG --overwrite

"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import List
import cv2
from tqdm import tqdm


def natural_key(text: str):
    """Sort helper so frame_10.png > frame_2.png."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', text)]


def frames_to_mp4(frames: List[Path], out_file: Path,
                  fps: int, codec: str):
    if not frames:
        raise RuntimeError("No PNG frames found")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot open {frames[0]}")
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(str(out_file),
                             cv2.VideoWriter_fourcc(*codec),
                             fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {out_file}")

    for f in tqdm(frames, desc=f"→ {out_file.name}", unit="frame"):
        img = cv2.imread(str(f))
        if img is None:
            raise RuntimeError(f"Cannot open {f}")
        if img.shape[0:2] != (h, w):
            raise ValueError(f"{f} has {img.shape[1]}×{img.shape[0]}; expected {w}×{h}")
        writer.write(img)
    writer.release()


def process_camera(cam_dir: Path, fps: int, codec: str, overwrite: bool):
    color_dir = cam_dir / "color"
    if not color_dir.is_dir():
        print(f"⚠️  {cam_dir}: no color/ folder → skipped")
        return

    # Find the parent folder that starts with "AP_"
    try:
        ap_parent = next(p for p in cam_dir.parents if p.name.startswith("AP_"))
    except StopIteration:
        ap_parent = None

    if ap_parent:
        out_file = cam_dir / f"{ap_parent.name}_{cam_dir.name}.mp4"
    else:
        out_file = cam_dir / f"{cam_dir.name}.mp4"

    if out_file.exists() and not overwrite:
        print(f"ℹ️  {out_file} exists – use --overwrite to rebuild")
        return

    frames = sorted(color_dir.glob("frame_*.png"),
                    key=lambda p: natural_key(p.name))
    if not frames:
        print(f"⚠️  {color_dir}: no frame_*.png files → skipped")
        return

    print(f"🎬  Converting {len(frames)} frames from {color_dir} …")
    frames_to_mp4(frames, out_file, fps, codec)
    print(f"✅  {out_file} written ({len(frames)} frames, {fps} fps)")


def main():
    ap = argparse.ArgumentParser(
        description="Convert RealSense color PNG frames to MP4 for every camX.")
    ap.add_argument("session_root", type=Path,
                    help="Path to the recording root")
    ap.add_argument("--fps", type=int, default=30,
                    help="Frames-per-second of the output video (default 30)")
    ap.add_argument("--codec", default="mp4v",
                    help="FOURCC codec (default mp4v = MPEG-4 Part 2)")
    ap.add_argument("--cams", nargs="+", metavar="CAM",
                    help="Subset of cameras to process (e.g. cam1 cam3)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing MP4s")
    args = ap.parse_args()

    root = args.session_root.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory")

    cam_dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("cam")])

    if args.cams:
        missing = set(args.cams) - {d.name for d in cam_dirs}
        if missing:
            sys.exit(f"ERROR: cameras not found: {', '.join(missing)}")
        cam_dirs = [d for d in cam_dirs if d.name in args.cams]

    if not cam_dirs:
        sys.exit("ERROR: no cam* directories found")

    for cam in cam_dirs:
        process_camera(cam, args.fps, args.codec, args.overwrite)


if __name__ == "__main__":
    main()
