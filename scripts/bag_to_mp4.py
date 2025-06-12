#!/usr/bin/env python3
"""
bag_to_mp4.py
---------------------
Convert an Intel RealSense *.bag* recording into a single MP4 video
containing ONLY the RGB stream.
Usage
-----
python bag_to_mp4.py --bag /path/to/file.bag [--out-dir ./outputs]
Dependencies
------------
pip install pyrealsense2 opencv-python numpy
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
def export_color_video(bag_path: Path, out_dir: Path) -> None:
    """Read *.bag* and write *_color.mp4* into *out_dir*."""
    if not bag_path.is_file():
        sys.exit(f"[ERROR] Bag file not found: {bag_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # -------- RealSense pipeline ---------------------------------------------
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(str(bag_path), repeat_playback=False)
    cfg.enable_stream(rs.stream.color)                     # <- only color
    print(f"[INFO] Opening {bag_path.name}")
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as e:
        sys.exit(f"[ERROR] Cannot open {bag_path.name}: {e}")
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)                          # fastest offline read
    color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    fps, width, height = color_prof.fps(), color_prof.width(), color_prof.height()
    print(f"[INFO] {width}×{height} @ {fps} FPS detected (color)")
    # -------- VideoWriter -----------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_path = out_dir / f"{bag_path.stem}_color.mp4"
    vw_color = cv2.VideoWriter(str(color_path), fourcc, fps, (width, height))
    if not vw_color.isOpened():
        sys.exit("[ERROR] Could not open output video file for writing.")
    # -------- Frame loop ------------------------------------------------------
    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frm = frames.get_color_frame()
            if not color_frm:
                continue
            color_img = np.asanyarray(color_frm.get_data())   # BGR8
            vw_color.write(color_img)
            frame_idx += 1
            if frame_idx % 300 == 0:                          # ~10 s at 30 FPS
                print(f"\r[INFO] Processed {frame_idx} frames …", end="", flush=True)
    except RuntimeError:
        # End of file
        print(f"\n[INFO] Finished – {frame_idx} frames exported.")
    finally:
        vw_color.release()
        pipeline.stop()
        print(f"[✓] RGB video ➜ {color_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RealSense .bag to RGB MP4")
    parser.add_argument("--bag", required=True, type=Path, help="Input .bag file")
    parser.add_argument("--out-dir", default=Path.cwd(), type=Path,
                        help="Directory for MP4 output")
    args = parser.parse_args()
    export_color_video(args.bag, args.out_dir.resolve())