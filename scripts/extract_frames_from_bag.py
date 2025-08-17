import os
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from pathlib import Path

def extract_frames(bag_path):
    bag_path = Path(bag_path)
    if not bag_path.exists() or not bag_path.suffix == '.bag':
        raise FileNotFoundError("Provide a valid RealSense .bag file.")

    # Create folders
    out_dir = bag_path.parent
    color_dir = out_dir / "color"
    depth_dir = out_dir / "depth"
    color_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    # Set up pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    profile = pipeline.get_active_profile()

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)  # Faster than real-time

    align = rs.align(rs.stream.color)

    frame_id = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Save frames
            color_filename = color_dir / f"frame_{frame_id:06d}.png"
            depth_filename = depth_dir / f"frame_{frame_id:06d}.tiff"

            cv2.imwrite(str(color_filename), color_image)
            cv2.imwrite(str(depth_filename), depth_image)

            print(f"Saved frame {frame_id}")
            frame_id += 1

    except RuntimeError:
        print("Finished processing bag file.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract color/depth frames from RealSense .bag")
    parser.add_argument("--bag", help="Path to .bag file recorded by RealSense SDK")
    args = parser.parse_args()

    extract_frames(args.bag)
