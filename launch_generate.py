#!/usr/bin/env python3

import pyrealsense2 as rs
from pathlib import Path
import argparse

KNOWN_SERIALS_FILE = "camera_serials.txt"
OUTPUT_LAUNCH_FILE = "multi_camera.launch"

def load_known_serials():
    known = {}
    with open(KNOWN_SERIALS_FILE, "r") as f:
        for line in f:
            cam, serial = line.strip().split(":")
            known[cam.strip()] = serial.strip()
    return known

def get_connected_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    return [dev.get_info(rs.camera_info.serial_number) for dev in devices]

def build_launch_block(cam_ns, serial):
    return f"""
  <group ns="{cam_ns}">
    <node name="realsense2_camera" pkg="realsense2_camera" type="realsense2_camera_node" output="screen">
      <param name="serial_no" value="{serial}" />
      <param name="enable_sync" value="true" />
      <param name="diagnostics_period" value="1.0" />
      <param name="enable_color" value="true" />
      <param name="enable_depth" value="true" />
      <param name="color_width" value="640" />
      <param name="color_height" value="480" />
      <param name="color_fps" value="30" />
      <param name="depth_width" value="640" />
      <param name="depth_height" value="480" />
      <param name="depth_fps" value="30" />
    </node>
  </group>
"""

def main(num_cams):
    known_serials = load_known_serials()
    connected_serials = get_connected_serials()
    print(f"[INFO] Connected RealSense serials: {connected_serials}")

    available = [(cam, s) for cam, s in known_serials.items() if s in connected_serials]

    if not available:
        print("[ERROR] None of the known cameras are currently connected.")
        return

    print(f"[INFO] {len(available)} known cameras detected.")
    for i, (cam, s) in enumerate(available):
        print(f"  {i+1}. {cam} → {s}")

    if num_cams > len(available):
        print(f"[WARN] Requested {num_cams} cameras but only {len(available)} are connected. Using all available.")
        num_cams = len(available)

    selected = available[:num_cams]

    launch_str = "<launch>\n"
    for cam, serial in selected:
        launch_str += build_launch_block(cam, serial)
    launch_str += "\n</launch>"

    with open(OUTPUT_LAUNCH_FILE, "w") as f:
        f.write(launch_str)

    print(f"[SUCCESS] Created launch file with {num_cams} cameras → {OUTPUT_LAUNCH_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi_camera.launch based on connected RealSense cameras.")
    parser.add_argument("--num-cameras", type=int, default=3, help="Number of cameras to use (max 4)")
    args = parser.parse_args()
    main(args.num_cameras)
