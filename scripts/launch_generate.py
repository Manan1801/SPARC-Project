#!/usr/bin/env python3

import pyrealsense2 as rs
from pathlib import Path
import argparse
import os

KNOWN_SERIALS_FILE = str(Path("~/Desktop/SPARC-Project/camera_serials.txt").expanduser())
OUTPUT_LAUNCH_FILE = str(Path("~/Desktop/SPARC-Project/launch/multi_camera.launch.py").expanduser())

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
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='{cam_ns}',
            name='realsense2_camera',
            output='screen',
            parameters=[{{
                'serial_no': '{serial}',
                'enable_sync': True,
                'diagnostics_period': 1.0,
                'enable_color': True,
                'enable_depth': True,
                'color_width': 640,
                'color_height': 480,
                'color_fps': 30,
                'depth_width': 640,
                'depth_height': 480,
                'depth_fps': 30
            }}]
        )
    """

def main(num_cams):
    known_serials = load_known_serials()
    connected_serials = get_connected_serials()
    print(f"[INFO] Connected RealSense serials: {connected_serials}")

    used_serials = set()
    available = []
    for cam, serial in known_serials.items():
        if serial in connected_serials:
            if serial not in used_serials:
                available.append((cam, serial))
                used_serials.add(serial)
            else:
                print(f"[WARN] Skipping {cam} → {serial} (duplicate serial)")
        else:
            print(f"[INFO] Skipping {cam} → {serial} (not connected)")

    if not available:
        print("[ERROR] None of the known cameras are currently connected.")
        return

    print(f"[INFO] {len(available)} known cameras detected.")
    for i, (cam, s) in enumerate(available):
        print(f"  {i+1}. {cam} → {s}")

    if num_cams > len(available):
        print(f"[WARN] Requested {num_cams} cameras but only {len(available)} are usable. Using all available.")
        num_cams = len(available)

    selected = available[:num_cams]
    os.makedirs(os.path.dirname(OUTPUT_LAUNCH_FILE), exist_ok=True)

    launch_file_content = """from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
"""
    for cam, serial in selected:
        launch_file_content += build_launch_block(cam, serial) + ",\n"

    launch_file_content += "    ])\n"

    with open(OUTPUT_LAUNCH_FILE, "w") as f:
        f.write(launch_file_content)

    print(f"[SUCCESS] Created launch file with {len(selected)} cameras → {OUTPUT_LAUNCH_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi_camera.launch.py based on connected RealSense cameras.")
    parser.add_argument("--num-cameras", type=int, default=3, help="Number of cameras to use (max 4)")
    args = parser.parse_args()
    main(args.num_cameras)
