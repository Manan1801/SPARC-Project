#!/usr/bin/env python3

import pyrealsense2 as rs
import os
import time
from datetime import datetime
import threading
import argparse
import json

CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

def load_serial_map():
    serial_map = {}
    with open(CAMERA_SERIALS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                label, serial = line.strip().split(":")
                serial_map[serial.strip()] = label.strip()
    return serial_map

def write_camera_info(profile, serial, label, output_path_txt, output_path_json):
    dev = profile.get_device()
    sensors = dev.query_sensors()

    with open(output_path_txt, "w") as f:
        f.write(f"Camera Label: {label}\n")
        f.write(f"Serial Number: {serial}\n")
        f.write(f"Firmware Version: {dev.get_info(rs.camera_info.firmware_version)}\n")
        f.write(f"USB Port ID: {dev.get_info(rs.camera_info.physical_port)}\n")
        f.write(f"Product Line: {dev.get_info(rs.camera_info.product_line)}\n\n")

        for sensor in sensors:
            f.write(f"[Sensor: {sensor.get_info(rs.camera_info.name)}]\n")
            for opt in sensor.get_supported_options():
                try:
                    val = sensor.get_option(opt)
                    f.write(f"  {opt.name}: {val}\n")
                except:
                    continue
            f.write("\n")

        f.write("[Active Streams]\n")

    intrinsics_data = {}
    for s in profile.get_streams():
        stream_type = s.stream_type().name
        fmt = s.format().name
        stream_key = f"{stream_type.lower()}_{fmt.lower()}"

        intr = s.as_video_stream_profile().get_intrinsics()

        # Add to text
        with open(output_path_txt, "a") as f:
            f.write(f"Stream: {stream_type} ({fmt})\n")
            f.write(f"  Resolution: {s.width()}x{s.height()} @ {s.fps()} FPS\n")
            f.write(f"  Intrinsics: fx={intr.fx}, fy={intr.fy}, cx={intr.ppx}, cy={intr.ppy}\n")
            f.write(f"  Distortion Model: {intr.model.name}\n")
            f.write(f"  Distortion Coeffs: {intr.coeffs}\n\n")

        # Add to JSON
        intrinsics_data[stream_key] = {
            "width": s.width(),
            "height": s.height(),
            "fps": s.fps(),
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "model": intr.model.name,
            "coeffs": intr.coeffs
        }

    with open(output_path_json, "w") as jf:
        json.dump(intrinsics_data, jf, indent=4)

def record_camera(serial, label, duration_min, base_dir):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)

    cam_dir = os.path.join(base_dir, label)
    os.makedirs(cam_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = os.path.join(cam_dir, f"{label}_{timestamp}.bag")
    config.enable_record_to_file(bag_name)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        print(f"[INFO] Starting recording for {label} ({serial})")
        pipeline_profile = pipeline.start(config)

        # Save intrinsics
        info_txt = os.path.join(cam_dir, f"camera_info_{serial}.txt")
        info_json = os.path.join(cam_dir, f"camera_intrinsics_{serial}.json")
        write_camera_info(pipeline_profile, serial, label, info_txt, info_json)

        time.sleep(duration_min * 60)

    except Exception as e:
        print(f"[ERROR] {label} → {e}")
    finally:
        pipeline.stop()
        print(f"[INFO] Finished recording for {label}")

def main():
    parser = argparse.ArgumentParser(description="Record RealSense RGB+D to .bag with intrinsics")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in minutes")
    args = parser.parse_args()

    serial_map = load_serial_map()

    ctx = rs.context()
    connected_serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    print(f"[INFO] Connected RealSense serials: {connected_serials}")

    active = [(s, serial_map[s]) for s in connected_serials if s in serial_map]

    if not active:
        print("[ERROR] No known RealSense cameras found.")
        return

    threads = []
    for serial, label in active:
        t = threading.Thread(target=record_camera, args=(serial, label, args.duration, args.output_dir))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[✅] All recordings completed. Saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
