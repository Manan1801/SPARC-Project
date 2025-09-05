#!/usr/bin/env python3

import pyrealsense2 as rs
import os
import time
import threading
import json
import cv2
import numpy as np
from datetime import datetime

CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

def load_serial_map():
    serial_map = {}
    with open(CAMERA_SERIALS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                label, serial = line.strip().split(":")
                serial_map[serial.strip()] = label.strip()
    return serial_map

def write_camera_info(profile, serial, label, output_txt, output_json):
    device = profile.get_device()
    sensors = device.query_sensors()

    with open(output_txt, "w") as f:
        f.write(f"Camera Label: {label}\n")
        f.write(f"Serial Number: {serial}\n")
        f.write(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}\n")
        f.write(f"USB Port ID: {device.get_info(rs.camera_info.physical_port)}\n")
        f.write(f"Product Line: {device.get_info(rs.camera_info.product_line)}\n\n")

        for sensor in sensors:
            f.write(f"[Sensor: {sensor.get_info(rs.camera_info.name)}]\n")
            for opt in sensor.get_supported_options():
                try:
                    val = sensor.get_option(opt)
                    f.write(f"  {opt.name}: {val}\n")
                except Exception:
                    continue
            f.write("\n")
        f.write("[Active Streams]\n")

    intrinsics_data = {}
    for s in profile.get_streams():
        try:
            video_profile = s.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            stream_type = s.stream_type().name
            fmt = s.format().name
            stream_key = f"{stream_type.lower()}_{fmt.lower()}"

            with open(output_txt, "a") as f:
                f.write(f"Stream: {stream_type} ({fmt})\n")
                f.write(f"  Resolution: {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()} FPS\n")
                f.write(f"  Intrinsics: fx={intr.fx}, fy={intr.fy}, cx={intr.ppx}, cy={intr.ppy}\n")
                f.write(f"  Distortion Model: {intr.model.name}\n")
                f.write(f"  Distortion Coeffs: {intr.coeffs}\n\n")

            intrinsics_data[stream_key] = {
                "width": video_profile.width(),
                "height": video_profile.height(),
                "fps": video_profile.fps(),
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.ppx,
                "cy": intr.ppy,
                "model": intr.model.name,
                "coeffs": intr.coeffs
            }
        except Exception as e:
            print(f"[WARN] Skipping stream: {e}")

    # Add alignment metadata
    intrinsics_data["alignment"] = {
        "depth_to_color": True,
        "alignment_target": "color",
        "use_intrinsics": "color"
    }

    with open(output_json, "w") as jf:
        json.dump(intrinsics_data, jf, indent=4)

def record_camera(serial, label, duration_min, base_dir):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    cam_dir = os.path.join(base_dir, label)
    color_dir = os.path.join(cam_dir, "color")
    depth_dir = os.path.join(cam_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    try:
        print(f"[INFO] 🎥 Starting recording for {label} ({serial})")
        pipeline_profile = pipeline.start(config)

        align = rs.align(rs.stream.color)
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()

        info_txt = os.path.join(cam_dir, f"camera_info_{serial}.txt")
        info_json = os.path.join(cam_dir, f"camera_intrinsics_{serial}.json")
        write_camera_info(pipeline_profile, serial, label, info_txt, info_json)

        start_time = time.time()
        duration_sec = duration_min * 60
        frame_count = 0

        while (time.time() - start_time) < duration_sec:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Apply filters to depth
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Save color frame
            color_filename = os.path.join(color_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(color_filename, color_image)

            # Save depth frame
            depth_filename = os.path.join(depth_dir, f"frame_{frame_count:06d}.tiff")
            cv2.imwrite(depth_filename, depth_raw, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

            frame_count += 1

    except Exception as e:
        print(f"[ERROR] {label} → {e}")
    finally:
        pipeline.stop()
        print(f"[INFO] ✅ Finished recording for {label}")

def main(base_dir, duration_min):
    serial_map = load_serial_map()
    ctx = rs.context()
    connected_serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    print(f"[INFO] 📸 Connected RealSense serials: {connected_serials}")

    active = [(s, serial_map[s]) for s in connected_serials if s in serial_map]

    if not active:
        print("[ERROR] No known RealSense cameras found.")
        return

    threads = []
    for serial, label in active:
        t = threading.Thread(target=record_camera, args=(serial, label, duration_min, base_dir))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[🏁] All camera recordings completed. Saved in: {base_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record RealSense RGB & Depth frames for all connected cameras.")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--duration", type=float, required=True, help="Duration in minutes")
    args = parser.parse_args()
    main(args.output_dir, args.duration)
