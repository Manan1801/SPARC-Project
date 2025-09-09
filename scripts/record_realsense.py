#!/usr/bin/env python3

import pyrealsense2 as rs
import os
import time
from datetime import datetime
import threading
import sys
import json
import signal

CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

# ── Cooperative control ────────────────────────────────────────────────────────
pause_event = threading.Event()   # set() => paused (logical; bag file still records)
stop_event  = threading.Event()   # set() => stop ASAP

def _sig_pause(signum, frame):
    if not pause_event.is_set():
        print("\n[⏸] Received SIGUSR1 → PAUSE (bag keeps writing; countdown paused).", flush=True)
    pause_event.set()

def _sig_resume(signum, frame):
    if pause_event.is_set():
        print("\n[▶] Received SIGUSR2 → RESUME.", flush=True)
    pause_event.clear()

def _sig_stop(signum, frame):
    print("\n[⛔] Received SIGINT → Graceful shutdown requested.", flush=True)
    stop_event.set()

signal.signal(signal.SIGUSR1, _sig_pause)
signal.signal(signal.SIGUSR2, _sig_resume)
signal.signal(signal.SIGINT,  _sig_stop)

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        try:
            video_profile = s.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            stream_type = s.stream_type().name
            fmt = s.format().name
            stream_key = f"{stream_type.lower()}_{fmt.lower()}"

            with open(output_path_txt, "a") as f:
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
            print(f"[WARN] Failed to extract stream info: {e}")

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

    # Streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        print(f"[INFO] Starting recording for {label} ({serial}) → {os.path.basename(bag_name)}")
        pipeline_profile = pipeline.start(config)

        info_txt  = os.path.join(cam_dir, f"camera_info_{serial}.txt")
        info_json = os.path.join(cam_dir, f"camera_intrinsics_{serial}.json")
        write_camera_info(pipeline_profile, serial, label, info_txt, info_json)

        # ACTIVE duration countdown (bag keeps writing during cooperative pause)
        target_active_sec = int(duration_min * 60)
        active_elapsed = 0
        last_tick = time.time()

        while not stop_event.is_set() and active_elapsed < target_active_sec:
            if pause_event.is_set():
                last_tick = time.time()
                time.sleep(0.05)
                continue
            # Optional: fetch frames to keep device warm (not required for recorder)
            time.sleep(0.05)
            now = time.time()
            active_elapsed += (now - last_tick)
            last_tick = now

    except Exception as e:
        print(f"[ERROR] {label} → {e}")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        print(f"[INFO] Finished recording for {label}")

def main(base_dir, duration_min):
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
        t = threading.Thread(target=record_camera, args=(serial, label, duration_min, base_dir), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[✅] All recordings completed. Saved in: {base_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record RealSense .bag files (cooperative pause/resume via SIGUSR1/SIGUSR2; bag file continues during cooperative pause).")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--duration", type=float, required=True, help="ACTIVE duration in minutes")
    args = parser.parse_args()
    main(args.output_dir, args.duration)
