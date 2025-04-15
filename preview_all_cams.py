#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import time
import threading
import numpy as np
import psutil
from datetime import datetime

CAMERA_SERIALS_FILE = "camera_serials.txt"
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_camera_serials():
    serial_map = {}
    with open(CAMERA_SERIALS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                label, serial = line.strip().split(":")
                serial_map[serial.strip()] = label.strip()
    return serial_map

def system_status():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_io_counters()
    return f"CPU: {cpu:.1f}%  RAM: {ram:.1f}%", disk.write_bytes

class CameraStream:
    def __init__(self, serial, label):
        self.serial = serial
        self.label = label
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.frames = []
        self.last_ts = None
        self.fps = 0.0
        self.dropped = 0
        self.lock = threading.Lock()

    def start(self):
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()

    def update(self):
        while True:
            try:
                frameset = self.pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    continue
                ts = color_frame.get_timestamp()
                frame = np.asanyarray(color_frame.get_data())
                now = time.time()

                with self.lock:
                    self.frames.append(frame)
                    if len(self.frames) > 5:
                        self.frames = self.frames[-5:]
                    if self.last_ts:
                        time_diff = ts - self.last_ts
                        expected_diff = 1000.0 / 30.0
                        if time_diff > expected_diff * 1.5:
                            self.dropped += 1
                    self.last_ts = ts
                    self.fps = 1000.0 / time_diff if time_diff > 0 else 0
            except Exception as e:
                print(f"[ERROR] {self.label}: {e}")
                break

    def get_latest_frame(self):
        with self.lock:
            if self.frames:
                return self.frames[-1].copy()
            return np.zeros((480, 640, 3), dtype=np.uint8)

def draw_overlay(img, label, fps, dropped, sys_text=None):
    overlay = img.copy()
    cv2.putText(overlay, f"{label}", (10, 25), FONT, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 50), FONT, 0.7, (255, 255, 255), 1)
    cv2.putText(overlay, f"Dropped: {dropped}", (10, 75), FONT, 0.7, (0, 0, 255), 1)
    if sys_text:
        cv2.putText(overlay, sys_text, (10, 100), FONT, 0.6, (255, 200, 0), 1)
    return overlay

def create_preview_grid(cams, cols=2):
    frames = [cam.get_latest_frame() for cam in cams]
    overlays = []
    sys_text, _ = system_status()
    for cam, f in zip(cams, frames):
        overlays.append(draw_overlay(f, cam.label, cam.fps, cam.dropped, sys_text))

    while len(overlays) % cols != 0:
        overlays.append(np.zeros_like(overlays[0]))

    rows = len(overlays) // cols
    grid_rows = [np.hstack(overlays[i*cols:(i+1)*cols]) for i in range(rows)]
    grid = np.vstack(grid_rows)
    return grid

def main():
    serial_map = load_camera_serials()
    ctx = rs.context()
    connected_serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]

    print(f"[INFO] Connected RealSense devices: {connected_serials}")

    cameras = []
    threads = []

    for serial in connected_serials:
        if serial in serial_map:
            cam = CameraStream(serial, serial_map[serial])
            cam.start()
            t = threading.Thread(target=cam.update, daemon=True)
            t.start()
            cameras.append(cam)
            threads.append(t)

    if not cameras:
        print("[ERROR] No known cameras connected.")
        return

    try:
        while True:
            grid = create_preview_grid(cameras, cols=2)
            cv2.imshow("RealSense Grid View", grid)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
