#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import threading
import numpy as np
import sys
import os
import time

FONT = cv2.FONT_HERSHEY_SIMPLEX

class BagStream:
    def __init__(self, cam_id, bag_path):
        self.cam_id = cam_id
        self.bag_path = bag_path
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(bag_path, repeat_playback=False)
        self.config.disable_all_streams()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)

        self.color_frame = None
        self.depth_frame = None
        self.fps = 0.0
        self.dropped = 0
        self.last_ts = None

        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.show_depth = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.pipeline.start(self.config)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()

    def update(self):
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            try:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue

                ts = color.get_timestamp()

                with self.lock:
                    self.color_frame = color
                    self.depth_frame = depth

                    if self.last_ts is not None:
                        time_diff = ts - self.last_ts
                        expected = 1000.0 / 30.0
                        if time_diff > expected * 1.5:
                            self.dropped += 1
                        self.fps = 1000.0 / time_diff if time_diff > 0 else 0.0
                    self.last_ts = ts
            except Exception as e:
                print(f"[ERROR] {self.cam_id}: {e}")
                break

    def get_frame(self):
        with self.lock:
            if self.show_depth and self.depth_frame:
                depth_image = np.asanyarray(self.depth_frame.get_data())
                return cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
            elif self.color_frame:
                return np.asanyarray(self.color_frame.get_data())
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)

def draw_overlay(img, label, fps, dropped, stream_type):
    overlay = img.copy()
    cv2.putText(overlay, f"{label}", (10, 25), FONT, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 50), FONT, 0.7, (255, 255, 255), 1)
    cv2.putText(overlay, f"Dropped: {dropped}", (10, 75), FONT, 0.7, (0, 0, 255), 1)
    cv2.putText(overlay, f"Stream: {stream_type}", (10, 100), FONT, 0.6, (200, 200, 200), 1)
    return overlay

def build_grid(streams, cols=2):
    overlays = []
    for cam in streams:
        frame = cam.get_frame()
        overlay = draw_overlay(frame, cam.cam_id, cam.fps, cam.dropped,
                               "DEPTH" if cam.show_depth else "COLOR")
        overlays.append(overlay)

    while len(overlays) % cols != 0:
        overlays.append(np.zeros_like(overlays[0]))

    rows = len(overlays) // cols
    return np.vstack([np.hstack(overlays[i*cols:(i+1)*cols]) for i in range(rows)])

def main(bag_files):
    if not bag_files:
        print("[ERROR] No bag files provided.")
        return

    print(f"[INFO] Launching preview for {len(bag_files)} bag files...")
    streams = []
    for idx, path in enumerate(bag_files):
        cam_id = f"cam{idx+1}"
        cam = BagStream(cam_id, path)
        cam.start()
        streams.append(cam)

    print("[INFO] Press SPACE or 'p' to Pause/Resume All")
    print("[INFO] Press 1, 2, 3... to toggle color/depth stream for each cam")

    paused = False

    try:
        while True:
            grid = build_grid(streams, cols=2)
            cv2.imshow("RealSense Bag Playback Grid", grid)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('p') or key == ord(' '):
                paused = not paused
                for cam in streams:
                    cam.paused = paused
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if 0 <= idx < len(streams):
                    streams[idx].show_depth = not streams[idx].show_depth
    finally:
        for cam in streams:
            cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[USAGE] realsense_preview_grid.py <cam1.bag> <cam2.bag> ...")
        sys.exit(1)
    main(sys.argv[1:])
