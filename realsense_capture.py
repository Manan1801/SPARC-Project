#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import numpy as np
import sys
import time

class RealSenseCapture:
    def __init__(self, bag_file=None, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.bag_file = bag_file

        # Internal flag to track pause state
        self._paused = False

        if bag_file:
            print(f"Reading from file: {bag_file}")
            # We'll load the .bag file without repeating
            self.config.enable_device_from_file(bag_file, repeat_playback=False)
        else:
            # Live camera usage
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")

        # Setup playback controls if using a .bag
        self.playback = None
        if self.bag_file:
            dev = self.profile.get_device()
            try:
                self.playback = dev.as_playback()
                print("Playback device ready for pause/resume control.")
                # --- Enable real-time playback so frames come at recorded speed ---
                self.playback.set_real_time(True)
            except Exception as e:
                print("Warning: could not set up playback device:", e)

        # Keep track of the last valid frames, so we can keep displaying them if paused
        self.last_color_image = None
        self.last_depth_image = None
        self.last_depth_colormap = None

    def pause_playback(self):
        """Pauses .bag playback if available, and marks our internal flag."""
        if self.playback:
            self.playback.pause()
        self._paused = True
        print("Playback paused.")

    def resume_playback(self):
        """Resumes .bag playback if available, and clears our internal flag."""
        if self.playback:
            self.playback.resume()
        self._paused = False
        print("Playback resumed.")

    def is_paused(self):
        """Returns our local paused state."""
        return self._paused

    def get_frames(self):
        """
        Yields (color_image, depth_image, depth_colormap) in real-time:
          - If paused, yields the last frames so main loop can still run cv2.waitKey.
          - Otherwise, pulls new frames from RealSense at the recorded rate.
        """
        try:
            while True:
                if self.is_paused():
                    # If paused, yield the last frames so main loop can still run cv2.waitKey
                    if self.last_color_image is not None:
                        yield self.last_color_image, self.last_depth_image, self.last_depth_colormap
                    time.sleep(0.01)
                    continue

                # Attempt to grab frames with no specific timeout,
                # letting RealSense deliver them at recorded speed
                try:
                    frames = self.pipeline.wait_for_frames()
                except RuntimeError as e:
                    print(f"[ERROR] Could not retrieve frames: {e}")
                    break

                if not frames or frames.size() == 0:
                    print("[INFO] No more frames arrived. Possibly end of file.")
                    break

                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    # Possibly near end-of-file or a bad frame
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Convert from RGB -> BGR if your .bag was recorded in RGB8
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                # Create a colorized depth frame
                depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.05)
                depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                # Update last valid frames
                self.last_color_image = color_image
                self.last_depth_image = depth_image
                self.last_depth_colormap = depth_colormap

                yield color_image, depth_image, depth_colormap

        finally:
            self.pipeline.stop()

def main():
    """
    Usage:
      python realsense_capture.py path/to/file.bag  (pause/resume with 'p')
      python realsense_capture.py                   (live camera, no pause)

    This version ensures real-time playback for .bag files
    so frames come at the recorded speed rather than as fast as possible.
    """
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    capture = RealSenseCapture(bag_file=bag_file)
    paused = False

    for color_img, depth_raw, depth_map in capture.get_frames():
        # Show color + depth side by side
        cv2.imshow("Color", color_img)
        cv2.imshow("Depth", depth_map)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                capture.pause_playback()
            else:
                capture.resume_playback()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
