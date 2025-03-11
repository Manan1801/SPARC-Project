#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import numpy as np
import sys

class RealSenseCapture:
    def __init__(self, bag_file=None, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.bag_file = bag_file

        if bag_file:
            # Reading from a .bag file
            print(f"Reading from file: {bag_file}")
            self.config.enable_device_from_file(bag_file, repeat_playback=False)
        else:
            # Live camera usage: explicitly enable streams
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start pipeline
        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")

        # For .bag playback, we can access the playback device to control pause/resume
        self.playback = None
        if self.bag_file:
            dev = self.profile.get_device()
            # Attempt to cast the device to a playback object for pause/resume
            try:
                self.playback = dev.as_playback()
                print("Playback device ready for pause/resume control.")
            except Exception as e:
                print("Warning: could not set up playback device:", e)

    def pause_playback(self):
        """Pauses the .bag playback if available."""
        if self.playback:
            self.playback.pause()
            print("Playback paused.")

    def resume_playback(self):
        """Resumes the .bag playback if available."""
        if self.playback:
            self.playback.resume()
            print("Playback resumed.")

    def get_frames(self):
        """
        Yields (color_image, depth_image, depth_colormap) for each frame:
          - color_image: BGR image for OpenCV
          - depth_image: 16-bit raw depth
          - depth_colormap: 8-bit color-mapped depth for visualization
        """
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert frames to NumPy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # If your .bag was recorded in RGB8, convert from RGB -> BGR
                # Comment out this line if the .bag or live stream is already BGR8
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                # Create an 8-bit color map of the depth image for better visualization
                depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.05)
                depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                yield color_image, depth_image, depth_colormap

        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.pipeline.stop()

def main():
    """
    Usage:
      python realsense_capture.py path/to/file.bag  (pause/resume is available)
      python realsense_capture.py                   (live camera, no pausing)
    """
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    capture = RealSenseCapture(bag_file=bag_file)
    paused = False

    for color_img, depth_img, depth_map in capture.get_frames():
        cv2.imshow("Color", color_img)
        cv2.imshow("Depth", depth_map)

        key = cv2.waitKey(1) & 0xFF
        # 'ESC' to quit
        if key == 27:  # ord('\x1b')
            break
        # 'p' to toggle pause/resume
        elif key == ord('p'):
            paused = not paused
            if paused:
                capture.pause_playback()
            else:
                capture.resume_playback()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
