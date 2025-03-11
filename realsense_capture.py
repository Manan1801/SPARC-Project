#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import numpy as np
import sys

class RealSenseCapture:
    def __init__(self, bag_file=None, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if bag_file:
            # Reading from a .bag file; no need to enable specific streams
            # because the .bag metadata already defines them
            print(f"Reading from file: {bag_file}")
            self.config.enable_device_from_file(bag_file, repeat_playback=False)
        else:
            # Live camera usage: explicitly enable streams
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start the pipeline
        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Get depth scale (to convert raw depth to meters, if needed)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")

    def get_frames(self):
        """
        Yields (color_frame, depth_frame) as NumPy arrays on each iteration.
        """
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert frames to NumPy
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # If your .bag is recorded in RGB8 (not BGR8), 
                # convert from RGB to BGR for proper OpenCV visualization:
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                yield (color_image, depth_image)
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.pipeline.stop()

def main():
    """
    Example usage:
      python realsense_capture.py path/to/file.bag
    or:
      python realsense_capture.py  (for live camera)
    """
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    capture = RealSenseCapture(bag_file=bag_file)

    # For demonstration: display frames in real-time
    for color_frame, depth_frame in capture.get_frames():
        cv2.imshow("Color", color_frame)
        cv2.imshow("Depth", depth_frame)

        # Press 'ESC' to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
