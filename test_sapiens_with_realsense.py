#!/usr/bin/env python3

import sys
import cv2
import numpy as np

# Import the RealSense capture class and the Sapiens keypoint class
from realsense_capture import RealSenseCapture
from sapiens_keypoint_inference import Sapiens2DKeypoint

def main():
    """
    Usage:
      python test_sapiens_with_realsense.py [path/to/file.bag]
    If no .bag file is specified, it will attempt to read from a live RealSense camera.
    
    Controls:
      - Press 'p' to pause/resume playback (only works for .bag files).
      - Press ESC to exit.
    """
    # Optional command-line argument for bag file
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    # 1) Initialize RealSense capture
    capture = RealSenseCapture(bag_file=bag_file)

    # 2) Initialize Sapiens 2D keypoint detector
    #    Must match your new constructor: 
    #    __init__(pose_config, pose_checkpoint, device="cuda")
    keypoint_detector = Sapiens2DKeypoint(
        pose_config="/home/hpm_mv_2/Desktop/sapiens/pose/configs/sapiens_pose/coco_wholebody/sapiens_1b-210e_coco_wholebody-1024x768.py",
        pose_checkpoint="/home/hpm_mv_2/Desktop/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth",
        device="cuda"
    )

    paused = False

    for color_img, depth_raw, depth_map in capture.get_frames():
        # 3) Run pose estimation
        pose_results = keypoint_detector.infer_keypoints(color_img)

        # 4) Visualize the pose results
        #    pose_results is a list of PoseDataSample objects (one for each bounding box).
        for data_sample in pose_results:
            # Some items might not have 'pred_instances' if no keypoints were predicted
            if not hasattr(data_sample, 'pred_instances'):
                continue

            # data_sample.pred_instances.keypoints shape: (nK, 2) or (1, nK, 2)
            # depends on your model. Often it's (N, K, 2) if multiple persons per box.
            keypoints = data_sample.pred_instances.keypoints

            # If shape is (K, 2), we can do:
            # for x, y in keypoints:
            #   cv2.circle(...)

            # Or if shape is (N, K, 2), we need an extra loop:
            for person_kpts in keypoints:
                for (x, y) in person_kpts:
                    cv2.circle(color_img, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 5) Display color image with keypoints & depth map
        cv2.imshow("Color with Keypoints", color_img)
        cv2.imshow("Depth", depth_map)

        # 6) Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                capture.pause_playback()
            else:
                capture.resume_playback()

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
