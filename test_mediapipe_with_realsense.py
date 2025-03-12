#!/usr/bin/env python3

import sys
import cv2
from realsense_capture import RealSenseCapture
from mediapipe_hand_keypoint import MediaPipeHandKeypoint

def main():
    """
    Usage:
      python test_mediapipe_with_realsense.py [path/to/file.bag]
    If no .bag is specified, it uses a live RealSense camera.
    
    Controls:
      - Press 'p' to pause/resume playback (when using a .bag).
      - Press ESC to exit.
    """
    # 1) Optional command-line argument for .bag file
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    # 2) Initialize RealSense capture
    capture = RealSenseCapture(bag_file=bag_file)
    paused = False

    # 3) Initialize MediaPipe Hand Keypoint detector
    keypoint_detector = MediaPipeHandKeypoint(
        model_path="/home/hpm_mv_2/Desktop/hand_landmarker.task",  # adjust path if needed
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        fps=30.0
    )

    print("[INFO] Press 'p' to pause/resume if using a .bag, ESC to exit.")

    # 4) Main loop
    for color_img, depth_raw, depth_map in capture.get_frames():
        # color_img: BGR color frame from RealSense
        # depth_raw: raw depth (16-bit)
        # depth_map: colorized depth for visualization

        # (A) Run MediaPipe hand detection
        hand_landmarks_list = keypoint_detector.infer_keypoints(color_img)

        # (B) Create an overlay image for alpha blending
        overlay = color_img.copy()

        # (C) Draw smaller, semi-transparent circles on the overlay
        h, w, _ = color_img.shape
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                for lmk in hand_landmarks:
                    px = int(lmk.x * w)
                    py = int(lmk.y * h)
                    # Smaller radius and draw onto overlay
                    cv2.circle(overlay, (px, py), 2, (0, 255, 0), -1)

        # (D) Blend overlay back into color_img for semi-transparency
        alpha = 0.4  # adjust for more or less transparency
        cv2.addWeighted(overlay, alpha, color_img, 1 - alpha, 0, color_img)

        # (E) Display the color image + depth map
        cv2.imshow("Color with Hand Keypoints", color_img)
        cv2.imshow("Depth", depth_map)

        # (F) Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):  # pause/resume
            paused = not paused
            if paused:
                capture.pause_playback()
            else:
                capture.resume_playback()

    # 5) Cleanup
    keypoint_detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
