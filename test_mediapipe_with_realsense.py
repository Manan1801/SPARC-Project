#!/usr/bin/env python3

import sys
import cv2
import os
import csv
import time

from realsense_capture import RealSenseCapture
from mediapipe_hand_keypoint import MediaPipeHandKeypoint

def main():
    """
    Usage:
      python test_mediapipe_with_realsense_separated.py [path/to/file.bag]

    This script:
      1) Reads frames from RealSense (live or .bag).
      2) Runs MediaPipe to detect hands, converting normalized coords to pixels.
      3) Saves all keypoints in a CSV file (frame_index, hand_label, landmark_index, px, py).
      4) Opens an optional preview window to show color + depth, but closing or pausing that
         does NOT stop the pipeline from capturing frames.
      5) Additionally, you can toggle CSV saving ON/OFF at any time by pressing 's'.

    Controls in preview window:
      - Press 'p' to pause/resume the PREVIEW ONLY (pipeline & CSV keep running).
      - Press 'c' to close the preview window (pipeline & CSV keep running).
      - Press 's' to toggle CSV saving (pipeline & preview remain unaffected).
      - Press ESC to end the pipeline entirely (closes CSV and preview).
    """

    # ------------------------------------------------
    # 1) Optional command-line argument for .bag file
    # ------------------------------------------------
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    # ------------------------------------------------
    # 2) Initialize RealSense capture
    # ------------------------------------------------
    capture = RealSenseCapture(bag_file=bag_file)
    # We do not pause the pipeline on user input. The pipeline runs from start to end.

    # ------------------------------------------------
    # 3) Initialize MediaPipe Hand Keypoint detector
    # ------------------------------------------------
    keypoint_detector = MediaPipeHandKeypoint(
        model_path="/home/hpm_mv_2/Desktop/hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        fps=30.0
    )

    print("[INFO] Pipeline started. Press ESC to stop everything.")
    print("[INFO] Press 'p' to pause/resume the PREVIEW ONLY.")
    print("[INFO] Press 'c' to close the preview window, pipeline continues.")
    print("[INFO] Press 's' to toggle CSV saving on/off.")

    # ------------------------------------------------
    # 4) Prepare CSV output for storing keypoints
    # ------------------------------------------------
    csv_filename = "hand_keypoints_realsense.csv"
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_index", "hand_label", "landmark_index", "pixel_x", "pixel_y"])

    # CSV saving can be toggled
    saving_csv = True

    # ------------------------------------------------
    # 5) Variables for controlling preview
    # ------------------------------------------------
    preview_on = True      # Whether preview window is open
    preview_paused = False # If the user wants to pause preview

    frame_index = 0

    # ------------------------------------------------
    # 6) Main loop reading frames from RealSense
    # ------------------------------------------------
    for color_img, depth_raw, depth_map in capture.get_frames():
        # color_img: BGR from RealSense
        # depth_map: colorized depth for preview

        # (A) Run MediaPipe
        hand_landmarks_list, handedness_list = keypoint_detector.infer_keypoints(color_img)

        # (B) Save to CSV if saving_csv == True
        h, w, _ = color_img.shape
        if hand_landmarks_list and saving_csv:
            for (hand_landmarks, classification_list) in zip(hand_landmarks_list, handedness_list):
                if classification_list:
                    hand_label = classification_list[0].category_name  # "Left" or "Right"
                else:
                    hand_label = "Unknown"

                for lm_idx, lmk in enumerate(hand_landmarks):
                    px = int(lmk.x * w)
                    py = int(lmk.y * h)
                    csv_writer.writerow([frame_index, hand_label, lm_idx, px, py])

        # (C) If preview is on
        if preview_on:
            if not preview_paused:
                # We'll do a small overlay for visualization
                display_color = color_img.copy()
                alpha = 0.3
                overlay = display_color.copy()

                # Draw circles if any hands
                if hand_landmarks_list:
                    for (hand_landmarks, _) in zip(hand_landmarks_list, handedness_list):
                        for lmk in hand_landmarks:
                            px = int(lmk.x * w)
                            py = int(lmk.y * h)
                            cv2.circle(overlay, (px, py), 2, (0,255,0), -1)

                cv2.addWeighted(overlay, alpha, display_color, 1 - alpha, 0, display_color)

                # Show color & depth
                cv2.imshow("Color (Preview)", display_color)
                cv2.imshow("Depth (Preview)", depth_map)

                # Keyboard input for preview
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC => stop pipeline
                    print("[INFO] ESC pressed, stopping pipeline + CSV.")
                    break
                elif key == ord('p'):
                    preview_paused = not preview_paused
                    print("[INFO] Preview paused =", preview_paused)
                elif key == ord('c'):
                    # Close the preview windows but keep pipeline
                    print("[INFO] Closing preview windows, continuing in background.")
                    cv2.destroyAllWindows()
                    preview_on = False
                elif key == ord('s'):
                    saving_csv = not saving_csv
                    if saving_csv:
                        print("[INFO] CSV saving turned ON.")
                    else:
                        print("[INFO] CSV saving turned OFF.")
            else:
                # If preview is paused
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC => stop
                    print("[INFO] ESC pressed, stopping pipeline + CSV.")
                    break
                elif key == ord('p'):
                    preview_paused = not preview_paused
                    print("[INFO] Preview paused =", preview_paused)
                elif key == ord('c'):
                    print("[INFO] Closing preview windows, continuing in background.")
                    cv2.destroyAllWindows()
                    preview_on = False
                elif key == ord('s'):
                    saving_csv = not saving_csv
                    if saving_csv:
                        print("[INFO] CSV saving turned ON.")
                    else:
                        print("[INFO] CSV saving turned OFF.")

        frame_index += 1

    # ------------------------------------------------
    # 7) Cleanup
    # ------------------------------------------------
    keypoint_detector.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Pipeline ended. Keypoints saved to {csv_filename}")


if __name__ == "__main__":
    main()
