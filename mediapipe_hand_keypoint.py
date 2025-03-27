#!/usr/bin/env python3

import sys
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeHandKeypoint:
    """
    A simple OOP-style class to load and run the MediaPipe Hand Landmarker (hand mesh).
    """
    def __init__(
        self,
        model_path="/home/hpm_mv_2/Desktop/hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        fps=30.0
    ):
        """
        Args:
            model_path (str): Path to the .task model file for MediaPipe Hand Landmarker.
            num_hands (int): Maximum number of hands to detect.
            min_hand_detection_confidence (float): Minimum detection confidence.
            min_hand_presence_confidence (float): Minimum confidence for landmarks.
            min_tracking_confidence (float): Confidence for landmark tracking.
            fps (float): Estimated frames per second for timestamp calculations.
        """
        self.fps = fps
        self.frame_index = 0

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Create the landmarker
        self.landmarker = HandLandmarker.create_from_options(options)

    def infer_keypoints(self, color_frame: np.ndarray):
        """
        Perform hand keypoint detection on a single frame.

        Args:
            color_frame (np.ndarray): BGR image from OpenCV.

        Returns:
            (hand_landmarks, handedness):
                hand_landmarks -> list of lists (each sub-list has 21 Landmarks)
                handedness     -> list of lists (each sub-list has classification(s) for left/right)
        """
        h, w, _ = color_frame.shape
        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Convert frame index to a timestamp in ms
        timestamp_ms = int(self.frame_index * (1000.0 / self.fps))
        self.frame_index += 1

        # Run detection
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Return both landmarks & handedness
        return results.hand_landmarks, results.handedness

    def close(self):
        """Release any resources if needed."""
        if self.landmarker:
            self.landmarker.close()


def process_frame(
    frame: np.ndarray,
    keypoint_detector: MediaPipeHandKeypoint,
    csv_writer: csv.writer,
    frame_index: int
):
    """
    Given a single frame, perform inference and draw the landmarks/labels on it.
    Also write results to the provided CSV writer.

    Args:
        frame (np.ndarray): The BGR image to process.
        keypoint_detector (MediaPipeHandKeypoint): The detector instance.
        csv_writer (csv.writer): Writer for saving results.
        frame_index (int): The current frame index, used in the CSV output.

    Returns:
        Annotated frame with keypoints drawn.
    """
    hand_landmarks_list, handedness_list = keypoint_detector.infer_keypoints(frame)
    h, w, _ = frame.shape

    # If any hands found
    if hand_landmarks_list:
        # Each element in hand_landmarks_list corresponds to an element in handedness_list
        for hand_idx, (hand_landmarks, classification_list) in enumerate(
            zip(hand_landmarks_list, handedness_list)
        ):
            if classification_list:
                label = classification_list[0].category_name  # e.g. "Left" or "Right"
            else:
                label = "Unknown"

            # Each hand_landmarks is a list of 21 Landmarks
            for lm_idx, landmark in enumerate(hand_landmarks):
                px = int(landmark.x * w)
                py = int(landmark.y * h)

                # Draw circle on the frame
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

                # Save to CSV
                csv_writer.writerow([frame_index, label, lm_idx, px, py])

    return frame


def main():
    """
    Usage:
        python mediapipe_hand_keypoints.py [path/to/video or 0 for webcam or path/to/image]

    Press 'q' to quit the display window.

    This version handles both images and videos.
    """
    # Parse input argument (video file, image file, or camera index)
    if len(sys.argv) > 1:
        source = sys.argv[1]
        try:
            source = int(source)
            # If this is successful, 'source' is an integer => webcam
        except ValueError:
            # Otherwise, it remains a string => file path
            pass
    else:
        source = 0  # default to webcam

    # Prepare CSV output
    csv_filename = "hand_keypoints.csv"
    # Overwrite if it already exists
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    # Write header: frame_index, hand_label, landmark_index, pixel_x, pixel_y
    csv_writer.writerow(["frame_index", "hand_label", "landmark_index", "pixel_x", "pixel_y"])

    # Create our hand keypoint detector
    keypoint_detector = MediaPipeHandKeypoint(
        model_path="/home/hpm_mv_2/Desktop/hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        fps=30.0
    )

    print("[INFO] Press 'q' to quit.")

    # --- CHECK IF SOURCE IS AN IMAGE FILE ---
    # We'll attempt to read an image from 'source' if it's a string.
    frame_index = 0
    if isinstance(source, str):
        # Try reading as an image
        image = cv2.imread(source)
        if image is not None:
            # Valid image => process single frame
            print(f"[INFO] Detected single image input: {source}")
            annotated_image = process_frame(image, keypoint_detector, csv_writer, frame_index)
            # Display the result
            cv2.imshow("MediaPipe Hand Keypoints (Image)", annotated_image)
            cv2.waitKey(0)  # wait for any key press
            cv2.destroyAllWindows()

            # Cleanup
            keypoint_detector.close()
            csv_file.close()
            print(f"[INFO] Saved keypoints to {csv_filename}")
            return
        else:
            print(f"[INFO] '{source}' is not recognized as a valid image. Attempting video capture...")

    # --- OTHERWISE, TREAT SOURCE AS VIDEO OR WEBCAM ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        keypoint_detector.close()
        csv_file.close()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames available or cannot read.")
            break

        annotated_frame = process_frame(frame, keypoint_detector, csv_writer, frame_index)

        # Show the annotated frame
        cv2.imshow("MediaPipe Hand Keypoints", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    # Cleanup
    cap.release()
    keypoint_detector.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved keypoints to {csv_filename}")


if __name__ == "__main__":
    main()
