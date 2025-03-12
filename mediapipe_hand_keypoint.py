#!/usr/bin/env python3

import sys
import cv2
import mediapipe as mp
import numpy as np

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
            A list of hand landmarks (each a list of 21 Landmarks).
            So the shape is (num_hands_found, 21).
            Each Landmark has .x, .y, .z (normalized).
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
        # results.hand_landmarks -> list of lists, each inner list has 21 Landmarks
        return results.hand_landmarks

    def close(self):
        """Release any resources if needed."""
        if self.landmarker:
            self.landmarker.close()


def main():
    """
    Usage:
        python mediapipe_hand_keypoints.py [path/to/video or 0 for webcam]

    Press 'q' to quit the display window.
    """
    # Parse input argument (video file or camera index)
    if len(sys.argv) > 1:
        source = sys.argv[1]
        try:
            source = int(source)
        except ValueError:
            pass
    else:
        source = 0  # default to webcam

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    # Create our hand keypoint detector
    # Adjust the path to your actual .task file
    keypoint_detector = MediaPipeHandKeypoint(
        model_path="/home/hpm_mv_2/Desktop/hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        fps=30.0
    )

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames available or cannot read.")
            break

        # Inference for hand landmarks
        hand_landmarks_list = keypoint_detector.infer_keypoints(frame)

        # Draw the landmarks on the frame
        h, w, _ = frame.shape
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                # Each hand_landmarks is a list of 21 Landmarks
                # Landmark.x, Landmark.y in [0,1], so convert to pixel coords
                for landmark in hand_landmarks:
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        # Show the annotated frame
        cv2.imshow("MediaPipe Hand Keypoints", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    keypoint_detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
