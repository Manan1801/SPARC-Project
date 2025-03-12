#!/usr/bin/env python3
"""
test_pipeline.py
Demonstrates integrating:
  1) RealSenseCapture (RGB+D frames)
  2) MediaPipeHandKeypoint (2D detection)
  3) HaMeR3D (3D single-frame inference)
  4) ThreeDAlignment (Alignment step)

Instead of showing multiple OpenCV windows, we save 5 different videos:
  1) color.avi        -> The color frames with 2D landmarks drawn
  2) depth.avi        -> The colorized depth frames
  3) overlay.avi      -> HaMeR overlay frames
  4) mesh.avi         -> HaMeR mesh-only frames
  5) aligned.avi      -> The final aligned (reprojected) mesh frames

Press 'q' in the terminal to exit the loop (or wait for the bag/video to finish).
"""

import sys
import cv2
import numpy as np
import pyrealsense2 as rs

# 1) Import your classes
from realsense_capture import RealSenseCapture
from mediapipe_hand_keypoint import MediaPipeHandKeypoint
from hamer_inference import HaMer3D
from threeD_alignment import ThreeDAlignment


def project_vertices_to_image(
    verts_3d: np.ndarray,  # shape (N, 3), in camera space
    intrinsics: rs.intrinsics
) -> np.ndarray:
    """
    Given Nx3 aligned vertices in camera coordinates [X, Y, Z],
    project them onto the color image using the RealSense intrinsics.
    Returns an array of shape (N, 2) with pixel coordinates (u,v).
    """
    projected_points = []
    for (X, Y, Z) in verts_3d:
        if Z <= 0:
            # Vertex is behind the camera or invalid
            projected_points.append([-1, -1])
            continue
        # Simple pinhole model (ignoring distortion for brevity):
        u = int((X / Z) * intrinsics.fx + intrinsics.ppx)
        v = int((Y / Z) * intrinsics.fy + intrinsics.ppy)
        projected_points.append([u, v])
    return np.array(projected_points)


def main():
    """
    Main demonstration of a combined pipeline that writes 5 separate videos instead of displaying multiple windows.
    1) color.avi        -> color frames with 2D landmarks
    2) depth.avi        -> colorized depth frames
    3) overlay.avi      -> HaMeR overlay frames
    4) mesh.avi         -> HaMeR mesh-only frames
    5) aligned.avi      -> final reprojected aligned mesh frames
    """
    # -------------------------------------------------------
    # A) Initialize RealSense capture (from .bag or live)
    # -------------------------------------------------------
    bag_file = None
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]

    print("[INFO] Starting RealSenseCapture...")
    realsense = RealSenseCapture(bag_file=bag_file)

    # Retrieve intrinsics + depth scale
    profile = realsense.profile
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    depth_scale = realsense.depth_scale

    print(f"[INFO] RealSense Depth Scale: {depth_scale}")
    print(f"[INFO] RealSense Intrinsics: {intrinsics}")

    # -------------------------------------------------------
    # B) Initialize MediaPipe for 2D keypoints
    # -------------------------------------------------------
    print("[INFO] Initializing MediaPipeHandKeypoint...")
    mp_hand = MediaPipeHandKeypoint(
        model_path="/home/hpm_mv_2/Desktop/hand_landmarker.task",  # Adjust path if needed
        num_hands=1,   # For demo, handle 1 hand
        fps=30.0
    )

    # -------------------------------------------------------
    # C) Initialize HaMer3D for single-frame usage
    # -------------------------------------------------------
    print("[INFO] Initializing HaMer3D for single-frame inference...")
    hamer_3d = HaMer3D(
        checkpoint="/home/hpm_mv_2/Desktop/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
        body_detector="vitdet",
        conf_threshold=0.5,
        rescale_factor=2.0
    )

    # -------------------------------------------------------
    # D) Initialize ThreeDAlignment
    # -------------------------------------------------------
    print("[INFO] Initializing ThreeDAlignment for alignment step...")
    aligner = ThreeDAlignment(intrinsics, depth_scale)

    # -------------------------------------------------------
    # E) Setup video writers
    # -------------------------------------------------------
    # We don't know FPS exactly if it's a bag file; RealSense typically ~30fps. 
    # We'll guess 30 or try to get from metadata.
    fps_guess = 30
    width, height = 640, 480  # We expect the frames are 640x480 from realsense_capture

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_color   = cv2.VideoWriter("color.avi",   fourcc, fps_guess, (width, height))
    out_depth   = cv2.VideoWriter("depth.avi",   fourcc, fps_guess, (width, height))
    out_overlay = cv2.VideoWriter("overlay.avi", fourcc, fps_guess, (width, height))
    out_mesh    = cv2.VideoWriter("mesh.avi",    fourcc, fps_guess, (width, height))
    out_aligned = cv2.VideoWriter("aligned.avi", fourcc, fps_guess, (width, height))

    print("[INFO] Starting main loop. Press 'q' in the terminal to quit...")
    frame_counter = 0

    try:
        for color_frame, depth_frame, depth_colormap in realsense.get_frames():
            frame_counter += 1
            # Quick check for user input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed, exiting loop.")
                break

            # 1) 2D detection with MediaPipe
            hand_landmarks_list = mp_hand.infer_keypoints(color_frame)

            # Draw 2D landmarks on color_frame
            if hand_landmarks_list:
                h, w, _ = color_frame.shape
                x_vals, y_vals = [], []
                for landmark in hand_landmarks_list[0]:  # single hand
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    x_vals.append(px)
                    y_vals.append(py)
                    cv2.circle(color_frame, (px, py), 4, (0,255,0), -1)

                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)

                # Suppose MediaPipe's index-0 is the wrist
                wrist_lm = hand_landmarks_list[0][0]  
                wrist_px = (int(wrist_lm.x * w), int(wrist_lm.y * h))

                # 2) Single-frame HaMeR inference (3D)
                overlay_bgr, mesh_bgr, all_verts_list, all_camt, all_right = \
                    hamer_3d.process_single_frame(color_frame)

                # If HaMeR found a hand
                if len(all_verts_list) > 0:
                    # We'll align the FIRST hand mesh only
                    pred_vertices = all_verts_list[0]
                    wrist_idx_3d = 0  # guess

                    # 3) alignment: wrist-based + ICP
                    hand_bbox = (x_min, y_min, x_max, y_max)
                    aligned_verts = aligner.align_frame(
                        color_image=color_frame,
                        depth_image=depth_frame,
                        wrist_px=wrist_px,
                        hamer_vertices=pred_vertices,
                        wrist_idx=wrist_idx_3d,
                        hand_bbox=hand_bbox,
                        use_icp=True
                    )

                    # 4) Re-render the aligned mesh by projecting each aligned vertex to 2D
                    aligned_overlay = color_frame.copy()
                    projected_pts = project_vertices_to_image(aligned_verts, intrinsics)
                    for (u, v) in projected_pts:
                        if 0 <= u < w and 0 <= v < h:
                            cv2.circle(aligned_overlay, (u,v), 3, (0,0,255), -1)

                else:
                    # If HaMeR didn't detect any hands
                    overlay_bgr = np.zeros_like(color_frame)
                    mesh_bgr    = np.zeros_like(color_frame)
                    aligned_overlay = np.zeros_like(color_frame)
            else:
                # No 2D hand => no bounding box => no alignment => black placeholders
                overlay_bgr = np.zeros_like(color_frame)
                mesh_bgr    = np.zeros_like(color_frame)
                aligned_overlay = np.zeros_like(color_frame)

            # -------------------------------------------------------
            # Write out frames to each video
            # -------------------------------------------------------
            # 1) color.avi (with MP landmarks)
            out_color.write(color_frame)
            # 2) depth.avi (colorized depth)
            out_depth.write(depth_colormap)
            # 3) overlay.avi
            out_overlay.write(overlay_bgr)
            # 4) mesh.avi
            out_mesh.write(mesh_bgr)
            # 5) aligned.avi
            out_aligned.write(aligned_overlay)

            if frame_counter % 30 == 0:
                print(f"[INFO] Processed frame {frame_counter}...")

    finally:
        # Cleanup
        mp_hand.close()
        out_color.release()
        out_depth.release()
        out_overlay.release()
        out_mesh.release()
        out_aligned.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_counter} frames. Exiting.")
        print("[INFO] Videos saved: color.avi, depth.avi, overlay.avi, mesh.avi, aligned.avi")

if __name__ == "__main__":
    main()
