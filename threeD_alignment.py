#!/usr/bin/env python3
"""
3D Alignment Example: Step 4 with RealSense depth, wrist-based alignment, and optional ICP.

Usage in a larger pipeline:
  1) Read frames from RealSense (via realsense_capture.py), get color and depth images + intrinsics.
  2) Detect 2D hand landmarks with MediaPipeHandKeypoint (mediapipe_hand_keypoint.py).
  3) Get 3D hand meshes from HaMeR (hamer_inference.py).
  4) Use this script's 3DAlignment class to align those meshes with real 3D data from depth.

Example:
  from realsense_capture import RealSenseCapture
  from mediapipe_hand_keypoint import MediaPipeHandKeypoint
  from hamer_inference import HaMer3D
  from 3d_alignment import ThreeDAlignment

  # Pseudocode pipeline:
  1) realsense = RealSenseCapture(bag_file='some_file.bag')
  2) mp_hand   = MediaPipeHandKeypoint(...)
  3) hamer     = HaMer3D(...)  # or a method that returns predicted 3D vertices

  # In your loop: get frames from realsense, do 2D detection => 3D pred => alignment
"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

class ThreeDAlignment:
    """
    Class that aligns HaMeR's 3D hand mesh to real depth data from a RealSense camera.
    Provides:
      1) Wrist-based alignment
      2) Optional ICP refinement
    """

    def __init__(self, intrinsics: rs.intrinsics, depth_scale: float):
        """
        Args:
            intrinsics (rs.intrinsics): RealSense color intrinsics (from aligned frames).
            depth_scale (float): Depth scale to convert raw depth units to meters.
        """
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale

    def get_3d_wrist(self, wrist_px: tuple, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert a 2D wrist pixel to a 3D wrist coordinate using RealSense intrinsics + depth map.

        Args:
            wrist_px (tuple): (x, y) in pixel coordinates (color-aligned).
            depth_image (np.ndarray): 2D array of depth (same size as color), each in raw units.

        Returns:
            np.ndarray: shape (3,) the [X, Y, Z] in camera space (meters).
                        Returns None if invalid depth at that pixel.
        """
        x, y = wrist_px
        if x < 0 or y < 0 or y >= depth_image.shape[0] or x >= depth_image.shape[1]:
            print("[WARN] Wrist pixel out of bounds.")
            return None

        depth_raw = depth_image[y, x]
        if depth_raw == 0:
            print("[WARN] No valid depth at wrist pixel (0).")
            return None

        depth_m = depth_raw * self.depth_scale
        # Deproject to 3D camera coords
        wrist_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [x, y], depth_m
        )
        return np.array(wrist_3d)  # shape (3,)

    def align_wrist(self, hamer_vertices: np.ndarray, wrist_idx: int, real_wrist_3d: np.ndarray) -> np.ndarray:
        """
        Translate the entire HaMeR mesh so its wrist aligns with the real wrist's 3D location.

        Args:
            hamer_vertices (np.ndarray): shape (N, 3), HaMeR mesh vertices in 3D (some camera-like coords).
            wrist_idx (int): index of the wrist vertex in hamer_vertices.
            real_wrist_3d (np.ndarray): shape (3,), the real wrist 3D coordinate from depth.

        Returns:
            np.ndarray: shape (N, 3), aligned HaMeR vertices.
        """
        if real_wrist_3d is None:
            print("[INFO] Real wrist is None, skipping alignment.")
            return hamer_vertices

        # Predicted wrist from HaMeR
        pred_wrist_3d = hamer_vertices[wrist_idx]
        # Offset
        offset = real_wrist_3d - pred_wrist_3d

        aligned_verts = hamer_vertices + offset
        return aligned_verts

    def extract_hand_pointcloud(
        self,
        depth_image: np.ndarray,
        bbox: tuple
    ) -> np.ndarray:
        """
        Extract real hand point cloud from depth by cropping to the bounding box, then deprojecting.

        Args:
            depth_image (np.ndarray): Aligned depth image.
            bbox (tuple): (x1, y1, x2, y2) bounding box in pixel coords.

        Returns:
            np.ndarray: shape (M, 3) real 3D points for the hand (camera space).
        """
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([max(0,x1), min(depth_image.shape[1]-1,x2)])
        y1, y2 = sorted([max(0,y1), min(depth_image.shape[0]-1,y2)])

        real_points = []
        for yy in range(y1, y2+1):
            for xx in range(x1, x2+1):
                depth_raw = depth_image[yy, xx]
                if depth_raw == 0:
                    continue
                z_m = depth_raw * self.depth_scale
                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    self.intrinsics, [xx, yy], z_m
                )
                real_points.append([X, Y, Z])

        if len(real_points) == 0:
            return np.empty((0, 3))
        return np.array(real_points)

    def run_icp(
        self,
        real_hand_points: np.ndarray,
        pred_vertices: np.ndarray,
        threshold: float = 0.02
    ):
        """
        Perform ICP between real hand point cloud and predicted HaMeR mesh vertices.

        Args:
            real_hand_points (np.ndarray): shape (M,3), real point cloud from depth.
            pred_vertices (np.ndarray): shape (N,3), HaMeR vertices (already wrist-aligned).
            threshold (float): max correspondence distance for ICP in meters.

        Returns:
            final_transform (np.ndarray): 4x4 transform from pred -> real.
            aligned_pred (np.ndarray): shape (N,3), transformed pred vertices.
        """
        if real_hand_points.shape[0] < 10 or pred_vertices.shape[0] < 2:
            print("[WARN] Not enough points for ICP, skipping.")
            return np.eye(4), pred_vertices

        real_pc_o3d = o3d.geometry.PointCloud()
        real_pc_o3d.points = o3d.utility.Vector3dVector(real_hand_points)

        pred_pc_o3d = o3d.geometry.PointCloud()
        pred_pc_o3d.points = o3d.utility.Vector3dVector(pred_vertices)

        trans_init = np.eye(4)

        result_icp = o3d.pipelines.registration.registration_icp(
            source=pred_pc_o3d,
            target=real_pc_o3d,
            max_correspondence_distance=threshold,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        final_transform = result_icp.transformation

        # Apply transform to the predicted cloud
        pred_pc_o3d.transform(final_transform)
        aligned_pred = np.asarray(pred_pc_o3d.points)

        return final_transform, aligned_pred

    def align_frame(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        wrist_px: tuple,
        hamer_vertices: np.ndarray,
        wrist_idx: int,
        hand_bbox: tuple = None,
        use_icp: bool = False
    ) -> np.ndarray:
        """
        Complete alignment routine for a single frame:
          1) Convert wrist pixel to real 3D wrist
          2) Wrist-based alignment of HaMeR mesh
          3) (Optional) Extract real hand point cloud from bounding box
          4) (Optional) ICP for finer alignment

        Args:
            color_image (np.ndarray): BGR or RGB image (unused except for debugging).
            depth_image (np.ndarray): 2D array of depth, same resolution as color_image.
            wrist_px (tuple): (x, y) for the wrist in pixel coords.
            hamer_vertices (np.ndarray): shape (N,3) from HaMeR.
            wrist_idx (int): index of wrist vertex in HaMeR's vertex array.
            hand_bbox (tuple): (x1,y1,x2,y2) bounding box for the hand in pixel coords (optional).
            use_icp (bool): If True, run ICP after wrist alignment for finer transformation.

        Returns:
            np.ndarray: shape (N,3), final aligned HaMeR vertices in camera space.
        """
        # 1) Get real 3D wrist
        real_wrist_3d = self.get_3d_wrist(wrist_px, depth_image)

        # 2) Wrist-based alignment
        aligned_verts = self.align_wrist(hamer_vertices, wrist_idx, real_wrist_3d)

        if not use_icp or hand_bbox is None:
            return aligned_verts

        # 3) Extract real hand point cloud from depth in bounding box
        real_hand_points = self.extract_hand_pointcloud(depth_image, hand_bbox)
        if real_hand_points.shape[0] == 0:
            print("[WARN] No valid real hand points in the bounding box. Skipping ICP.")
            return aligned_verts

        # 4) ICP
        _, final_verts = self.run_icp(real_hand_points, aligned_verts, threshold=0.02)
        return final_verts

# -------------------------------------------------------------------------
# Example usage code in main() if you want to test as a standalone script.
# In practice, you'll likely import ThreeDAlignment into your pipeline code.
# -------------------------------------------------------------------------
def main():
    import cv2

    print("[INFO] This is a demonstration of the 3DAlignment class with placeholders.")
    print("       In a real pipeline, integrate with realsense_capture, mediapipe, and hamer_inference.")
    print("       Press any key to exit this mock test.")

    # Mock intrinsics for 640x480
    intr = rs.intrinsics()
    intr.width = 640
    intr.height = 480
    intr.ppx = 320
    intr.ppy = 240
    intr.fx  = 600
    intr.fy  = 600
    intr.model = rs.distortion.inverse_brown_conrady
    intr.coeffs = [0,0,0,0,0]

    depth_scale = 0.001  # 1mm => 0.001m, example

    # Create alignment object
    aligner = ThreeDAlignment(intr, depth_scale)

    # Placeholder images:
    color_dummy = np.zeros((480,640,3), dtype=np.uint8)
    depth_dummy = np.zeros((480,640), dtype=np.uint16)
    # Let's pretend there's a "depth" of 1000 (mm) at center
    depth_dummy[240,320] = 1000

    # Fake HaMeR mesh: Nx3
    hamer_vertices = np.array([
        [0.0,0.0,0.0],  # wrist at index 0
        [0.1,0.0,0.0],
        [0.0,0.1,0.0],
        [0.0,0.0,0.1]
    ], dtype=np.float32)
    wrist_index = 0

    # Suppose our 2D wrist is at the center (320,240)
    wrist_px = (320,240)

    # Suppose our hand bbox is a 50-pixel box around the wrist
    x1,y1 = (270,190)
    x2,y2 = (370,290)
    hand_bbox = (x1,y1,x2,y2)

    # Perform alignment
    final_verts = aligner.align_frame(
        color_image=color_dummy,
        depth_image=depth_dummy,
        wrist_px=wrist_px,
        hamer_vertices=hamer_vertices,
        wrist_idx=wrist_index,
        hand_bbox=hand_bbox,
        use_icp=True
    )

    print("[INFO] Final aligned vertices:\n", final_verts)
    cv2.rectangle(color_dummy, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.circle(color_dummy, wrist_px, 5, (0,0,255), -1)
    cv2.imshow("Dummy color", color_dummy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
