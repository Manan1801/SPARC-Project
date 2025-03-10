#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def wrist_offset_alignment(joints_3d_pred, wrist_2d, depth_image, intrinsics, depth_scale):
    """
    - joints_3d_pred: Nx3 predicted joints in the model's coordinate system.
    - wrist_2d: (x, y) pixel of the wrist in the color image.
    - depth_image: 2D array of depth values.
    - intrinsics: RealSense intrinsics (rs.intrinsics).
    - depth_scale: real-world scaling factor (usually ~0.001).
    Returns:
       updated_joints_3d: Nx3 array of the aligned joints
    """
    x, y = int(wrist_2d[0]), int(wrist_2d[1])
    if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
        print("Wrist 2D is out of image bounds!")
        return joints_3d_pred

    depth_value = depth_image[y, x] * depth_scale
    if depth_value <= 0:
        print("No valid depth at wrist pixel!")
        return joints_3d_pred

    # Convert wrist pixel to 3D camera coordinates
    real_wrist_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
    # Suppose predicted wrist is the first joint: joints_3d_pred[0]
    # Or find the index of wrist in the skeleton. Let's assume index 0 is wrist.
    pred_wrist_3d = joints_3d_pred[0]

    # Offset = real_wrist - predicted_wrist
    offset = np.array(real_wrist_3d) - pred_wrist_3d
    updated_joints_3d = joints_3d_pred + offset
    return updated_joints_3d

def icp_registration(hand_mesh_vertices, hand_mesh_faces, depth_pcd):
    """
    - hand_mesh_vertices: Nx3 array of predicted mesh vertices
    - hand_mesh_faces: Mx3 face indices
    - depth_pcd: open3d.geometry.PointCloud from real hand points
    Returns:
       transformed_vertices: Nx3 aligned vertices
    """

    # Build open3d TriangleMesh
    pred_mesh = o3d.geometry.TriangleMesh()
    pred_mesh.vertices = o3d.utility.Vector3dVector(hand_mesh_vertices)
    pred_mesh.triangles = o3d.utility.Vector3iVector(hand_mesh_faces)
    pred_mesh.compute_vertex_normals()

    # Convert mesh vertices -> PointCloud for ICP
    pred_pcd = pred_mesh.sample_points_poisson_disk(number_of_points=5000)

    # Prepare for ICP
    threshold = 0.02  # distance threshold in meters, adjust as needed
    trans_init = np.eye(4)

    # Run ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source=pred_pcd,
        target=depth_pcd,
        max_correspondence_distance=threshold,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    transformation = result_icp.transformation

    # Apply transformation to the original mesh
    pred_mesh.transform(transformation)

    # Return new vertices
    transformed_vertices = np.asarray(pred_mesh.vertices)
    return transformed_vertices

def main_demo():
    # Example usage of the wrist offset
    joints_3d_pred = np.random.randn(21, 3)
    wrist_2d = (320, 240)
    depth_image = np.zeros((480, 640))  # dummy
    intrinsics = rs.intrinsics()        # dummy
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.fx = 600
    intrinsics.fy = 600
    intrinsics.ppx = 320
    intrinsics.ppy = 240
    intrinsics.model = rs.distortion.inverse_brown_conrady
    intrinsics.coeffs = [0,0,0,0,0]

    depth_scale = 0.001

    aligned_joints = wrist_offset_alignment(
        joints_3d_pred, wrist_2d, depth_image, intrinsics, depth_scale
    )
    print("Aligned Joints:\n", aligned_joints)

if __name__ == "__main__":
    main_demo()
