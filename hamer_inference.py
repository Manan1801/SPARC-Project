#!/usr/bin/env python3

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import csv

#####################################################################
# 1) Paths to HaMeR repo and checkpoint
#####################################################################
HAMER_REPO_PATH = "/home/hpm_mv_2/Desktop/hamer"
DEFAULT_CKPT_PATH = "/home/hpm_mv_2/Desktop/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"

# Add HaMeR repo path so "import hamer" works
sys.path.append(HAMER_REPO_PATH)

#####################################################################
# 2) Import HaMeR modules
#####################################################################
try:
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import download_models, load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from hamer.utils import recursive_to
    # from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy  # We no longer need this
except ImportError as e:
    print("ERROR: Could not import 'hamer' from:", HAMER_REPO_PATH)
    raise e

#####################################################################
# 3) Import vitpose_model
#####################################################################
try:
    from vitpose_model import ViTPoseModel
except ImportError as e:
    print("ERROR: Could not import 'vitpose_model.py'.")
    raise e

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# Hardcoded fallback video (if --video_in not provided)
HARDCODED_VIDEO = "/home/hpm_mv_2/Desktop/camera1.avi"


class HaMer3D:
    """
    Class to run HaMeR 3D hand mesh estimation on a video or single frames.
    We skip big body detection (ViTDet), so bounding boxes must be provided or dummy.
    """

    def __init__(
        self,
        input_video: str = None,
        checkpoint: str = DEFAULT_CKPT_PATH,
        body_detector: str = "vitdet",  # Ignored now
        conf_threshold: float = 0.5,    # Ignored
        rescale_factor: float = 2.0,
        out_overlay_name: str = None,
        out_mesh_name: str = None
    ):
        """
        Args:
            input_video (str): Path to input video (file or camera).
            checkpoint (str): Path to the HaMeR checkpoint (.ckpt).
            body_detector (str): (Ignored, we skip big detection).
            conf_threshold (float): (Ignored).
            rescale_factor (float): Factor for bounding box expansion.
            out_overlay_name (str): If specified, overrides overlay output name.
            out_mesh_name (str): If specified, overrides mesh output name.
        """
        # ----------------- Setup paths -----------------
        if input_video and input_video.strip():
            self.input_video_path = input_video.strip()
        elif HARDCODED_VIDEO.strip():
            self.input_video_path = HARDCODED_VIDEO
            print(f"[INFO] Using HARDCODED_VIDEO: {self.input_video_path}")
        else:
            self.input_video_path = input("Please specify input video path: ").strip()
            if not self.input_video_path:
                raise ValueError("[ERROR] No video path given.")

        self.ckpt_path = Path(checkpoint)
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(
                f"[ERROR] The checkpoint file does not exist:\n  {self.ckpt_path}"
            )

        # ----------------- Device & Model Load -----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        download_models(CACHE_DIR_HAMER)
        print(f"[INFO] Loading HaMeR checkpoint from: {self.ckpt_path}")
        self.hamer_model, self.model_cfg = load_hamer(str(self.ckpt_path))
        self.hamer_model.to(self.device)
        self.hamer_model.eval()

        # ----------------- Skipping big backbone logic -----------------
        # self.body_detector = body_detector
        # print("[INFO] No big body detection. Provide bounding boxes or skip ViTPose...")

        # ----------------- ViTPose (Optional) -----------------
        self.cpm = ViTPoseModel(self.device)

        # ----------------- Renderer -----------------
        self.renderer = Renderer(self.model_cfg, faces=self.hamer_model.mano.faces)

        # ----------------- Misc Config -----------------
        self.conf_threshold = conf_threshold
        self.rescale_factor = rescale_factor

        # Decide output video names
        self.vid_name = Path(self.input_video_path).stem
        if out_overlay_name is None:
            self.out_overlay_name = f"{self.vid_name}_overlay.avi"
        else:
            self.out_overlay_name = out_overlay_name
        if out_mesh_name is None:
            self.out_mesh_name = f"{self.vid_name}_mesh.avi"
        else:
            self.out_mesh_name = out_mesh_name

    def _process_frame_impl(self, frame_bgr: np.ndarray):
        """
        1) Provide dummy bounding box or from MediaPipe
        2) Use ViTPose to refine bounding box
        3) Run HaMeR => get pred_vertices & pred_joints
        4) Return overlay, mesh, pred_verts, pred_joints, etc.
        """

        h, w, _ = frame_bgr.shape
        # Step 1) Dummy bounding box in center
        center_x, center_y = w // 2, h // 2
        box_size = 100
        x_min = center_x - box_size
        x_max = center_x + box_size
        y_min = center_y - box_size
        y_max = center_y + box_size
        hand_bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        hand_score = np.array([1.0], dtype=np.float32)
        det_input = [np.concatenate([hand_bbox, hand_score[:, None]], axis=1)]

        # Step 2) ViTPose
        frame_rgb = frame_bgr[:, :, ::-1]
        vitposes_out = self.cpm.predict_pose(frame_rgb, det_input)
        if len(vitposes_out) == 0:
            black = np.zeros_like(frame_bgr)
            return frame_bgr, black, [], [], []

        # Extract bounding boxes from keypoints
        bboxes = []
        is_right = []
        for vitposes in vitposes_out:
            # Last 42 => 21 left + 21 right
            left_hand_keyp  = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            valid_l = left_hand_keyp[:, 2] > 0.5
            if valid_l.sum() > 3:
                x_min_v, y_min_v = left_hand_keyp[valid_l, 0].min(), left_hand_keyp[valid_l, 1].min()
                x_max_v, y_max_v = left_hand_keyp[valid_l, 0].max(), left_hand_keyp[valid_l, 1].max()
                bboxes.append([x_min_v, y_min_v, x_max_v, y_max_v])
                is_right.append(0)

            valid_r = right_hand_keyp[:, 2] > 0.5
            if valid_r.sum() > 3:
                x_min_v, y_min_v = right_hand_keyp[valid_r, 0].min(), right_hand_keyp[valid_r, 1].min()
                x_max_v, y_max_v = right_hand_keyp[valid_r, 0].max(), right_hand_keyp[valid_r, 1].max()
                bboxes.append([x_min_v, y_min_v, x_max_v, y_max_v])
                is_right.append(1)

        if len(bboxes) == 0:
            black = np.zeros_like(frame_bgr)
            return frame_bgr, black, [], [], []

        # Step 3) Create cropped dataset & run HaMeR
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils.renderer import cam_crop_to_full
        from hamer.utils import recursive_to

        boxes = np.array(bboxes)
        right_array = np.array(is_right)
        dataset = ViTDetDataset(
            self.model_cfg,
            frame_bgr,
            boxes,
            right_array,
            rescale_factor=self.rescale_factor
        )

        all_verts = []
        all_joints2d = []  # We'll store 21 joint coords in pixel space
        all_camt  = []
        all_right = []

        for i in range(len(dataset)):
            sample = dataset[i]

            # Convert sample['img'] to torch
            if isinstance(sample['img'], np.ndarray):
                arr = sample['img']
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2,0,1))
                sample['img'] = torch.from_numpy(arr).float()
            if sample['img'].ndim == 3:
                sample['img'] = sample['img'].unsqueeze(0)

            # Fix other numeric fields
            if isinstance(sample["box_center"], np.ndarray):
                bc = torch.from_numpy(sample["box_center"]).float()
                sample["box_center"] = bc.unsqueeze(0) if bc.ndim == 1 else bc
            if isinstance(sample["box_size"], np.ndarray):
                sample["box_size"] = torch.from_numpy(sample["box_size"]).float()
            if isinstance(sample["img_size"], np.ndarray):
                isz = torch.from_numpy(sample["img_size"]).float()
                sample["img_size"] = isz.unsqueeze(0) if isz.ndim == 1 else isz
            if isinstance(sample["right"], (float, np.float32, int)):
                sample["right"] = torch.tensor(sample["right"], dtype=torch.float32)

            for k in sample:
                if k not in ("img","box_center","box_size","img_size","right"):
                    val = sample[k]
                    if hasattr(val, "unsqueeze"):
                        sample[k] = val.unsqueeze(0)

            sample = recursive_to(sample, self.device)

            with torch.no_grad():
                out = self.hamer_model(sample)

            # Flip horizontal param if right hand
            pred_cam = out['pred_cam'].clone()
            multiplier = (2 * sample['right'] - 1)
            pred_cam[:, 1] *= multiplier

            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * sample['img_size'].max().item()
            )

            # Convert from cropped coords => full
            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                sample["box_center"],
                sample["box_size"],
                sample["img_size"],
                scaled_focal_length
            ).cpu().numpy()[0]

            # (A) Entire mesh
            pred_vertices = out['pred_vertices'][0].cpu().numpy()
            is_right_hand = float(sample['right'])
            pred_vertices[:,0] = (2*is_right_hand - 1)*pred_vertices[:,0]
            all_verts.append(pred_vertices)
            all_camt.append(pred_cam_t_full)
            all_right.append(is_right_hand)

            # (B) 21 keypoint landmarks => "pred_joints" (assuming out has that)
            if "pred_joints" not in out:
                # If your HaMeR doesn't output this, you can't get the 21 keypoints
                # You might need a different approach or a different branch
                all_joints2d.append(None)
            else:
                # shape (21, 3) => 3D joint coords in model space
                pred_joints_3d = out["pred_joints"][0].cpu().numpy()
                # Flip X if right hand
                pred_joints_3d[:,0] = (2*is_right_hand - 1)*pred_joints_3d[:,0]

                # Convert from cropped coords => full camera coords
                joints_cam = cam_crop_to_full(
                    torch.from_numpy(pred_joints_3d).unsqueeze(0).to(self.device),
                    sample["box_center"],
                    sample["box_size"],
                    sample["img_size"],
                    scaled_focal_length
                ).cpu().numpy()[0]  # shape (21,3)

                # Project to 2D
                # Typically we assume principal point = center of the image
                # e.g. (cx, cy) = (img_size_x/2, img_size_y/2)
                # but you can adapt if needed
                img_size = sample["img_size"][0].cpu().numpy()  # e.g. [width, height]
                cx = img_size[0] / 2.0
                cy = img_size[1] / 2.0
                f  = scaled_focal_length

                X = joints_cam[:,0]
                Y = joints_cam[:,1]
                Z = joints_cam[:,2]

                # Avoid division by zero
                eps = 1e-6
                Z[Z < eps] = eps

                px = f * X / Z + cx
                py = f * Y / Z + cy

                # store as Nx2
                joints_2d = np.stack([px, py], axis=-1)  # shape (21,2)
                all_joints2d.append(joints_2d)

        # Step 4) Render overlay + mesh
        hh, ww, _ = frame_bgr.shape
        cam_view = self.renderer.render_rgba_multiple(
            all_verts,
            cam_t=all_camt,
            render_res=[ww, hh],
            is_right=all_right,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1,1,1),
            focal_length=scaled_focal_length
        )

        frame_rgba = np.concatenate([
            frame_bgr[:,:,::-1].astype(np.float32)/255.0,
            np.ones((hh, ww, 1), dtype=np.float32)
        ], axis=2)

        alpha_mesh = cam_view[:,:,3:]
        mesh_rgb   = cam_view[:,:,:3]

        overlay_rgb = frame_rgba[:,:,:3]*(1 - alpha_mesh) + mesh_rgb*alpha_mesh
        overlay_bgr = (overlay_rgb[:,:,::-1]*255).astype(np.uint8)

        mesh_only_rgb = mesh_rgb * alpha_mesh
        mesh_only_bgr = (mesh_only_rgb[:,:,::-1]*255).astype(np.uint8)

        # Return overlay, mesh, plus the 2D joints
        return overlay_bgr, mesh_only_bgr, all_verts, all_joints2d, all_right

    def process_single_frame(self, frame_bgr: np.ndarray):
        """
        Run HaMeR inference on a single BGR frame.
        Returns (overlay_bgr, mesh_bgr, all_verts, all_joints2d, all_right).
        """
        return self._process_frame_impl(frame_bgr)

    def run_inference(self):
        """
        Process an entire video => produce 2 output videos + CSV of 21 keypoints in 2D.
        """
        cap = cv2.VideoCapture(str(self.input_video_path))
        if not cap.isOpened():
            raise IOError(f"[ERROR] Could not open video: {self.input_video_path}")

        # Prepare CSV for 21 keypoint landmarks
        csv_filename = f"{self.vid_name}_keypoints2d.csv"
        if os.path.exists(csv_filename):
            os.remove(csv_filename)

        csv_file = open(csv_filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        # columns: frame_index, hand_index, joint_index, pixel_x, pixel_y
        csv_writer.writerow(["frame_index", "hand_index", "joint_index", "pixel_x", "pixel_y"])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_overlay = cv2.VideoWriter(self.out_overlay_name, fourcc, fps, (width, height))
        out_mesh    = cv2.VideoWriter(self.out_mesh_name, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"[INFO] Processing frame {frame_count}...")

            overlay_bgr, mesh_bgr, _, all_joints2d_list, is_right_list = self._process_frame_impl(frame)
            # all_joints2d_list is a list of Nx(21,2) or None if no joints
            # Each element corresponds to one "hand" in that frame

            out_overlay.write(overlay_bgr)
            out_mesh.write(mesh_bgr)

            # Save 2D joints to CSV
            if all_joints2d_list:
                for hand_idx, joints2d in enumerate(all_joints2d_list):
                    if joints2d is None:
                        # If HaMeR didn't produce pred_joints for this hand
                        continue
                    # joints2d shape => (21,2)
                    for j_idx in range(joints2d.shape[0]):
                        px, py = joints2d[j_idx]
                        # Write row => frame_count, hand_idx, j_idx, px, py
                        csv_writer.writerow([frame_count, hand_idx, j_idx, f"{px:.2f}", f"{py:.2f}"])

        cap.release()
        out_overlay.release()
        out_mesh.release()
        csv_file.close()

        print("[INFO] Processing complete!")
        print(f"[INFO] Overlay video saved: {self.out_overlay_name}")
        print(f"[INFO] Mesh-only video saved: {self.out_mesh_name}")
        print(f"[INFO] 2D keypoints saved to {csv_filename}")


def main():
    parser = argparse.ArgumentParser(description='HaMeR Inference + 21 keypoints in 2D.')
    parser.add_argument('--video_in', type=str, default=None,
                        help='Path to input video. If omitted, fallback or prompt is used.')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT_PATH,
                        help='Path to HaMeR checkpoint (.ckpt file).')
    parser.add_argument('--body_detector', type=str, default='vitdet',
                        help='(Ignored) We commented out the big backbone code.')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='(Ignored) Confidence threshold for person detection.')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='Factor for bounding box expansion.')
    args = parser.parse_args()

    hamer_3d = HaMer3D(
        input_video=args.video_in,
        checkpoint=args.checkpoint,
        body_detector=args.body_detector,
        conf_threshold=args.conf_threshold,
        rescale_factor=args.rescale_factor
    )

    # Run inference => produce 2 output videos + CSV of 21 keypoints in 2D
    hamer_3d.run_inference()


if __name__ == "__main__":
    """
    Additional Notes:
    - We assume HaMeR outputs 21 "pred_joints" in out['pred_joints'] => shape (B,21,3).
    - We do the same camera transform + perspective projection as the mesh.
    - We store them in CSV => <video_stem>_keypoints2d.csv with columns:
        frame_index, hand_index, joint_index, pixel_x, pixel_y
    - If your model doesn't produce out['pred_joints'], you'll need to adapt or generate them.
    - For multiple hands, each "hand_idx" in all_joints2d_list is processed and appended to CSV.
    """
    main()
