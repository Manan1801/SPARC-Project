#!/usr/bin/env python3

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

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
    Class to run HaMeR 3D hand mesh estimation either:
      - On a full video (producing two output videos with overlay & mesh-only)
      - OR on a single frame in memory (returns overlay, mesh, and 3D data).

    The big 'human detection' part (ViTDet/RegNet) is now commented out.
    You must provide bounding boxes from MediaPipe or skip ViTPose entirely.
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

        # ----------------- Detector Setup (Commented Out) -----------------
        # Skipping big backbone (ViTDet or RegNet).
        # self.body_detector = body_detector
        # if self.body_detector == "vitdet":
        #     ...
        # else:
        #     ...
        # print("[INFO] Skipping big body detector! Provide your own bounding boxes or rely on MediaPipe...")

        # ----------------- ViTPose (Optional) -----------------
        self.cpm = ViTPoseModel(self.device)

        # ----------------- Renderer -----------------
        self.renderer = Renderer(self.model_cfg, faces=self.hamer_model.mano.faces)

        # ----------------- Misc Config -----------------
        self.conf_threshold = conf_threshold
        self.rescale_factor = rescale_factor

        # For video output
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
        Core logic:
         1) (Removed) Person detection
         2) (Optional) bounding boxes for ViTPose
         3) HaMeR 3D mesh
         4) Render overlay & mesh-only

        NOTE: We skip big detection. Provide your bounding boxes from MediaPipe if needed.
        """

        # Step 1) (Removed) Person detection. We'll define a dummy bounding box.
        h, w, _ = frame_bgr.shape
        center_x, center_y = w // 2, h // 2
        box_size = 100
        x_min = center_x - box_size
        x_max = center_x + box_size
        y_min = center_y - box_size
        y_max = center_y + box_size
        hand_bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        hand_score = np.array([1.0], dtype=np.float32)

        det_input = [np.concatenate([hand_bbox, hand_score[:, None]], axis=1)]

        # Step 2) Keypoint detection with ViTPose (optional)
        frame_rgb = frame_bgr[:, :, ::-1]
        vitposes_out = self.cpm.predict_pose(frame_rgb, det_input)
        if len(vitposes_out) == 0:
            black = np.zeros_like(frame_bgr)
            return frame_bgr, black, [], [], []

        # Extract bounding boxes from ViTPose keypoints
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

        # Step 3) Crop & run HaMeR
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
        all_camt  = []
        all_right = []

        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample['img'], np.ndarray):
                arr = sample['img']
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2,0,1))
                sample['img'] = torch.from_numpy(arr).float()
            if sample['img'].ndim == 3:
                sample['img'] = sample['img'].unsqueeze(0)

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

            pred_cam = out['pred_cam'].clone()
            multiplier = (2 * sample['right'] - 1)
            pred_cam[:, 1] *= multiplier

            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * sample['img_size'].max().item()
            )

            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                sample["box_center"],
                sample["box_size"],
                sample["img_size"],
                scaled_focal_length
            ).cpu().numpy()[0]

            pred_vertices = out['pred_vertices'][0].cpu().numpy()
            is_right_hand = float(sample['right'])
            pred_vertices[:,0] = (2*is_right_hand - 1)*pred_vertices[:,0]

            all_verts.append(pred_vertices)
            all_camt.append(pred_cam_t_full)
            all_right.append(is_right_hand)

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

        return overlay_bgr, mesh_only_bgr, all_verts, [], all_right

    def process_single_frame(self, frame_bgr: np.ndarray):
        """
        Run HaMeR inference on a single BGR frame.
        Returns (overlay_bgr, mesh_bgr, all_verts, all_camt, all_right).
        """
        return self._process_frame_impl(frame_bgr)

    def run_inference(self):
        """
        Process an entire video => produce 2 output videos:
            1) <video_stem>_overlay.avi
            2) <video_stem>_mesh.avi
        """
        cap = cv2.VideoCapture(str(self.input_video_path))
        if not cap.isOpened():
            raise IOError(f"[ERROR] Could not open video: {self.input_video_path}")

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

            overlay_bgr, mesh_bgr, _, _, _ = self._process_frame_impl(frame)
            out_overlay.write(overlay_bgr)
            out_mesh.write(mesh_bgr)

        cap.release()
        out_overlay.release()
        out_mesh.release()

        print("[INFO] Processing complete!")
        print(f"[INFO] Overlay video saved: {self.out_overlay_name}")
        print(f"[INFO] Mesh-only video saved: {self.out_mesh_name}")


def main():
    parser = argparse.ArgumentParser(description='HaMeR Inference without Detectron2.')
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

    # Run inference => produce 2 output videos
    hamer_3d.run_inference()


if __name__ == "__main__":
    """
    Instructions for Using This Script (HaMer3D OOP Pipeline)
    ---------------------------------------------------------
    1. Prerequisites:
    - Python 3.7+ environment with PyTorch, Detectron2, and the HaMeR dependencies installed.
    - vitpose_model.py present in the same directory or an accessible module path.
    - The HaMeR repository cloned locally, and the checkpoint file (e.g., hamer.ckpt) placed where needed.

    2. Description:
    - This script provides an OOP-style approach to run HaMeR for 3D hand mesh estimation on a video,
      or handle single frames in memory (via process_single_frame()).
    - It detects persons using Detectron2 (ViTDet or RegNetY), extracts hand bounding boxes via ViTPose,
      and then runs HaMeR to produce either:
        (a) Overlaid + mesh-only videos (video-based),
        (b) A single-frame result (overlay & mesh images + 3D data).

    3. Usage (video):
      python hamer_inference.py \
          --video_in /path/to/input_video.mp4 \
          --checkpoint /path/to/hamer.ckpt \
          --body_detector vitdet \
          --conf_threshold 0.5 \
          --rescale_factor 2.0

    Arguments:
      --video_in         Path to the input video file. If omitted, uses a fallback or prompts for input.
      --checkpoint       Path to the HaMeR .ckpt model file (default: /home/hpm_mv_2/Desktop/hamer/...).
      --body_detector    Choice of 'vitdet' or 'regnety' for the body detection model.
      --conf_threshold   Confidence threshold for person detection (default: 0.5).
      --rescale_factor   Scaling factor for hand bounding boxes (default: 2.0).

    4. Expected Outputs (video mode):
      - Two .avi files in the current directory:
            <video_stem>_overlay.avi : The original frames with mesh overlay
            <video_stem>_mesh.avi    : Mesh-only frames on a black background

    5. Single-Frame Usage:
      import cv2
      from hamer_inference import HaMer3D

      hamer_3d = HaMer3D(checkpoint="/path/to/hamer.ckpt", ...)
      frame_bgr = cv2.imread("some_image.jpg")
      overlay, mesh_only, all_verts, all_camt, all_right = hamer_3d.process_single_frame(frame_bgr)
      # Now do something with these results.

    6. Notes:
      - Adjust paths (HAMER_REPO_PATH, DEFAULT_CKPT_PATH, vitpose_model) as needed.
      - Ensure GPU drivers + CUDA version are compatible with your installed PyTorch.
      - For single-frame usage in a loop, just keep calling process_single_frame() repeatedly.
      - We no longer rely on Detectron2 for bounding boxes. Provide bounding boxes from MediaPipe or skip ViTPose.
      - The dummy bounding box is placed at the center of the frame. Replace that with actual bounding box logic from MediaPipe.
      - If you already have the entire 21 keypoints from MediaPipe, you can remove or skip the ViTPose step entirely and feed those keypoints directly into HaMeR (which requires deeper changes).

    Enjoy exploring 3D hand pose estimation with HaMeR! Enjoy your smaller VRAM usage without the big ViT-based body detector!
    """
    main()
