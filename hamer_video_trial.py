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
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
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
HARDCODED_VIDEO = "/home/hpm_mv_2/Desktop/camera1.avi"


def process_frame(
    frame_bgr: np.ndarray,
    detector: DefaultPredictor_Lazy,
    cpm: ViTPoseModel,
    hamer_model,
    model_cfg,
    renderer,
    device,
    rescale_factor=2.0,
    conf_threshold=0.5
):
    """
    1) Detect persons (Detectron2),
    2) Extract bounding boxes for hands (ViTPose),
    3) Run HaMeR for 3D mesh,
    4) Return overlay + mesh-only frames.
    """
    # 1) DETECT PERSON
    frame_rgb = frame_bgr[:, :, ::-1].copy()
    det_out = detector(frame_bgr)
    det_instances = det_out['instances']

    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > conf_threshold)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    if len(pred_bboxes) == 0:
        return frame_bgr, np.zeros_like(frame_bgr)

    # 2) KEYPOINTS (ViTPose)
    det_input = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
    vitposes_out = cpm.predict_pose(frame_rgb, det_input)

    # 3) EXTRACT HAND BBOXES
    bboxes = []
    is_right = []
    for vitposes in vitposes_out:
        left_hand_keyp  = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        valid_l = left_hand_keyp[:,2] > 0.5
        if valid_l.sum() > 3:
            x_min, y_min = left_hand_keyp[valid_l,0].min(), left_hand_keyp[valid_l,1].min()
            x_max, y_max = left_hand_keyp[valid_l,0].max(), left_hand_keyp[valid_l,1].max()
            bboxes.append([x_min, y_min, x_max, y_max])
            is_right.append(0)

        valid_r = right_hand_keyp[:,2] > 0.5
        if valid_r.sum() > 3:
            x_min, y_min = right_hand_keyp[valid_r,0].min(), right_hand_keyp[valid_r,1].min()
            x_max, y_max = right_hand_keyp[valid_r,0].max(), right_hand_keyp[valid_r,1].max()
            bboxes.append([x_min, y_min, x_max, y_max])
            is_right.append(1)

    if len(bboxes) == 0:
        return frame_bgr, np.zeros_like(frame_bgr)

    # 4) MAKE DATASET FOR HaMeR
    boxes = np.array(bboxes)
    right_array = np.array(is_right)
    dataset = ViTDetDataset(model_cfg, frame_bgr, boxes, right_array, rescale_factor=rescale_factor)

    all_verts = []
    all_camt  = []
    all_right = []

    for i in range(len(dataset)):
        sample = dataset[i]

        # Convert sample['img'] => torch tensor [3,H,W]
        if isinstance(sample['img'], np.ndarray):
            arr = sample['img']
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2,0,1))  # (H,W,3)->(3,H,W)
            sample['img'] = torch.from_numpy(arr).float()
        if sample['img'].ndim == 3:
            sample['img'] = sample['img'].unsqueeze(0)  # => [1,3,H,W]

        # box_center => shape [B,2]
        if isinstance(sample["box_center"], np.ndarray):
            sample["box_center"] = torch.from_numpy(sample["box_center"]).float()
        if sample["box_center"].ndim == 1:
            sample["box_center"] = sample["box_center"].unsqueeze(0)

        # box_size => typically shape (1,) or scalar is okay
        if isinstance(sample["box_size"], np.ndarray):
            sample["box_size"] = torch.from_numpy(sample["box_size"]).float()

        # img_size => shape [B,2]
        if isinstance(sample["img_size"], np.ndarray):
            sample["img_size"] = torch.from_numpy(sample["img_size"]).float()
        if sample["img_size"].ndim == 1:
            sample["img_size"] = sample["img_size"].unsqueeze(0)

        # Now fix sample['right'] => ensure it's a 1D torch tensor
        if isinstance(sample['right'], (np.float32, float, int)):
            # e.g. 0 or 1 or np.float32(0)
            sample['right'] = torch.tensor([sample['right']]).float()
        elif isinstance(sample['right'], np.ndarray):
            sample['right'] = torch.from_numpy(sample['right']).float()

        # If it's a 0D or shape [1], we do not necessarily need unsqueeze,
        # but let's keep it consistent => shape [1].
        if sample['right'].ndim == 0:
            sample['right'] = sample['right'].unsqueeze(0)

        # For other keys that might be Tensors
        for key in sample:
            if key not in ('img', 'box_center', 'box_size', 'img_size', 'right'):
                if hasattr(sample[key], 'unsqueeze'):
                    sample[key] = sample[key].unsqueeze(0)

        sample = recursive_to(sample, device)

        # 5) RUN HaMeR
        with torch.no_grad():
            out = hamer_model(sample)

        # Flip horizontal camera offset for right hand
        pred_cam = out['pred_cam'].clone()
        multiplier = (2 * sample['right'] - 1)
        pred_cam[:,1] *= multiplier

        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * sample['img_size'].max().item()
        )

        # Re-map local coords => full image
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            sample["box_center"],
            sample["box_size"],
            sample["img_size"],
            scaled_focal_length
        ).cpu().numpy()[0]

        pred_vertices = out['pred_vertices'][0].cpu().numpy()
        is_right_hand = sample['right'].cpu().numpy()[0]

        # Flip X coords if right hand
        pred_vertices[:,0] = (2*is_right_hand -1)*pred_vertices[:,0]

        all_verts.append(pred_vertices)
        all_camt.append(pred_cam_t_full)
        all_right.append(is_right_hand)

    # 6) RENDER FULL-FRAME
    h, w, _ = frame_bgr.shape
    cam_view = renderer.render_rgba_multiple(
        all_verts, cam_t=all_camt, render_res=[h, w],
        is_right=all_right,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1,1,1),
        focal_length=scaled_focal_length
    )

    frame_rgba = np.concatenate([
        frame_bgr[:,:,::-1].astype(np.float32)/255.0,
        np.ones((h,w,1), dtype=np.float32)
    ], axis=2)

    alpha_mesh = cam_view[:,:,3:]
    mesh_rgb   = cam_view[:,:,:3]

    overlay_rgb = frame_rgba[:,:,:3]*(1 - alpha_mesh) + mesh_rgb*alpha_mesh
    overlay_bgr = (overlay_rgb[:,:,::-1]*255).astype(np.uint8)

    mesh_only_rgb = mesh_rgb*alpha_mesh
    mesh_only_bgr = (mesh_only_rgb[:,:,::-1]*255).astype(np.uint8)

    return overlay_bgr, mesh_only_bgr


def main():
    parser = argparse.ArgumentParser(description='HaMeR Video Demo (SPARC-Project).')
    parser.add_argument('--video_in', type=str, default=None,
                        help='Path to input video. If not provided, fallback or prompt is used.')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT_PATH,
                        help='Path to HaMeR checkpoint in /home/hpm_mv_2/Desktop/hamer/_DATA/... .ckpt file')
    parser.add_argument('--body_detector', type=str, default='vitdet',
                        help='Choose "vitdet" or "regnety" as the body detector.')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for person detection.')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='Factor for bounding box expansion.')
    args, unknown = parser.parse_known_args()

    # 1) Decide on input video
    if args.video_in and args.video_in.strip():
        input_video_path = args.video_in.strip()
    elif HARDCODED_VIDEO.strip():
        input_video_path = HARDCODED_VIDEO
        print(f"[INFO] Using HARDCODED_VIDEO: {input_video_path}")
    else:
        input_video_path = input("Please specify input video path: ").strip()
        if not input_video_path:
            print("[ERROR] No video path given. Exiting.")
            return

    # 2) Check checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"[ERROR] The checkpoint file does not exist:\n  {ckpt_path}")
        sys.exit(1)

    # 3) Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4) Download + load HaMeR
    download_models(CACHE_DIR_HAMER)
    print(f"[INFO] Loading HaMeR checkpoint from: {ckpt_path}")
    hamer_model, model_cfg = load_hamer(str(ckpt_path))
    hamer_model.to(device)
    hamer_model.eval()

    # 5) Body detector
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_file = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_file))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # 6) ViTPose
    cpm = ViTPoseModel(device)

    # 7) Renderer
    renderer = Renderer(model_cfg, faces=hamer_model.mano.faces)

    # 8) Read video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_video_path}")
        sys.exit(1)

    vid_name = Path(input_video_path).stem
    out_overlay_name = f"{vid_name}_overlay.avi"
    out_mesh_name    = f"{vid_name}_mesh.avi"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_overlay = cv2.VideoWriter(out_overlay_name, fourcc, fps, (width, height))
    out_mesh    = cv2.VideoWriter(out_mesh_name, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"[INFO] Processing frame {frame_count}...")

        overlay_bgr, mesh_bgr = process_frame(
            frame,
            detector,
            cpm,
            hamer_model,
            model_cfg,
            renderer,
            device,
            rescale_factor=args.rescale_factor,
            conf_threshold=args.conf_threshold
        )
        out_overlay.write(overlay_bgr)
        out_mesh.write(mesh_bgr)

    cap.release()
    out_overlay.release()
    out_mesh.release()

    print("[INFO] Processing complete!")
    print(f"[INFO] Overlay video saved: {out_overlay_name}")
    print(f"[INFO] Mesh-only video saved: {out_mesh_name}")


if __name__ == "__main__":
    main()
