#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import torch

# Force torch.load to unpickle everything (weights_only=False)
old_torch_load = torch.load
def custom_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    kwargs["weights_only"] = False
    return old_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
torch.load = custom_torch_load

from mmpose.apis import init_pose_model as init_pose_estimator, inference_top_down_pose_model

def build_sapiens_model(config_path, checkpoint_path, device='cuda'):
    model = init_pose_estimator(config_path, 
                                checkpoint_path,
                                device=device
                                # override_ckpt_meta=True,  # If needed for new MMPose

                                # If you want to pass something else, keep cfg_options if needed:
                                # cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True)))
                                )   
    model.eval()

    cfg = model.cfg

    # If older code references cfg.data_cfg
    if not hasattr(cfg, 'data_cfg'):
        cfg.data_cfg = dict(
            image_size=[256, 256],  # adjust for your model
            num_joints=21,         # if your model is a 21-joint hand model
        )
        print("[WARN] Created fallback cfg.data_cfg with image_size=[256,256], num_joints=21")

    # If older code references cfg.test_pipeline
    if not hasattr(cfg, 'test_pipeline'):
        # Try copying from data.test.pipeline if it exists
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'test') and hasattr(cfg.data.test, 'pipeline'):
            cfg.test_pipeline = cfg.data.test.pipeline
            print("[INFO] Copied pipeline from cfg.data.test.pipeline")
        else:
            # Minimal fallback
            cfg.test_pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='TopDownAffine'),
                dict(type='ToTensor'),
            ]
            print("[WARN] Using minimal fallback pipeline for older MMPose code.")

    return model

class Sapiens2DKeypoint:
    def __init__(
        self,
        pose_config="/home/hpm_mv_2/Desktop/SPARC-Project/sapiens/pose/configs/sapiens_pose/coco_wholebody/sapiens_1b-210e_coco_wholebody-1024x768.py",
        pose_checkpoint="/home/hpm_mv_2/Desktop/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth",
        device="cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = build_sapiens_model(
            config_path=pose_config,
            checkpoint_path=pose_checkpoint,
            device=self.device,
        )

    def infer_keypoints(self, color_image, bboxes=None):
        h, w = color_image.shape[:2]
        if bboxes is None or len(bboxes) == 0:
            bboxes = [np.array([0, 0, w, h], dtype=np.float32)]
        elif isinstance(bboxes, np.ndarray):
            bboxes = [bbox for bbox in bboxes]

        # Convert to person_results
        person_results = []
        for box in bboxes:
            person_results.append({'bbox': box})

        # Convert color_image from BGR to RGB
        rgb_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        results, _ = inference_top_down_pose_model(
            self.model,
            rgb_img,
            person_results
        )
        # returns a list of PoseDataSample
        return results

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_image = cv2.imread(image_path)
        if test_image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return
    else:
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    keypoint_detector = Sapiens2DKeypoint()
    pose_results = keypoint_detector.infer_keypoints(test_image)

    for data_sample in pose_results:
        if not hasattr(data_sample, 'pred_instances'):
            continue
        keypoints = data_sample.pred_instances.keypoints
        for kpt in keypoints:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(test_image, (x, y), 3, (0,255,0), -1)

    cv2.imshow("Sapiens Keypoints Demo", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
