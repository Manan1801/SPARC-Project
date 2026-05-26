#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import certifi
from logger_utils import get_eye_logger


from control_flags import stop_event, pause_event

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def pixel_to_3d(u, v, depth, fx, fy, cx, cy):

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return X, Y, Z


def get_iris_center(face_landmarks, iris_indices, w, h):

    points = []

    for idx in iris_indices:

        lm = face_landmarks.landmark[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        points.append((x, y))

    center = np.mean(points, axis=0).astype(np.int32)

    return center, points


def eye_tracking_worker(cam_label, q_eye, out_dir):

    print(f"[INFO] Eye tracking started for {cam_label}")

    cam_dir = Path(out_dir) / cam_label
    cam_dir.mkdir(parents=True, exist_ok=True)

    logger = get_eye_logger(cam_dir,flush_sec=5)

    logger.info("Eye tracking started")

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    eye_dir = Path(out_dir) / cam_label / "eye_tracking"
    eye_dir.mkdir(parents=True, exist_ok=True)

    while not stop_event.is_set():

        if pause_event.is_set():
            time.sleep(0.02)
            continue

        try:
            pkt = q_eye.get(timeout=0.1)

        except Exception:
            continue

        if pkt is None:
            continue

        color_img = pkt.color
        depth_img = pkt.depth

        fx = pkt.fx
        fy = pkt.fy
        cx = pkt.cx
        cy = pkt.cy

        depth_scale = pkt.depth_scale_m

        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:

            # cv2.imshow(f"Eye Tracking - {cam_label}", color_img)

            # if cv2.waitKey(1) == 27:
            #     stop_event.set()

            continue

        h, w, _ = color_img.shape

        face_landmarks = results.multi_face_landmarks[0]

        left_center, left_points = get_iris_center(
            face_landmarks,
            LEFT_IRIS,
            w,
            h
        )

        lx, ly = left_center

        for (x, y) in left_points:
            cv2.circle(color_img, (x, y), 2, (0, 255, 0), -1)

        cv2.circle(color_img, (lx, ly), 4, (0, 0, 255), -1)

        if (
            lx < 0 or
            ly < 0 or
            lx >= depth_img.shape[1] or
            ly >= depth_img.shape[0]
        ):
            continue

        depth_raw = depth_img[ly, lx]

        if depth_raw == 0:
            continue

        depth_m = float(depth_raw) * float(depth_scale)

        X, Y, Z = pixel_to_3d(
            lx,
            ly,
            depth_m,
            fx,
            fy,
            cx,
            cy
        )

        text = (
            f"X={X:.3f}m "
            f"Y={Y:.3f}m "
            f"Z={Z:.3f}m"
        )

        cv2.putText(
            color_img,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        msg = (
            f"Eye3D => "
            f"X={X:.3f} "
            f"Y={Y:.3f} "
            f"Z={Z:.3f}"
        )

        print(f"[{cam_label}] {msg}")

        logger.info(msg)

        # cv2.imshow(f"Eye Tracking - {cam_label}", color_img)

        # key = cv2.waitKey(1)

        # if key == 27:
        #     stop_event.set()

    # cv2.destroyAllWindows()

    print(f"[INFO] Eye tracking stopped for {cam_label}")