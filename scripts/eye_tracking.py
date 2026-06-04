#!/usr/bin/env python3

import time
from pathlib import Path
import traceback

import cv2
import numpy as np
import mediapipe as mp
import certifi
from logger_utils import get_eye_logger
from eye_tracking_ui import draw_hud


from control_flags import stop_event, pause_event

# Left eye boundaries
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Right eye boundaries
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


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

    csv_path = eye_dir / "eye_tracking.csv"

    csv_file = open(csv_path, "w", buffering=1)  # Line-buffered for real-time writing

    csv_file.write(
        "timestamp_ns,frame_id,"
        "left_iris_x_px,left_iris_y_px,"
        "right_iris_x_px,right_iris_y_px,"
        "X,Y,Z,"
        "smooth_X,smooth_Y,smooth_Z,"
        "head_horizontal,head_vertical,"
        "gaze_horizontal,gaze_vertical\n"
    )

    csv_file.flush()

    prev_X = None
    prev_Y = None
    prev_Z = None

    alpha = 0.8
    flush_counter = 0

    while not stop_event.is_set():

        if pause_event.is_set():
            time.sleep(0.02)
            continue

        try:
            pkt = q_eye.get(timeout=0.1)

        except Exception as e:
            print(f"[QUEUE ERROR] {e}")
            traceback.print_exc()
            continue

        if pkt is None:
            continue

        try:

            color_img = pkt.color.copy()
            depth_img = pkt.depth

            fx = pkt.fx
            fy = pkt.fy
            cx = pkt.cx
            cy = pkt.cy

            depth_scale = pkt.depth_scale_m

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            h, w, _ = color_img.shape

            face_landmarks = results.multi_face_landmarks[0]

            NOSE_TIP = 1

            nose = face_landmarks.landmark[NOSE_TIP]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            face_center_x = w // 2
            face_center_y = h // 2

            head_yaw = (nose_x - face_center_x) / face_center_x
            head_pitch = (nose_y - face_center_y) / face_center_y

            head_yaw = np.clip(head_yaw, -1.0, 1.0)
            head_pitch = np.clip(head_pitch, -1.0, 1.0)

            # LEFT EYE
            left_eye_left = face_landmarks.landmark[LEFT_EYE_LEFT]
            left_eye_right = face_landmarks.landmark[LEFT_EYE_RIGHT]
            left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP]
            left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM]

            # RIGHT EYE
            right_eye_left = face_landmarks.landmark[RIGHT_EYE_LEFT]
            right_eye_right = face_landmarks.landmark[RIGHT_EYE_RIGHT]
            right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
            right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]

            left_center, left_points = get_iris_center(
                face_landmarks,
                LEFT_IRIS,
                w,
                h
            )

            lx, ly = left_center

            right_center, right_points = get_iris_center(
                face_landmarks,
                RIGHT_IRIS,
                w,
                h
            )

            rx, ry = right_center

    
            # Combined eye center
            eye_x = int((lx + rx) / 2)
            eye_y = int((ly + ry) / 2)

            # Eye boundary pixels
            left_eye_left_x = int(left_eye_left.x * w)
            left_eye_right_x = int(left_eye_right.x * w)
            left_eye_top_y = int(left_eye_top.y * h)
            left_eye_bottom_y = int(left_eye_bottom.y * h)

            right_eye_left_x = int(right_eye_left.x * w)
            right_eye_right_x = int(right_eye_right.x * w)
            right_eye_top_y = int(right_eye_top.y * h)
            right_eye_bottom_y = int(right_eye_bottom.y * h)

            # Safe ordering
            left_min_x = min(left_eye_left_x, left_eye_right_x)
            left_max_x = max(left_eye_left_x, left_eye_right_x)

            right_min_x = min(right_eye_left_x, right_eye_right_x)
            right_max_x = max(right_eye_left_x, right_eye_right_x)

            left_min_y = min(left_eye_top_y, left_eye_bottom_y)
            left_max_y = max(left_eye_top_y, left_eye_bottom_y)

            right_min_y = min(right_eye_top_y, right_eye_bottom_y)
            right_max_y = max(right_eye_top_y, right_eye_bottom_y)

            # Ratios
            left_horizontal_ratio = (
                (lx - left_min_x) /
                (left_max_x - left_min_x + 1e-6)
            )

            left_vertical_ratio = (
                (ly - left_min_y) /
                (left_max_y - left_min_y + 1e-6)
            )

            right_horizontal_ratio = (
                (rx - right_min_x) /
                (right_max_x - right_min_x + 1e-6)
            )

            right_vertical_ratio = (
                (ry - right_min_y) /
                (right_max_y - right_min_y + 1e-6)
            )

            horizontal_ratio = (
                left_horizontal_ratio +
                right_horizontal_ratio
            ) / 2.0

            vertical_ratio = (
                left_vertical_ratio +
                right_vertical_ratio
            ) / 2.0

            # Bounds check
            if (
                eye_x < 0 or
                eye_y < 0 or
                eye_x >= depth_img.shape[1] or
                eye_y >= depth_img.shape[0]
            ):
                continue

            patch = depth_img[
                max(0, eye_y - 2):min(depth_img.shape[0], eye_y + 3),
                max(0, eye_x - 2):min(depth_img.shape[1], eye_x + 3)
            ]

            valid = patch[patch > 0]

            if len(valid) == 0:
                continue

            depth_raw = np.median(valid)

            if depth_raw == 0:
                continue

            depth_m = float(depth_raw) * float(depth_scale)

            if depth_m < 0.15 or depth_m > 2.0:
                continue

            X, Y, Z = pixel_to_3d(
                eye_x,
                eye_y,
                depth_m,
                fx,
                fy,
                cx,
                cy
            )

            # Smoothing
            if prev_X is None:
                smooth_X = X
                smooth_Y = Y
                smooth_Z = Z
            else:
                smooth_X = alpha * prev_X + (1 - alpha) * X
                smooth_Y = alpha * prev_Y + (1 - alpha) * Y
                smooth_Z = alpha * prev_Z + (1 - alpha) * Z

            prev_X = smooth_X
            prev_Y = smooth_Y
            prev_Z = smooth_Z

            # Head pose
            head_horizontal = "CENTER"
            head_vertical = "CENTER"

            if head_yaw < -0.15:
                head_horizontal = "LEFT"
            elif head_yaw > 0.15:
                head_horizontal = "RIGHT"

            if head_pitch < -0.10:
                head_vertical = "UP"
            elif head_pitch > 0.10:
                head_vertical = "DOWN"
            # Gaze
            # gaze_horizontal = "CENTER"
            # gaze_vertical = "CENTER"

            # if horizontal_ratio < 0.35:
            #     gaze_horizontal = "LEFT"
            # elif horizontal_ratio > 0.65:
            #     gaze_horizontal = "RIGHT"

            # if vertical_ratio < 0.30:
            #     gaze_vertical = "UP"
            # elif vertical_ratio > 0.70:
            #     gaze_vertical = "DOWN"

            # ==========================================================
            # IMPROVED GAZE ESTIMATION
            # ==========================================================

            eye_gaze_x = (horizontal_ratio - 0.5) * 2.0
            eye_gaze_y = (vertical_ratio - 0.5) * 2.0

            # combine eye movement + head movement

            gaze_x = eye_gaze_x + 0.5 * head_yaw
            gaze_y = eye_gaze_y + 0.5 * head_pitch

            gaze_x = np.clip(gaze_x, -1.0, 1.0)
            gaze_y = np.clip(gaze_y, -1.0, 1.0)

            gaze_dx = int(gaze_x * 170)
            gaze_dy = int(gaze_y * 170)

            gaze_horizontal = "CENTER"
            gaze_vertical = "CENTER"

            if gaze_x < -0.35:
                gaze_horizontal = "LEFT"

            elif gaze_x > 0.35:
                gaze_horizontal = "RIGHT"

            if gaze_y < -0.25:
                gaze_vertical = "UP"

            elif gaze_y > 0.25:
                gaze_vertical = "DOWN"

            # Yaw and pitch in degrees
            eye_yaw_deg = gaze_x * 35.0
            eye_pitch_deg = -gaze_y * 25.0

            tracking_confidence = min(
                1.0,
                max(0.0, len(valid) / 25.0)
            )

            # HUD
            try:

              draw_hud(
                    color_img,

                    smooth_X,
                    smooth_Y,
                    smooth_Z,

                    head_horizontal,
                    head_vertical,

                    gaze_horizontal,
                    gaze_vertical,

                    horizontal_ratio,
                    vertical_ratio,

                    lx,
                    ly,

                    rx,
                    ry,

                    gaze_dx,
                    gaze_dy,

                    head_yaw,
                    head_pitch,

                    eye_yaw_deg,
                    eye_pitch_deg
                )

            except Exception as e:
                print("[HUD ERROR]")
                print(e)
                traceback.print_exc()

            pkt.eye_overlay = color_img

            # SAVE FRAME
            if pkt.frame_id % 5 == 0:

                frame_path = eye_dir / f"eye_{pkt.frame_id:06d}.jpg"

                ok = cv2.imwrite(str(frame_path), color_img)

                if not ok:
                    print(f"[SAVE ERROR] Failed to save {frame_path}")

            msg = (
                f"Eye3D => "
                f"X={smooth_X:.3f} "
                f"Y={smooth_Y:.3f} "
                f"Z={smooth_Z:.3f}"
            )

            print(f"[{cam_label}] {msg}")

            logger.info(msg)

            logger.periodic_flush()

            csv_file.write(
                f"{pkt.t_ns},"
                f"{pkt.frame_id},"
                f"{lx},{ly},"
                f"{rx},{ry},"
                f"{X:.5f},{Y:.5f},{Z:.5f},"
                f"{smooth_X:.5f},{smooth_Y:.5f},{smooth_Z:.5f},"
                f"{head_horizontal},{head_vertical},"
                f"{gaze_horizontal},{gaze_vertical}\n"
            )

            flush_counter += 1

            if flush_counter >= 30:
                csv_file.flush()
                flush_counter = 0

        except Exception as e:

            print(f"[EYE TRACK ERROR] {e}")

            traceback.print_exc()

            continue

        # cv2.imshow(f"Eye Tracking - {cam_label}", color_img)

        # key = cv2.waitKey(1)

        # if key == 27:
        #     stop_event.set()

    # cv2.destroyAllWindows()

    csv_file.close()
    face_mesh.close()
    logger.close()

    print(f"[INFO] Eye tracking stopped for {cam_label}")