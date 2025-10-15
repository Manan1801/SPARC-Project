#!/usr/bin/env python3
"""
Real-time unified pipeline: capture (all cams), optional per-cam processing, optional audio,
streaming CSV append, optional debug preview, and debounced logging.

Controls:
  • SPACE  → toggle Pause/Resume (applies to capture + processing + audio)
  • ESC or Ctrl+C → Stop gracefully

Notes
- Captures from ALL mapped/connected RealSense cameras.
- Use --process-cams to select which cam labels to PROCESS (others will still capture & save if --save-every > 0).
- Hand labeling: MediaPipe handedness with optional global flip baseline (--force-flip flip|same),
  then stabilized by proximity to a two-hand anchor (same logic as your batch script).
- 3D projection uses COLOR intrinsics with depth aligned to color.
- Movement is computed frame-to-frame for keypoints [0..4] per hand (wrist + thumb).
"""

import os, sys, json, time, threading, queue, signal, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# 3rd-party
import numpy as np
import cv2
import pandas as pd
import pyrealsense2 as rs
import mediapipe as mp

# ---------- Constants ----------
CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

WRIST_ID = 0
NUM_LANDMARKS = 21
WRIST_SEP_PX = 30  # keep hard-coded (per your instruction)
KEYPOINTS_MOV = [0, 1, 2, 3, 4]  # compute movement only for these
HANDS = ("L", "R")

# anchor reset if we haven't seen BOTH hands for this many consecutive processed frames
ANCHOR_MISS_RESET = 300

COLOR_LEFT  = (0, 255, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)

mp_hands = mp.solutions.hands

# ---------- Cooperative control ----------
pause_event = threading.Event()  # set => paused
stop_event  = threading.Event()  # set => stop ASAP

def _sig_stop(signum, frame):
    print("\n[⛔] SIGINT → stopping…", flush=True)
    stop_event.set()

signal.signal(signal.SIGINT, _sig_stop)

# ---------- Keyboard controls (SPACE toggle pause, ESC stop) ----------
def toggle_pause(reason="keyboard"):
    if not pause_event.is_set():
        pause_event.set()
        print(f"[⏸] Pause ({reason})", flush=True)
    else:
        pause_event.clear()
        print(f"[▶] Resume ({reason})", flush=True)

def start_keyboard_listener():
    """Terminal listener; works even with --viz-live off. Skips if stdin not a TTY."""
    if not sys.stdin.isatty():
        return None
    import termios, tty, select
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    def _run():
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == ' ':
                    toggle_pause("space")
                elif ch == '\x1b':  # ESC
                    print("[⛔] ESC pressed → stopping.", flush=True)
                    stop_event.set()
                    break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True, name="kbd-listener")
    t.start()
    return t

# ---------- Debounced logger ----------
class DebouncedLogger:
    def __init__(self, log_path: Path, flush_interval_sec: int = 5):
        self.log_path = Path(log_path)
        self._lines: List[str] = []
        self._last_msg: Optional[str] = None
        self._repeat: int = 0
        self._lock = threading.Lock()
        self._flush_interval = max(1, int(flush_interval_sec))
        self._last_flush = time.monotonic()

    def _flush_repeat(self):
        if self._last_msg is not None:
            if self._repeat > 1:
                self._lines.append(f"[WARN x{self._repeat}] {self._last_msg}")
            else:
                self._lines.append(f"[WARN] {self._last_msg}")
        self._last_msg = None
        self._repeat = 0

    def warn(self, msg: str):
        with self._lock:
            if msg == self._last_msg:
                self._repeat += 1
            else:
                self._flush_repeat()
                self._last_msg = msg
                self._repeat = 1

    def info(self, msg: str):
        with self._lock:
            self._flush_repeat()
            self._lines.append(f"[INFO] {msg}")

    def periodic_flush(self, force: bool = False):
        now = time.monotonic()
        with self._lock:
            if force or (now - self._last_flush) >= self._flush_interval:
                self._flush_repeat()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    if self._lines:
                        f.write("\n".join(self._lines) + "\n")
                        self._lines.clear()
                self._last_flush = now

# ---------- Helpers ----------
def load_serial_map() -> Dict[str, str]:
    """
    camera_serials.txt format:
      cam1: <serial>
      cam2: <serial>
      cam3: <serial>
    We'll invert to serial->label.
    """
    serial_to_label: Dict[str, str] = {}
    if not os.path.isfile(CAMERA_SERIALS_FILE):
        print(f"[WARN] camera_serials file missing: {CAMERA_SERIALS_FILE}")
        return serial_to_label
    with open(CAMERA_SERIALS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                label, serial = line.strip().split(":")
                serial_to_label[serial.strip()] = label.strip()
    return serial_to_label

def write_camera_info(profile: rs.pipeline_profile, serial: str, label: str,
                      out_txt: Path, out_json: Path) -> Tuple[dict, float]:
    device = profile.get_device()
    sensors = device.query_sensors()

    intr_json = {}
    with open(out_txt, "w") as f:
        f.write(f"Camera Label: {label}\n")
        f.write(f"Serial Number: {serial}\n")
        f.write(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}\n")
        f.write(f"USB Port ID: {device.get_info(rs.camera_info.physical_port)}\n")
        f.write(f"Product Line: {device.get_info(rs.camera_info.product_line)}\n\n")
        for sensor in sensors:
            f.write(f"[Sensor: {sensor.get_info(rs.camera_info.name)}]\n")
            for opt in sensor.get_supported_options():
                try:
                    val = sensor.get_option(opt)
                    f.write(f"  {opt.name}: {val}\n")
                except Exception:
                    continue
            f.write("\n")
        f.write("[Active Streams]\n")

    for s in profile.get_streams():
        try:
            vs = s.as_video_stream_profile()
            intr = vs.get_intrinsics()
            stream_key = f"{s.stream_type().name.lower()}_{s.format().name.lower()}"
            intr_json[stream_key] = {
                "width": vs.width(),
                "height": vs.height(),
                "fps": vs.fps(),
                "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                "model": intr.model.name, "coeffs": list(intr.coeffs),
            }
        except Exception as e:
            print(f"[WARN] Skipping stream info: {e}")

    # depth scale
    depth_scale = 0.001
    try:
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
    except Exception:
        pass

    intr_json["alignment"] = {"depth_to_color": True, "alignment_target": "color", "use_intrinsics": "color"}
    intr_json["depth_scale_m"] = depth_scale

    with open(out_json, "w") as jf:
        json.dump(intr_json, jf, indent=2)

    return intr_json, depth_scale

@dataclass
class FramePacket:
    cam_label: str
    frame_id: int
    t_ns: int
    rs_ts_ms: float
    color: np.ndarray            # HxWx3 BGR
    depth: np.ndarray            # HxW uint16 (aligned to color)
    fx: float; fy: float; cx: float; cy: float
    depth_scale_m: float         # meters per unit

# ---------- Hand helpers ----------
def mp_landmarks_to_pixels(landmarks, w, h):
    pts = np.zeros((NUM_LANDMARKS, 2), dtype=float)
    for i, lm in enumerate(landmarks):
        pts[i] = [lm.x * w, lm.y * h]
    return pts

def euclid(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def euclid2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def get_mediapipe_detections(results, w, h):
    """
    Returns: [{'pts': (21,2), 'label': 'L'/'R', 'score': float}, ...]
    """
    out = []
    if not results.multi_hand_landmarks:
        return out
    for hand_landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        pts = mp_landmarks_to_pixels(hand_landmarks.landmark, w, h)
        label_str = handed.classification[0].label
        score = float(handed.classification[0].score)
        label = 'L' if label_str.lower().startswith('l') else 'R'
        out.append({'pts': pts, 'label': label, 'score': score})
    return out

def collapse_overlap_mediapipe(dets, frame_id, logger: DebouncedLogger):
    """If both detected but wrists too close, keep higher score."""
    if len(dets) == 2:
        w0 = dets[0]['pts'][WRIST_ID]; w1 = dets[1]['pts'][WRIST_ID]
        d = float(np.hypot(w0[0]-w1[0], w0[1]-w1[1]))
        if d < WRIST_SEP_PX:
            keep_idx = 0 if dets[0]['score'] >= dets[1]['score'] else 1
            logger.warn(f"Frame {frame_id}: wrist distance {d:.1f}px < {WRIST_SEP_PX}px → collapse 2→1 (keep {keep_idx}).")
            return [dets[keep_idx]]
    return dets

def collapse_overlap_raw(raw_pts, prev_L, prev_R, logger: DebouncedLogger, frame_id):
    """
    If exactly two raw detections and wrists are very close, keep the one closer to previous anchors.
    """
    if len(raw_pts) == 2:
        w0 = raw_pts[0][WRIST_ID]; w1 = raw_pts[1][WRIST_ID]
        d = euclid(w0, w1)
        if d < WRIST_SEP_PX:
            if prev_L is not None and prev_R is not None:
                d0 = min(euclid2(w0, prev_L), euclid2(w0, prev_R))
                d1 = min(euclid2(w1, prev_L), euclid2(w1, prev_R))
                keep_idx = 0 if d0 <= d1 else 1
            else:
                keep_idx = 0
            logger.warn(f"Frame {frame_id}: raw wrists {d:.1f}px < {WRIST_SEP_PX}px → collapse 2→1 (keep {keep_idx}).")
            return [raw_pts[keep_idx]]
    return raw_pts

def label_by_proximity(current_pts_list, prev_left_wrist, prev_right_wrist):
    """
    Assign L/R by nearest wrist to the anchors (prev_left_wrist/prev_right_wrist).
    """
    L = None; R = None
    if len(current_pts_list) == 2:
        w0 = current_pts_list[0][WRIST_ID]
        w1 = current_pts_list[1][WRIST_ID]
        d0 = euclid2(w0, prev_left_wrist)
        d1 = euclid2(w1, prev_left_wrist)
        if d0 <= d1:
            L, R = current_pts_list[0], current_pts_list[1]
        else:
            L, R = current_pts_list[1], current_pts_list[0]
    elif len(current_pts_list) == 1:
        w = current_pts_list[0][WRIST_ID]
        dL = euclid2(w, prev_left_wrist)
        dR = euclid2(w, prev_right_wrist)
        if dL <= dR:
            L, R = current_pts_list[0], None
        else:
            L, R = None, current_pts_list[0]
    else:
        L, R = None, None
    return {'L': L, 'R': R}

def draw_annotations(img, pts_L, pts_R, frame_id):
    if pts_L is not None:
        for (x, y) in pts_L:
            cv2.circle(img, (int(x), int(y)), 2, COLOR_LEFT, -1)
        wx, wy = pts_L[WRIST_ID]
        cv2.putText(img, f"L (frame {frame_id})", (int(wx)+5, int(wy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LEFT, 1, cv2.LINE_AA)
    if pts_R is not None:
        for (x, y) in pts_R:
            cv2.circle(img, (int(x), int(y)), 2, COLOR_RIGHT, -1)
        wx, wy = pts_R[WRIST_ID]
        cv2.putText(img, f"R (frame {frame_id})", (int(wx)+5, int(wy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RIGHT, 1, cv2.LINE_AA)
    cv2.putText(img, f"id:{frame_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
    return img

# ---------- 3D projection ----------
def project_xy_to_xyz(px: float, py: float, depth_img: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float, depth_scale_m: float) -> Optional[Tuple[float,float,float]]:
    h, w = depth_img.shape[:2]
    x = int(round(px)); y = int(round(py))
    if not (0 <= x < w and 0 <= y < h):
        return None
    raw = int(depth_img[y, x])
    if raw <= 0:
        return None
    z_mm = raw * depth_scale_m * 1000.0
    X_mm = (x - cx) * z_mm / fx
    Y_mm = (y - cy) * z_mm / fy
    return (round(X_mm, 2), round(Y_mm, 2), round(z_mm, 2))

# ---------- CSV streaming ----------
class CSVStream:
    def __init__(self, csv_path: Path, flush_every: int = 30):
        self.csv_path = Path(csv_path)
        self.flush_every = max(1, int(flush_every))
        self._fh = None
        self._lock = threading.Lock()
        self._count = 0
        self._header_written = False

    def _ensure_open(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self._fh is None:
            self._fh = open(self.csv_path, "a", buffering=1)

    def write_header_if_needed(self):
        with self._lock:
            if not self._header_written:
                cols = self.build_columns()
                self._ensure_open()
                self._fh.write(",".join(cols) + "\n")
                self._header_written = True

    @staticmethod
    def build_columns():
        cols = ["frame_id", "t_ns"]
        for lid in range(NUM_LANDMARKS):
            cols += [f"x_{lid}_L_px", f"y_{lid}_L_px"]
        for lid in range(NUM_LANDMARKS):
            cols += [f"x_{lid}_R_px", f"y_{lid}_R_px"]
        # 3D (only for KP_MOV per hand)
        for h in HANDS:
            for lid in KEYPOINTS_MOV:
                cols += [f"{h}_{lid}_X_mm", f"{h}_{lid}_Y_mm", f"{h}_{lid}_Z_mm"]
        # per-frame movement for KP_MOV
        for h in HANDS:
            for lid in KEYPOINTS_MOV:
                cols += [f"{h}_{lid}_move_mm"]
        return cols

    def append_row(self, row_vals: List[str]):
        with self._lock:
            self._ensure_open()
            self._fh.write(",".join(map(str, row_vals)) + "\n")
            self._count += 1
            if self._count % self.flush_every == 0:
                self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh:
                try:
                    self._fh.flush()
                    self._fh.close()
                finally:
                    self._fh = None

# ---------- Processor thread ----------
def processor_worker(cam_label: str,
                     q: "queue.Queue[FramePacket]",
                     out_dir: Path,
                     force_flip: str,
                     stride: int,
                     viz_live: str,
                     viz_save_every: int,
                     csv_flush_every: int,
                     log_flush_sec: int):
    # per-cam dirs & logger
    cam_dir = out_dir / cam_label
    (cam_dir / "CSV").mkdir(parents=True, exist_ok=True)
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cam_dir / "color_mp").mkdir(parents=True, exist_ok=True)
    logger = DebouncedLogger(cam_dir / "logs" / "hand_detection_rt.log", flush_interval_sec=log_flush_sec)
    csv_stream = CSVStream(cam_dir / "CSV" / "hand_landmark_rt.csv", flush_every=csv_flush_every)
    csv_stream.write_header_if_needed()

    # state
    last_xyz: Dict[str, Optional[Tuple[float,float,float]]] = {f"{h}_{k}": None for h in HANDS for k in KEYPOINTS_MOV}
    prev_two_left_wrist: Optional[Tuple[float,float]] = None
    prev_two_right_wrist: Optional[Tuple[float,float]] = None
    have_anchor = False
    consecutive_no_two = 0

    # MediaPipe hands instance local to this thread
    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands_model:

        processed = 0
        saved_counter = 0

        while not stop_event.is_set():
            try:
                pkt: FramePacket = q.get(timeout=0.1)
            except queue.Empty:
                logger.periodic_flush()
                continue

            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.05)
            if stop_event.is_set():
                break

            processed += 1
            if stride > 1 and (processed - 1) % stride != 0:
                continue  # processing drop (saving is in capture thread)

            img_bgr = pkt.color
            h, w = img_bgr.shape[:2]
            frame_id = pkt.frame_id

            # MediaPipe
            results = hands_model.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            dets = get_mediapipe_detections(results, w, h)
            dets = collapse_overlap_mediapipe(dets, frame_id, logger)

            # Prepare both labeled and raw lists
            L_lab = None; R_lab = None
            raw_pts = []
            for d in dets[:2]:
                raw_pts.append(d['pts'])
                if d['label'] == 'L':
                    L_lab = d['pts']
                elif d['label'] == 'R':
                    R_lab = d['pts']

            # Apply global flip baseline to LABELED path
            if force_flip == "flip":
                L_lab, R_lab = R_lab, L_lab

            # --- Stabilized labeling ---
            if have_anchor:
                # collapse near-duplicate by proximity to anchors
                raw_pts = collapse_overlap_raw(raw_pts, prev_two_left_wrist, prev_two_right_wrist, logger, frame_id)
                labeled = label_by_proximity(raw_pts, prev_two_left_wrist, prev_two_right_wrist)
                L = labeled['L']; R = labeled['R']

                # If both are present, refresh anchors (keep them tracking hands)
                if L is not None and R is not None:
                    prev_two_left_wrist  = tuple(L[WRIST_ID])
                    prev_two_right_wrist = tuple(R[WRIST_ID])
                    consecutive_no_two = 0
                else:
                    consecutive_no_two += 1
                    if consecutive_no_two > ANCHOR_MISS_RESET:
                        logger.warn(f"Lost two-hand reference for {consecutive_no_two} frames → resetting anchor.")
                        have_anchor = False
                        prev_two_left_wrist = prev_two_right_wrist = None
                        consecutive_no_two = 0

            else:
                # No anchor yet → rely on (flipped) MediaPipe labels.
                L, R = L_lab, R_lab
                if L is not None and R is not None:
                    prev_two_left_wrist  = tuple(L[WRIST_ID])
                    prev_two_right_wrist = tuple(R[WRIST_ID])
                    have_anchor = True
                    consecutive_no_two = 0
                    logger.info(f"Anchor set at frame {frame_id} (two-hand reference acquired).")

            # Build CSV row
            row: List[float] = [frame_id, pkt.t_ns]

            # 2D px (L then R)
            for lid in range(NUM_LANDMARKS):
                if L is not None:
                    row += [float(L[lid,0]), float(L[lid,1])]
                else:
                    row += ["", ""]
            for lid in range(NUM_LANDMARKS):
                if R is not None:
                    row += [float(R[lid,0]), float(R[lid,1])]
                else:
                    row += ["", ""]

            # 3D for selected KP + per-frame movement
            moves: Dict[str, float] = {}
            for hand_label, pts in (("L", L), ("R", R)):
                for lid in KEYPOINTS_MOV:
                    tag = f"{hand_label}_{lid}"
                    if pts is None:
                        row += ["", "", ""]
                        moves[tag] = 0.0
                        continue
                    xyz = project_xy_to_xyz(
                        pts[lid,0], pts[lid,1], pkt.depth,
                        pkt.fx, pkt.fy, pkt.cx, pkt.cy, pkt.depth_scale_m
                    )
                    if xyz is None:
                        row += ["", "", ""]
                        moves[tag] = 0.0
                    else:
                        row += [xyz[0], xyz[1], xyz[2]]
                        prev = last_xyz[tag]
                        if prev is None:
                            moves[tag] = 0.0
                        else:
                            dx = xyz[0] - prev[0]
                            dy = xyz[1] - prev[1]
                            dz = xyz[2] - prev[2]
                            moves[tag] = round(float(np.sqrt(dx*dx + dy*dy + dz*dz)), 4)
                        last_xyz[tag] = xyz  # update on valid only

            for hand_label in HANDS:
                for lid in KEYPOINTS_MOV:
                    row += [moves[f"{hand_label}_{lid}"]]

            csv_stream.append_row([str(v) for v in row])

            # Visualization (optional)
            if viz_live == "window" or viz_save_every > 0:
                annotated = draw_annotations(img_bgr.copy(), L, R, frame_id)

                if viz_live == "window":
                    cv2.imshow(f"cam:{cam_label}", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # space
                        toggle_pause("space (window)")
                    elif key == 27:  # esc
                        print("[⛔] ESC pressed in window → stopping.", flush=True)
                        stop_event.set()

                if viz_save_every > 0:
                    saved_counter += 1
                    if (saved_counter % viz_save_every) == 0:
                        out_path = cam_dir / "color_mp" / f"frame_{frame_id:06d}.png"
                        cv2.imwrite(str(out_path), annotated)

            logger.periodic_flush()

        # cleanup
        csv_stream.close()
        if viz_live == "window":
            try:
                cv2.destroyWindow(f"cam:{cam_label}")
            except Exception:
                pass

# ---------- Capture thread ----------
def capture_worker(serial: str,
                   cam_label: str,
                   out_dir: Path,
                   duration_sec: float,
                   save_every: int,
                   filters_on: bool,
                   proc_queue: Optional["queue.Queue[FramePacket]"],
                   backpressure: str):
    cam_dir = out_dir / cam_label
    color_dir = cam_dir / "color"
    depth_dir = cam_dir / "depth"
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    logger = DebouncedLogger(cam_dir / "logs" / "capture_rt.log", flush_interval_sec=5)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(cfg)
    except Exception as e:
        print(f"[ERROR] Failed to start RealSense {cam_label} ({serial}): {e}")
        return

    align = rs.align(rs.stream.color)
    if filters_on:
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole = rs.hole_filling_filter()
    else:
        spatial = temporal = hole = None

    # Write camera info & intrinsics json
    info_txt  = cam_dir / f"camera_info_{serial}.txt"
    info_json = cam_dir / f"camera_intrinsics_{serial}.json"
    intr_json, depth_scale_m = write_camera_info(profile, serial, cam_label, info_txt, info_json)

    # Use color intrinsics
    color_key = next((k for k in intr_json.keys() if k.startswith("color_")), None)
    if color_key is None:
        fx = 604.7; fy = 604.9; cx = 313.8; cy = 252.7  # fallback
    else:
        fx = float(intr_json[color_key]["fx"]); fy = float(intr_json[color_key]["fy"])
        cx = float(intr_json[color_key]["cx"]); cy = float(intr_json[color_key]["cy"])

    print(f"[INFO] 🎥 {cam_label} → depth_scale={depth_scale_m:.6f} m/unit | fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    frame_id = 0
    active_elapsed = 0.0
    last_tick = time.monotonic()
    t0_ns = time.monotonic_ns()

    while not stop_event.is_set() and active_elapsed < duration_sec:
        if pause_event.is_set():
            last_tick = time.monotonic()
            time.sleep(0.02)
            continue

        try:
            frames = pipeline.wait_for_frames()
        except Exception:
            continue

        aligned = align.process(frames)
        c = aligned.get_color_frame()
        d = aligned.get_depth_frame()
        if not c or not d:
            continue

        if filters_on:
            try:
                d = spatial.process(d)
                d = temporal.process(d)
                d = hole.process(d)
            except Exception:
                pass

        color_img = np.asanyarray(c.get_data())
        depth_img = np.asanyarray(d.get_data())

        # Save raw
        if save_every > 0 and (frame_id % save_every == 0):
            cv2.imwrite(str(color_dir / f"frame_{frame_id:06d}.png"), color_img)
            cv2.imwrite(str(depth_dir / f"frame_{frame_id:06d}.tiff"), depth_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        # Publish to processing (if enabled for this cam)
        if proc_queue is not None:
            pkt = FramePacket(
                cam_label=cam_label,
                frame_id=frame_id,
                t_ns=time.monotonic_ns(),
                rs_ts_ms=c.get_timestamp() if hasattr(c, "get_timestamp") else 0.0,
                color=color_img,
                depth=depth_img,
                fx=fx, fy=fy, cx=cx, cy=cy,
                depth_scale_m=depth_scale_m
            )
            try:
                if backpressure == "block":
                    proc_queue.put(pkt, timeout=0.01)
                else:
                    proc_queue.put_nowait(pkt)
            except queue.Full:
                try:
                    _ = proc_queue.get_nowait()
                except Exception:
                    pass
                try:
                    proc_queue.put_nowait(pkt)
                except Exception:
                    pass

        frame_id += 1

        now = time.monotonic()
        active_elapsed += (now - last_tick)
        last_tick = now

    try:
        pipeline.stop()
    except Exception:
        pass
    logger.info(f"Finished capture {cam_label}: frames={frame_id}, active_elapsed={active_elapsed:.2f}s")
    logger.periodic_flush(force=True)

# ---------- Audio (minimal) ----------
try:
    from mic_config import VALID_MIC_IDS as _VALID_MIC_IDS
except Exception:
    _VALID_MIC_IDS = []

def audio_worker(device_str: str, out_dir: Path, duration_sec: float, rate: int):
    if duration_sec <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    wav_tmp = out_dir / f"mic_{device_str.replace(':','').replace(',','')}_{ts}.part"
    wav_final = Path(str(wav_tmp).replace(".part", ".wav"))

    import subprocess, signal as pysignal
    cmd = ["arecord", "-D", device_str, "-f", "cd", "-c", "1", "-r", str(rate), "-t", "wav", str(wav_tmp)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    active_elapsed = 0.0
    last = time.monotonic()
    paused_child = False

    try:
        while not stop_event.is_set() and active_elapsed < duration_sec:
            now = time.monotonic()
            if pause_event.is_set():
                if proc.poll() is None and not paused_child:
                    os.kill(proc.pid, pysignal.SIGSTOP); paused_child = True
                last = now
                time.sleep(0.05)
                continue
            else:
                if proc.poll() is None and paused_child:
                    os.kill(proc.pid, pysignal.SIGCONT); paused_child = False
                active_elapsed += (now - last)
                last = now
                time.sleep(0.02)

        if proc and proc.poll() is None:
            if paused_child:
                os.kill(proc.pid, pysignal.SIGCONT); time.sleep(0.05)
            os.kill(proc.pid, pysignal.SIGINT)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
        if wav_tmp.exists():
            wav_tmp.replace(wav_final)
            print(f"[✅] Audio saved: {wav_final}")
    except Exception as e:
        print(f"[ERROR] audio({device_str}): {e}")
        try:
            proc.kill()
        except Exception:
            pass

# ---------- Args ----------
def build_argparser():
    ap = argparse.ArgumentParser(description="Real-time unified capture+process+audio pipeline")
    ap.add_argument("--output-dir", required=True, help="Base output directory")
    ap.add_argument("--duration-sec", type=float, required=True, help="ACTIVE duration in seconds")
    ap.add_argument("--save-every", type=int, default=1, help="Save raw color/depth every Nth frame (0=off)")
    ap.add_argument("--filters", choices=["on","off"], default="off", help="Depth filters on/off")
    ap.add_argument("--viz-live", choices=["off","window"], default="off", help="Preview window")
    ap.add_argument("--viz-save-every", type=int, default=3, help="Save annotated preview every N frames (0=off)")
    ap.add_argument("--force-flip", choices=["flip","same"], default="flip", help="Global handedness flip baseline")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--backpressure", choices=["drop-latest","block"], default="drop-latest", help="Processor queue policy")
    ap.add_argument("--csv-flush", type=int, default=30, help="CSV flush interval (frames)")
    ap.add_argument("--log-flush-sec", type=int, default=5, help="Logger flush interval")
    ap.add_argument("--health-interval-sec", type=int, default=5, help="(reserved) health log cadence")
    ap.add_argument("--process-cams", nargs="*", help="Labels to process (capture happens for all). If flag is present with no labels → process none.")
    ap.add_argument("--audio-out", default=None, help="Audio output directory (default: <output-dir>/audio)")
    ap.add_argument("--audio-duration-sec", type=float, default=0, help="Active duration for audio (0=off)")
    ap.add_argument("--rate", type=int, choices=[44100,48000], default=44100, help="Audio sample rate")
    ap.add_argument("--rt-publish", choices=["on","off"], default="off", help="(reserved) RT audio publish (queue)")
    ap.add_argument("--audio-backpressure", choices=["drop-latest","block"], default="drop-latest", help="(reserved)")
    return ap

# ---------- Main ----------
def main():
    args = build_argparser().parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _kbd = start_keyboard_listener()

    # Discover cams
    serial_to_label = load_serial_map()
    ctx = rs.context()
    connected = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    active = [(s, serial_to_label.get(s, f"cam_{s[-4:]}")) for s in connected]
    if not active:
        print("[ERROR] No RealSense cameras found.")
        return
    print(f"[INFO] Connected cams: {', '.join([f'{lab}({s})' for s,lab in active])}")

    # Decide which cams to process
    if args.process_cams is None:
        process_set = set(lab for _, lab in active)
    elif len(args.process_cams) == 0:
        process_set = set()
    else:
        process_set = set([lab.strip() for lab in args.process_cams if lab.strip()])

    print(f"[INFO] Processing cams: {sorted(process_set) if process_set else 'NONE (capture-only)'}")

    # Spawn per-cam queues & threads
    cap_threads = []
    proc_threads = []
    queues: Dict[str, queue.Queue] = {}

    for serial, label in active:
        q_cam = None
        if label in process_set:
            q_cam = queue.Queue(maxsize=4)
            queues[label] = q_cam
            t_proc = threading.Thread(
                target=processor_worker,
                args=(label, q_cam, out_dir, args.force_flip, max(1,args.stride),
                      args.viz_live, max(0,args.viz_save_every), max(1,args.csv_flush), max(1,args.log_flush_sec)),
                daemon=True, name=f"proc-{label}"
            )
            t_proc.start()
            proc_threads.append(t_proc)

        t_cap = threading.Thread(
            target=capture_worker,
            args=(serial, label, out_dir, float(args.duration_sec), max(0,args.save_every),
                  (args.filters=="on"), q_cam, args.backpressure),
            daemon=True, name=f"cap-{label}"
        )
        t_cap.start()
        cap_threads.append(t_cap)

    # Audio
    audio_threads = []
    aud_dir = Path(args.audio_out) if args.audio_out else (out_dir / "audio")
    if args.audio_duration_sec > 0 and _VALID_MIC_IDS:
        for dev in _VALID_MIC_IDS:
            t = threading.Thread(target=audio_worker, args=(dev, aud_dir, float(args.audio_duration_sec), int(args.rate)),
                                 daemon=True, name=f"aud-{dev}")
            t.start()
            audio_threads.append(t)
        print(f"[INFO] 🎙 Audio enabled on {len(audio_threads)} mic(s) → {aud_dir}")
    elif args.audio_duration_sec > 0:
        print("[WARN] audio requested but mic_config.VALID_MIC_IDS is empty; skipping audio.")

    # Wait
    try:
        for t in cap_threads:
            t.join()
        stop_event.set()
        for t in proc_threads:
            t.join()
        for t in audio_threads:
            t.join()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("[🏁] Done.")

if __name__ == "__main__":
    main()
