#!/usr/bin/env python3
"""
Batch, multi-session version.

Folder layout (root_dir):
  /root_dir/
    1/
      audio/  cam1/  cam2/  cam3/
                 └── <COLOR FRAMES FOLDER>  (contains frame_*.png)
    2/
      audio/  cam1/  cam2/  cam3/
                 └── <COLOR FRAMES FOLDER>
    ...
    22/

Per session:
  • Phase 1 → write previews to cam2/sample_color_mp/, then prompt: "same" or "flip".
  • Phase 2 → final outputs to cam2/{color_mp, CSV, logs}.

Requirements:
  pip install opencv-python mediapipe pandas
"""

import argparse
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import defaultdict

# ----------------------- Config -----------------------
WRIST_ID = 0
NUM_LANDMARKS = 21
FIRST_TRUSTED_N = 100  # frames index 0..99

# <<< EDIT THIS VALUE IF YOU WANT A DIFFERENT COLLAPSE THRESHOLD >>>
WRIST_SEP_PX = 30  # pixels: collapse two wrists to one if closer than this

COLOR_LEFT  = (0, 255, 0)   # BGR
COLOR_RIGHT = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)

mp_hands = mp.solutions.hands

# ----------------------- Debounced Logger -----------------------
class DebouncedLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lines = []
        self._last_msg = None
        self._repeat = 0
        self.counters = defaultdict(int)

    def _flush_repeat(self):
        if self._last_msg is not None:
            if self._repeat > 1:
                self._lines.append(f"[WARN x{self._repeat}] {self._last_msg}")
            else:
                self._lines.append(f"[WARN] {self._last_msg}")
        self._last_msg = None
        self._repeat = 0

    def warn(self, key: str, msg: str):
        self.counters[key] += 1
        if msg == self._last_msg:
            self._repeat += 1
        else:
            self._flush_repeat()
            self._last_msg = msg
            self._repeat = 1

    def info(self, msg: str):
        self._flush_repeat()
        self._lines.append(f"[INFO] {msg}")

    def summary(self, summary_dict: dict):
        self._flush_repeat()
        self._lines.append("\n===== SUMMARY =====")
        for k, v in summary_dict.items():
            self._lines.append(f"{k}: {v}")
        if self.counters:
            self._lines.append("\n-- Warning counts by type --")
            for k, v in sorted(self.counters.items()):
                self._lines.append(f"{k}: {v}")

    def write(self):
        self._flush_repeat()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("\n".join(self._lines) + "\n")

# ----------------------- Utilities -----------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def list_frames(color_dir: Path):
    return sorted([p for p in color_dir.glob("frame_*.png")], key=lambda p: natural_key(p.name))

def extract_frame_id(name: str):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None

def find_color_dir(cam2_dir: Path) -> Path | None:
    """
    Strictly use cam2/color as the raw color frames folder.
    Returns the path only if cam2/color exists and contains frame_*.png.
    """
    color = cam2_dir / "color"
    if not color.is_dir():
        return None
    # Require actual raw frames here
    if not any(color.glob("frame_*.png")):
        return None
    return color

def mp_landmarks_to_pixels(landmarks, image_w, image_h):
    pts = np.zeros((NUM_LANDMARKS, 2), dtype=float)
    for i, lm in enumerate(landmarks):
        pts[i] = [lm.x * image_w, lm.y * image_h]
    return pts

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
    cv2.putText(img, f"id: {frame_id}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
    return img

def build_csv_columns():
    cols = ["frame_id"]
    for lid in range(NUM_LANDMARKS):
        cols.append(f"x_{lid}_L_px")
        cols.append(f"y_{lid}_L_px")
    for lid in range(NUM_LANDMARKS):
        cols.append(f"x_{lid}_R_px")
        cols.append(f"y_{lid}_R_px")
    return cols

def make_row(frame_id, pts_L, pts_R):
    row = [frame_id]
    for lid in range(NUM_LANDMARKS):
        if pts_L is not None:
            row.extend([pts_L[lid,0], pts_L[lid,1]])
        else:
            row.extend([np.nan, np.nan])
    for lid in range(NUM_LANDMARKS):
        if pts_R is not None:
            row.extend([pts_R[lid,0], pts_R[lid,1]])
        else:
            row.extend([np.nan, np.nan])
    return row

def euclid(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def euclid2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

# ----------------------- Labeling helpers -----------------------
def get_mediapipe_detections(results, w, h):
    """
    Returns list of dicts:
      [{'pts': (21,2) float array, 'label': 'L'/'R', 'score': float}, ...]
    """
    out = []
    if not results.multi_hand_landmarks:
        return out
    for hand_landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        pts = mp_landmarks_to_pixels(hand_landmarks.landmark, w, h)
        label_str = handed.classification[0].label  # 'Left'/'Right'
        score = float(handed.classification[0].score)
        label = 'L' if label_str.lower().startswith('l') else 'R'
        out.append({'pts': pts, 'label': label, 'score': score})
    return out

def collapse_overlap_mediapipe(dets, log, frame_id):
    """
    If exactly two detections and wrist distance < WRIST_SEP_PX:
      keep the one with higher handedness score, drop the other.
    """
    if len(dets) == 2:
        w0 = dets[0]['pts'][WRIST_ID]; w1 = dets[1]['pts'][WRIST_ID]
        d = euclid(w0, w1)
        if d < WRIST_SEP_PX:
            keep_idx = 0 if dets[0]['score'] >= dets[1]['score'] else 1
            drop_idx = 1 - keep_idx
            log.warn("overlap_collapse",
                     f"Frame {frame_id}: wrist distance {d:.2f}px < {WRIST_SEP_PX:.2f}px → "
                     f"collapsing 2→1 (kept idx {keep_idx}, dropped idx {drop_idx}).")
            return [dets[keep_idx]]
    return dets

def collapse_overlap_raw(raw_pts, prev_L, prev_R, log, frame_id):
    """
    raw_pts: list of (21,2) arrays (len 0..2).
    If exactly two and wrist distance < WRIST_SEP_PX:
      keep the one closer to either prev_L or prev_R (whichever is smaller).
      If prev refs are missing, keep index 0.
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
            drop_idx = 1 - keep_idx
            log.warn("overlap_collapse",
                     f"Frame {frame_id}: wrist distance {d:.2f}px < {WRIST_SEP_PX:.2f}px → "
                     f"collapsing 2→1 (kept idx {keep_idx}, dropped idx {drop_idx}).")
            return [raw_pts[keep_idx]]
    return raw_pts

def label_by_proximity(current_pts_list, prev_left_wrist, prev_right_wrist):
    L = None
    R = None
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

def flip_LR(L, R):
    return R, L

# ----------------------- Phase 1: preview first N -----------------------
def phase1_preview_first_n(frames, out_preview_dir, hands_detector, log):
    """
    Returns cached_firstN: list of dict per previewed frame:
      { 'frame_id': int, 'L': np.ndarray|None, 'R': np.ndarray|None }
    """
    cached = []
    out_preview_dir.mkdir(parents=True, exist_ok=True)

    for idx, fpath in enumerate(frames[:FIRST_TRUSTED_N]):
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            log.warn("io_error", f"[Phase1] Failed to read image: {fpath.name}")
            continue

        h, w = img_bgr.shape[:2]
        frame_id = extract_frame_id(fpath.name)
        results = hands_detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        dets = get_mediapipe_detections(results, w, h)
        dets = collapse_overlap_mediapipe(dets, log, frame_id)

        L, R = None, None
        for d in dets:
            if d['label'] == 'L':
                L = d['pts']
            elif d['label'] == 'R':
                R = d['pts']

        annotated = draw_annotations(img_bgr.copy(), L, R, frame_id)
        (out_preview_dir / fpath.name).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_preview_dir / fpath.name), annotated)

        cached.append({'frame_id': frame_id, 'L': L, 'R': R})

    return cached

# ----------------------- Phase 2: final run -----------------------
def phase2_full_run(frames, cached_firstN, user_choice, parent_dir, hands_detector, log):
    out_img_dir = parent_dir / "color_mp"
    out_csv_dir = parent_dir / "CSV"
    out_logs_dir = parent_dir / "logs"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)

    cols = build_csv_columns()
    csv_rows = []

    # Stats
    total = 0
    cnt_two = 0
    cnt_one = 0
    cnt_zero = 0
    cnt_used_mediapipe = 0
    cnt_used_proximity = 0
    cnt_missing_prev_two = 0
    cnt_overlap_collapsed = 0

    prev_two_left_wrist = None
    prev_two_right_wrist = None

    for idx, fpath in enumerate(frames):
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            log.warn("io_error", f"[Phase2] Failed to read image: {fpath.name}")
            continue
        h, w = img_bgr.shape[:2]
        frame_id = extract_frame_id(fpath.name)
        total += 1

        results = hands_detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        raw_pts = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                raw_pts.append(mp_landmarks_to_pixels(hand_landmarks.landmark, w, h))

        n_detected_orig = len(raw_pts)
        if idx >= FIRST_TRUSTED_N:
            raw_pts = collapse_overlap_raw(raw_pts, prev_two_left_wrist, prev_two_right_wrist, log, frame_id)
            if len(raw_pts) == 1 and n_detected_orig == 2:
                cnt_overlap_collapsed += 1

        n_detected = len(raw_pts)
        if n_detected == 0:
            cnt_zero += 1

        # Decide labeling
        if idx < FIRST_TRUSTED_N:
            cached = next((c for c in cached_firstN if c['frame_id'] == frame_id), None)
            if cached is None:
                dets = get_mediapipe_detections(results, w, h)
                dets = collapse_overlap_mediapipe(dets, log, frame_id)
                L, R = None, None
                for d in dets:
                    if d['label'] == 'L':
                        L = d['pts']
                    elif d['label'] == 'R':
                        R = d['pts']
                cnt_used_mediapipe += 1
            else:
                L, R = cached['L'], cached['R']
                if user_choice == "flip":
                    L, R = flip_LR(L, R)

            if L is not None and R is not None:
                cnt_two += 1
                prev_two_left_wrist  = tuple(L[WRIST_ID])
                prev_two_right_wrist = tuple(R[WRIST_ID])
            elif (L is not None) or (R is not None):
                cnt_one += 1

        else:
            if prev_two_left_wrist is None or prev_two_right_wrist is None:
                cnt_missing_prev_two += 1
                log.warn("no_prev_two",
                         f"No previous two-hand reference for frame {frame_id}; falling back to MediaPipe labels.")
                dets = get_mediapipe_detections(results, w, h)
                dets = collapse_overlap_mediapipe(dets, log, frame_id)
                L, R = None, None
                for d in dets:
                    if d['label'] == 'L':
                        L = d['pts']
                    elif d['label'] == 'R':
                        R = d['pts']
                cnt_used_mediapipe += 1

                if L is not None and R is not None:
                    cnt_two += 1
                    prev_two_left_wrist  = tuple(L[WRIST_ID])
                    prev_two_right_wrist = tuple(R[WRIST_ID])
                elif (L is not None) or (R is not None):
                    cnt_one += 1
            else:
                labeled_dict = label_by_proximity(raw_pts, prev_two_left_wrist, prev_two_right_wrist)
                L, R = labeled_dict['L'], labeled_dict['R']
                cnt_used_proximity += 1

                if L is not None and R is not None:
                    cnt_two += 1
                    prev_two_left_wrist  = tuple(L[WRIST_ID])
                    prev_two_right_wrist = tuple(R[WRIST_ID])
                elif (L is not None) or (R is not None):
                    cnt_one += 1

        # Warnings
        if n_detected_orig > 2:
            log.warn("too_many_hands",
                     f"Detected {n_detected_orig} hands in frame {frame_id}; using only top 2.")
        if n_detected == 1 and (L is None and R is None):
            log.warn("label_fail_one",
                     f"One hand detected in frame {frame_id} but labeling failed; check confidences.")
        if n_detected == 2 and (L is None or R is None):
            log.warn("label_fail_two",
                     f"Two hands detected in frame {frame_id} but labeling incomplete.")

        # Save annotated final image
        annotated = draw_annotations(img_bgr.copy(), L, R, frame_id)
        (out_img_dir / fpath.name).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_dir / fpath.name), annotated)

        # CSV row
        csv_rows.append(make_row(frame_id, L, R))

    # Write CSV and logs
    df = pd.DataFrame(csv_rows, columns=build_csv_columns())
    df.sort_values("frame_id", inplace=True)
    out_csv_path = (parent_dir / "CSV" / "hand_landmark.csv")
    df.to_csv(out_csv_path, index=False)

    summary = {
        "User choice": user_choice,
        "FIRST_TRUSTED_N": FIRST_TRUSTED_N,
        "WRIST_SEP_PX": WRIST_SEP_PX,
        "Total frames processed": total,
        "Frames with 2 hands (final labels)": cnt_two,
        "Frames with 1 hand (final labels)": cnt_one,
        "Frames with 0 hands": cnt_zero,
        "Frames using MediaPipe handedness": cnt_used_mediapipe,
        "Frames using proximity relabeling": cnt_used_proximity,
        "Overlap collapses (2→1)": cnt_overlap_collapsed,
        "Fallbacks (no previous two-hand reference)": cnt_missing_prev_two,
        "Output images dir": str(parent_dir / "color_mp"),
        "Output CSV": str(out_csv_path),
        "Logs": str(parent_dir / "logs" / "hand_detection.log"),
    }
    log.summary(summary)
    log.write()

# ----------------------- Helper: Already Processed Sessions  -----------------------
def session_already_processed(cam2_dir: Path, color_dir: Path) -> bool:
    """
    Heuristics to decide if a session is already fully processed:
      - CSV exists and is non-empty
      - color_mp exists and has >= as many annotated frames as input frames
      - logs/hand_detection.log exists and contains a SUMMARY line
    """
    csv_path = cam2_dir / "CSV" / "hand_landmark.csv"
    out_img_dir = cam2_dir / "color_mp"
    log_path = cam2_dir / "logs" / "hand_detection.log"

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return False
    if not out_img_dir.is_dir():
        return False

    n_in = len(list_frames(color_dir))
    n_out = len(sorted(out_img_dir.glob("frame_*.png")))
    if n_out < n_in:
        return False

    if not log_path.is_file() or log_path.stat().st_size == 0:
        return False
    try:
        tail = log_path.read_text(errors="ignore")[-2000:]
        if "===== SUMMARY =====" not in tail and "SUMMARY" not in tail:
            return False
    except Exception:
        return False

    return True

# ----------------------- Session Process -----------------------
def process_session(session_dir: Path, hands_detector):
    cam2 = session_dir / "cam2"
    if not cam2.is_dir():
        print(f"[SKIP] {session_dir.name}: missing cam2/")
        return

    color_dir = find_color_dir(cam2)
    if color_dir is None:
        print(f"[SKIP] {session_dir.name}: no folder under cam2/ with frame_*.png")
        return

    frames = list_frames(color_dir)
    if not frames:
        print(f"[SKIP] {session_dir.name}: found color dir but no frames")
        return

    # >>> NEW: skip if already processed
    if session_already_processed(cam2, color_dir):
        print(f"[SKIP] {session_dir.name}: already processed (CSV+color_mp+SUMMARY present).")
        return
    # <<<

    parent = color_dir.parent  # cam2/
    out_preview_dir = parent / "sample_color_mp"
    out_logs_dir = parent / "logs"
    log = DebouncedLogger(out_logs_dir / "hand_detection.log")

    # Phase 1
    log.info(f"[{session_dir.name}] Phase 1: Preview first {FIRST_TRUSTED_N} frames in {out_preview_dir} "
             f"(overlap threshold {WRIST_SEP_PX:.2f}px)")
    cached_firstN = phase1_preview_first_n(frames, out_preview_dir, hands_detector, log)
    log.info(f"[{session_dir.name}] Preview ready. Inspect 'sample_color_mp/'.")
    log.write()

    # Wait
    # prompt = (f'[{session_dir.name}] Enter "same" to keep labels, or "flip" to swap L/R for the first '
    #           f'{FIRST_TRUSTED_N} frames: ')
    # choice = input(prompt).strip().lower()
    choice = "flip"  # or "same"

    # Phase 2
    log.info(f"[{session_dir.name}] Phase 2: Final outputs with user choice = {choice}")
    phase2_full_run(frames, cached_firstN, choice, parent, hands_detector, log)

    print(f"[DONE] {session_dir.name}")


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Batch two-phase hand labeling over sessions 1..22 (cam2 only)")
    ap.add_argument("--root_dir", required=True, help="Path to directory containing session folders 1..22")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    # Gather numeric session dirs (supports 01..22 and 1..22). We'll process in numeric order and silently skip missing ones.
    session_dirs = sorted(
        [p for p in root_dir.iterdir()
         if p.is_dir() and re.fullmatch(r"\d+", p.name) and 1 <= int(p.name) <= 22],
        key=lambda p: int(p.name)
    )
    if not session_dirs:
        raise RuntimeError(f"No numeric session folders (1..22) found under {root_dir}")

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands_detector:
        for sdir in session_dirs:
            print(f"\n===== Processing session: {sdir.name} =====")
            process_session(sdir, hands_detector)


if __name__ == "__main__":
    main()
