#!/usr/bin/env python3
# capture_worker.py
import time
import threading
from pathlib import Path
from typing import Optional, Dict
import queue as pyqueue

import numpy as np
import cv2
import pyrealsense2 as rs

from tunables import FALLBACK_DEPTH_SCALE
from control_flags import pause_event, stop_event
from logger_utils import DebouncedLogger
from types_shared import FramePacket
from camera_utils import write_camera_info  # assumed present in your modularization

# Keep a shared notion of cam2's depth scale for cross-cam consistency
_CAM2_DEPTH_SCALE_LOCK = threading.Lock()
_CAM2_DEPTH_SCALE: Optional[float] = None  # learned during run from cam2


def capture_worker(
    serial: str,
    cam_label: str,
    out_dir: Path,
    duration_sec: float,
    save_every: int,
    filters_on: bool,
    q_mov: Optional["pyqueue.Queue[FramePacket]"],
    q_emo: Optional["pyqueue.Queue[FramePacket]"],
    backpressure: str,
):
    """
    Captures color+depth from a RealSense, aligns depth to color, optionally filters,
    publishes FramePacket(s) to movement / emotion queues, and optionally saves raw frames.
    Honors global pause/stop (SPACE / ESC) via control_flags.
    """
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

    # Write camera info & intrinsics json (returns a scale; we may override below)
    info_txt = cam_dir / f"camera_info_{serial}.txt"
    info_json = cam_dir / f"camera_intrinsics_{serial}.json"
    intr_json, depth_scale_m = write_camera_info(profile, serial, cam_label, info_txt, info_json)

    # Prefer cam2's scale for others if available; ensure fallback constant
    with _CAM2_DEPTH_SCALE_LOCK:
        if cam_label == "cam2":
            _CAM2_DEPTH_SCALE = depth_scale_m
        else:
            if (depth_scale_m is None) or (depth_scale_m <= 0):
                if _CAM2_DEPTH_SCALE is not None:
                    depth_scale_m = _CAM2_DEPTH_SCALE
                else:
                    depth_scale_m = FALLBACK_DEPTH_SCALE

    # Use color intrinsics
    color_key = next((k for k in intr_json.keys() if k.startswith("color_")), None)
    if color_key is None:
        fx = 604.7
        fy = 604.9
        cx = 313.8
        cy = 252.7  # conservative fallback
    else:
        fx = float(intr_json[color_key]["fx"])
        fy = float(intr_json[color_key]["fy"])
        cx = float(intr_json[color_key]["cx"])
        cy = float(intr_json[color_key]["cy"])

    print(
        f"[INFO] 🎥 {cam_label} → depth_scale={depth_scale_m:.12f} m/unit | "
        f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
    )

    frame_id = 0
    active_elapsed = 0.0
    last_tick = time.monotonic()

    def _publish(qtarget: Optional["pyqueue.Queue[FramePacket]"], pkt: FramePacket):
        if qtarget is None:
            return
        try:
            if backpressure == "block":
                qtarget.put(pkt, timeout=0.01)
            else:
                qtarget.put_nowait(pkt)
        except pyqueue.Full:
            # drop-then-insert (keeps freshest)
            try:
                _ = qtarget.get_nowait()
            except Exception:
                pass
            try:
                qtarget.put_nowait(pkt)
            except Exception:
                pass

    try:
        while not stop_event.is_set() and active_elapsed < duration_sec:
            # Pause support (SPACE)
            if pause_event.is_set():
                # Do not accumulate active time while paused
                last_tick = time.monotonic()
                time.sleep(0.02)
                continue

            try:
                frames = pipeline.wait_for_frames()
            except Exception:
                # transient device hiccup
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
                cv2.imwrite(
                    str(depth_dir / f"frame_{frame_id:06d}.tiff"),
                    depth_img,
                    [cv2.IMWRITE_TIFF_COMPRESSION, 1],
                )

            # Publish to processing pipelines (if enabled for this cam)
            pkt = FramePacket(
                cam_label=cam_label,
                frame_id=frame_id,
                t_ns=time.monotonic_ns(),
                rs_ts_ms=c.get_timestamp() if hasattr(c, "get_timestamp") else 0.0,
                color=color_img,
                depth=depth_img,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_scale_m=depth_scale_m,
            )
            _publish(q_mov, pkt)
            _publish(q_emo, pkt)

            frame_id += 1

            now = time.monotonic()
            active_elapsed += (now - last_tick)
            last_tick = now

            if stop_event.is_set():
                break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        logger.info(f"Finished capture {cam_label}: frames={frame_id}, active_elapsed={active_elapsed:.2f}s")
        logger.periodic_flush(force=True)
