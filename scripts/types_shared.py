#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

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
