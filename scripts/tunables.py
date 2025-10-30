#!/usr/bin/env python3

import os
from pathlib import Path

# ---------- Paths ----------
CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

# ---------- Hand landmarks & processing ----------
WRIST_ID = 0
NUM_LANDMARKS = 21
WRIST_SEP_PX = 30
KEYPOINTS_MOV = [0, 1, 2, 3, 4]  # wrist + thumb ids
HANDS = ("L", "R")

# Anchor reset if both-hands reference is lost for this many processed frames
ANCHOR_MISS_RESET = 300

# ---------- Colors (BGR) ----------
COLOR_LEFT  = (0, 255, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)

# ---------- Chart & CSV cadences ----------
PLOT_UPDATE_INTERVAL = 7        # frames
CUM_CSV_INTERVAL_SEC = 10.0     # seconds

# ---------- Depth scale ----------
FALLBACK_DEPTH_SCALE = 0.0010000000474974513  # meters per unit

# ---------- Misc ----------
DEFAULT_CELL_W = 640
DEFAULT_CELL_H = 360

# ---------- Audio device whitelist ----------
VALID_MIC_IDS = [
    'hw:2,0',
    'hw:3,0',
]

# ================== Event trigger tunables (NEW) =======================================
# Throttle logs/overlays to once per this many seconds (per slot)
SPEED_TRIGGER_CADENCE_WINDOW_S = 30.0  # adjust as you like

# Default path to the expected reference CSV (can be overridden per trigger)
# CSV headers: time_s, lower_bound, upper_bound
# SPEED_TRIGGER_REFCSV_PATH = "~/Desktop/SPARC-Project/right_wrist_speed_bounds.csv"
SPEED_TRIGGER_REFCSV_PATH = "/home/robotics/Desktop/SPARC-Project/right_wrist_speed_bounds.csv"

# How long the "High/Low Speed" label stays visible on the PreviewGrid
SPEED_TRIGGER_OVERLAY_TTL_S = 6.0  # 5–7 seconds as requested

# ========================================================================================
