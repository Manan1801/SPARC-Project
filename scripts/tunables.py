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
# Example device IDs (use `arecord -l` to list):
# 'hw:<card_number>,<device_number>'
VALID_MIC_IDS = [
    'hw:2,0',
    'hw:3,0',
    ]