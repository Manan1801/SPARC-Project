#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate data-pipeline

# ------------------------------------------------------------------------------------
# ⛔ Detect if a recording session is in progress to prevent camera conflicts
# ------------------------------------------------------------------------------------
if pgrep -f "rosbag_record_per_camera.py" > /dev/null || pgrep -f "record.sh" > /dev/null; then
  echo "[⚠️  WARNING] Detected that a recording is already in progress (rosbag or record.sh)."
  echo "Running preview while recording may cause frame drops, USB overload, or data loss."
  read -p "Do you still want to launch the preview? (y/N): " yn
  if [[ ! "$yn" =~ ^[Yy]$ ]]; then
    echo "[INFO] Aborting preview for safety."
    exit 1
  fi
fi

echo "[INFO] Starting live camera previews..."
python3 ~/Desktop/SPARC-Project/scripts/preview_all_cams.py
