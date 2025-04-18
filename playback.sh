#!/bin/bash

# ------------------------------------------------------------
# Usage: ./playback.sh
# Plays back RealSense .bag files recorded using SDK
# ------------------------------------------------------------

# Activate environment
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
eval "$(conda shell.bash hook)"
conda activate data-pipeline

echo "[INFO] Locating latest realsense_recording_* directory..."
latest_dir=$(ls -td ~/realsense_recording_* | head -1)

if [ ! -d "$latest_dir" ]; then
  echo "[ERROR] No recording directory found."
  exit 1
fi

echo "[INFO] Found recording: $latest_dir"

# ------------------------------------------
# 1. Detect .bag files in: $latest_dir/cam*/cam*.bag
# ------------------------------------------
echo "[INFO] Searching for .bag files in camera folders..."
declare -a bag_files=()
for cam_dir in "$latest_dir"/cam*/; do
  bag=$(find "$cam_dir" -maxdepth 1 -type f -name "*.bag" | head -1)
  if [ -f "$bag" ]; then
    echo "[INFO] Found: $bag"
    bag_files+=("$bag")
  fi
done

if [ ${#bag_files[@]} -eq 0 ]; then
  echo "[ERROR] No RealSense .bag files found in cam folders."
  exit 1
fi

# ------------------------------------------
# 2. Plot system usage if available
# ------------------------------------------
sys_log=$(find "$latest_dir" -type f -name "system_monitor_*.log" ! -empty | tail -1)
if [ -f "$sys_log" ]; then
  echo "[INFO] Plotting system usage from $sys_log..."
  python3 ~/Desktop/SPARC-Project/scripts/plot_monitor_log.py "$sys_log"
else
  echo "[WARN] No valid system_monitor log found."
fi

# ------------------------------------------
# 3. Preview .bag files (interactive + toggle control)
# ------------------------------------------
echo "[INFO] Starting RealSense bag preview with controls..."
python3 ~/Desktop/SPARC-Project/scripts/realsense_preview_grid.py "${bag_files[@]}"

echo "[✅] Playback and monitoring completed."
