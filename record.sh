#!/bin/bash

# ---------------------------------------------------
# Usage: ./record.sh [duration_minutes]
# Example: ./record.sh 5 → records for 5 minutes
# ---------------------------------------------------

if [ -z "$1" ]; then
  echo "[USAGE] ./record.sh <duration_in_minutes>"
  exit 1
fi

duration_minutes=$1
duration_seconds=$((duration_minutes * 60))

timestamp=$(date +%Y%m%d_%H%M%S)
base_dir="/media/hpm_mv_2/One Touch/SPARC/realsense_recording_$timestamp"
mkdir -p "$base_dir"
cd "$base_dir"

# ⛔ Detect preview session
if pgrep -f "preview_all_cams.py" > /dev/null; then
  echo "[⚠️  WARNING] Detected preview session is running."
  read -p "Do you want to auto-terminate it and proceed? (y/N): " yn
  if [[ "$yn" =~ ^[Yy]$ ]]; then
    pkill -f "preview_all_cams.py"
    sleep 2
  else
    echo "[INFO] Aborting recording."
    exit 1
  fi
fi

# Step 1: Generate camera launch file
echo "[INFO] Generating multi_camera.launch..."
python3 ~/launch_generate.py --num-cameras 3

# Step 2: Start monitoring + diagnostics
python3 ~/monitor_resources.py > "$base_dir/system_monitor_$timestamp.log" &
monitor_pid=$!

python3 ~/diagnostics_logger.py "$base_dir" &
diag_pid=$!

roslaunch multi_camera.launch &
ros_pid=$!

# Step 3: Wait for cameras and start timed rosbag record
sleep 5
echo "[INFO] Starting rosbag recording for $duration_minutes minutes..."
timeout "$duration_seconds"s python3 ~/rosbag_record_per_camera.py "$base_dir" &
rosbag_pid=$!

# Step 4: Wait for duration or user interrupt
trap_handler() {
  echo ""
  echo "[⚠️  INTERRUPTED] Stopping all processes early..."
  kill $rosbag_pid $ros_pid $diag_pid $monitor_pid 2>/dev/null
  exit 0
}

trap trap_handler SIGINT

echo "[INFO] Recording... (Ctrl+C to stop early)"
sleep "$duration_seconds"

# Step 5: Stop all remaining processes
echo "[INFO] Time's up! Stopping all processes..."
kill $rosbag_pid $ros_pid $diag_pid $monitor_pid 2>/dev/null

echo "[✅] Recording complete. Data saved in $base_dir"
