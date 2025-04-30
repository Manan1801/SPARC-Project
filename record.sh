#!/bin/bash

# ---------------------------------------------------
# Usage: ./record.sh [duration_minutes]
# Example: ./record.sh 5 → records for 5 minutes
# ---------------------------------------------------

if [ -z "$1" ]; then
  echo "[ERROR] Duration missing. Usage: ./record.sh <duration_in_minutes>"
  exit 1
fi

duration_minutes=$1
timestamp=$(date +%Y%m%d_%H%M%S)
base_dir="$HOME/realsense_recording_$timestamp"
echo "[INFO] 📁 Creating output directory: $base_dir"
mkdir -p "$base_dir" || { echo "[ERROR] Failed to create directory: $base_dir"; exit 1; }
cd "$base_dir" || { echo "[ERROR] Cannot access: $base_dir"; exit 1; }

# 🎥 Ask for camera recording mode
echo ""
echo "--------------------------------------------------"
echo "🎥 CAMERA RECORDING MODE SELECTION"
echo "--------------------------------------------------"
echo "  [0] Save individual frames (capture_realsense.py)"
echo "  [1] Save as raw .bag files (record_realsense.py)"
echo "--------------------------------------------------"
read -p "[?] Enter 0 or 1 to select recording mode: " record_mode
echo ""

# ⛔ Check and kill preview if running
if pgrep -f "preview_all_cams.py" > /dev/null; then
  echo "[⚠️  WARNING] Preview already running."
  read -p "[?] Terminate it and continue? (y/N): " yn
  if [[ "$yn" =~ ^[Yy]$ ]]; then
    pkill -f "preview_all_cams.py"
    sleep 2
  else
    echo "[INFO] Aborting recording."
    exit 1
  fi
fi

# 🧠 Activate conda
eval "$(conda shell.bash hook)"
conda activate data-pipeline || exit 1

# 🧠 Launch system monitor
monitor_log="$base_dir/system_monitor_$(date +%Y%m%d_%H%M%S).log"
echo -e "\n[INFO] 📊 Starting system monitor..."
python3 ~/Desktop/SPARC-Project/scripts/monitor_resources.py > "$monitor_log" 2>&1 &
monitor_pid=$!

# ⏱ Buffer audio to 5s/min extra
audio_duration_sec=$((duration_minutes * 60 + duration_minutes * 5))

# 🎤 Launch audio mic recording
audio_dir="$base_dir/audio"
mkdir -p "$audio_dir"
echo -e "\n[INFO] 🎧 Starting audio recording for $duration_minutes min (+buffer = $audio_duration_sec sec)..."
python3 ~/Desktop/SPARC-Project/scripts/record_audio_mics.py "$audio_dir" --duration "$audio_duration_sec" &
audio_pid=$!

if [ "$record_mode" == "0" ]; then
  echo -e "\n[INFO] 🎥 Starting RealSense frame capturing..."
  python3 ~/Desktop/SPARC-Project/scripts/capture_realsense.py "$base_dir" --duration "$duration_minutes" &
  video_pid=$!
elif [ "$record_mode" == "1" ]; then
  echo -e "\n[INFO] 🎥 Starting RealSense .bag file recording..."
  python3 ~/Desktop/SPARC-Project/scripts/record_realsense.py "$base_dir" --duration "$duration_minutes" &
  video_pid=$!
else
  echo "[ERROR] Invalid input. Must be 0 (frames) or 1 (.bag). Exiting."
  kill $monitor_pid $audio_pid 2>/dev/null
  exit 1
fi

# 🕒 Timer — stays on one line until completion
display_timer() {
  duration_sec=$((duration_minutes * 60))
  for ((elapsed=0; elapsed<=duration_sec; elapsed++)); do
    mins=$((elapsed / 60))
    secs=$((elapsed % 60))
    printf "\r[⏳] Recording... Elapsed: %02d:%02d / %02d:00" "$mins" "$secs" "$duration_minutes"
    sleep 1
  done
  echo ""
}
display_timer &
timer_pid=$!

# 🚨 Setup interrupt cleanup
trap_handler() {
  echo -e "\n[⚠️] Interrupted. Cleaning up..."
  kill $monitor_pid $audio_pid $video_pid 2>/dev/null
  exit 0
}
trap trap_handler SIGINT

# ✅ Wait for all processes
wait $audio_pid $video_pid
kill $monitor_pid $timer_pid 2>/dev/null
echo -e "\n[✅] All recordings completed."
echo "[📁] Saved in: $base_dir"
