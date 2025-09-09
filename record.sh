#!/bin/bash

# ---------------------------------------------------
# Usage: ./record.sh [duration_minutes]
# Controls:
#   SPACE = Start / Pause / Resume (cooperative: USR1/USR2)
#   ESC   = Stop early (graceful: INT)
# ---------------------------------------------------

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "[ERROR] Duration missing. Usage: ./record.sh <duration_in_minutes>"
  exit 1
fi

duration_minutes="$1"
duration_sec=$((duration_minutes * 60))
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
if [[ "$record_mode" != "0" && "$record_mode" != "1" ]]; then
  echo "[ERROR] Invalid input. Must be 0 (frames) or 1 (.bag)."
  exit 1
fi

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

# 🧠 Activate conda (silence libmamba/libarchive errors)
eval "$(conda shell.bash hook)"
export CONDA_SOLVER=classic
conda activate data-pipeline || exit 1

# Paths for quiet logs
audio_log="$base_dir/audio.log"
video_log="$base_dir/video.log"
monitor_log="$base_dir/system_monitor_$(date +%Y%m%d_%H%M%S).log"

# 🧠 Launch system monitor (already logged to file)
echo -e "\n[INFO] 📊 Starting system monitor (log: $monitor_log)"
python3 ~/Desktop/SPARC-Project/scripts/monitor_resources.py > "$monitor_log" 2>&1 &
monitor_pid=$!

# ⏱ Buffer audio to 5s/min extra (ACTIVE time)
audio_duration_sec=$((duration_minutes * 60 + duration_minutes * 5))

# PIDs filled after start
audio_pid=""
video_pid=""

print_controls() {
  cat <<EOF

────────────────────────────────────────────────────────
🎛️  CONTROLS (Recording does NOT start until you press SPACE)

   SPACE → Start / Pause / Resume
   ESC   → Stop early (graceful finalize & save)

Target ACTIVE duration: ${duration_minutes} min (${duration_sec}s)
Output directory      : $base_dir
Mode                  : $( [[ "$record_mode" == "0" ]] && echo "Frames (capture_realsense.py)" || echo ".bag (record_realsense.py)" )
Child logs            : audio → $audio_log
                        video → $video_log
────────────────────────────────────────────────────────

EOF
}

start_recording() {
  # 🎤 Audio (quiet → audio.log)
  audio_dir="$base_dir/audio"
  mkdir -p "$audio_dir"
  echo -e "\n[INFO] 🎧 Starting audio (ACTIVE ${duration_sec}s). Logs: $audio_log"
  python3 ~/Desktop/SPARC-Project/scripts/record_audio_mics.py "$audio_dir" --duration "$audio_duration_sec" \
    >"$audio_log" 2>&1 &
  audio_pid=$!

  # 🎥 Video (quiet → video.log)
  if [ "$record_mode" == "0" ]; then
    echo -e "[INFO] 🎥 Starting RealSense frames. Logs: $video_log"
    python3 ~/Desktop/SPARC-Project/scripts/capture_realsense.py "$base_dir" --duration "$duration_minutes" \
      >"$video_log" 2>&1 &
    video_pid=$!
  else
    echo -e "[INFO] 🎥 Starting RealSense .bag. Logs: $video_log"
    python3 ~/Desktop/SPARC-Project/scripts/record_realsense.py "$base_dir" --duration "$duration_minutes" \
      >"$video_log" 2>&1 &
    video_pid=$!
  fi
}

# ── Cooperative pause/resume via USR1/USR2 ────────────────────────────────────
pause_children() {
  [[ -n "$audio_pid" ]] && kill -USR1 "$audio_pid" 2>/dev/null || true
  [[ -n "$video_pid" ]] && kill -USR1 "$video_pid" 2>/dev/null || true
}

resume_children() {
  [[ -n "$audio_pid" ]] && kill -USR2 "$audio_pid" 2>/dev/null || true
  [[ -n "$video_pid" ]] && kill -USR2 "$video_pid" 2>/dev/null || true
}

# ── Graceful stop (SIGINT) then gentle fallback ───────────────────────────────
graceful_stop_children() {
  for pid in "$audio_pid" "$video_pid"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  end=$((SECONDS+6))
  for pid in "$audio_pid" "$video_pid"; do
    if [[ -n "${pid}" ]]; then
      while kill -0 "$pid" 2>/dev/null && [ $SECONDS -lt $end ]; do sleep 0.2; done
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

cleanup_and_exit() {
  echo -e "\n[INFO] Cleaning up..."
  graceful_stop_children
  kill "$monitor_pid" 2>/dev/null || true
  echo -e "\n[✅] Recording finalized."
  echo "[📁] Saved in: $base_dir"
  echo "[ℹ️ ] Logs:  $audio_log"
  echo "             $video_log"
  exit "${1:-0}"
}

trap "echo -e '\n[⚠️] Interrupted (Ctrl+C). Finalizing...'; cleanup_and_exit 0" SIGINT

print_controls

state="idle"      # idle | running | paused
elapsed=0         # ACTIVE seconds
last_tick=$(date +%s)

# ── Single-loop UI: poll key + update timer (quiet) ───────────────────────────
while true; do
  # Non-blocking single-char read with a small timeout
  if IFS= read -rsn1 -t 0.1 key; then
    if [[ "$key" == $'\x1b' ]]; then
      echo -e "\n[⛔] ESC → Finalizing and saving..."
      cleanup_and_exit 0
    elif [[ "$key" == " " ]]; then
      case "$state" in
        idle)
          start_recording
          state="running"
          last_tick=$(date +%s)
          echo "[▶️ ] Started."
          ;;
        running)
          pause_children
          state="paused"
          echo -e "\n[⏸️ ] Paused at $(printf '%02d:%02d' $((elapsed/60)) $((elapsed%60)))"
          ;;
        paused)
          resume_children
          state="running"
          last_tick=$(date +%s)
          echo "[⏯️ ] Resumed."
          ;;
      esac
    fi
  fi

  # Timer / health — single, quiet status line
  if [[ "$state" == "running" ]]; then
    now=$(date +%s)
    if (( now > last_tick )); then
      delta=$((now - last_tick))
      elapsed=$((elapsed + delta))
      last_tick=$now
    fi
    printf "\r[⏳] Recording... Elapsed: %02d:%02d / %02d:00   (logs: audio.log | video.log) " \
      $((elapsed/60)) $((elapsed%60)) "$duration_minutes"

    # Health checks
    if [[ -n "$video_pid" ]] && ! kill -0 "$video_pid" 2>/dev/null; then
      echo -e "\n[ERROR] Video process exited unexpectedly. See $video_log"
      cleanup_and_exit 1
    fi
    if [[ -n "$audio_pid" ]] && ! kill -0 "$audio_pid" 2>/dev/null; then
      echo -e "\n[ERROR] Audio process exited unexpectedly. See $audio_log"
      cleanup_and_exit 1
    fi

    # Done?
    if (( elapsed >= duration_sec )); then
      echo -e "\n[✅] Target ACTIVE duration reached."
      cleanup_and_exit 0
    fi
  elif [[ "$state" == "paused" ]]; then
    printf "\r[⏸️ ] Paused at %02d:%02d / %02d:00  (SPACE=Resume, ESC=Stop)                  " \
      $((elapsed/60)) $((elapsed%60)) "$duration_minutes"
  else
    printf "\r[🕹️ ] Press SPACE to start • ESC to stop                                         "
  fi
done
