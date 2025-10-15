#!/usr/bin/env bash
# run_rt_pipeline.sh — Interactive wrapper for realtime_capture.py

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="${SCRIPT_PATH:-./scripts/realtime_capture.py}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] python not found (looked for '$PYTHON_BIN'). Set PYTHON_BIN env var if needed."
  exit 1
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] Script not found at: $SCRIPT_PATH"
  echo "        Set SCRIPT_PATH env var or place this launcher next to realtime_capture.py"
  exit 1
fi

ask() {
  local prompt="${1}"; shift || true
  local default="${1:-}"; shift || true
  local reply
  if [[ -n "$default" ]]; then
    read -r -p "$prompt [$default]: " reply || true
    echo "${reply:-$default}"
  else
    read -r -p "$prompt: " reply || true
    echo "$reply"
  fi
}

ask_yn() {
  local prompt="$1"; shift
  local def="${1:-y}"
  local ans
  while true; do
    ans="$(ask "$prompt (y/n)" "$def")"
    case "${ans,,}" in
      y|yes) echo "y"; return 0 ;;
      n|no)  echo "n"; return 0 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

ask_choice() {
  local prompt="$1"; shift
  local choices="$1"; shift
  local def="${1:-}"
  local ans
  IFS='|' read -r -a opts <<< "$choices"
  while true; do
    ans="$(ask "$prompt (${choices//|//})" "$def")"
    for o in "${opts[@]}"; do
      if [[ "${ans,,}" == "${o,,}" ]]; then
        echo "$o"; return 0
      fi
    done
    echo "Please choose one of: $choices"
  done
}

cat <<'BANNER'
────────────────────────────────────────────────────────────────
   Real-Time Unified Pipeline — Interactive Launcher
   (captures all cams; optionally process selected cams)
────────────────────────────────────────────────────────────────
Controls during run:
  • Pause/Resume : SPACE
  • Stop         : ESC or Ctrl+C
(Advanced: you can still send SIGUSR1/SIGUSR2 if needed.)
────────────────────────────────────────────────────────────────
BANNER

OUTPUT_DIR="$(ask "Output directory" "./run_$(date +%Y%m%d_%H%M%S)")"
DUR_SEC="$(ask "Active recording duration in seconds" "60")"

echo
echo "Which camera(s) to PROCESS (capture still happens for all)"
echo "  - Enter comma-separated labels (e.g., cam2 or cam1,cam3)"
echo "  - Enter 'all' to process all"
echo "  - Enter 'none' for capture-only (no RT processing)"
PROC_INPUT="$(ask "Process cams" "cam2")"

declare -a PROC_CAMS_ARR=()
PROC_MODE="some"
case "${PROC_INPUT,,}" in
  ""|"all") PROC_MODE="all" ;;
  "none")   PROC_MODE="none" ;;
  *)
    IFS=',' read -r -a rawcams <<< "$PROC_INPUT"
    for c in "${rawcams[@]}"; do
      c_trim="$(echo "$c" | xargs)"
      [[ -n "$c_trim" ]] && PROC_CAMS_ARR+=("$c_trim")
    done
    if [[ "${#PROC_CAMS_ARR[@]}" -eq 0 ]]; then PROC_MODE="all"; fi
    ;;
esac

SAVE_EVERY="$(ask "Save raw frames? Enter N (save every Nth frame; 0 = OFF)" "1")"
FILTERS="$(ask_choice "Depth filters" "on|off" "off")"
VIZ_LIVE="$(ask_choice "Live preview window" "off|window" "off")"
FORCE_FLIP="$(ask_choice "Handedness flip baseline" "flip|same" "flip")"

AUDIO_ENABLE="$(ask_yn "Record microphones too?" "n")"
if [[ "$AUDIO_ENABLE" == "y" ]]; then
  RATE="$(ask_choice "Audio sample rate" "44100|48000" "44100")"
else
  RATE="44100"
fi

ADVANCED="$(ask_yn "Enter Advanced Controls?" "n")"

STRIDE="1"
BKP_POLICY="drop-latest"
VIZ_SAVE_EVERY="3"
CSV_FLUSH="30"
LOG_FLUSH_SEC="5"
HEALTH_SEC="5"
AUDIO_OUT="$OUTPUT_DIR/audio"
AUDIO_DUR="$DUR_SEC"

if [[ "$ADVANCED" == "y" ]]; then
  echo
  echo "── Advanced Controls ──────────────────────────────"
  STRIDE="$(ask "Processing stride (process every Nth frame)" "1")"
  BKP_POLICY="$(ask_choice "Frame backpressure policy" "drop-latest|block" "drop-latest")"
  VIZ_SAVE_EVERY="$(ask "Save annotated previews every N frames (0 = OFF)" "3")"
  CSV_FLUSH="$(ask "CSV flush interval (frames)" "30")"
  LOG_FLUSH_SEC="$(ask "Logger flush interval (seconds)" "5")"
  HEALTH_SEC="$(ask "Health/telemetry interval (seconds)" "5")"
  if [[ "$AUDIO_ENABLE" == "y" ]]; then
    AUDIO_OUT="$(ask "Audio output directory" "$AUDIO_OUT")"
    AUDIO_DUR="$(ask "Audio active duration in seconds" "$AUDIO_DUR")"
  fi
  echo "───────────────────────────────────────────────────"
fi

declare -a CMD
CMD+=("$PYTHON_BIN" "$SCRIPT_PATH")
CMD+=("--output-dir" "$OUTPUT_DIR")
CMD+=("--duration-sec" "$DUR_SEC")
CMD+=("--save-every" "$SAVE_EVERY")
CMD+=("--filters" "$FILTERS")
CMD+=("--viz-live" "$VIZ_LIVE")
CMD+=("--force-flip" "$FORCE_FLIP")
CMD+=("--stride" "$STRIDE")
CMD+=("--backpressure" "$BKP_POLICY")
CMD+=("--viz-save-every" "$VIZ_SAVE_EVERY")
CMD+=("--csv-flush" "$CSV_FLUSH")
CMD+=("--log-flush-sec" "$LOG_FLUSH_SEC")
CMD+=("--health-interval-sec" "$HEALTH_SEC")

if [[ "$PROC_MODE" == "none" ]]; then
  CMD+=("--process-cams")
elif [[ "$PROC_MODE" == "some" ]]; then
  CMD+=("--process-cams")
  for lab in "${PROC_CAMS_ARR[@]}"; do CMD+=("$lab"); done
fi

if [[ "$AUDIO_ENABLE" == "y" ]]; then
  CMD+=("--audio-out" "$AUDIO_OUT")
  CMD+=("--audio-duration-sec" "$AUDIO_DUR")
  CMD+=("--rate" "$RATE")
else
  CMD+=("--audio-duration-sec" "0")
fi

echo
echo "──────────────── RUN SUMMARY ────────────────"
echo "Output dir         : $OUTPUT_DIR"
echo "Duration (sec)     : $DUR_SEC"
echo "Process cams       : ${PROC_MODE^^}"
if [[ "$PROC_MODE" == "some" ]]; then
  echo "  • Labels         : ${PROC_CAMS_ARR[*]}"
fi
echo "Save-every (raw)   : $SAVE_EVERY"
echo "Depth filters      : $FILTERS"
echo "Live preview       : $VIZ_LIVE"
echo "Force flip         : $FORCE_FLIP"
echo "Stride             : $STRIDE"
echo "Backpressure       : $BKP_POLICY"
echo "Viz save-every     : $VIZ_SAVE_EVERY"
echo "CSV flush (frames) : $CSV_FLUSH"
echo "Log flush (sec)    : $LOG_FLUSH_SEC"
echo "Health interval    : $HEALTH_SEC"
echo "Audio enabled      : $([[ "$AUDIO_ENABLE" == "y" ]] && echo "yes" || echo "no")"
if [[ "$AUDIO_ENABLE" == "y" ]]; then
  echo "  • Rate (Hz)      : $RATE"
  echo "  • Audio out      : $AUDIO_OUT"
  echo "  • Audio dur (s)  : $AUDIO_DUR"
fi
echo "─────────────────────────────────────────────"
echo "Command to run:"
printf '  %q ' "${CMD[@]}"; echo
echo "─────────────────────────────────────────────"

read -r -p "Proceed? (y/n) [y]: " CONFIRM
CONFIRM="${CONFIRM:-y}"
if [[ "${CONFIRM,,}" != "y" ]]; then
  echo "Aborted."
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
if [[ "$AUDIO_ENABLE" == "y" ]]; then mkdir -p "$AUDIO_OUT"; fi

exec "${CMD[@]}"
