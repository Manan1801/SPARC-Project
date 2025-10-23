#!/usr/bin/env bash
# realtime.sh — Interactive wrapper for realtime_capture.py (movement + emotion)

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
   (captures all cams; selectively process Movement and/or Emotion)
────────────────────────────────────────────────────────────────
Controls during run:
  • Pause/Resume : SPACE
  • Close grid   : q     (pipeline continues)
  • Reopen grid  : g     (press in terminal)
  • Stop         : ESC or Ctrl+C
────────────────────────────────────────────────────────────────
BANNER

OUTPUT_DIR="$(ask "Output directory" "./run_$(date +%Y%m%d_%H%M%S)")"
DUR_SEC="$(ask "Active recording duration in seconds" "60")"

echo
echo "Which camera(s) to PROCESS for MOVEMENT?"
echo "  - Enter comma-separated labels (e.g., cam2 or cam1,cam3)"
echo "  - Enter 'all' to process all"
echo "  - Enter 'none' for capture-only (no RT movement)"
PROC_MOV_INPUT="$(ask "Movement cams" "cam2")"

declare -a PROC_MOV_ARR=()
PROC_MOV_MODE="some"
case "${PROC_MOV_INPUT,,}" in
  ""|"all") PROC_MOV_MODE="all" ;;
  "none")   PROC_MOV_MODE="none" ;;
  *)
    IFS=',' read -r -a rawcams <<< "$PROC_MOV_INPUT"
    for c in "${rawcams[@]}"; do
      c_trim="$(echo "$c" | xargs)"
      [[ -n "$c_trim" ]] && PROC_MOV_ARR+=("$c_trim")
    done
    if [[ "${#PROC_MOV_ARR[@]}" -eq 0 ]]; then PROC_MOV_MODE="all"; fi
    ;;
esac

echo
echo "Which camera(s) to PROCESS for EMOTION (valence/arousal)?"
echo "  - Enter comma-separated labels (e.g., cam3 or cam1,cam2)"
echo "  - Enter 'none' to disable emotion processing"
PROC_EMO_INPUT="$(ask "Emotion cams" "none")"

declare -a PROC_EMO_ARR=()
PROC_EMO_MODE="some"
case "${PROC_EMO_INPUT,,}" in
  ""|"none") PROC_EMO_MODE="none" ;;
  *)
    IFS=',' read -r -a rawemo <<< "$PROC_EMO_INPUT"
    for c in "${rawemo[@]}"; do
      c_trim="$(echo "$c" | xargs)"
      [[ -n "$c_trim" ]] && PROC_EMO_ARR+=("$c_trim")
    done
    if [[ "${#PROC_EMO_ARR[@]}" -eq 0 ]]; then PROC_EMO_MODE="none"; fi
    ;;
esac

SAVE_EVERY="$(ask "Save raw frames? Enter N (save every Nth frame; 0 = OFF)" "1")"
FILTERS="$(ask_choice "Depth filters" "on|off" "off")"
VIZ_LIVE="$(ask_choice "Live preview window (unified 2x2 grid)" "off|window" "off")"
FORCE_FLIP="$(ask_choice "Handedness flip baseline (movement)" "flip|same" "flip")"

AUDIO_ENABLE="$(ask_yn "Record microphones too?" "n")"
if [[ "$AUDIO_ENABLE" == "y" ]]; then
  RATE="$(ask_choice "Audio sample rate" "44100|48000" "44100")"
else
  RATE="44100"
fi

ADVANCED="$(ask_yn "Enter Advanced Controls?" "n")"

# Movement advanced defaults
STRIDE="1"
BKP_POLICY="drop-latest"
VIZ_SAVE_EVERY="3"
CSV_FLUSH="30"
LOG_FLUSH_SEC="5"

# Emotion advanced defaults
EMO_HISTORY="240"
EMO_STRIDE="1"
EMO_CSV_FLUSH="30"

AUDIO_OUT="$OUTPUT_DIR/audio"
AUDIO_DUR="$DUR_SEC"

if [[ "$ADVANCED" == "y" ]]; then
  echo
  echo "── Advanced Controls (Movement) ──────────────"
  STRIDE="$(ask "Movement processing stride (every Nth frame)" "1")"
  BKP_POLICY="$(ask_choice "Frame backpressure policy" "drop-latest|block" "drop-latest")"
  VIZ_SAVE_EVERY="$(ask "Save annotated movement previews every N frames (0 = OFF)" "3")"
  CSV_FLUSH="$(ask "Movement CSV flush interval (frames)" "30")"
  LOG_FLUSH_SEC="$(ask "Logger flush interval (seconds)" "5")"
  echo "── Advanced Controls (Emotion) ───────────────"
  EMO_HISTORY="$(ask "Emotion plot history length (frames)" "240")"
  EMO_STRIDE="$(ask "Emotion processing stride (every Nth frame)" "1")"
  EMO_CSV_FLUSH="$(ask "Emotion CSV flush interval (frames)" "30")"
  if [[ "$AUDIO_ENABLE" == "y" ]]; then
    echo "── Advanced Controls (Audio) ─────────────────"
    AUDIO_OUT="$(ask "Audio output directory" "$AUDIO_OUT")"
    AUDIO_DUR="$(ask "Audio active duration in seconds" "$AUDIO_DUR")"
  fi
  echo "──────────────────────────────────────────────"
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

# Movement cams:
if [[ "$PROC_MOV_MODE" == "none" ]]; then
  CMD+=("--process-mov-cams")
elif [[ "$PROC_MOV_MODE" == "some" ]]; then
  CMD+=("--process-mov-cams")
  for lab in "${PROC_MOV_ARR[@]}"; do CMD+=("$lab"); done
fi
# If PROC_MOV_MODE == "all", omit the flag (Python defaults to ALL)

# Emotion cams:
if [[ "$PROC_EMO_MODE" == "some" ]]; then
  CMD+=("--process-emo-cams")
  for lab in "${PROC_EMO_ARR[@]}"; do CMD+=("$lab"); done
fi
# If PROC_EMO_MODE == "none", omit the flag (Python defaults to NONE for emotion)

# Emotion tunables
CMD+=("--emo-history" "$EMO_HISTORY")
CMD+=("--emo-stride" "$EMO_STRIDE")
CMD+=("--emo-csv-flush" "$EMO_CSV_FLUSH")

# Audio
if [[ "$AUDIO_ENABLE" == "y" ]]; then
  CMD+=("--audio-out" "$AUDIO_OUT")
  CMD+=("--audio-duration-sec" "$AUDIO_DUR")
  CMD+=("--rate" "$RATE")
else
  CMD+=("--audio-duration-sec" "0")
fi

echo
echo "──────────────── RUN SUMMARY ────────────────"
echo "Output dir              : $OUTPUT_DIR"
echo "Duration (sec)          : $DUR_SEC"
echo "Movement cams           : ${PROC_MOV_MODE^^}"
if [[ "$PROC_MOV_MODE" == "some" ]]; then
  echo "  • Labels              : ${PROC_MOV_ARR[*]}"
fi
echo "Emotion cams            : ${PROC_EMO_MODE^^}"
if [[ "$PROC_EMO_MODE" == "some" ]]; then
  echo "  • Labels              : ${PROC_EMO_ARR[*]}"
fi
echo "Save-every (raw)        : $SAVE_EVERY"
echo "Depth filters           : $FILTERS"
echo "Live preview            : $VIZ_LIVE"
echo "Force flip (movement)   : $FORCE_FLIP"
echo "Stride (movement)       : $STRIDE"
echo "Backpressure            : $BKP_POLICY"
echo "Viz save-every (move)   : $VIZ_SAVE_EVERY"
echo "CSV flush (move)        : $CSV_FLUSH"
echo "Log flush (sec)         : $LOG_FLUSH_SEC"
echo "Emotion history (frames): $EMO_HISTORY"
echo "Emotion stride          : $EMO_STRIDE"
echo "Emotion CSV flush       : $EMO_CSV_FLUSH"
echo "Audio enabled           : $([[ "$AUDIO_ENABLE" == "y" ]] && echo "yes" || echo "no")"
if [[ "$AUDIO_ENABLE" == "y" ]]; then
  echo "  • Rate (Hz)           : $RATE"
  echo "  • Audio out           : $AUDIO_OUT"
  echo "  • Audio dur (s)       : $AUDIO_DUR"
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
