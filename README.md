# 📦 SPARC Data Collection Pipeline

This repository automates synchronized **RGB-D + Audio** data collection from up to **4 fixed RealSense D435i cameras**, saving:

- 📹 RGB + Depth streams (`.bag` raw files)
- 🔊 Multi-mic audio recordings (simultaneous `.wav` per mic)
- 🧠 Per-camera intrinsics (`.txt` and `.json`)
- 📊 System performance logs (CPU, RAM, Disk I/O)
- 🖼 Live RGB stream previews before recording
- 🔁 Playback with play/pause + depth/color toggle
- 💾 Data transfer and backup with SHA256 verification

---

## 🚀 Features

- 🔌 Plug in any **N (1–4)** of 4 known RealSense cameras
- 🎙 Records **multi-mic audio** in sync with video
- 🧾 Saves **raw .bag** files using RealSense SDK for max compatibility
- 🎥 **Preview and playback** with play/pause & depth/color toggle
- 📝 Per-camera `camera_info_<serial>.txt` and `camera_intrinsics_<serial>.json`
- 📊 Logs system CPU, RAM, and Disk I/O
- ⏱ Supports **timed recording** via `./record.sh <minutes>`
- 🧮 Adds buffer time to audio to prevent early cutoff
- 🕒 One-line live timer display during recording
- 🧪 SHA256 checksum verification during backup (`transfer.sh`)
- ⛔ Prevents preview–recording conflicts

---

## 🧰 Setup Instructions

### 1. Clone the repo & install dependencies

```bash
git clone <your_repo_url>
cd SPARC-Project
conda create -n data-pipeline python=3.10 -y
conda activate data-pipeline
pip install -r requirements.txt
conda deactivate
```

---

## 📁 File Overview

| File                            | Description |
|---------------------------------|-------------|
| `record.sh`                     | 🔴 Main script: records video + audio + system |
| `preview.sh`                    | 👁 Live preview with dropped frames and FPS |
| `playback.sh`                   | 🔁 Playback grid + system monitor visualization |
| `transfer.sh`                   | 💾 Copies latest session to external drive (with checksum) |
| `camera_serials.txt`            | 📋 Maps `cam1–cam4` to RealSense serials |
| `scripts/record_realsense.py`  | 🎥 Saves `.bag` + intrinsics in `.txt` + `.json` |
| `scripts/record_audio_mics.py` | 🔊 Records `.wav` from physical mics |
| `scripts/monitor_resources.py` | 📊 Logs CPU/RAM/Disk usage |
| `scripts/plot_monitor_log.py`  | 📈 Displays resource usage from logs |
| `scripts/realsense_preview_grid.py` | 👁 Playback viewer with per-cam control |
| `scripts/preview_all_cams.py`  | 🖼 Live RGB preview with overlay info |

---

## 📝 `camera_serials.txt` Format

```
cam1:902322073807
cam2:942422061612
cam3:947722072361
cam4:938322070387
```

Edit this file with the actual serial numbers for your RealSense cameras.

---

## ▶️ Usage Instructions

### 1. Preview Cameras (Optional)

```bash
./preview.sh
```

- ✅ Live grid view of all connected cams
- ✅ Displays FPS + dropped frames
- ✅ ESC to exit
- ⛔ Aborts if recording is in progress

---

### 2. Start Full Recording

```bash
./record.sh 10
```

- Records for 10 minutes (video + audio)
- Adds audio buffer (5s/min) to ensure sync
- Shows a live one-line timer while recording
- Intrinsics stored for each cam (`.txt` and `.json`)
- CPU/RAM/Disk usage logged in background

---

### 3. Output Folder Structure

```
realsense_recording_<timestamp>/
├── cam1/
│   ├── cam1_<timestamp>.bag
│   ├── camera_info_<serial>.txt
│   ├── camera_intrinsics_<serial>.json
├── cam2/
│   └── ...
├── audio/
│   ├── mic_hw20_<timestamp>.wav
│   ├── mic_hw30_<timestamp>.wav
├── system_monitor_<timestamp>.log
```

---

### 4. Playback & System Plot

```bash
./playback.sh
```

- ✅ Displays `.bag` video stream(s) in grid
- ✅ Shows system resource usage chart
- 🎛 Controls:
  - `SPACE` or `p`: pause/resume playback
  - `1`, `2`, `3`, ...: toggle depth/color for individual cams

---

### 5. Transfer Session to External HDD (Optional)

```bash
./transfer.sh
```

- 🧠 Auto-detects latest `realsense_recording_*` folder
- Prompts for naming (month + serial + date)
- Copies folder to:
  ```
  /media/robotics/One\ Touch/SPARC-Data/April_Pilot/
  ```
- 🔐 Computes & verifies SHA256 checksum for full folder integrity

---

## ✅ Tips

- All scripts support `Ctrl+C` safe termination
- Run from root folder (`SPARC-Project/`) to avoid path issues
- Keep terminal open during recording for timer display

---

## 🔧 Dependencies

All dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

---

For support, contact the Robotics Lab at IIT Gandhinagar.