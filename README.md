# 📦 SPARC Data Collection Pipeline

This repository automates synchronized **RGB-D + Audio** data collection from up to **4 fixed RealSense D435i cameras**, saving:

- 📹 RGB + Depth streams (as **`.bag`** or **per-frame PNG+TIFF**)
- 🔊 Multi-mic audio recordings (simultaneous `.wav` per mic)
- 🧠 Per-camera intrinsics (`.txt` and `.json`)
- 📊 System performance logs (CPU, RAM, Disk I/O)
- 🖼 Live RGB stream previews before recording
- 🔁 Playback with play/pause + depth/color toggle
- 💾 Data transfer and backup with **SHA256 verification** (single or batch mode)

---

## 🚀 Features

- 🔌 Plug in any **N (1–4)** of 4 known RealSense cameras
- 🎙 Records **multi-mic audio** in sync with video
- 💼 Supports **two camera recording modes**:
  - Mode 0 → Per-frame RGB+Depth as `.png`/`.tiff`
  - Mode 1 → Native `.bag` format using RealSense SDK
- 🧾 Saves per-camera `camera_info_<serial>.txt` and `camera_intrinsics_<serial>.json`
- 📺 **Preview and playback** with play/pause & depth/color toggle
- 📊 Logs system CPU, RAM, and Disk I/O
- ⏱ Supports **timed recording** via `./record.sh <minutes>`
- 🔊 Adds buffer time to audio to prevent early cutoff
- ⏲ One-line live timer display during recording
- 🔐 SHA256 checksum verification during transfer with retry + validation logic
- 🔄 Two transfer modes:
  - Mode 0: Archive, extract, and deep-verify
  - Mode 1: File-by-file copy with verification
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

| File                             | Description |
|----------------------------------|-------------|
| `record.sh`                      | 🔴 Main recording script with interactive mode selection |
| `preview.sh`                     | 👁 Live preview with dropped frames and FPS |
| `playback.sh`                    | 🔁 Playback grid + system monitor visualization |
| `transfer.sh`                    | 💾 Robust transfer with retry, checksum, archive/file modes |
| `camera_serials.txt`             | 📋 Maps `cam1–cam4` to RealSense serials |
| `scripts/capture_realsense.py`  | 📸 Frame-by-frame RGB+Depth recorder (Mode 0) |
| `scripts/record_realsense.py`   | 🎥 `.bag` recording with intrinsics (Mode 1) |
| `scripts/record_audio_mics.py`  | 🔊 Records `.wav` from physical mics |
| `scripts/monitor_resources.py`  | 📊 Logs CPU/RAM/Disk usage |
| `scripts/plot_monitor_log.py`   | 📈 Displays resource usage from logs |
| `scripts/realsense_preview_grid.py` | 👁 Playback viewer with per-cam control |
| `scripts/preview_all_cams.py`   | 🖼 Live RGB preview with overlay info |

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

- Prompts for **recording mode**:
  - `0`: Frame-by-frame PNG/TIFF
  - `1`: Native `.bag` format
- Records for 10 minutes (video + audio)
- Adds audio buffer (5s/min) to ensure sync
- Shows a live one-line timer while recording
- Intrinsics stored per cam (`.txt`, `.json`)
- CPU/RAM/Disk usage logged in background

---

### 3. Output Folder Structure

#### ▶ Mode 0: Frame-by-frame `.png` + `.tiff`

```
realsense_recording_<timestamp>/
├── cam1/
│   ├── color/
│   │   ├── frame_000000.png
│   │   └── ...
│   ├── depth/
│   │   ├── frame_000000.tiff
│   │   └── ...
│   ├── camera_info_<serial>.txt
│   └── camera_intrinsics_<serial>.json
├── cam2/
│   └── ...
├── audio/
│   ├── mic_hw20_<timestamp>.wav
│   └── mic_hw30_<timestamp>.wav
├── system_monitor_<timestamp>.log
```

#### ▶ Mode 1: `.bag` format

```
realsense_recording_<timestamp>/
├── cam1/
│   ├── cam1_<timestamp>.bag
│   ├── camera_info_<serial>.txt
│   └── camera_intrinsics_<serial>.json
├── cam2/
│   └── ...
├── audio/
│   ├── mic_hw20_<timestamp>.wav
│   └── mic_hw30_<timestamp>.wav
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
  - `1`, `2`, `3`, ...: toggle depth/color for each cam

---

### 5. Transfer Session to External HDD

```bash
./transfer.sh
```

- 🧠 Auto-detects latest `realsense_recording_*` folder
- Prompts for serial number and folder name format
- Prompts for **transfer mode**:
  - `0`: Archive → verify → extract → deep verify
  - `1`: File-by-file copy with SHA256 verify per file
- 🛡️ SHA256 hash comparison with retry for failed files
- 🎉 Live progress display for all operations

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