# 📦 SPARC Data Collection Pipeline

This branch automates synchronized **RGB-D + Audio** data collection from up to **4 fixed RealSense D435i cameras**, saving:

- 📹 RGB + Depth streams (`.bag` raw files)
- 🔊 Multi-mic audio recordings (simultaneous `.wav` per mic)
- 🧠 Per-camera intrinsics (`.txt` and `.json`)
- 📊 System performance logs (CPU, RAM, Disk I/O)
- 🖼 Live RGB stream previews before recording
- 🔁 Playback with play/pause + depth/color toggle

---

## 🚀 Features

- 🔌 Plug in any **N (1–4)** of 4 known RealSense cameras
- 🎙 Records **multi-mic audio** in sync with video
- 🧠 Auto-generates `multi_camera.launch.py` from connected serials
- 🧾 Saves **raw .bag** files using RealSense SDK for max compatibility
- 🎥 **Preview and playback** with play/pause & depth/color toggle
- 📝 Per-camera `camera_info_<serial>.txt` and `camera_intrinsics_<serial>.json`
- 📊 Logs system CPU, RAM, and Disk I/O
- ⏱ Supports **timed recording** via `./record.sh <minutes>`
- ⛔ Prevents conflicts between preview and recording

---

## 🧰 Requirements

- Ubuntu 22.04  
- ROS 2 Humble  
- Intel RealSense SDK  
- Python packages:
  ```bash
  pip install psutil pyrealsense2 pandas matplotlib sounddevice soundfile
  ```

---

## 📁 File Overview

| File                            | Description |
|---------------------------------|-------------|
| `record.sh`                     | 🔴 Main script: records video (RealSense) + audio (mic) + system |
| `preview.sh`                    | 👁 Live grid preview before recording |
| `playback.sh`                   | 🔁 Plays back camera `.bag` + system plots |
| `camera_serials.txt`            | 📋 Maps cam1–cam4 to RealSense serials |
| `launch/`                       | 🛠 Launch files auto-generated for ROS2 (not used for recording) |
| `scripts/launch_generate.py`   | 🧠 Auto-generates `multi_camera.launch.py` |
| `scripts/record_realsense.py`  | 📹 Starts RealSense SDK `.bag` + intrinsics dump |
| `scripts/record_audio_mics.py` | 🔊 Records `.wav` files from physical mics |
| `scripts/monitor_resources.py` | 📊 Logs CPU, RAM, disk stats during session |
| `scripts/plot_monitor_log.py`  | 📈 Displays interactive system log graphs |
| `scripts/realsense_preview_grid.py` | 🎥 Playback visualizer (toggle + pause) |
| `scripts/preview_all_cams.py`  | 👁 Live RGB preview with FPS/dropped overlay |

---

## 📝 `camera_serials.txt` Format

```
cam1:902322073807
cam2:942422061612
cam3:947722072361
cam4:938322070387
```

Update this with your actual RealSense serial numbers.

---

## ▶️ Usage Instructions

### 1. Preview Camera Feeds (Optional)

```bash
./preview.sh
```

✅ Grid-style RGB window with FPS & dropped frames  
✅ ESC to exit safely  
⛔ Warns if a recording is in progress

---

### 2. Start Timed Recording (Video + Audio)

```bash
./record.sh 10
```

- Records **all connected cameras and mics** for **10 minutes**
- Video: `.bag` (RealSense SDK)
- Audio: `.wav` per mic (saved to `audio/` subfolder)
- Logs system stats and saves intrinsics

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

### 4. Playback and Visualization

```bash
./playback.sh
```

✅ Plays all `.bag` camera streams in sync  
✅ Interactive system usage plot  
✅ Controls:
- `SPACE` or `p` to pause/resume
- `1`, `2`, `3`, ... to toggle depth/color for each cam

---

For issues, contact the Robotics Lab at IIT Gandhinagar.
```
