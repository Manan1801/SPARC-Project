# 📦 SPARC Data Collection Pipeline

This branch automates synchronized RGB-D data collection from up to **4 fixed RealSense D435i cameras**, saving:

- 📹 RGB + Depth streams  
- 🧠 Camera intrinsics  
- 🛠 RealSense diagnostics (e.g., dropped frames)  
- 📊 System performance logs (CPU, RAM, Disk I/O)  
- 🖼 Live RGB stream previews before recording  

---

## 🚀 Features

- 🔌 Plug in any **N (1–4)** of 4 pre-defined RealSense cameras  
- 🧠 Auto-generates `multi_camera.launch` using connected known serials  
- 📷 Displays real-time previews in a single grid-style window  
- 📝 Records `.bag` files per camera with selected topics  
- 🩺 Logs diagnostics like dropped frames  
- 📊 Logs system CPU, RAM, and Disk I/O  
- ⏱ Supports **timed recording** with early `Ctrl+C` interrupt  
- ⛔ Prevents conflicts between preview and recording  

---

## 🧰 Requirements

- Ubuntu 20.04  
- ROS Noetic  
- Intel RealSense SDK + `realsense2_camera` ROS wrapper  
- Python packages:
  ```bash
  pip install psutil pyrealsense2
  ```

---

## 📁 File Overview

| File                     | Description |
|--------------------------|-------------|
| `record.sh`              | 🔴 Timed recording script with safety checks |
| `preview.sh`             | 👁 Launches real-time camera preview |
| `preview_all_cams.py`    | 🧠 Streams all RGB cameras in grid view with overlay |
| `launch_generate.py`     | 📦 Auto-generates ROS launch file from connected known cameras |
| `camera_serials.txt`     | 📋 Maps cam1–cam4 to fixed RealSense serial numbers |
| `multi_camera.launch`    | 🔄 Auto-generated ROS launch file |
| `rosbag_record_per_camera.py` | 📦 Records selected topics from each camera |
| `diagnostics_logger.py`  | 🩺 Logs dropped frames and diagnostics per cam |
| `monitor_resources.py`   | 📊 Logs CPU, RAM, Disk during session |
| `plot_logs.py`           | 📈 Visualizes system + diagnostics logs after session |

---

## 📝 `camera_serials.txt` Format

```
cam1:902322073807
cam2:942422061612
cam3:947722072361
cam4:938322070387
```

Update this file with your actual camera serials.

---

## ▶️ Usage Instructions

### 1. (Optional) Use External Drive

By default, recordings are saved to:
```
/media/hpm_mv_2/One Touch/SPARC/realsense_recording_<timestamp>/
```

To save locally instead, edit the path inside `record.sh`:
```bash
base_dir="$HOME/realsense_recording_$timestamp"
```

---

### 2. Preview Camera Feeds (Optional)

Check camera positioning and dropped frames:
```bash
./preview.sh
```

✅ Grid-style OpenCV window  
✅ ESC to exit safely  
⛔ Warns if a recording is in progress

---

### 3. Start Timed Recording

```bash
./record.sh 10
```

This records for **10 minutes** and then stops automatically.  
✅ You can press `Ctrl+C` at any time to stop early.  
⛔ Warns (or kills) preview if still running.

---

### 4. Output Example

```
/media/hpm_mv_2/One Touch/SPARC/realsense_recording_20250414_1830/
├── cam1_20250414_1830.bag
├── cam2_20250414_1830.bag
├── cam3_20250414_1830.bag
├── cam1_diagnostics.log
├── cam2_diagnostics.log
├── system_monitor_20250414_1830.log
```

---

### 5. Visualize Logs (Optional)

```bash
python3 plot_logs.py
```

✅ Enter paths for system and diagnostics logs  
✅ Shows performance + frame drop plots

---

## ✅ Tips

- To record with fewer cameras, change in `record.sh`:
  ```bash
  python3 ~/launch_generate.py --num-cameras 2
  ```
- Always close preview before starting a recording — or let `record.sh` auto-close it

---

For issues, reach out to the Robotics Lab at IIT Gandhinagar.
