
# 🧠 SPARC Real-Time Unified Pipeline  
**Multi-Camera • Hand Movement • Emotion • Object Interaction • Event Triggers**

---

## 🌐 Overview

The **SPARC Real-Time Unified Pipeline** provides synchronized **multi-camera RGB-D capture**, **hand & emotion analysis**, and **object interaction tracking** — all in **real time**.  
It’s designed for **Human-Robot Interaction** and **Learning Behavior Studies** under the SPARC project.

> 🎥 Captures from 1–4 Intel RealSense cameras  
> 🖐️ Tracks hand movements (MediaPipe)  
> 🙂 Maps facial emotions to valence–arousal space  
> 🎯 Detects object interactions (HSV + ArUco)  
> ⚡ Triggers real-time speed and untouched events  
> 🧩 Publishes triggers via ROS 2 Humble  

---

## 🗂️ Project Structure

```

SPARC-Project/
├── realtime.sh                        # Interactive launcher
├── camera_serials.txt                  # Known RealSense serials
└── sync-data-collection/
└── scripts/
├── realtime_capture.py         # Main orchestrator
├── capture_worker.py           # Camera threads
├── camera_utils.py             # Camera discovery & serial mapping
├── movement_processor.py       # Hand & movement analysis
├── emotion_processor.py        # Face & valence–arousal mapping
├── emotion_mapping.py          # Emotion mapping helpers
├── object_worker.py            # Object tracking + COM logic
├── object_interaction.py       # Object detection & states
├── event_triggers.py           # Speed & untouched triggers
├── preview_grid.py             # Unified live preview grid
├── ros_publisher_node.py       # ROS 2 trigger publisher
├── projection.py               # RGB→Depth projection utilities
├── hand_utils.py               # MediaPipe Hands helpers
├── logger_utils.py             # Debounced logging utilities
├── control_flags.py            # Shared pause/stop events
├── tunables.py                 # Centralized constants & parameters
└── types_shared.py             # Shared data structures

````


## 🧩 Environment Setup

```bash
# Clone repository
git clone https://github.com/safi-harshil/SPARC-Project.git
cd SPARC-Project
git checkout sync-data-collection

# Create conda environment
conda create -n realtime_v2 python=3.10.12 -y
conda activate realtime_v2

# Install dependencies
pip install numpy opencv-python mediapipe \
            pyrealsense2 matplotlib tqdm pandas sounddevice soundfile \
            scikit-learn scipy
````

Make launcher executable:

```bash
chmod +x realtime.sh
```

---

## 🎥 Camera Configuration

Edit **`camera_serials.txt`** with one serial per line (from `realsense-viewer`):

```
cam1:241222073458
cam2:243522075041
cam3:250122074372
cam4:241222074947
```

The pipeline auto-detects connected cameras and maps them to `cam1`–`cam4`.

---

## 🚀 Running the Pipeline

From the repo root:

```bash
./realtime.sh
```

You’ll see:

```
───────────────────────────────────────────────────────────────
   Real-Time Unified Pipeline — Interactive Launcher
───────────────────────────────────────────────────────────────
Controls during run:
  • SPACE → Pause/Resume
  • q     → Close grid (pipeline continues)
  • g     → Reopen grid
  • ESC/Ctrl+C → Graceful stop
───────────────────────────────────────────────────────────────
```

Follow prompts to choose:

* Output directory
* Recording duration (seconds)
* Frame-saving interval (0 = off)
* Cameras for 🖐️ movement, 🙂 emotion, 🎯 object tracking
* Enable/disable event checkers
* Optional advanced settings (depth filters, stride, audio, etc.)

Confirm the summary → press **Enter** to launch.

---

## ⌨️ Keyboard Controls During Run

| Key              | Action                                  |
| ---------------- | --------------------------------------- |
| **SPACE**        | Pause/Resume pipeline                   |
| **q**            | Close preview grid (pipeline continues) |
| **g**            | Reopen preview grid                     |
| **ESC / Ctrl+C** | Gracefully stop all threads             |

---

## 🧾 Output Structure

```
run_YYYYMMDD_HHMMSS/
├── cam1/
│   ├── color/ , depth/ , preview/
│   ├── csv/
│   │   ├── movement_landmarks.csv
│   │   ├── movement_xyz.csv
│   │   └── emotion.csv
│   └── logs/speed_trigger.txt
├── cam2/ … (same pattern)
├── logs/object_trigger.txt
├── audio/mic_*.wav
└── meta/run_config.json
```

---

## 🧠 Internal Flow

1. **Camera Discovery** → `camera_utils.py`
2. **Parallel Capture Threads** → `capture_worker.py`
3. **Movement Analysis** → `movement_processor.py` (MediaPipe Hands)
4. **Emotion Mapping** → `emotion_processor.py` & `emotion_mapping.py`
5. **Object Interaction Tracking** → `object_worker.py` & `object_interaction.py`
6. **Event Triggers** → `event_triggers.py` (speed & untouched events)
7. **Unified Preview Grid** → `preview_grid.py` (video + plots)
8. **ROS 2 Publishing** → `ros_publisher_node.py` (optional)

---

## 🧩 ROS 2 Integration (Optional)

In another terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
conda activate realtime_v2
python sync-data-collection/scripts/ros_publisher_node.py
```

Publishes trigger events (speed/object) to ROS topics such as `/sparc/triggers`.

---

## 🎛️ Parameter Tuning

All tunables live in **`scripts/tunables.py`**:

| Category   | Examples                           |
| ---------- | ---------------------------------- |
| 🎥 Camera  | Resolution, FPS, Depth filters     |
| ✋ Movement | `R0_EVENT_MIN_MMPS`, window sizes  |
| 🙂 Emotion | History length, smoothing          |
| 🎯 Object  | HSV ranges, COM thresholds         |
| ⚡ Triggers | Untouched window, percentages      |
| 🖥️ UI     | Cell sizes, plot intervals, colors |

Modify, save, and rerun `./realtime.sh` to apply.

---

## 🧩 Typical Workflow

1. Connect RealSense cameras
2. Verify streams in `realsense-viewer`
3. Update `camera_serials.txt`
4. Launch pipeline → `./realtime.sh` (Automatically runs inside `realtime_v2` env)
5. Monitor preview grid and event logs
6. (Optional) Run ROS publisher

---

## 🧹 Troubleshooting

| Issue                   | Solution                                               |
| ----------------------- | ------------------------------------------------------ |
| **Blank preview grid**  | Check camera connections & serials                     |
| **No event triggers**   | Enable checkers + verify `tunables.py` thresholds      |
| **Frame drops / lag**   | Lower FPS or use `drop-latest` policy                  |
| **Audio not recording** | Ensure a physical mic is detected                      |
| **ROS topics missing**  | Match `ROS_DOMAIN_ID` and source ROS in both terminals |

---

## 📂 Example Run Summary

```
──────────────── RUN SUMMARY ────────────────
Output dir        : ./run_20251102_163045
Duration (sec)    : 60
Save-every (raw)  : 1
Live preview      : on
Movement cams     : cam2
Emotion cams      : cam1
Object cams       : cam2
Speed-trigger     : ENABLED
Object-trigger    : ENABLED
─────────────────────────────────────────────
```

---

## ✅ Pre-Run Checklist

* `realsense-viewer` detects all cameras
* ROS 2 Humble sourced
* Sufficient disk space
* Good lighting for object detection

---

## 🧠 Credits

Developed as part of the **SPARC Project** 
IIT Gandhinagar • Robotics Lab 
