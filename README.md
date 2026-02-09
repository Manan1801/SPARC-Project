# SPARC-Project — LAK Branch

## 1. Project Overview

The LAK branch of the SPARC-Project is intended for **offline analysis of recorded assembly-task data**, with a focus on extracting, transforming, and summarizing **hand movement and object interaction metrics** from preprocessed RGB-D inputs.

This branch provides a structured pipeline to convert hand landmarks into 3D motion representations, compute movement- and interaction-based features, and generate analytical visualizations and summary statistics that can be used for downstream interpretation, comparison, or reporting.

It is designed for **post-hoc analysis only** and assumes that all required inputs such as RGB frames, depth data, and landmark detections are already available.

---

## 2. Repository Structure

```
SPARC-Project/
├── README.md
├── scripts/
│   ├── clustering.py
│   ├── compute_movement.py
│   ├── compute_touch_metrics.py
│   ├── cumulative_heatmaps.py
│   ├── cumulative_heatmaps_fixseg.py
│   ├── cumulative_movement_csv.py
│   ├── cumulative_plot.py
│   ├── ellipsoid_png.py
│   ├── heatmaps.py
│   ├── labels_flip.py
│   ├── landmark_xyz.py
│   ├── png2mp4.py
│   ├── process_handmesh.py
│   ├── volume_lineplot.py
│   ├── wrist_relative_transforms.py
│   └── visualize_localframe.py
```

* All analysis and processing logic resides in the `scripts/` directory.
* Input data is **not versioned** in this repository and is expected to exist externally.
* Scripts typically write outputs such as CSV files, plots, videos, and logs either next to the input session data or to a user-specified output directory.

---

## 3. Setup Instructions

### 3.1 System Requirements

* Linux-based system recommended
* Sufficient disk space for intermediate CSVs, plots, and generated videos
* No GPU is required for this branch

### 3.2 Environment Setup

* Python 3.8 or newer
* Use of a virtual environment or Conda environment is recommended

Common dependencies used across scripts include:

* numpy
* pandas
* matplotlib
* seaborn
* scipy
* scikit-learn
* plotly
* opencv-python

Exact dependencies may vary depending on which scripts are executed.

### 3.3 Data Prerequisites

Before running any scripts, the following inputs must already be available:

* RGB image frames (PNG)
* Depth frames aligned with RGB frames (TIFF or equivalent)
* Hand landmark CSV files with consistent column naming
* Optional object interaction or untouched logs

All scripts assume a **consistent session-level folder structure** and **synchronized frame indexing across modalities**.

A typical example structure for a single session may look like:

```
<DATA_ROOT>/
├── session_01/
│   ├── cam1/
│   │   ├── color/
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── depth/
│   │   │   ├── 000000.tiff
│   │   │   ├── 000001.tiff
│   │   │   └── ...
│   │   ├── CSV/
│   │   │   ├── hand_landmarks.csv
│   │   ├── logs/
│   │   │   └── object_untouched.log
│   └── cam2/
│       ├── color/
│       ├── depth/
```

To get a good understanding of the required session-level folder structure refer to `data-collection-pipeline` or `sync-data-collection` branch.

Key assumptions:

* RGB and depth frames share the **same frame index** (for example, `000123.png` corresponds to `000123.tiff`).
* Hand landmark CSVs reference frames using the same indexing scheme.
* Scripts operate at the **session and camera level** and rely on this consistency rather than hard-coded paths.

---

## 4. Script Overview

* **`process_handmesh.py`**
  Cleans and post-processes hand landmark detections and applies two-phase left/right labeling with overlap suppression and batch support.

* **`labels_flip.py`**
  Flips left/right hand labels from a specified frame onward to correct labeling inconsistencies.

* **`landmark_xyz.py`**
  Converts 2D hand landmark pixel coordinates into 3D coordinates using aligned depth data.

* **`wrist_relative_transforms.py`**
  Transforms 3D hand landmarks into a wrist-centered local coordinate frame using translation and rotation for frame-invariant analysis.

* **`compute_movement.py`**
  Computes frame-to-frame movement and speed metrics at keypoint and part levels, with optional IQR-based outlier filtering.

* **`cumulative_movement_csv.py`**
  Generates cumulative movement features with dynamically constructed headers for structured analysis.

* **`cumulative_plot.py`**
  Produces cumulative movement plots with optional regression fits and exports regression slopes to CSV.

* **`heatmaps.py`**
  Generates 2D and 3D trajectory visualizations and spatial heatmaps of hand motion.

* **`cumulative_heatmaps.py`**
  Builds cumulative 3D heatmaps and fits ellipsoids to hand movement distributions.

* **`cumulative_heatmaps_fixseg.py`**
  Computes ellipsoid volumes over fixed temporal segments and appends results to CSV outputs.

* **`ellipsoid_png.py`**
  Produces static 3D scatter and ellipsoid visualizations exported as PNG images.

* **`volume_lineplot.py`**
  Plots ellipsoid volumes across temporal segments for comparative analysis.

* **`visualize_localframe.py`**
  Provides an interactive 3D Plotly-based visualization of hand movements in the wrist-centered local coordinate frame.

* **`compute_touch_metrics.py`**
  Computes object-level interaction metrics such as first-touch timing and untouched duration.

* **`clustering.py`**
  Performs PCA-based dimensionality reduction and clustering with visualization outputs.

* **`png2mp4.py`**
  Converts sequences of PNG frames into MP4 videos for visualization and presentation.

---

For questions, clarification, or issues specific to this branch, please contact the branch owner directly.
