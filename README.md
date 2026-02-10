# SPARC-Project — analysis-experiments Branch

---

## 1. Project Overview

This branch contains **experimental, trial-and-error analysis pipelines** developed during the early and intermediate stages of the SPARC-Project. It serves as a workspace where multiple ideas for hand tracking, movement analysis, validation, visualization, and unsupervised pattern discovery were **implemented, tested, revised, or abandoned** before converging on more stable pipelines used in later work.

The scripts in this branch explore different approaches to:

* Hand landmark processing and validation
* Depth-aware and bounding-box–based analysis
* Movement computation across multiple temporal granularities
* Statistical summarization and visualization of hand motion
* Clustering and exploratory analysis of movement-derived feature spaces

In addition to signal processing and visualization experiments, this branch includes clustering trials that investigate how different feature representations, normalization strategies, dimensionality reduction techniques, and clustering algorithms behave on movement data. These experiments are exploratory in nature and were used to assess structure and separability rather than to produce finalized models.

Some concepts and scripts from this branch later informed structured pipelines used in formal submissions, but **this branch itself should be treated as exploratory and non-final**.

It is intended for **offline experimentation and reference**, not as a clean or production-ready pipeline.

---

## 2. Repository Structure

```
SPARC-Project/
├── README.md
├── scripts/
│   ├── boxplots_chunk.py
│   ├── calculate_rms_error.py
│   ├── calibrate_offset.py
│   ├── compute_movement.py
│   ├── compute_movement_10s.py
│   ├── compute_movement_parts.py
│   ├── compute_movement_segments.py
│   ├── cumulative_movement.py
│   ├── depth_preview.py
│   ├── extract_frames_from_bag.py
│   ├── extract_landmarks_mm_xyz.py
│   ├── frame_scale.py
│   ├── generate_handmap.py
│   ├── hand_bbox_depth_skin.py
│   ├── interactive_boxplot.py
│   ├── plot_EN_cumulative.py
│   ├── plot_LnNL_cumulative.py
│   ├── plot_cumulative_movement.py
│   ├── png2mp4_multicam.py
│   ├── process_hand_mesh_batch.py
│   ├── process_hand_mesh_batch_righthand.py
│   ├── process_hand_mesh_frames_handedness.py
│   ├── process_hand_mesh_video.py
│   └── sam_hand_segmentation.py
├── cluster_trials/
│   ├── build_dataset.py
│   ├── cluster.py
│   ├── cluster_hdbscan.py
│   ├── cluster_kmeans.py
│   ├── feature_norm.py
│   ├── pca.py
│   └── plot_cluster.py
```

* Core experimental scripts live under the `scripts/` directory.
* Clustering-specific exploratory experiments are grouped under `cluster_trials/`.
* Input data is **not versioned** and must be supplied externally.
* Outputs include CSVs, diagnostic logs, plots, and videos depending on the experiment.

---

## 3. Setup Instructions

### 3.1 System Requirements

* Linux-based system recommended
* Large storage capacity for extracted frames and intermediate results
* GPU not required

### 3.2 Environment Setup

* Python 3.8 or newer
* Virtual environment or Conda environment recommended

Common dependencies include:

* numpy
* pandas
* matplotlib
* seaborn
* scipy
* scikit-learn
* plotly
* opencv-python

Some scripts assume MediaPipe-based landmark outputs generated outside this branch.

---

## 4. Script Overview

### Hand Processing and Validation

* **`process_hand_mesh_batch.py`**, **`process_hand_mesh_video.py`**
  Apply hand landmark detection across image sequences or videos, handling batch execution, frame iteration, and intermediate output generation.

* **`process_hand_mesh_batch_righthand.py`**
  Specialized batch-processing variant that assumes a dominant right hand to reduce left–right ambiguity in single-hand interaction scenarios.

* **`process_hand_mesh_frames_handedness.py`**
  Implements experimental heuristics to stabilize handedness assignment across frames and minimize frequent label flipping.

* **`calculate_rms_error.py`**
  Computes RMS error metrics to evaluate hand landmark accuracy and consistency, including comparisons against reference annotations or datasets.

* **`calibrate_offset.py`**, **`frame_scale.py`**
  Utilities for calibrating pixel-to-metric offsets and normalizing spatial scales prior to 3D reconstruction and movement analysis.

* **`extract_landmarks_mm_xyz.py`**
  Converts 2D pixel-space hand landmarks into metric 3D coordinates using aligned depth information.

---

### Movement Analysis Experiments

* **`compute_movement.py`**
  Computes baseline frame-to-frame hand movement and speed metrics from 3D landmark trajectories.

* **`compute_movement_10s.py`**
  Aggregates movement statistics over fixed 10-second temporal windows to study short-term motion patterns.

* **`compute_movement_segments.py`**
  Divides task duration into equal temporal segments and computes movement metrics per segment for comparative analysis.

* **`compute_movement_parts.py`**
  Computes movement statistics for specific hand parts or landmark groups rather than the full hand.

* **`cumulative_movement.py`**
  Aggregates movement over time into cumulative trajectories to study long-term motion progression.

---

### Visualization and Exploration

* **`plot_cumulative_movement.py`**
  Generates line plots visualizing cumulative hand movement over time.

* **`plot_EN_cumulative.py`**, **`plot_LnNL_cumulative.py`**
  Produces comparative cumulative plots for different participant groups or experimental conditions.

* **`boxplots_chunk.py`**, **`interactive_boxplot.py`**
  Generates chunk-wise statistical summaries using static and interactive boxplots for exploratory analysis.

* **`generate_hand_heatmap.py`**
  Creates 2D and 3D spatial heatmaps of hand motion, with optional statistical overlays such as ellipsoids or covariance ellipses.

* **`hand_bbox_depth_skin.py`**
  Explores depth-aware and bounding-box–based hand region extraction using skin and depth cues.

* **`depth_preview.py`**
  Utility script for visual inspection and debugging of depth frames.

---

### Clustering Analysis Experiments (`cluster_trials/`)

* **`build_dataset.py`**
  Constructs clustering-ready datasets from extracted movement and feature CSVs.

* **`feature_norm.py`**
  Applies feature normalization and scaling prior to dimensionality reduction or clustering.

* **`pca.py`**
  Performs Principal Component Analysis for dimensionality reduction and exploratory visualization.

* **`cluster.py`**
  Generic clustering driver script for running and evaluating multiple clustering configurations.

* **`cluster_kmeans.py`**
  Implements K-Means–based clustering experiments on reduced or normalized feature spaces.

* **`cluster_hdbscan.py`**
  Implements density-based clustering experiments using HDBSCAN for structure discovery.

* **`plot_cluster.py`**
  Visualizes clustering outputs and embeddings for qualitative inspection and comparison.

---

### Utilities

* **`extract_frames_from_bag.py`**
  Extracts synchronized RGB frames from Intel RealSense recorded `.bag` files (RealSense SDK format) to generate image sequences for downstream processing and analysis.

* **`png2mp4_multicam.py`**
  Converts per-camera image frame sequences into MP4 videos, supporting multi-camera recordings for visualization and review.

* **`sam_hand_segmentation.py`**
  Applies Segment Anything Model–based hand segmentation to generate hand masks for experimental region-based and depth-aware analysis.

---

For structured and actively maintained analysis pipelines, refer to newer branches such as `LAK`.
