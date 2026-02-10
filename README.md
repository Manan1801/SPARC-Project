# SPARC-Project — davis-mediapipe-2d Branch

## Overview

This branch contains **early 2D hand landmark analysis experiments** based on **MediaPipe Hand Mesh**, focused on extracting, visualizing, and analyzing **pixel-space hand keypoints** from processed videos.

The work in this branch was primarily exploratory and used to understand:

* MediaPipe hand landmark extraction in 2D image space
* CSV-based storage of hand keypoint coordinates
* Per-frame and aggregate movement changes in 2D
* Skeleton visualization and coloring for multiple detected hands
* Pixel-to-real-world scaling experiments for later 3D analysis

This branch represents an initial analysis phase and predates the more structured 3D and depth-aware pipelines developed later in the project.

---

## Contents

The branch includes:

* Jupyter notebooks for:

  * MediaPipe hand landmark extraction and visualization
  * CSV generation of 2D hand keypoints
  * Movement change and delta-based analysis
  * Pixel-to-real-world scaling experiments
* Large CSV datasets generated from expert and novice task recordings

Data-heavy outputs were committed during early experimentation and are retained here for historical reference.

---

## Usage Notes

* Notebooks may assume **local paths**, specific datasets, or older dependency versions.
* No fixed environment or dependency locking is provided.
* This branch is **not intended for reuse as a pipeline**.

---

## Recommendation

For maintained and actively used analysis pipelines, refer to newer branches such as:

* `analysis-core`
* `analysis-experiments`

---

This branch is preserved for traceability and methodological context, not active development.
