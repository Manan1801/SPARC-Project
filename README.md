# SPARC-Project — hamer-analysis Branch

## Overview

This branch contains **early exploratory experiments and trial code** related to **HaMeR-based 3D hand pose estimation**, **MediaPipe hand keypoints**, and **Intel RealSense camera integration**.

The work in this branch was primarily focused on:

* Understanding and testing HaMeR and FrankMocap pipelines
* Integrating 2D hand detections with HaMeR-based 3D inference
* Debugging and resolving shape, coordinate, and rendering mismatches
* Comparing MediaPipe and HaMeR hand keypoints using RMS error metrics
* Experimenting with RealSense `.bag` file playback, visualization, and capture control

This branch represents a trial-and-error phase of development. Many scripts were iteratively modified, partially abandoned, or replaced as understanding of the models and data improved.

---

## Contents 

The branch includes:

* Jupyter notebooks for initial HaMeR and FrankMocap trials
* Python scripts for:

  * HaMeR inference and video-based testing
  * MediaPipe hand keypoint extraction and comparison
  * RealSense camera capture and `.bag` file playback
  * RMS error computation and visualization
  * Alignment and coordinate transformation experiments

Data files and large intermediate outputs used during testing were intentionally removed to keep the branch lightweight.

---

## Usage Notes

* Scripts may rely on **specific local paths**, datasets, or environment setups that are no longer present.
* Dependency versions are **not fixed or guaranteed**.
* Some scripts assume availability of external models or SDKs (HaMeR, MediaPipe, Intel RealSense).

---

## Recommendation

For structured, maintained, and actively used pipelines, refer to newer branches such as:

* `analysis-core`
* `analysis-experiments`

---

This branch is preserved for completeness and traceability, not active development.
