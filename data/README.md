# Data

This folder holds datasets used to validate detection thresholds (EAR, PERCLOS, MOR) against real data before relying on them for live detection. These are not used to train any new models — MediaPipe and YOLO are pretrained.

## MRL Eye Dataset
- **Location:** `data/mrl_eye/` (not tracked in Git — ~328MB, 84,902 images)
- **Source:** MRL Infrared Eye Images Dataset for Drowsiness Detection
- **Structure:** `train/`, `val/`, `test/` folders, each split into `awake/` and `sleepy/`
- **Used for:** validating EAR and PERCLOS thresholds (Week 2, Day 9)

## YawDD
- **Location:** `data/yawdd/` (to be added)
- **Used for:** validating MOR / yawn detection thresholds (Week 2, Day 10)

## NTHU Driver Drowsiness Dataset (optional)
- **Location:** `data/nthu/` (to be added, optional)
- **Used for:** end-to-end pipeline testing