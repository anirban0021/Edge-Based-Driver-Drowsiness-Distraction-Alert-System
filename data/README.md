# Data

This folder holds datasets used to validate detection thresholds (EAR, PERCLOS, MOR) against real data before relying on them for live detection. These are not used to train any new models — MediaPipe and YOLO are pretrained.

## MRL Eye Dataset
- **Location:** `data/mrl_eye/` (not tracked in Git — ~328MB, 84,902 images)
- **Source:** MRL Infrared Eye Images Dataset for Drowsiness Detection
- **Structure:** `train/`, `val/`, `test/` folders, each split into `awake/` and `sleepy/`
- **Used for:** validating EAR and PERCLOS thresholds (Week 2, Day 9)

## YawDD
- **Location:** `data/yawdd/` (not tracked in Git — ~5GB)
- **Source:** YawDD (Yawning Detection Dataset)
- **Structure:** `Dash/` and `Mirror/` folders (two camera angles), each split by subject with videos labeled by gender and glasses (e.g. `MaleGlasses`, `MaleNoGlasses`); `Table1`/`Table2` hold metadata; see `Readme_YawDD.pdf` for full labeling details
- **Used for:** validating MOR / yawn detection thresholds (Week 2, Day 10)

## NTHU Driver Drowsiness Dataset (optional)
- **Location:** `data/nthu/` (to be added, optional)
- **Used for:** end-to-end pipeline testing