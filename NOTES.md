## Day 6 — EAR observations
- Eyes open (normal): ~0.28–0.30
- Eyes closed (blink): ~0.04–0.15

## Day 7 — Calibration results
- Baseline (open) EAR: <0.212>
- Closed threshold (0.75x baseline): <0.054>
- Open/closed classification: correct in manual test? yes

## Day 8 — PERCLOS test results
- Holding eyes closed 2+ seconds: triggered? yes
- Normal blinking: false-triggered? no
- Window size used: 90 frames (~3-6 seconds depending on your FPS)

## Day 9 — Validation results (self-recorded, not MRL dataset)
**Note:** MRL Eye Dataset images are pre-cropped eye-only patches with no full face,
incompatible with our MediaPipe-based full-face landmark pipeline. Validated against
controlled self-recordings instead (logs/day9_validation.csv).

- 0-10s (eyes open): false positives? no
- 10-20s (normal blinking): false positives? no
- 20-30s (deliberate closure): correctly detected? yes
- closed_ratio used: 0.75 | perclos_threshold: 20% | continuous_threshold: 2.0s

## Day 10 — Yawn detection results
- MOR threshold used: 0.6
- Real yawn detected correctly? yes
- Talking falsely triggered a yawn? no
- Adjustments made: none