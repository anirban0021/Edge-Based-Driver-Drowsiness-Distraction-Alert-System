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

## Day 11 — Head pose sign convention
- Turning head to my right → yaw is: positive
- Turning head to my left → yaw is: negative
- Looking down → pitch is: positive
- Looking up → pitch is: negative

## Day 12 — Distraction threshold results

**Initial issue:** raw yaw/pitch from solvePnP showed a baseline offset even when
facing the camera dead-on (~25° yaw, ~23° pitch at rest), because the webcam sits
below eye level and the generic 3D face model doesn't perfectly match this face's
proportions. This caused "DISTRACTED" to fire while facing forward.

**Fixes applied:**
1. Switched `cv2.solvePnP` flag from `SOLVEPNP_ITERATIVE` to `SOLVEPNP_EPNP` for
   more stable pose estimates from a sparse 6-point set.
2. Added exponential smoothing (`smoothing=0.7`) in `HeadPoseEstimator` to reduce
   single-frame jitter/spikes.
3. Added a 3-second head-pose calibration step (`calibrate_head_pose()`), measuring
   this user's neutral "looking straight" yaw/pitch at startup. All live readings
   are now expressed as *deviation from this baseline*, not raw angles.

**Calibration result (this session):**
- Baseline yaw: 26.9°
- Baseline pitch: 22.9°

**Test results after fix:**
- Facing forward normally: relative yaw ~-13.6°, relative pitch ~22.4° → correctly
  stayed green, no false "DISTRACTED" trigger.
- Quick glance (<1s): false-triggered? no
- Sustained turn (1+s): correctly detected? yes
- Sustained downward look (1+s): correctly detected? pending re-test with an
  exaggerated "chin to chest" motion — initial casual downward glances measured
  ~22° relative pitch, which stayed under the 25° threshold (may be correct
  behavior — a slight glance down arguably *should* stay below threshold).

**Adjustment under consideration:**
- `pitch_threshold` may need lowering from `25.0` to `18.0` if the webcam's low
  mounting position means genuine "looking down at phone" gestures don't produce
  a large enough relative pitch swing. To be confirmed with a clearer deliberate
  test before finalizing.

- yaw_threshold: 25° | pitch_threshold: 25° (pending possible reduction to 18°) | sustain_duration: 1.0s