# PuReST Soft Flow Lite for Pi 5

The native engine now uses a CPU-only adaptation of the neighboring project's
`PuReST_SoftFlow_Standalone`. It preserves validated PuReST detection,
multi-threshold recovery, short-gap sparse optical flow, and independently
confirmed reacquisition. It replaces the RITnet hard mask, soft probability
map, and segmented-eye openness checks with grayscale evidence. There is no
RITnet checkpoint, PyTorch, neural inference, or learned segmentation stage.

This is an adaptation of the reference's recovery design, not the same model
with equivalent accuracy. The original uses future observations to confirm
gaps and interpolate axes; the rig's live/offline engine is causal in both modes.

## Recording, idle preview and final results

The server always invokes Soft Flow Lite on the saved MP4 after recording
closes. The app remains in `analysis` until the offline pupilogram/metrics
are saved and promoted, then sees `done` and `analysis_source: offline`.
Live preview/startup/trace failures do not bypass this pass. A failed offline
pass is reported explicitly, rather than silently falling back to live output.
Legacy `report: auto/live` and `offline: false` settings are migrated while
preserving the rig's remaining settings.

Manual Stop now closes the recording, saves frame/flash timing and analyzes
the partial experiment. An interrupted flash is retained in its command log.

Idle camera frames carry a dedicated preview flag (stream flags bit 1).
This preserves two-observation confirmation when the preview intentionally
samples a 30 fps camera at 5 fps: sequence gaps and 200 ms spacing no longer
reset it every time. Idle mode never runs optical-flow gap interpolation.
Recording mode retains its strict dropped-frame and time-gap checks. Eye-box
refresh also accounts for elapsed time so decimated boxes do not expire
before the next scheduled search. Mode transitions reset temporal evidence.

`/preview.jpg` and `/stream.mjpg` now draw fresh eye boxes and pupil circles;
`?overlay=0` provides raw IR. `/monitor` (also `/`) adds browser pupilograms,
metrics, progress, diagnostics and a saved-session reanalysis button. It uses
Pillow and browser canvas, so the Pi server needs no Matplotlib or Python
OpenCV. Native binaries older than 2.3.0 are rejected with a rebuild instruction.

## Finding eyes in a full-face camera frame

Automatic localization is the default. Bundled OpenCV Haar cascades run in
C++ on a grayscale image capped at 320 pixels on its longest side. The first
search tries the largest face's upper region, then a direct eye-pair search
if the face is cropped. A pair must have similar scales, plausible separation,
and similar vertical positions; a single candidate is not enough. This assumes
one dominant frontal face. It is not a general multi-person identity tracker.

Searches run every six processed frames. While a pair remains fresh, the next
search is restricted to its neighborhood; a failed local search triggers
face/full-eye reacquisition. Crop jitter is suppressed, and undetected boxes
expire after 500 ms. Search retries remain rate-limited even with no eyes.
No automatic fallback measures pupils in static frame halves. Each crop
change resets the pupil tracker to avoid mixing coordinate systems.

Labels follow subject anatomy: left = screen-right on a non-mirrored camera.
Set `eye_localization.swap_eyes: true` for mirrored video or reversed LED
assignment. `eye_localization.mode: "fixed"` restores manually calibrated
`eye_roi` crops for pre-cropped recordings and synthetic fixtures.

The TFT draws dynamic eye boxes. `/purest` reports current boxes, localization
settings, shared localization time, per-eye measurement time and dropped frames.
Changing localization settings rebuilds the live engine immediately while idle.

## Processing each eye

1. Resize only the eye crop, preserving aspect ratio, so its longest side is
   at most 320 pixels. Never upscale. Replace bright local reflection outliers
   with an 11-pixel median only where their intensity exceeds the local median
   by more than 12 grayscale levels. Run the existing PuReST detector/tracker.
   This repair is small and CPU-only; it is not neural segmentation.
2. Validate confidence, size, crop boundaries, aspect ratio, dark interior,
   brighter surrounding ring, and temporal consistency. Reject a mixed
   pupil/iris interior. Reset PuReST after a rejected fit.
3. On failure, search a compact area near the last observed pupil using three
   grayscale darkness thresholds, a small opening filter, contours, and ellipse
   fits. Fit at most 12 substantial components per threshold. Require fill,
   contrast, size and temporal agreement; observed recovery requires aspect
   ratio >= 0.82. Confidence here is a heuristic evidence score, not a learned
   probability and not calibrated to PuReST confidence.
4. If a plausible visible region remains but an observed ellipse fails, track
   at most 32 corners with sparse pyramidal Lucas–Kanade on a compact patch.
   Require at least five tracks, forward/backward agreement, low residuals,
   bounded motion, and current contrast evidence. Estimates last at most three
   frames and 150 ms from the last observed anchor. The previous diameter is
   held only as an estimate; it is never reported as an observed measurement.
5. After loss, search the full bounded eye crop every third frame. A grayscale
   recovery candidate must agree with a candidate in the next processed frame
   before acceptance; the first frame remains missing. A validated PuReST fit
   also needs a second consistent observation after independent loss. No stale
   center guides a long loss. Consecutive accepted diameters must agree within
   0.85–1.15, with bounded center displacement; large jumps remain missing.
6. Reset all temporal evidence on session reset, skipped sequence numbers,
   backward/non-increasing timestamps, a gap over 150 ms, or changed crop size.
   Missing/uncertain image evidence terminates flow immediately.

Each eye has a separate worker. OpenCV internal threading stays disabled to
avoid oversubscription. Live queues remain bounded and report dropped frames;
offline decoding still visits every frame. Results are mapped back to the
original camera pixels, including rotated ellipses and resize rounding.

## Results and compatibility

The app's `analyzer: purest` and five-field pupilogram samples stay compatible.
`analyzer_config.variant` is `softflow_lite`. Each eye also has a parallel
`tracking` array with `frame`, `source`, `estimated`, `visibility`,
`estimate_px`, `eye_box` (source x/y/width/height), and `localization`. Sources are `purest`, `soft_contrast`, `reacquired`,
`flow_estimate`, or `missing`.

Native CSVs append provenance, dynamic eye boxes, localization source/time,
and `Timestamp_ns`. Streaming `P` lines contain 22 whitespace-separated
fields: the legacy 13 fields, three provenance fields, four eye-box integers,
localization source and localization microseconds. The Python reader accepts
legacy 13/16-field and current 22-field formats. For automatic mode the legacy
static `roi_px`/`roi_normalised` metadata is null; use each frame's eye box.
Flow estimates always have `valid: false`, a null measured diameter, and a
separate estimate. They are excluded from PLR metrics even when the user sets
the confidence threshold to zero. The TFT continues to mark them as lost.

`coverage` still means frames processed, not successful pupil detections.
Check `valid` samples separately; an offline rerun cannot fix genuinely absent
image evidence.

## Build and measure on the Pi

The installer already supplies OpenCV development libraries and rebuilds the
engine. For an existing installation:

```bash
bash scripts/build_native.sh
sudo systemctl restart plr.service
.venv/bin/python tools/benchmark_tracker.py sessions/SESSION_ID/full_video.mp4 \
  --frames 600 --fps 30 --output /tmp/plr-tracker-benchmark.json
curl -s localhost:5000/purest
```

The benchmark uses device settings and CPU placement. It reports both-eye
wall throughput, per-eye median/p95 processing times, sources, and frames over
the target budget. At 30 fps each eye has 33.3 ms; at 60 fps it has 16.7 ms.
Measure with representative eye crops and then check live processing times
and shared localization time/dropped frames with camera recording and TFT active. Offline throughput
alone does not establish sustained live performance or temperature behavior.

Settings under `purest`:

| Setting | Default | Range / meaning |
|---|---:|---|
| `work_size` | 320 | 64–640, longest processed eye side |
| `flow_frames` | 3 | 0–5, maximum consecutive estimates; 0 disables LK |
| `max_gap_ms` | 150 | 10–500, reset gap and total flow horizon |
| `reacquire_every` | 3 | 1–30, full-eye recovery search interval |
| `min_contrast` | 12 | 1–100, ring minus pupil grayscale intensity |
| `confidence_threshold` | 0.70 | confidence gate for observed ellipses |
| `glint_kernel` | 11 | odd 3–15; 0 disables bright-outlier cleanup |
| `min_pupil_px` / `max_pupil_px` | unset | limits in source-camera pixels |

For a rig missing its frame budget, try `work_size: 256` and `flow_frames: 0`
using the benchmark overrides or device settings, then recheck accuracy.
Tight eye ROIs and measured pixel limits help suppress iris-sized fits. Smaller
processing images can lose very small pupils; increasing contrast thresholds
can reject dim but real pupils.

## Full-face sample validation, 2026-10-16

The user-supplied `full_video.mp4` (640 × 480, 971 decoded frames, about 33 s)
was processed end to end, with automatic localization and default settings.
Source SHA-256: `6e869b54e5b7598022ec9c47c04c773a6319431e7fbd8d7d88962d3fafffebff`.
The input and face-containing media are local artifacts, not repository files.
The trace, audit overlay, and both pupilograms are produced with:

```bash
python tools/analyze_video.py /path/full_video.mp4 --output outputs/sample
```

The native build, CTest eye-pair geometry checks and 58 Python tests passed.
The optional full-video test was enabled; it verifies bilateral measurement,
centers inside localized crops, and reversed labels with `swap_eyes`.

The sample report at this revision accepted 794 / 971 left-eye measurements
(81.77%) and 782 / 971 right-eye measurements (80.54%). Both eye boxes were
available in 942 frames. Raw gaps are preserved; no smoothing or interpolation
is used to improve the plotted coverage. Reflection repair addressed a large
rejection rate from this recording's bright pupil glints. Two-frame
reacquisition and stricter temporal gates reject isolated blink/iris jumps.
There is no manually labeled pupil ground truth for this clip: coverage and
visual overlay review are not a calibrated accuracy measurement.

The run took about 7.3 seconds including process startup on an ARM Mac;
individual pupil processing p95 was 1.2–1.5 ms, but occasional localization
searches were much slower (about 93 ms p95 among nonzero-timed calls). These
numbers are host-specific. Native queues bound latency and report drops;
verify the full pipeline on the actual Pi with camera/TFT/LED load.

## Earlier close-up validation, 2026-10-12

The following is historical validation before automatic localization,
reflection cleanup and tighter reacquisition gates were introduced. It is not
an accuracy evaluation of the current full-face pipeline.


The build and 45 hardware-free tests passed on the development Mac with
OpenCV 5. Tests cover bilateral size accuracy, live/offline agreement, overload
accounting, blank/flat-eye rejection, recovery after blinks, flow limits and
disablement, resets/discontinuities, and excluding estimates from measurements.
The build also retains the OpenCV 4 path used by Raspberry Pi OS; that path
has not been executed on a Pi in this change.

Both annotated `GS_F_08_005-1` eye crops were processed in full (3,818 frames
each), with the entire crop assigned to each worker. The previous tracker was
built from this repository's pre-change source with the same compiler/OpenCV.
Its accepted samples use the existing 0.70 Python confidence gate. Defaults
were used; no pupil pixel limits were supplied.

| Measure | Old left | Lite left | Old right | Lite right |
|---|---:|---:|---:|---:|
| Accepted manually annotated frames | 353 / 444 | 297 / 444 | 1,304 / 1,544 | 1,127 / 1,544 |
| Diameter MAE on accepted manual frames (px) | 7.95 | 2.46 | 9.37 | 2.77 |
| False detections on annotated blink frames | 8 / 120 | 6 / 120 | 33 / 151 | 21 / 151 |
| Per-eye p95 processing time (ms, Mac) | 23.43 | 6.79 | 22.78 | 6.23 |

Lite used 451 / 745 grayscale recoveries and 7 / 8 flow-only estimates for
left / right. Timing runs processed the two videos concurrently, so these
are development smoke measurements, not an isolated hardware benchmark.
MAE is conditional on acceptance and does not penalize missing samples; it
must be read alongside accepted-frame counts. The accepted sets differ.

Grayscale visibility does **not** replicate segmented-eye openness. Blink
false detections remain, and stricter validation loses some visible pupils.
These results are from one annotated recording and do not establish general
accuracy or Pi 5 speed. Run the benchmark and review your rig's traces before
relying on it for new recordings.

## Reference and licenses

The design reference is `PLR/PuReST_SoftFlow_Standalone/README.md`,
`src/recover_soft_flow.py`, `src/standalone_reacquisition.py`, and
`src/locate_region.py`. The grayscale recovery is a new implementation; neural
probability thresholds and original measured results are not transferable.
The vendored PuRe/PuReST algorithms and their existing academic/non-commercial
licenses remain in `native/purest/third_party/pupil-detection-methods`.

The bundled face/eye cascades are from OpenCV. Their embedded redistribution
licenses and exact SHA-256 hashes are preserved under `native/purest/models`.
See the [OpenCV cascade tutorial](https://docs.opencv.org/4.13.0/db/d28/tutorial_cascade_classifier.html)
for the underlying face/eye window classifier API.
