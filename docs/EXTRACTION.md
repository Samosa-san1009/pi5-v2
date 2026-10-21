# Extraction after PuReST Soft Flow Lite

After the saved video has been analyzed, `build_results()` now passes each
whole-eye trace through the project's root `extraction/` implementation.
The native detector, recording, LED schedule and live overlay are unchanged.
`analysis_source` remains `offline`; `postprocessing.method` is
`extraction-v1.0.0`. This uses the root method, not the separate experiments
or ensemble. Source hashes and adaptations are recorded in
`plr/analysis/extraction/SOURCE.md`.

## Processing and timing

The reused `preprocess()` masks invalid/nonpositive values and declared
absent-pupil intervals, pads those intervals by 50 ms before/100 ms after,
and linearly interpolates bounded missing intervals of up to five frames
at 30 fps (167 ms). It never bridges a declared blink or extrapolates the
recording ends. A three-sample median and fifth-order, zero-phase 1.5 Hz
Butterworth filter run separately within contiguous segments. Segments
shorter than the filter padding requirement stay unavailable. Filter edges
and interpolation are flagged. Native flow-only estimates are excluded
from the input measurements; interpolation does not increase observed coverage.

The root filter requires near-uniform sampling. If sensor/source intervals
vary by more than 5%, the adapter bins samples onto a median-period grid.
Each source frame can occupy only one slot; empty slots remain missing until
the root preprocessing applies its bounded interpolation rule. Source-frame
and source-time fields, dropped/colliding-source counts and empty-grid counts
are exported. The original pupilogram `samples` and timestamps stay intact.
Processed `Frame` indices refer to the processing grid; use `source_frame`
for a matched original video frame. Invalid times produce a visible error.

Flash onset and offset come from `events.json`, mapped onto that signal's
time axis. No dataset timestamp file or fixed eight-color sequence is assumed.
Each physical LED's sequence is supplied independently, allowing simultaneous
left/right flashes. Their original event IDs remain separate and are flagged
`bilateral_stimulation_not_isolated_consensual`: those trials cannot isolate
a consensual-only response. The API retains direct/consensual keys for app
compatibility.

## MCA and MCV

The unchanged `compute_mca()` uses a preceding 0.667 s median baseline and
finds the minimum within 3.333 s after onset, **bounded by flash offset**.
It requires at least 80% observed coverage in both windows and no unfilled
or unfilterable gaps. Short flashes can therefore yield a boundary minimum
or unavailable phase. No later minimum is substituted silently.

MCA is baseline minus selected minimum; relative MCA is percent of baseline.
The unchanged `compute_mcv()` uses the maximum negative interior central
derivative during the selected constriction phase. A separate 167 ms
averaged velocity is exported. Missing phases/nonresponses have null MCV,
not a fabricated zero. Latency uses the selected phase start relative to
the sampled onset; exact and sampled event times are retained separately.

Extraction statuses (`stimulus_consistent`, `uncertain`,
`no_clear_constriction`, `unmeasurable`) and every quality flag are retained
in JSON and CSV. API `status: ok` means extraction returned a measurable
amplitude/nonresponse; it does not override `mca_status`/`mcv_status` or
their flags. Values are camera pixels and px/s. Smoothing and its thresholds
are project choices, not a validation of physiological accuracy.

## Outputs and browser

The existing five-field raw sample shape is preserved. Schema 4 adds
`pupilogram.eyes.<eye>.processed` (metadata, all processing stages and masks)
and metric selection details. The monitor at `/monitor` displays the extraction
folder's plot style for both eyes: MCA baseline/minimum/arrow in the upper
panel, peak MCV/tangent and averaged segment in the lower panel. A table shows
all events, including unavailable values, and the monitor lists quality reasons.

Each session has `extraction_offline/` and a promoted `extraction/` containing:

```text
left_smoothed.csv/json     right_smoothed.csv/json
left_mca.csv               right_mca.csv
left_mcv.csv               right_mcv.csv
left_pupillogram.png        right_pupillogram.png
```

Optional live diagnostic results use `extraction_live/`. Plots are generated
after recording, with a non-interactive Matplotlib backend; the camera callback
does not filter signals or render figures. End-to-end processing remains on
the Pi CPU. Runtime has not been benchmarked on physical Pi hardware.

Download artifacts with
`/sessions/<id>/extraction/<left|right>/<smoothed.csv|smoothed.json|mca.csv|mcv.csv|pupillogram.png>`.
Add `?source=offline` or `?source=live` for diagnostic versions. Historical
sessions need **Analyze saved video again** before these artifacts exist.

## Update an existing Pi

```bash
git pull --ff-only
sudo apt-get update
sudo apt-get install -y python3-scipy python3-matplotlib
sudo systemctl restart plr.service
```

Run inside the Pi repo. Its installer-created `.venv` uses system packages;
the updated `scripts/install.sh` also installs these dependencies on new rigs.
Desktop tests install `requirements-dev.txt`. Desktop video review tools use
`requirements-analysis.txt`.

## Standalone sample with no flash log

`tools/analyze_video.py` now also saves extraction CSVs and smoothed plots
under its output folder's `extraction/`. With no flash timings it does
smoothing/interpolation only, labels the plots accordingly and leaves
MCA/MCV files empty. It does not infer stimulus times from diameter changes.

On the supplied `full_video.mp4`, 971 decoded source frames map to a
993-slot 30 Hz grid (22 empty source-time slots). Left: 784 observed,
108 interpolated, 872 filterable slots. Right: 766 observed, 126 interpolated,
874 filterable slots. Neither eye uses MCA/MCV without a flash log.

Checks include the original extraction numerical regression cases and new
tests for exact preprocessing parity, irregular timestamps, short/long gaps,
blink masking, simultaneous LED events, nonresponses, CSV/PNG exports and
the artifact API, plus the existing native/camera/server regression suite.
