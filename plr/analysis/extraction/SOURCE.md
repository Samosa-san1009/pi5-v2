# Extraction source

Copied from the project `extraction/` folder on 2026-10-19.
`source_hashes.json` records SHA-256 of the original five files.
Only `from common` imports became package-relative; the preprocessing,
MCA and MCV functions are unchanged. The dataset CLI helpers are retained
for provenance; the Pi calls the functions directly through
`plr.analysis.postprocess`. No dataset folder is needed on the Pi.

The production root extraction method is used, rather than the separate
`experiments/` or `ensemble/` alternatives.

`plot_pupillogram.py` is also copied from extraction. Its imports are relative,
its cache is redirected to the temporary directory, hex-color rig stimuli
are accepted, event IDs label the selections, and an empty stimulus schedule
produces a smoothing-only plot. MCA/MCV annotation and tangent calculations
are unchanged. Plot sizes/fonts suit the browser; simultaneous event labels
are grouped and identical markers are drawn once. Unknown rig colors use a
visible blue so white LEDs do not disappear against the plot background.
The original extraction numerical tests are retained in
`tests/test_extraction_reference.py` with package imports.
