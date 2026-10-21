#!/usr/bin/env python3
"""Mask invalid data, interpolate bounded short gaps, median-filter and low-pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from .common import DEFAULT_OUTPUT, PROJECT, VERSION, number, read_csv, runs, write_csv


def preprocess(rows, max_gap_sec=5/30, cutoff_hz=1.5, blink_before_sec=0.05,
               blink_after_sec=0.10):
    if len(rows) < 2:
        raise ValueError("At least two frames are required")
    frame_ids = np.array([int(row["Frame"]) for row in rows])
    if not np.array_equal(frame_ids, np.arange(len(rows))):
        raise ValueError("Input must retain all consecutive zero-based video frames")
    times = np.array([number(row.get("timestamp_ms"))/1000 for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Missing or non-increasing timestamps")
    dt = float(np.median(np.diff(times)))
    fps = 1/dt
    if np.max(np.abs(np.diff(times)-dt)) > dt*0.05:
        raise ValueError("Sampling intervals vary by >5%; resampling is required before filtering")
    if not 0 < cutoff_hz < fps/2:
        raise ValueError("Cutoff must be between zero and Nyquist")
    original = np.array([number(row.get("diameter_px")) for row in rows])
    valid = np.array([str(row.get("Valid")) == "1" for row in rows])
    absent = np.array([row.get("visibility_state") == "absent" for row in rows])
    blink = np.zeros(len(rows), dtype=bool)
    for start, end in runs(absent):
        blink[max(0, start-int(np.ceil(blink_before_sec*fps))):
              min(len(rows), end+int(np.ceil(blink_after_sec*fps)))] = True
    valid &= np.isfinite(original) & (original > 0) & ~blink
    inferred = valid & np.array([row.get("result_source") == "optical_flow_short_gap" for row in rows])
    observed = valid & ~inferred
    cleaned = np.where(valid, original, np.nan)
    filled = cleaned.copy()
    interpolated = np.zeros(len(rows), dtype=bool)
    max_gap = max(0, int(np.floor(max_gap_sec*fps + 1e-5)))
    for start, end in runs(~np.isfinite(cleaned)):
        if end-start <= max_gap and start > 0 and end < len(rows):
            # Do not reconstruct a declared blink or its adjacent masked edges.
            if np.any(blink[start:end]):
                continue
            filled[start:end] = np.interp(times[start:end], times[[start-1, end]], cleaned[[start-1, end]])
            interpolated[start:end] = True
    smoothed = np.full(len(rows), np.nan)
    median_values = np.full(len(rows), np.nan)
    edge = np.zeros(len(rows), dtype=bool)
    segments = np.full(len(rows), -1, dtype=int)
    sos = butter(5, cutoff_hz, btype="low", fs=fps, output="sos")
    padlen = 3*(2*len(sos)+1-min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    short_segments = 0
    for segment_id, (start, end) in enumerate(runs(np.isfinite(filled))):
        if end-start <= padlen:
            short_segments += 1
            continue
        # Nearest-edge padding avoids the zero-padding artifacts of medfilt.
        median_values[start:end] = median_filter(filled[start:end], size=3, mode="nearest")
        smoothed[start:end] = sosfiltfilt(sos, median_values[start:end], padlen=padlen)
        edge[start:min(start+padlen, end)] = True
        edge[max(start, end-padlen):end] = True
        segments[start:end] = segment_id
    output = []
    for i, row in enumerate(rows):
        output.append({"Frame": i, "time_sec": times[i], "diameter_original_px": original[i],
                       "diameter_valid_px": cleaned[i], "diameter_interpolated_px": filled[i],
                       "diameter_median_px": median_values[i], "diameter_smoothed_px": smoothed[i],
                       "is_observed": int(observed[i]), "is_inferred": int(inferred[i]),
                       "is_interpolated": int(interpolated[i]), "blink_masked": int(blink[i]),
                       "filter_edge": int(edge[i]), "segment_id": segments[i],
                       "result_source": row.get("result_source", "")})
    metadata = {"fps": fps, "frames": len(rows), "cutoff_hz": cutoff_hz, "butterworth_order": 5,
                "median_kernel": 3, "filter_padlen_frames": int(padlen), "max_gap_sec": max_gap_sec,
                "blink_before_sec": blink_before_sec, "blink_after_sec": blink_after_sec,
                "short_unfilterable_segments": short_segments, "observed_frames": int(observed.sum()),
                "inferred_frames": int(inferred.sum()), "interpolated_frames": int(interpolated.sum()),
                "smoothed_frames": int(np.isfinite(smoothed).sum()), "version": VERSION}
    return output, metadata


def discover(input_root):
    root = Path(input_root)
    selected = {}
    excluded = []
    candidates = sorted((root/"dataset_batch").glob("*/detections.csv"))
    candidates += sorted((root/"random_5/detections").glob("*.csv"))
    # Support a standalone run's outputs/<ID>_<eye>/detections.csv too.
    candidates += sorted(root.glob("*/detections.csv"))
    for path in candidates:
        stem = path.parent.name if path.name == "detections.csv" else path.stem
        if not (stem.endswith("_left") or stem.endswith("_right")):
            excluded.append({"path": str(path.resolve()), "reason": "noncanonical_eye_name"})
            continue
        participant, eye = stem.rsplit("_", 1)
        key = (participant, eye)
        if key in selected:
            # One result per participant/eye; compare content instead of silently overwriting.
            if path.read_bytes() != selected[key].read_bytes():
                raise ValueError(f"Conflicting final detections for {key}: {selected[key]} and {path}")
            excluded.append({"path": str(path.resolve()), "reason": "identical_duplicate"})
        else:
            selected[key] = path
    return selected, excluded


def smooth_file(source, participant, eye, output=DEFAULT_OUTPUT, **options):
    source = Path(source)
    rows, meta = preprocess(read_csv(source), **options)
    path = Path(output)/participant/f"{eye}_smoothed.csv"
    meta.update({"participant_id": participant, "eye": eye, "source_csv": str(source.resolve()),
                 "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write_csv(path, rows)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT/"PuReST_SoftFlow_Standalone/outputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff-hz", type=float, default=1.5)
    parser.add_argument("--max-gap-sec", type=float, default=5/30)
    args = parser.parse_args()
    selected, excluded = discover(args.input_root)
    for (participant, eye), source in selected.items():
        print(smooth_file(source, participant, eye, args.output,
                          cutoff_hz=args.cutoff_hz, max_gap_sec=args.max_gap_sec), flush=True)
    print(f"Smoothed {len(selected)} eyes; excluded {len(excluded)} duplicate/noncanonical files")


if __name__ == "__main__":
    main()
