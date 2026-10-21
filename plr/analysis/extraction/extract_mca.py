#!/usr/bin/env python3
"""Extract baseline-to-minimum amplitude for each of eight stimulus onsets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (DEFAULT_OUTPUT, DEFAULT_TIMESTAMPS, quality_flags, schedule,
                    runs, signal_data, smooth_paths, supports_observed, write_csv)


def compute_mca(data, meta, stimuli, baseline_sec=20/30, response_sec=100/30,
                minimum_coverage=0.8):
    diam = data["diameter_smoothed_px"]
    times = data["time_sec"]
    fps, n = meta["fps"], len(diam)
    baseline_frames = max(3, int(round(baseline_sec*fps)))
    response_frames = max(3, int(round(response_sec*fps)))
    results = []
    for order, color, onset, offset in stimuli:
        start = max(0, onset-baseline_frames)
        stop = min(n, offset, onset+response_frames)
        row = {"participant_id": meta["participant_id"], "eye": meta["eye"],
               "stimulus_order": order, "stimulus": color, "onset_frame": onset,
               "offset_frame": offset, "onset_sec": times[onset] if 0 <= onset < n else np.nan,
               "baseline_start_frame": start, "response_end_frame_exclusive": stop,
               "baseline_observed_coverage": 0., "response_observed_coverage": 0.,
               "baseline_px": np.nan, "baseline_noise_px": np.nan,
               "minimum_diameter_px": np.nan, "minimum_frame": np.nan, "minimum_sec": np.nan,
               "time_to_minimum_sec": np.nan, "mca_px": np.nan,
               "relative_constriction_percent": np.nan, "constriction_start_frame": np.nan,
               "constriction_latency_sec": np.nan, "response_threshold_px": np.nan,
               "status": "unmeasurable", "quality_flags": ""}
        flags = []
        previous_offset = stimuli[order-2][3] if order > 1 else 0
        if not (baseline_frames <= onset < stop and onset < offset <= n) or previous_offset >= onset:
            row["quality_flags"] = "incomplete_or_invalid_stimulus_interval"
            results.append(row)
            continue
        if previous_offset < 0:
            flags.append("previous_stimulus_offset_invalid_review")
        base = diam[start:onset]
        response = diam[onset:stop]
        bc = float(np.mean(data["is_observed"][start:onset]))
        rc = float(np.mean(data["is_observed"][onset:stop]))
        row.update(baseline_observed_coverage=bc, response_observed_coverage=rc)
        if bc < minimum_coverage or rc < minimum_coverage:
            flags.append("insufficient_observed_coverage")
        if not np.isfinite(base).all() or not np.isfinite(response).all():
            flags.append("unfilled_or_unfilterable_gap")
        if "insufficient_observed_coverage" in flags or "unfilled_or_unfilterable_gap" in flags:
            row["quality_flags"] = ";".join(flags)
            results.append(row)
            continue
        baseline = float(np.median(base))
        if baseline <= 0:
            row["quality_flags"] = "nonpositive_baseline"
            results.append(row)
            continue
        minimum_index = onset+int(np.argmin(response))
        minimum = float(diam[minimum_index])
        amplitude = baseline-minimum
        noise = float(1.4826*np.median(np.abs(base-baseline)))
        # Exploratory response-detection rule; retain numerical amplitudes even below it.
        threshold = max(3*noise, 0.01*baseline)
        flags.extend(quality_flags(data, start, stop))
        row.update(baseline_px=baseline, baseline_noise_px=noise,
                   minimum_diameter_px=minimum, minimum_frame=minimum_index,
                   minimum_sec=times[minimum_index],
                   time_to_minimum_sec=times[minimum_index]-times[onset],
                   mca_px=amplitude, relative_constriction_percent=100*amplitude/baseline,
                   response_threshold_px=threshold)
        if not supports_observed(data, minimum_index):
            flags.append("minimum_lacks_observed_support")
        if minimum_index <= onset+1:
            flags.append("minimum_at_stimulus_onset_review")
        if minimum_index >= stop-2:
            flags.append("minimum_at_response_boundary")
        baseline_trend = float(np.median(base[len(base)//2:])-np.median(base[:len(base)//2]))
        if abs(baseline_trend) > max(threshold, 0.2*abs(amplitude)):
            flags.append("baseline_trend_review")
        if amplitude <= threshold:
            row.update(status="no_clear_constriction", quality_flags=";".join(flags+["below_exploratory_noise_threshold"]))
            results.append(row)
            continue
        # Locate the last peak preceding a sustained descent to the response minimum.
        # Short gaps are filled before filtering; large gaps were rejected above.
        grad = np.gradient(response, times[onset:stop])
        noise_velocity = float(1.4826*np.median(np.abs(np.gradient(base, times[start:onset]))))
        velocity_threshold = max(noise_velocity, 0.01*baseline)
        run_length = max(2, int(round(0.1*fps)))
        descent = grad < -velocity_threshold
        sustained_descents = [(a, b) for a, b in runs(descent[:minimum_index-onset+1]) if b-a >= run_length]
        if len(sustained_descents) > 1:
            # Separate falls can reflect hippus, an artifact or a later response.
            for (_, previous_end), (next_start, _) in zip(sustained_descents, sustained_descents[1:]):
                if np.max(response[previous_end:next_start+1])-response[previous_end-1] > threshold:
                    flags.append("multiple_descent_phases_review")
                    break
        candidates = [k for k in range(max(0, minimum_index-onset-run_length+1))
                      if descent[k:k+run_length].all()]
        if candidates:
            first = candidates[0]
            preceding_peak = int(np.argmax(response[:first+1]))
            constriction_start = onset+preceding_peak
            latency = float(times[constriction_start]-times[onset])
            row.update(constriction_start_frame=constriction_start, constriction_latency_sec=latency)
            if latency < 0.1:
                flags.append("early_or_preexisting_drop_review")
            elif latency > 1.0:
                flags.append("delayed_drop_review")
        else:
            flags.append("no_sustained_constriction_detected")
        # This remains a stimulus-consistent estimate, not proof of stimulus causation.
        row.update(status="uncertain" if flags else "stimulus_consistent",
                   quality_flags=";".join(dict.fromkeys(flags)))
        results.append(row)
    return results


def extract_file(path, timestamps=DEFAULT_TIMESTAMPS, **options):
    path = Path(path)
    data, meta = signal_data(path)
    rows = compute_mca(data, meta, schedule(meta["participant_id"], timestamps), **options)
    destination = path.with_name(f"{meta['eye']}_mca.csv")
    write_csv(destination, rows)
    return destination, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timestamps", type=Path, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--baseline-sec", type=float, default=20/30)
    parser.add_argument("--response-sec", type=float, default=100/30)
    args = parser.parse_args()
    for path in smooth_paths(args.output):
        destination, _ = extract_file(path, args.timestamps, baseline_sec=args.baseline_sec,
                                     response_sec=args.response_sec)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
