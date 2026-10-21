#!/usr/bin/env python3
"""Extract maximum contraction velocity within the selected constriction phase."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (DEFAULT_OUTPUT, number, quality_flags, read_csv, signal_data,
                    smooth_paths, supports_observed, write_csv)


def compute_mcv(data, meta, mca_rows, velocity_window_sec=5/30):
    diam, times = data["diameter_smoothed_px"], data["time_sec"]
    width = max(1, int(round(velocity_window_sec*meta["fps"])))
    results = []
    for mca in mca_rows:
        row = {key: mca[key] for key in ("participant_id", "eye", "stimulus_order", "stimulus", "onset_frame")}
        row.update(constriction_start_frame=mca["constriction_start_frame"], minimum_frame=mca["minimum_frame"],
                   mcv_px_per_sec=np.nan, mcv_peak_frame=np.nan, mcv_peak_sec=np.nan,
                   mcv_latency_sec=np.nan, mcv_averaged_px_per_sec=np.nan,
                   averaged_start_frame=np.nan, averaged_end_frame=np.nan,
                   averaging_interval_frames=width, averaging_interval_sec=width/meta["fps"],
                   mcv_baseline_normalized_per_sec=np.nan, status="unmeasurable", quality_flags="")
        flags = str(mca.get("quality_flags", "")).split(";") if mca.get("quality_flags") else []
        if mca["status"] in ("unmeasurable", "no_clear_constriction"):
            row.update(status=mca["status"], quality_flags=";".join(flags))
            results.append(row)
            continue
        start, end = number(mca["constriction_start_frame"]), number(mca["minimum_frame"])
        if not np.isfinite(start) or not np.isfinite(end) or end-start < 2:
            row["quality_flags"] = ";".join(flags+["constriction_phase_unresolved"])
            results.append(row)
            continue
        start, end = int(start), int(end)
        segment = diam[start:end+1]
        if not np.isfinite(segment).all():
            row["quality_flags"] = ";".join(flags+["gap_in_constriction_phase"])
            results.append(row)
            continue
        # Interior central differences ensure derivative support stays within the phase.
        contraction_velocity = -np.gradient(segment, times[start:end+1])
        local = 1+int(np.argmax(contraction_velocity[1:-1]))
        peak = start+local
        velocity = float(max(0., contraction_velocity[local]))
        onset = int(mca["onset_frame"])
        row.update(mcv_px_per_sec=velocity, mcv_peak_frame=peak, mcv_peak_sec=times[peak],
                   mcv_latency_sec=times[peak]-times[onset],
                   mcv_baseline_normalized_per_sec=velocity/number(mca["baseline_px"]))
        if not all(supports_observed(data, j) for j in range(peak-1, peak+2)):
            flags.append("velocity_peak_lacks_observed_support")
        if end-start >= width:
            starts = np.arange(start, end-width+1)
            slopes = (diam[starts]-diam[starts+width])/(times[starts+width]-times[starts])
            best = int(starts[int(np.argmax(slopes))])
            row.update(mcv_averaged_px_per_sec=float(max(0., np.max(slopes))),
                       averaged_start_frame=best, averaged_end_frame=best+width)
        else:
            flags.append("constriction_too_short_for_averaged_velocity")
        flags.extend(quality_flags(data, start, end+1))
        row.update(status="uncertain" if flags else "stimulus_consistent",
                   quality_flags=";".join(dict.fromkeys(flags)))
        results.append(row)
    return results


def extract_file(path, velocity_window_sec=5/30):
    path = Path(path)
    data, meta = signal_data(path)
    rows = compute_mcv(data, meta, read_csv(path.with_name(f"{meta['eye']}_mca.csv")), velocity_window_sec)
    destination = path.with_name(f"{meta['eye']}_mcv.csv")
    write_csv(destination, rows)
    return destination, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--velocity-window-sec", type=float, default=5/30)
    args = parser.parse_args()
    for path in smooth_paths(args.output):
        print(extract_file(path, args.velocity_window_sec)[0], flush=True)


if __name__ == "__main__":
    main()
