"""Adapt rig traces/timing to the project's unchanged extraction functions."""
from __future__ import annotations

import math
import json
import threading
from pathlib import Path

import numpy as np

from .extraction.smoothen import preprocess
from .extraction.extract_mca import compute_mca
from .extraction.extract_mcv import compute_mcv

METHOD = "extraction-v1.0.0"
PARAMETERS = {"cutoff_hz": 1.5, "max_gap_sec": 5/30, "baseline_sec": 20/30,
              "response_sec": 100/30, "minimum_observed_coverage": .8,
              "velocity_window_sec": 5/30}
_PLOT_LOCK = threading.Lock()


def serializable(value):
    """Strict JSON: absent numerical estimates are null, never NaN or zero."""
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _input_rows(trace, times):
    n = len(trace["valid"])
    visibility = trace.get("visibility", ["unknown"] * n)
    sources = trace.get("tracking_source", ["purest"] * n)
    estimated = trace.get("estimated", [False] * n)
    return [{"Frame": i, "timestamp_ms": times[i]*1000,
             "diameter_px": trace["diameter_px"][i],
             "Valid": "1" if trace["valid"][i] and not estimated[i] else "0",
             "visibility_state": str(visibility[i]), "result_source": str(sources[i]),
             "source_frame": i, "source_time_sec": times[i]}
            for i in range(n)]


def _regularize(rows):
    """For irregular sensor times, bin to the median-period grid once per frame.

    A source measurement can occupy only one grid slot. Empty slots stay invalid
    for extraction's bounded interpolation; no gap is pre-filled here. Raw
    samples/times remain untouched in pupilogram.eyes.*.samples.
    """
    times = np.array([r["timestamp_ms"]/1000 for r in rows])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Extraction requires at least two strictly increasing finite frame times")
    dt = float(np.median(np.diff(times)))
    if np.max(np.abs(np.diff(times)-dt)) <= dt*.05:
        return rows, {"resampled": False, "source_frames": len(rows)}
    count = int(round((times[-1]-times[0])/dt)) + 1
    if count > max(10*len(rows), 1000):
        raise ValueError("Sensor timestamp gaps are too large to form a supported extraction grid")
    grid = times[0] + np.arange(count)*dt
    slots = [{} for _ in grid]
    blink_slots = set()
    for row, t in zip(rows, times):
        j = int(round((t-times[0])/dt))
        if not 0 <= j < count:
            continue
        if row["visibility_state"] == "absent":
            blink_slots.add(j)
        prior = slots[j]
        if abs(t-grid[j]) <= .5*dt and (not prior or abs(t-grid[j]) < abs(prior["source_time_sec"]-grid[j])):
            slots[j] = row
    output = []
    for i, (t, row) in enumerate(zip(grid, slots)):
        item = {"diameter_px": np.nan, "Valid": "0", "visibility_state": "unknown",
                "result_source": "missing_sensor_frame", "source_frame": None, "source_time_sec": None,
                **row, "Frame": i, "timestamp_ms": t*1000}
        if i in blink_slots:
            item["visibility_state"] = "absent"
        output.append(item)
    return output, {"resampled": True, "source_frames": len(rows), "grid_period_sec": dt,
                    "grid_frames": count, "empty_grid_frames": sum(not r for r in slots),
                    "unassigned_source_frames": len(rows)-sum(bool(r) for r in slots),
                    "resampling_method": "nearest_slot_once_per_source_frame"}


def extract_eye(trace, times, events, session_id, eye):
    if trace is None:
        return {"status": "error", "error": "eye trace unavailable", "method": METHOD}, []
    try:
        inputs, alignment = _regularize(_input_rows(trace, times))
        rows, meta = preprocess(inputs, cutoff_hz=PARAMETERS["cutoff_hz"], max_gap_sec=PARAMETERS["max_gap_sec"])
        meta.update(participant_id=session_id, eye=eye, method=METHOD, alignment=alignment,
                    analysis_parameters=PARAMETERS)
        data = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "result_source"}
        t = data["time_sec"]
        pairs = [None] * len(events)
        # Both LEDs may flash together. Each physical LED has its own sequence;
        # compute_mca's previous-offset check must not treat its partner as a
        # preceding stimulus. Keep the two original event identities intact.
        for led in (1, 2):
            indices = [i for i, e in enumerate(events) if e["led_index"] == led]
            stimuli = []
            for order, i in enumerate(indices, 1):
                e = events[i]
                onset = int(np.searchsorted(t, e["onset_s"]))
                offset = int(np.searchsorted(t, e["offset_s"]))
                if e["offset_s"] > t[-1] + 1.5/meta["fps"]:
                    offset = len(t)+1  # preserve an incomplete recording as invalid
                stimuli.append((order, e.get("hex_color", "flash"), onset, offset))
            amplitudes = compute_mca(data, meta, stimuli, baseline_sec=PARAMETERS["baseline_sec"],
                                     response_sec=PARAMETERS["response_sec"],
                                     minimum_coverage=PARAMETERS["minimum_observed_coverage"])
            velocities = compute_mcv(data, meta, amplitudes, velocity_window_sec=PARAMETERS["velocity_window_sec"])
            for i, a, v in zip(indices, amplitudes, velocities):
                event = events[i]
                extra = []
                if alignment["resampled"]:
                    extra.append("sensor_times_resampled")
                if any(j != i and other["led_index"] != led and
                       max(event["onset_s"], other["onset_s"]) < min(event["offset_s"], other["offset_s"])
                       for j, other in enumerate(events)):
                    extra.append("bilateral_stimulation_not_isolated_consensual")
                for row in (a, v):
                    row.update(event_id=event.get("event_id"), led_index=led)
                    row["quality_flags"] = ";".join(filter(None, [row["quality_flags"], *extra]))
                    if extra and row["status"] == "stimulus_consistent":
                        row["status"] = "uncertain"
                pairs[i] = {"mca": a, "mcv": v}
        for row, source in zip(rows, inputs):
            row.update(source_frame=source["source_frame"], source_time_sec=source["source_time_sec"])
        available = any(np.isfinite(r["diameter_smoothed_px"]) for r in rows)
        result = {"status": "ok" if available else "unavailable", "method": METHOD,
                  "metadata": meta, "samples": rows}
        if not available:
            result["error"] = "No contiguous valid segment is long enough for the extraction filter"
        return serializable(result), serializable(pairs)
    except ValueError as exc:
        return {"status": "error", "method": METHOD, "error": str(exc), "samples": []}, []


def response_metrics(pair, processed, event):
    if pair is None:
        return {"status": "error", "error": processed.get("error", "extraction unavailable"),
                "formula_version": METHOD, "mca_px": None, "mcv_px_s": None, "latency_ms": None}
    a, v = pair["mca"], pair["mcv"]
    flags = list(dict.fromkeys(filter(None, (a["quality_flags"]+";"+v["quality_flags"]).split(";"))))
    onset_time = a["onset_sec"]
    latency = (a["constriction_latency_sec"]*1000 if a["constriction_latency_sec"] is not None else None)
    minimum_time = a["minimum_sec"]
    start_frame = a["constriction_start_frame"]
    start_time = (processed["samples"][int(start_frame)]["time_sec"] if start_frame is not None else None)
    return {"status": "error" if a["status"] == "unmeasurable" else "ok",
            "error": "; ".join(flags) if a["status"] == "unmeasurable" else None,
            "unit": "px", "formula_version": METHOD, "mca_px": a["mca_px"],
            "constriction_amplitude_px": a["mca_px"], "mca_pct": a["relative_constriction_percent"],
            "baseline_diameter_px": a["baseline_px"], "min_diameter_px": a["minimum_diameter_px"],
            "mcv_px_s": v["mcv_px_per_sec"], "mcv_averaged_px_s": v["mcv_averaged_px_per_sec"],
            "mcv_baseline_normalized_per_sec": v["mcv_baseline_normalized_per_sec"],
            "latency_ms": latency,
            "constriction_time_ms": (minimum_time-start_time)*1000 if start_time is not None else None,
            "constriction_onset_ms": start_time*1000 if start_time is not None else None,
            "flash_onset_ms": event["onset_s"]*1000,
            "sampled_flash_onset_ms": onset_time*1000 if onset_time is not None else None,
            "mca_status": a["status"], "mcv_status": v["status"], "quality_flags": flags,
            "baseline_observed_coverage": a["baseline_observed_coverage"],
            "response_observed_coverage": a["response_observed_coverage"],
            "selection": pair}


def write_extraction(directory, eye, processed, pairs):
    """CSV audit stages and the extraction folder's two-panel annotated plot."""
    from .extraction.common import write_csv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = processed.get("samples", [])
    metadata = processed.get("metadata", {})
    metadata = {**metadata, "eye": eye, "processing_status": processed["status"],
                "error": processed.get("error"), "stimulus_count": len(pairs)}
    path = directory / f"{eye}_smoothed.csv"
    write_csv(path, rows)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    for key in ("mca", "mcv"):
        write_csv(directory / f"{eye}_{key}.csv", [p[key] for p in pairs if p])
    plot_path = directory / f"{eye}_pupillogram.png"
    if not rows:
        plot_path.unlink(missing_ok=True)
        return
    # Matplotlib is imported only after recording. Live camera/preview paths do
    # not render figures. A lock protects its global state from tool callers.
    with _PLOT_LOCK:
        from .extraction.plot_pupillogram import plot_file
        plot_file(path)
