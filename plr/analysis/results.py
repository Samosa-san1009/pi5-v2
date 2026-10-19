"""Pupilogram + flash-aligned PLR metrics for one session.

Input: one PuReST trace per eye (live or offline, already matched to video
frames), the session's frames.csv (sensor timestamp of every video frame) and
events.json (optically measured flash onsets).

Output: pupilogram<suffix>.json and metrics<suffix>.json, in the JSON shape
the app already displays:

  pupilogram.eyes.<left|right>.samples[]: frame, time_s, diameter_px, confidence, valid
  metrics.events[]: event_id, status, stimulus, direct{mca_px, mcv_px_s, latency_ms}, consensual{...}
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .plr_metrics import calculate_plr_metrics

EYE_OF_LED = {1: "left", 2: "right"}


def load_frame_times(session_dir: Path) -> tuple[list[int], np.ndarray]:
    """Sensor timestamps (ns) and times from the first frame (s)."""
    path = Path(session_dir) / "frames.csv"
    stamps, times = [], []
    if path.is_file():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                stamps.append(int(row["sensor_timestamp_ns"]))
                times.append(float(row["time_s"]))
    return stamps, np.asarray(times, dtype=float)


def nominal_fps(times: np.ndarray, fallback: float = 30.0) -> float:
    if len(times) > 1:
        period = float(np.median(np.diff(times)))
        if period > 0:
            return 1.0 / period
    return fallback


def build_results(session_dir: Path, session_id: str, traces: dict, source: str, settings: dict,
                  participant: dict | None = None, suffix: str = "", engine: dict | None = None) -> dict:
    session_dir = Path(session_dir)
    timeline = json.loads((session_dir / "events.json").read_text(encoding="utf-8"))
    events = sorted(timeline.get("events", []), key=lambda e: e["onset_s"])
    _, times = load_frame_times(session_dir)
    fps = nominal_fps(times, float(settings["camera"]["fps"]))
    width, height = settings["camera"]["resolution"]
    window_pre = float(settings["session"].get("window_pre_s", 1.5))
    window_post = float(settings["session"].get("window_post_s", 3.0))

    failures = {}
    for eye in ("left", "right"):
        trace = traces.get(eye)
        if trace is None:
            failures[eye] = "no PuReST trace"
        elif not trace["valid"].any():
            failures[eye] = "Soft Flow Lite found no accepted pupil measurements for this eye (check focus, illumination and eye boxes)"

    metric_events = []
    for event in events:
        direct_eye = EYE_OF_LED[event["led_index"]]
        other_eye = "right" if direct_eye == "left" else "left"
        direct = _window(traces.get(direct_eye), times, fps, event["onset_s"], window_pre, window_post,
                         failures.get(direct_eye))
        consensual = _window(traces.get(other_eye), times, fps, event["onset_s"], window_pre, window_post,
                             failures.get(other_eye))
        statuses = {direct["status"], consensual["status"]}
        status = "error" if statuses == {"error"} else ("partial" if "error" in statuses else "ok")
        metric_events.append({"event_id": event.get("event_id"), "status": status, "stimulus": event,
                              "direct": direct, "consensual": consensual})

    ok_eyes = sum(1 for eye in ("left", "right") if eye not in failures)
    event_errors = sum(e["status"] != "ok" for e in metric_events)
    status = "error" if ok_eyes == 0 else ("partial" if ok_eyes < 2 or event_errors else "ok")

    roi_px = {eye: _roi_px(settings["eye_roi"][eye], width, height) for eye in ("left", "right")}
    analyzer_config = {
        eye: {"analyzer": "purest", "eye": eye, "source": source,
              "roi_px": roi_px[eye] if settings.get("eye_localization", {}).get("mode", "fixed") == "fixed" else None,
              "variant": "softflow_lite", "flow_estimates_in_metrics": False,
              "eye_localization": settings.get("eye_localization", {"mode": "fixed"}),
              "roi_normalised": settings["eye_roi"][eye] if settings.get("eye_localization", {}).get("mode", "fixed") == "fixed" else None,
              "roi_definition": "tracking.eye_box for each frame" if settings.get("eye_localization", {}).get("mode") == "auto" else "fixed eye_roi",
              "confidence_threshold": settings["purest"]["confidence_threshold"],
              "coverage": (traces.get(eye) or {}).get("coverage")}
        for eye in ("left", "right")
    }
    pupilogram = {
        "schema_version": 3,
        "session_id": session_id,
        "analyzer": "purest",
        "analysis_source": source,
        "analyzer_config": analyzer_config,
        "unit": "px",
        "time_base": "sensor_timestamps" if len(times) else "nominal_fps",
        "source_video": {"path": str(session_dir / "full_video.mp4"), "fps": round(fps, 6),
                         "frames_with_timestamps": int(len(times))},
        "eyes": {eye: _trace_output(traces.get(eye), times, fps, analyzer_config[eye]) for eye in ("left", "right")},
        "failures": failures,
    }
    metrics = {
        "schema_version": 3,
        "session_id": session_id,
        "participant": participant or {},
        "status": status,
        "unit": "px",
        "analyzer": "purest",
        "analysis_source": source,
        "coverage": {eye: (traces.get(eye) or {}).get("coverage") for eye in ("left", "right")},
        "engine": engine or {},
        "definitions": {
            "mca": "baseline median minus minimum post-flash diameter",
            "mcv": "largest constriction speed from onset to minimum",
            "latency": "constriction onset minus optically measured flash onset",
        },
        "flash_timing": timeline.get("optical"),
        "warnings": timeline.get("warnings", []),
        "events": metric_events,
    }
    _write(session_dir / f"pupilogram{suffix}.json", pupilogram)
    _write(session_dir / f"metrics{suffix}.json", metrics)
    return {"status": status, "source": source, "coverage": metrics["coverage"]}


def _times(n: int, times: np.ndarray, fps: float) -> np.ndarray:
    if len(times) >= n:
        return times[:n]
    return np.arange(n, dtype=float) / fps


def _window(trace, times, fps, onset_s, pre, post, failure) -> dict:
    if trace is None or failure:
        return {"status": "error", "error": failure or "eye trace unavailable"}
    n = len(trace["valid"])
    t = _times(n, times, fps)
    mask = (t >= max(0.0, onset_s - pre)) & (t <= onset_s + post) & trace["valid"]
    if not mask.any():
        return {"status": "error", "error": "no valid samples in event window"}
    try:
        result = calculate_plr_metrics(
            np.flatnonzero(mask), trace["diameter_px"][mask], fps=fps, flash_onset_s=onset_s,
            smoothing_window=int(os.environ.get("PLR_SMOOTHING_WINDOW", "5")), times_s=t[mask])
        result.pop("diameter_series", None)
        return {"status": "ok", "time_base": "sensor_timestamps" if len(times) else "nominal_fps", **result}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _trace_output(trace, times, fps, config):
    if trace is None:
        return None
    n = len(trace["valid"])
    t = _times(n, times, fps)
    samples = []
    for i in range(n):
        d = trace["diameter_px"][i]
        c = trace["confidence"][i]
        valid = bool(trace["valid"][i]) and bool(np.isfinite(d))
        samples.append({
            "frame": i,
            "time_s": round(float(t[i]), 6),
            "diameter_px": round(float(d), 4) if valid else None,
            "confidence": round(float(c), 4) if np.isfinite(c) else None,
            "valid": valid,
        })
    tracking = []
    estimates = trace.get("estimate_px", np.full(n, np.nan))
    sources = trace.get("tracking_source", ["purest"] * n)
    estimated = trace.get("estimated", [False] * n)
    visibility = trace.get("visibility", ["unknown"] * n)
    eye_boxes = trace.get("eye_boxes", np.zeros((n, 4), dtype=int))
    localization = trace.get("localization", ["fixed"] * n)
    for i in range(n):
        estimate = estimates[i]
        tracking.append({"frame": i, "source": str(sources[i]),
                         "estimated": bool(estimated[i]),
                         "visibility": str(visibility[i]),
                         "eye_box": [int(v) for v in eye_boxes[i]], "localization": str(localization[i]),
                         "estimate_px": round(float(estimate), 4) if np.isfinite(estimate) else None})
    return {"fps": round(float(fps), 6), "time_base": "sensor_timestamps" if len(times) else "nominal_fps",
            "analyzer": "purest", "analyzer_config": config, "coverage": trace.get("coverage"),
            "samples": samples, "tracking": tracking}


def _roi_px(roi, width, height):
    x, y, w, h = roi
    return {"x": int(round(x * width)), "y": int(round(y * height)),
            "width": int(round(w * width)), "height": int(round(h * height))}


def _write(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
