"""Derive PLR summary metrics from per-frame diameter predictions.

Definitions follow Hall & Chilcott, "Eyeing up the Future of the Pupillary
Light Reflex in Neurodiagnostics", Diagnostics 2018, 8, 19:

  * baseline (D0):  median smoothed diameter before flash onset
  * latency (tL):   elapsed time from flash onset to the START of
                     constriction (NOT to the minimum diameter). Onset is
                     found as the first sustained downward run inside a
                     bounded search window right after the flash, so a
                     later, smaller secondary dip further down the trace
                     can never be mistaken for the real constriction.
  * Dmin:           minimum smoothed diameter, searched only inside a
                     bounded window that starts at constriction onset --
                     again to keep a secondary/later dip from being picked.
  * constriction time (tC): time from onset to Dmin
  * MCA:            D0 - Dmin (also reported normalized to D0, as a %,
                     since the paper notes raw MCA is confounded by
                     baseline pupil size and should be baseline-normalized)
  * MCV:            peak constriction velocity (px/s) between onset and
                     Dmin -- steepest negative slope of the smoothed trace
  * PIPR:           post-illumination pupil response -- see
                     calculate_pipr_metrics() below.

This module is deliberately agnostic about whether it's fed a short
per-flash clip or a window sliced out of one long, continuous, unsegmented
recording -- callers now do the latter, slicing a
bounded window of (frame, diameter) pairs around each flash event out of
one whole-session trace, but keeping the ORIGINAL absolute frame numbers.
Because flash_onset_s is also expressed in that same absolute time base,
everything below still works unmodified.

Values remain in pixels until a validated pixel-to-mm calibration exists.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _moving_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or len(values) < 2:
        return values.copy()

    radius = window // 2
    smoothed = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed[index] = float(np.median(values[start:end]))
    return smoothed


def _find_constriction_onset(
    times_s: np.ndarray,
    smoothed: np.ndarray,
    post_indices: np.ndarray,
    flash_onset_s: float,
    onset_search_window_s: float,
    onset_run_frames: int,
    onset_slope_px_s: float,
) -> Optional[int]:
    """Return the index (into the full arrays) of the first frame that
    starts a sustained downward run, searched only within
    [flash_onset_s, flash_onset_s + onset_search_window_s]. This is what
    keeps a second, smaller, later dip from ever being visible to the
    latency/MCA/MCV calculation: we simply never look past this window
    for the *onset*.

    Returns None if no sustained downward run is found in the window
    (caller should fall back to treating flash_onset as onset).
    """
    window_end_s = flash_onset_s + onset_search_window_s
    candidates = [i for i in post_indices if times_s[i] <= window_end_s]
    if len(candidates) < onset_run_frames + 1:
        return None

    for run_start_pos in range(len(candidates) - onset_run_frames):
        is_sustained_drop = True
        for step in range(onset_run_frames):
            a = candidates[run_start_pos + step]
            b = candidates[run_start_pos + step + 1]
            dt = times_s[b] - times_s[a]
            if dt <= 0:
                is_sustained_drop = False
                break
            slope = (smoothed[b] - smoothed[a]) / dt
            if slope > -abs(onset_slope_px_s):
                is_sustained_drop = False
                break
        if is_sustained_drop:
            return candidates[run_start_pos]
    return None


def calculate_plr_metrics(
    frame_indices: Iterable[int],
    diameters_px: Iterable[float],
    fps: float,
    flash_onset_s: float,
    smoothing_window: int = 5,
    onset_search_window_s: float = 0.8,
    constriction_search_window_s: float = 1.2,
    onset_run_frames: int = 2,
    onset_slope_px_s: float = 2.0,
    times_s: Iterable[float] | None = None,
) -> dict:
    """Return aggregate metrics and a serializable per-frame time series.

    onset_search_window_s: only look for constriction ONSET within this
        many seconds after flash_onset_s. Keeps a later, unrelated dip
        from ever being mistaken for the reflex.
    constriction_search_window_s: once onset is found, only look for
        Dmin within this many seconds AFTER onset. Same purpose, applied
        to the minimum-diameter search.
    onset_run_frames: number of consecutive frame-to-frame drops required
        to call something a "sustained" downward run (vs. single-frame
        detector noise).
    onset_slope_px_s: minimum downward slope (px/s) for a frame-to-frame
        step to count toward a sustained run.
    """
    frames = np.asarray(list(frame_indices), dtype=int)
    values = np.asarray(list(diameters_px), dtype=float)
    # times_s (optional) gives each frame's measured time from the camera's
    # sensor timestamps. Without it, times are derived from a constant fps.
    explicit_times = None if times_s is None else np.asarray(list(times_s), dtype=float)

    if len(frames) == 0 or len(values) == 0 or len(frames) != len(values):
        raise ValueError("Predictions must contain matching non-empty frame and diameter arrays")
    if explicit_times is not None and len(explicit_times) != len(frames):
        raise ValueError("times_s must match frame_indices")
    if fps <= 0:
        raise ValueError("fps must be positive")

    valid = np.isfinite(values)
    frames = frames[valid]
    values = values[valid]
    if explicit_times is not None:
        explicit_times = explicit_times[valid]
    if len(values) == 0:
        raise ValueError("Predictions contain no finite diameter values")

    order = np.argsort(frames)
    frames = frames[order]
    values = values[order]

    if explicit_times is not None:
        times_s = explicit_times[order]
    else:
        times_s = frames.astype(float) / float(fps)
    smoothed = _moving_median(values, smoothing_window)

    pre_mask = times_s < flash_onset_s
    post_mask = times_s >= flash_onset_s

    # If a window has no usable pre-flash frames, use its first few
    # predictions as a clearly marked fallback instead of failing outright.
    baseline_source = smoothed[pre_mask]
    baseline_fallback = False
    if len(baseline_source) == 0:
        baseline_source = smoothed[: min(5, len(smoothed))]
        baseline_fallback = True

    post_indices = np.flatnonzero(post_mask)
    if len(post_indices) == 0:
        raise ValueError("Window contains no predictions at or after flash onset")

    baseline = float(np.median(baseline_source))

    onset_index = _find_constriction_onset(
        times_s, smoothed, post_indices, flash_onset_s,
        onset_search_window_s, onset_run_frames, onset_slope_px_s,
    )
    onset_fallback_used = onset_index is None
    if onset_index is None:
        # No clean sustained run detected (e.g. noisy trace) -- fall back
        # to flash onset itself rather than silently scanning the whole
        # window, which is exactly the behaviour that caused the original bug.
        onset_index = int(post_indices[0])

    onset_time_s = float(times_s[onset_index])
    latency_ms = max(0.0, (onset_time_s - flash_onset_s) * 1000.0)

    # Search for Dmin only within a bounded window AFTER onset.
    window_end_s = onset_time_s + constriction_search_window_s
    dmin_candidates = [i for i in post_indices if onset_time_s <= times_s[i] <= window_end_s]
    if not dmin_candidates:
        dmin_candidates = [i for i in post_indices if times_s[i] >= onset_time_s] or list(post_indices)

    dmin_candidates = np.asarray(dmin_candidates, dtype=int)
    minimum_relative_index = int(np.argmin(smoothed[dmin_candidates]))
    minimum_index = int(dmin_candidates[minimum_relative_index])
    minimum = float(smoothed[minimum_index])

    constriction_time_ms = max(0.0, (times_s[minimum_index] - onset_time_s) * 1000.0)

    # MCV: steepest negative slope of the smoothed trace between onset and Dmin.
    mcv_px_s = 0.0
    window_indices = [i for i in post_indices if onset_time_s <= times_s[i] <= times_s[minimum_index]]
    for a, b in zip(window_indices, window_indices[1:]):
        dt = times_s[b] - times_s[a]
        if dt <= 0:
            continue
        slope = (smoothed[b] - smoothed[a]) / dt
        mcv_px_s = min(mcv_px_s, slope)   # most negative = fastest constriction
    mcv_px_s = abs(mcv_px_s)

    mca_px = baseline - minimum
    mca_pct = (mca_px / baseline * 100.0) if baseline > 0 else 0.0

    series = [
        {
            "frame": int(frame),
            "time_ms": round(float(time_s * 1000.0), 3),
            "diameter_px": round(float(raw), 6),
            "smoothed_diameter_px": round(float(smooth), 6),
        }
        for frame, time_s, raw, smooth in zip(frames, times_s, values, smoothed)
    ]

    return {
        "unit": "px",
        "baseline_diameter_px": round(baseline, 6),
        "min_diameter_px": round(minimum, 6),
        "constriction_amplitude_px": round(mca_px, 6),
        "mca_px": round(mca_px, 6),
        "mca_pct": round(mca_pct, 3),
        "mcv_px_s": round(mcv_px_s, 3),
        "latency_ms": round(latency_ms, 3),
        "constriction_time_ms": round(constriction_time_ms, 3),
        "constriction_onset_ms": round(onset_time_s * 1000.0, 3),
        "onset_fallback_used": onset_fallback_used,
        "flash_onset_ms": round(float(flash_onset_s * 1000.0), 3),
        "fps": round(float(fps), 6),
        "baseline_fallback_used": baseline_fallback,
        "formula_version": "v2-bounded-onset",
        "diameter_series": series,
    }


def calculate_pipr_metrics(
    frame_indices: Iterable[int],
    diameters_px: Iterable[float],
    fps: float,
    flash_onset_s: float,
    flash_offset_s: float,
    smoothing_window: int = 5,
    early_window_s: tuple = (0.0, 1.7),
    late_timepoints_s: tuple = (6.0, 30.0),
) -> dict:
    """Post-illumination pupil response.

    Per Hall & Chilcott 2018: early PIPR (<1.7s post-offset) reflects
    combined rod/cone + ipRGC input; late PIPR (up to ~3 min, reported at
    fixed checkpoints within whatever window is available) is driven by
    melanopsin (ipRGCs) alone. Both are reported as % of baseline (D0)
    diameter. This now always runs against a window sliced out of the one
    continuous whole-session recording (no more short segmented clips), so
    the redilation tail is available as long as the caller's window
    extends far enough past flash_offset_s (session.window_post_s).
    """
    frames = np.asarray(list(frame_indices), dtype=int)
    values = np.asarray(list(diameters_px), dtype=float)
    if len(frames) == 0 or len(frames) != len(values):
        raise ValueError("Predictions must contain matching non-empty frame and diameter arrays")
    if fps <= 0:
        raise ValueError("fps must be positive")

    valid = np.isfinite(values)
    frames = frames[valid]
    values = values[valid]
    order = np.argsort(frames)
    frames = frames[order]
    values = values[order]
    if len(values) == 0:
        raise ValueError("Predictions contain no finite diameter values")

    times_s = frames.astype(float) / float(fps)
    smoothed = _moving_median(values, smoothing_window)

    pre_mask = times_s < flash_onset_s
    baseline_source = smoothed[pre_mask]
    baseline_fallback = False
    if len(baseline_source) == 0:
        baseline_source = smoothed[: min(5, len(smoothed))]
        baseline_fallback = True
    baseline = float(np.median(baseline_source))

    clip_start_s = float(times_s[0]) if len(times_s) else 0.0
    clip_end_s = float(times_s[-1]) if len(times_s) else 0.0
    post_offset_available_s = max(0.0, clip_end_s - flash_offset_s)

    def _pct_at_offset(delta_s: float) -> Optional[float]:
        target_time = flash_offset_s + delta_s
        if target_time > clip_end_s:
            return None
        window_lo, window_hi = target_time - 0.15, target_time + 0.15
        mask = (times_s >= window_lo) & (times_s <= window_hi)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(times_s - target_time)))
            value = float(smoothed[nearest])
        else:
            value = float(np.mean(smoothed[mask]))
        return round(value / baseline * 100.0, 3) if baseline > 0 else None

    early_lo, early_hi = early_window_s
    early_mask = (times_s >= flash_offset_s + early_lo) & (times_s <= flash_offset_s + early_hi)
    early_pipr_pct = (
        round(float(np.mean(smoothed[early_mask])) / baseline * 100.0, 3)
        if np.any(early_mask) and baseline > 0
        else None
    )

    late_checkpoints = {
        f"pipr_pct_at_{int(t)}s": _pct_at_offset(t) for t in late_timepoints_s
    }

    return {
        "unit": "px",
        "baseline_diameter_px": round(baseline, 6),
        "baseline_fallback_used": baseline_fallback,
        "flash_onset_ms": round(flash_onset_s * 1000.0, 3),
        "flash_offset_ms": round(flash_offset_s * 1000.0, 3),
        "window_duration_s": round(clip_end_s - clip_start_s, 3),
        "post_offset_available_s": round(post_offset_available_s, 3),
        "early_pipr_pct": early_pipr_pct,
        "early_pipr_window_s": list(early_window_s),
        **late_checkpoints,
        "formula_version": "pipr-v1",
        "fps": round(float(fps), 6),
    }
