"""Shared IO and quality checks for stimulus-specific PLR extraction."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_TIMESTAMPS = PROJECT / "PLR_DATASET/timestamps/timestamps.csv"
COLORS = ("Red", "Green", "Blue", "Yellow", "No Light", "White", "Purple", "Indigo")
VERSION = "1.0.0"


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                             else value for key, value in row.items()})


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def signal_data(path):
    rows = read_csv(path)
    names = ("Frame", "time_sec", "diameter_smoothed_px", "diameter_valid_px",
             "is_observed", "is_inferred", "is_interpolated", "blink_masked",
             "filter_edge", "segment_id")
    data = {name: np.array([number(row[name]) for row in rows]) for name in names}
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    return data, meta


def schedule(participant_id, path=DEFAULT_TIMESTAMPS):
    with Path(path).open(newline="") as stream:
        matches = [row for row in csv.reader(stream) if row and row[0].strip() == participant_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one timestamp row for {participant_id}; found {len(matches)}")
    values = matches[0][1:]
    if len(values) != 16 or any(not value.strip() for value in values):
        raise ValueError(f"{participant_id}: expected 16 transition frame entries")
    transitions = [int(value) for value in values]
    # Preserve erroneous entries; the affected stimulus is labelled unmeasurable.
    # Never sort timestamps, because sorting would silently change colour identity.
    return [(i + 1, color, transitions[2*i], transitions[2*i+1]) for i, color in enumerate(COLORS)]


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts, ends))  # end exclusive


def quality_flags(data, start, stop):
    flags = []
    if np.any(data["blink_masked"][start:stop]):
        flags.append("blink_in_analysis_interval")
    if np.any(data["is_interpolated"][start:stop]):
        flags.append("short_gap_interpolated")
    if np.any(data["is_inferred"][start:stop]):
        flags.append("optical_flow_inferred")
    if np.any(data["filter_edge"][start:stop]):
        flags.append("filter_segment_edge")
    return flags


def supports_observed(data, index, radius=2):
    n = len(data["Frame"])
    lo, hi = max(0, index-radius), min(n, index+radius+1)
    return bool(data["is_observed"][index] and np.mean(data["is_observed"][lo:hi]) >= 0.8)


def smooth_paths(output):
    return sorted(Path(output).glob("*/[lr]*_smoothed.csv"))
