#!/usr/bin/env python3
"""Measure both-eye throughput on the machine that will run the rig.

    .venv/bin/python tools/benchmark_tracker.py sessions/<id>/full_video.mp4
    # For a single-eye dataset crop:
    .venv/bin/python tools/benchmark_tracker.py eye.mp4 --single-eye

Uses current device settings. Counts flow estimates separately from observed
diameters; video decode and both eye workers are included in wall throughput.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plr import purest  # noqa: E402
from plr.settings import SettingsStore  # noqa: E402


def summarize(rows, fps):
    ms = np.asarray([int(r["Proc_us"]) / 1000 for r in rows])
    period = 1000 / fps
    return {
        "frames": len(rows),
        "observed": sum(r["Valid"] == "1" for r in rows),
        "flow_estimates": sum(r.get("Estimated") == "1" for r in rows),
        "sources": dict(Counter(r.get("Source", "purest") for r in rows)),
        "median_ms": round(float(np.median(ms)), 3) if len(ms) else None,
        "p95_ms": round(float(np.percentile(ms, 95)), 3) if len(ms) else None,
        "over_frame_budget": int((ms > period).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=float, help="target frame rate; defaults to camera setting")
    parser.add_argument("--work-size", type=int, help="override processed eye size for this run")
    parser.add_argument("--single-eye", action="store_true", help="process the whole crop in both workers")
    parser.add_argument("--output", type=Path, help="save the JSON report")
    args = parser.parse_args()
    settings = SettingsStore().get()
    if args.frames <= 0 or not args.video.is_file():
        parser.error("use an existing video and a positive --frames count")
    fps = args.fps if args.fps is not None else float(settings["camera"]["fps"])
    if not np.isfinite(fps) or fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.work_size is not None:
        if not 64 <= args.work_size <= 640:
            parser.error("--work-size must be between 64 and 640")
        settings["purest"]["work_size"] = args.work_size
    if args.single_eye:
        settings["eye_roi"] = {e: [0, 0, 1, 1] for e in purest.EYES}
        settings["eye_localization"]["mode"] = "fixed"
    status = purest.binary_status()
    if not status["ready"]:
        parser.error(status["error"])
    with tempfile.TemporaryDirectory(prefix="plr-benchmark-") as tmp:
        paths = {e: Path(tmp) / f"{e}.csv" for e in purest.EYES}
        cmd = [str(purest.binary_path()), "--input", str(args.video.resolve()),
               "--out-left", str(paths["left"]), "--out-right", str(paths["right"]),
               "--max-frames", str(args.frames), *purest._engine_args(settings)]
        # Match the rig's CPU placement, but measure at live priority.
        cmd = purest._wrap_priority(cmd, settings["purest"].get("cpus"),
                                    settings["purest"].get("live_nice", 0))
        start = time.monotonic()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        seconds = time.monotonic() - start
        rows = {}
        for eye, path in paths.items():
            with path.open(newline="") as handle:
                rows[eye] = list(csv.DictReader(handle))
    count = min(len(r) for r in rows.values())
    report = {
        "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "variant": "softflow_lite", "video": str(args.video.resolve()),
        "eye_localization": settings["eye_localization"],
        "single_eye_duplicated": args.single_eye, "settings": settings["purest"],
        "target_fps": fps, "frame_budget_ms": round(1000 / fps, 3),
        "wall_seconds": round(seconds, 3), "both_eye_frames_per_second": round(count / seconds, 3),
        "eyes": {e: summarize(r, fps) for e, r in rows.items()},
        "localization_p95_ms": round(float(np.percentile([int(r.get("Localization_us", 0)) / 1000 for r in rows["left"]], 95)), 3) if rows["left"] else None,
        "note": "Offline throughput is a benchmark; verify live /purest dropped_frames under camera load.",
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
