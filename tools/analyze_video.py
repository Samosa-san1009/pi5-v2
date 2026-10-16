#!/usr/bin/env python3
"""Run automatic face/eye localization and export both pupilograms + an audit overlay.

Desktop dependencies: pip install -r requirements-dev.txt
Example: python tools/analyze_video.py /path/full_video.mp4 --output /tmp/plr-report
The original face video and generated media stay local; they are not Git assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plr import purest  # noqa: E402
from plr.settings import SettingsStore, validate  # noqa: E402


def measured(row):
    return row["Valid"] == "1" and row.get("Estimated", "0") != "1"


def number(row, key):
    return float(row[key]) if row.get(key) not in (None, "") else float("nan")


def plot(rows, output, title):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plr-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#f7f9fc")
    for ax, eye, color in zip(axes, purest.EYES, ("#2563eb", "#d74646")):
        records = rows[eye]
        t = np.array([int(r["Timestamp_ns"]) / 1e9 for r in records])
        t -= t[0]
        d = np.array([number(r, "Diameter_px") if measured(r) else np.nan for r in records])
        ax.plot(t, d, color=color, lw=1, marker=".", ms=2.5, label="Observed pupil diameter")
        estimates = np.array([number(r, "Diameter_px") if r.get("Estimated") == "1" else np.nan for r in records])
        if np.isfinite(estimates).any():
            ax.scatter(t, estimates, color="#8a94a6", marker="x", s=16, label="Flow estimate (excluded)")
        ax.set_title(f"Subject's {eye} eye  •  {np.isfinite(d).sum()} / {len(d)} observed frames ({100*np.isfinite(d).mean():.1f}%)",
                     loc="left", fontsize=12, weight="bold", color="#172033")
        ax.set_ylabel("Diameter (camera px)")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    axes[-1].set_xlabel("Time in source video (s)")
    fig.suptitle(title + "\nAutomatic eye boxes → Soft Flow Lite · gaps are rejected measurements", fontsize=15, weight="bold")
    fig.savefig(output / "pupilogram.png", dpi=170)
    fig.savefig(output / "pupilogram.svg")
    plt.close(fig)


def render(video, rows, output):
    import cv2

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for the audit overlay")
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    # Overlay is a constant-rate review copy. CSV/plot preserve original PTS.
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "overlay.mp4")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames, index = [], 0
    picks = {min(1, len(rows["left"]) - 1), len(rows["left"]) // 5, 2 * len(rows["left"]) // 5,
             3 * len(rows["left"]) // 5, 4 * len(rows["left"]) // 5, len(rows["left"]) - 1}
    try:
        while index < len(rows["left"]):
            ok, frame = capture.read()
            if not ok:
                break
            for eye, color in (("left", (235, 110, 30)), ("right", (70, 70, 240))):
                row = rows[eye][index]
                box = [int(row[k]) for k in ("Eye_x", "Eye_y", "Eye_width", "Eye_height")]
                x, y, w, h = box
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
                    label = eye.upper() + (f" {number(row, 'Diameter_px'):.1f}px" if measured(row) else " missing")
                    cv2.putText(frame, label, (x, max(34, y-7)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
                if measured(row):
                    # Plot a circle at measured major diameter for a readable center/size audit.
                    center = (round(number(row, "Center_x")), round(number(row, "Center_y")))
                    cv2.circle(frame, center, max(1, round(number(row, "Diameter_px") / 2)), (70, 255, 70), 1, cv2.LINE_AA)
                    cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 5, 1)
            seconds = int(rows["left"][index]["Timestamp_ns"]) / 1e9
            cv2.rectangle(frame, (0, 0), (width, 24), (18, 22, 30), -1)
            cv2.putText(frame, f"Soft Flow Lite | {seconds:.2f}s | L/R = subject anatomy", (8, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1, cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
            if index in picks:
                frames.append(frame.copy())
            index += 1
    finally:
        capture.release()
        proc.stdin.close()
    if proc.wait() or index != len(rows["left"]):
        raise RuntimeError("overlay ended early")
    if len(frames) == 6:
        cv2.imwrite(str(output / "detection_contact_sheet.jpg"), np.vstack((np.hstack(frames[:3]), np.hstack(frames[3:]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swap-eyes", action="store_true")
    parser.add_argument("--fixed", action="store_true", help="use calibrated eye_roi instead of automatic localization")
    parser.add_argument("--glint-kernel", type=int)
    parser.add_argument("--skip-overlay", action="store_true")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error("video does not exist")
    settings = SettingsStore().get()
    settings["eye_localization"]["mode"] = "fixed" if args.fixed else "auto"
    if args.swap_eyes:
        settings["eye_localization"]["swap_eyes"] = True
    if args.glint_kernel is not None:
        settings["purest"]["glint_kernel"] = args.glint_kernel
    validate(settings)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {eye: args.output / f"{eye}.csv" for eye in purest.EYES}
    cmd = [str(purest.binary_path()), "--input", str(args.video.resolve()),
           "--out-left", str(paths["left"]), "--out-right", str(paths["right"]), *purest._engine_args(settings)]
    start = time.monotonic()
    subprocess.run(cmd, check=True)
    elapsed = time.monotonic() - start
    rows = {}
    for eye, path in paths.items():
        with path.open(newline="") as handle:
            rows[eye] = list(csv.DictReader(handle))
    if not rows["left"] or len(rows["left"]) != len(rows["right"]):
        raise RuntimeError("native output frame counts differ or are empty")
    summary = {"video": args.video.name, "input_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
               "host": platform.platform(), "native_version": subprocess.check_output([str(purest.binary_path()), "--version"], text=True).strip(),
               "wall_seconds": round(elapsed, 3), "bilateral_frames_per_second": round(len(rows["left"]) / elapsed, 2),
               "eye_localization": settings["eye_localization"], "purest": settings["purest"], "eyes": {},
               "notes": ["No manual pupil ground truth is available for this video; coverage is not accuracy.",
                         "Left/right are anatomical for a non-mirrored camera; use --swap-eyes for mirrored input.",
                         "Flow estimates are excluded from measurements. Overlay is a nominal-fps review copy; CSV/plot retain source PTS.",
                         "Runtime measured on the reported host, not on a Raspberry Pi."]}
    for eye, records in rows.items():
        valid = [r for r in records if measured(r)]
        diameters = np.array([number(r, "Diameter_px") for r in valid])
        summary["eyes"][eye] = {
            "frames": len(records), "localized_frames": sum(int(r["Eye_width"]) > 0 for r in records),
            "observed_frames": len(valid), "observed_percent": round(100 * len(valid) / len(records), 2),
            "sources": dict(Counter(r["Source"] for r in records)),
            "diameter_p05_median_p95_px": np.round(np.percentile(diameters, [5, 50, 95]), 3).tolist() if len(diameters) else [],
            "pupil_p95_ms": round(float(np.percentile([int(r["Proc_us"]) / 1000 for r in records], 95)), 3),
        }
    localization_ms = [int(r["Localization_us"]) / 1000 for r in rows["left"] if int(r["Localization_us"]) > 0]
    summary["localization_search_p95_ms"] = round(float(np.percentile(localization_ms, 95)), 3) if localization_ms else 0
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, args.output, args.video.name + " · bilateral pupilogram")
    if not args.skip_overlay:
        render(args.video, rows, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
