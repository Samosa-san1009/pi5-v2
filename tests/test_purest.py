"""PuReST engine tests (need the native build; no Pi hardware).

    bash scripts/build_native.sh && python3 -m pytest tests -q
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plr import purest  # noqa: E402
from plr.analysis.results import build_results  # noqa: E402
from plr.settings import DEFAULTS  # noqa: E402

pytestmark = pytest.mark.skipif(not purest.binary_status()["ready"], reason="native PuReST not built")
W, H = 640, 480


def settings(**purest_overrides):
    s = copy.deepcopy(DEFAULTS)
    s["purest"]["cpus"] = []            # no pinning on a test machine
    s["purest"].update(purest_overrides)
    return s


def radius_at(i: int, side: str) -> float:
    """Known pupil radius: constricts after frame 30 (left) / 40 (right)."""
    start = 30 if side == "left" else 40
    return 30.0 if i < start else 30.0 - 10.0 * (1 - np.exp(-(i - start) / 6.0))


def eye_frame(i: int, rng=None) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.full((H, W), 150.0, np.float32)
    for side, cx in (("left", W / 4), ("right", 3 * W / 4)):
        cy = H / 2
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img -= 22.0 * np.clip((48 - d) / 6 + 0.5, 0, 1)
        pupil = np.clip((radius_at(i, side) - d) / 1.5 + 0.5, 0, 1)
        img = img * (1 - pupil) + 20 * pupil
    if rng is not None:
        img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    return [eye_frame(i, rng) for i in range(90)]


@pytest.fixture(scope="module")
def video(tmp_path_factory, frames):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("vid") / "full_video.mp4"
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                             "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "10",
                             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def test_offline_tracks_both_eyes_every_frame(video, tmp_path, frames):
    # Pixel limits as a user would set them: pupils here are 40-60 px, the iris 96 px.
    traces = purest.run_offline(video, tmp_path, settings(min_pupil_px=20, max_pupil_px=80),
                                frame_count=len(frames))
    for side in ("left", "right"):
        t = traces[side]
        assert t["coverage"] == 1.0 and t["valid"].all()
        truth = np.array([2 * radius_at(i, side) for i in range(len(frames))])
        assert np.nanmedian(np.abs(t["diameter_px"] - truth)) < 1.5
        # full-frame coordinates: left pupil in the left half, right in the right half
        assert abs(np.nanmedian(t["center_x"]) - (W / 4 if side == "left" else 3 * W / 4)) < 2


def _stream(frames, extra=()):
    cmd = [str(purest.binary_path()), "--stream", *extra]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = bytearray()
    for i, f in enumerate(frames):
        data += purest._HEADER.pack(b"PLR1", 1 if i == 0 else 0, i, 1000 + i, W, H) + f.tobytes()
    out, err = proc.communicate(bytes(data), timeout=60)
    return out.decode().splitlines(), proc.returncode


def test_stream_mode_matches_offline_and_reports_every_frame(video, tmp_path, frames):
    lines, code = _stream(frames, extra=("--queue-depth", "200"))   # burst input: no drops
    assert code == 0 and lines[0].startswith("READY") and lines[-1] == f"BYE {len(frames)}"
    results = {"left": {}, "right": {}}
    for line in lines:
        p = line.split()
        if p[0] == "P":
            results[p[1]][int(p[2])] = float(p[7])
    offline = purest.run_offline(video, tmp_path, settings(min_pupil_px=20, max_pupil_px=80),
                                 frame_count=len(frames))
    for side in ("left", "right"):
        assert sorted(results[side]) == list(range(len(frames)))      # one result per frame per eye
        live = np.array([results[side][i] for i in range(len(frames))])
        assert np.nanmedian(np.abs(live - offline[side]["diameter_px"])) < 1.0


def test_stream_drops_instead_of_blocking_when_overloaded(frames):
    lines, code = _stream(frames[:40], extra=("--queue-depth", "2", "--simulate-load-us", "30000"))
    assert code == 0
    for side in ("left", "right"):
        done = {int(l.split()[2]) for l in lines if l.startswith(f"P {side} ")}
        dropped = {int(l.split()[2]) for l in lines if l.startswith(f"D {side} ")}
        assert dropped and not (done & dropped)
        assert done | dropped == set(range(40))                       # every frame accounted for


class _Feeder:
    """Feeds an engine like the camera does (camera-thread semantics)."""

    def __init__(self, engine, frames, fps=60):
        self.engine, self.frames, self.period = engine, frames, 1.0 / fps

    def run(self, start_seq=0):
        stamps = []
        for i, f in enumerate(self.frames):
            ts = 10_000_000_000 + (start_seq + i) * 33_333_333
            if self.engine.wants(ts):
                self.engine.submit(f, start_seq + i, ts)
            stamps.append(ts)
            time.sleep(self.period)
        return stamps


def test_live_engine_session_covers_every_frame(frames):
    engine = purest.LiveEngine(settings(idle_preview_fps=0))
    engine.start()
    try:
        assert engine.ready
        assert not engine.wants(1)                       # idle preview off -> not fed
        assert engine.begin_session()
        stamps = _Feeder(engine, frames).run()
        collected = engine.end_session()
        traces = purest.live_traces(collected, stamps, 0.7)
        for side in ("left", "right"):
            assert traces[side]["coverage"] == 1.0
            truth = np.array([2 * radius_at(i, side) for i in range(len(frames))])
            assert np.nanmedian(np.abs(traces[side]["diameter_px"] - truth)) < 1.5
        assert engine.status()["processing_time"]["left"] is not None
    finally:
        engine.stop()


def test_live_engine_overload_is_accounted_and_lowers_coverage(frames, monkeypatch):
    monkeypatch.setenv("PLR_PUREST_SIMULATE_LOAD_US", "40000")
    engine = purest.LiveEngine(settings(idle_preview_fps=0, queue_depth=2))
    engine.start()
    try:
        engine.begin_session()
        stamps = _Feeder(engine, frames[:60], fps=60).run()
        t0 = time.monotonic()
        collected = engine.end_session(timeout_s=10)
        assert time.monotonic() - t0 < 10                 # finished, did not hit the timeout
        for side in ("left", "right"):
            answered = set(collected["results"][side]) | collected["dropped"][side]
            assert answered == set(collected["submitted"])
            assert collected["dropped"][side]
        traces = purest.live_traces(collected, stamps, 0.7)
        assert all(traces[s]["coverage"] < 0.98 for s in ("left", "right"))   # -> offline is used
    finally:
        engine.stop()


def test_results_have_the_shape_the_app_reads(tmp_path, frames):
    n = len(frames)
    t0 = 5_000_000_000
    stamps = [t0 + i * 33_333_333 for i in range(n)]
    with open(tmp_path / "frames.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["frame", "sensor_timestamp_ns", "time_s"])
        for i, ts in enumerate(stamps):
            w.writerow([i, ts, f"{(ts - t0) / 1e9:.6f}"])
    onset = 30 / 30.0 - 0.2
    events = {"events": [{"event_id": "flash_001", "led_index": 1, "eye": "left", "hex_color": "#FFFFFF",
                          "onset_s": onset, "offset_s": onset + 1, "timestamp_source": "endstop_optical"}],
              "optical": {"available": True}, "warnings": []}
    (tmp_path / "events.json").write_text(json.dumps(events))
    traces = {}
    for side in ("left", "right"):
        t = purest._empty_trace(n)
        for i in range(n):
            purest._fill(t, i, True, 0.9, 2 * radius_at(i, side), 2 * radius_at(i, side), 100, 100, 0.7)
        t["coverage"] = 1.0
        traces[side] = t
    out = build_results(tmp_path, "sid", traces, "live", settings(), {"name": "x"}, suffix="_live")
    assert out["status"] == "ok"
    pupilogram = json.loads((tmp_path / "pupilogram_live.json").read_text())
    metrics = json.loads((tmp_path / "metrics_live.json").read_text())
    sample = pupilogram["eyes"]["left"]["samples"][0]
    assert set(sample) == {"frame", "time_s", "diameter_px", "confidence", "valid"}
    assert pupilogram["analyzer"] == "purest" and metrics["analysis_source"] == "live"
    event = metrics["events"][0]
    assert event["stimulus"]["led_index"] == 1
    for key in ("direct", "consensual"):
        assert event[key]["status"] == "ok"
        assert {"mca_px", "mcv_px_s", "latency_ms"} <= set(event[key])
    assert event["direct"]["mca_px"] == pytest.approx(20, abs=1.5)


def test_pixel_limits_reject_iris_sized_outlines(video, tmp_path, frames):
    """With max_pupil_px below the iris size, no iris-sized sample survives."""
    traces = purest.run_offline(video, tmp_path, settings(max_pupil_px=80), frame_count=len(frames))
    for side in ("left", "right"):
        d = traces[side]["diameter_px"]
        assert np.nanmax(d) <= 80 * 1.15 and np.isfinite(d).mean() > 0.95


def test_csv_threshold_marks_low_confidence_invalid(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("Frame,Valid,Confidence,Outline_confidence,Diameter_px,Minor_px,Center_x,Center_y,Angle_deg,Proc_us\n"
                    "0,1,0.9,0.8,40,38,10,10,0,100\n1,1,0.5,0.4,41,39,10,10,0,100\n2,0,0,0,,,,,,100\n")
    t = purest.read_csv_trace(path, 4, 0.7)
    assert list(t["valid"]) == [True, False, False, False]
    assert t["diameter_px"][0] == 40 and np.isnan(t["diameter_px"][1])
    assert t["coverage"] == 0.75
