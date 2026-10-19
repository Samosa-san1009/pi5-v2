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
    s["eye_localization"]["mode"] = "fixed"  # these fixtures are pre-cropped synthetic eyes
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
        assert t["coverage"] == 1.0 and t["valid"][1:].all()
        assert not t["valid"][0]  # independent initial detection needs a second observation
        truth = np.array([2 * radius_at(i, side) for i in range(len(frames))])
        assert np.nanmedian(np.abs(t["diameter_px"] - truth)) < 1.5
        # full-frame coordinates: left pupil in the left half, right in the right half
        assert abs(np.nanmedian(t["center_x"]) - (W / 4 if side == "left" else 3 * W / 4)) < 2


def _stream(frames, extra=(), seqs=None, timestamps=None, resets=None, preview=False):
    cmd = [str(purest.binary_path()), "--stream", "--eye-mode", "fixed", *extra]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = bytearray()
    for i, f in enumerate(frames):
        seq = seqs[i] if seqs is not None else i
        ts = timestamps[i] if timestamps is not None else 1_000_000_000 + i * 33_333_333
        reset = resets[i] if resets is not None else i == 0
        data += purest._HEADER.pack(b"PLR1", int(reset) | (2 if preview else 0), seq, ts, W, H) + f.tobytes()
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
    assert pupilogram["analyzer_config"]["left"]["variant"] == "softflow_lite"
    assert len(pupilogram["eyes"]["left"]["tracking"]) == n
    assert pupilogram["eyes"]["left"]["tracking"][0]["source"] == "purest"
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


def _left_samples(lines):
    return [l.split() for l in lines if l.startswith("P left ")]


def test_blank_and_flat_eyelid_are_not_recovered():
    clear = eye_frame(0)
    blank = np.full_like(clear, 150)
    slit = blank.copy()
    slit[235:245, 100:220] = 20
    slit[235:245, 420:540] = 20
    frames = [clear] * 3 + [blank] * 3 + [slit] * 4 + [clear] * 3
    lines, code = _stream(frames, ("--queue-depth", "100", "--min-pupil-px", "20", "--max-pupil-px", "80"))
    assert code == 0
    samples = _left_samples(lines)
    assert all(p[4] == "0" and p[14] == "0" for p in samples[3:10])
    assert all(p[15] == "absent" for p in samples[3:6])
    assert samples[-1][4] == "1"  # recovery after a blink


def _glint_frames():
    yy, xx = np.mgrid[:H, :W]
    clear = eye_frame(0)
    damaged = clear.copy()
    for cx in (W // 4, 3 * W // 4):
        damaged[(xx - cx - 27) ** 2 + (yy - H // 2) ** 2 < 8 ** 2] = 230
    return [clear] * 3 + [damaged] * 7 + [clear] * 3


def test_flow_estimates_are_bounded_and_never_measurements():
    args = ("--queue-depth", "100", "--confidence", ".99", "--min-pupil-px", "20", "--max-pupil-px", "80", "--glint-kernel", "0")
    lines, code = _stream(_glint_frames(), args)
    assert code == 0
    samples = _left_samples(lines)
    estimates = [p for p in samples if p[13] == "flow_estimate"]
    assert estimates and len(estimates) <= 3
    assert all(p[4] == "0" and p[14] == "1" for p in estimates)
    assert all(p[14] == "0" for p in samples[6:10])
    lines, code = _stream(_glint_frames(), (*args, "--flow-frames", "0"))
    assert code == 0 and not any(p[14] == "1" for p in _left_samples(lines))


@pytest.mark.parametrize("boundary", ["sequence", "timestamp", "reset"])
def test_flow_does_not_cross_discontinuities(boundary):
    frames = _glint_frames()[:4]
    seqs = [0, 1, 2, 10] if boundary == "sequence" else None
    stamps = [1_000_000_000, 1_033_333_333, 1_066_666_666, 1_500_000_000] if boundary == "timestamp" else None
    resets = [True, False, False, True] if boundary == "reset" else None
    lines, code = _stream(frames, ("--queue-depth", "100", "--confidence", ".99",
                                  "--min-pupil-px", "20", "--max-pupil-px", "80", "--glint-kernel", "0"), seqs, stamps, resets)
    assert code == 0 and _left_samples(lines)[-1][14] == "0"


def test_csv_flow_estimate_is_excluded_even_with_zero_threshold(tmp_path):
    path = tmp_path / "flow.csv"
    path.write_text("Frame,Valid,Confidence,Outline_confidence,Diameter_px,Minor_px,Center_x,Center_y,Angle_deg,Proc_us,Source,Estimated,Visibility\n"
                    "0,1,0.9,0.8,40,38,10,10,0,100,purest,0,visible\n"
                    "1,1,0.6,0,42,38,11,10,0,100,flow_estimate,1,visible\n")
    trace = purest.read_csv_trace(path, 2, 0)
    assert list(trace["valid"]) == [True, False]
    assert np.isnan(trace["diameter_px"][1]) and trace["estimate_px"][1] == 42
    assert trace["tracking_source"][1] == "flow_estimate"


def test_auto_mode_does_not_measure_frame_halves_when_eyes_are_missing():
    lines, code = _stream([np.full((H, W), 140, np.uint8)] * 8,
                          ("--eye-mode", "auto", "--queue-depth", "100"))
    assert code == 0
    for line in lines:
        if line.startswith("P "):
            p = line.split()
            assert len(p) == 22 and p[4] == "0" and p[13] == "missing"
            assert p[18:20] == ["0", "0"] and p[20] == "missing"


def test_missing_localizer_models_fail_clearly(tmp_path):
    proc = subprocess.run([str(purest.binary_path()), "--stream", "--cascade-dir", str(tmp_path)],
                          input=b"", capture_output=True, timeout=10)
    assert proc.returncode != 0
    assert b"eye localization cascades missing" in proc.stderr
    assert b"READY" not in proc.stdout


def test_isolated_size_jump_is_rejected():
    yy, xx = np.mgrid[:H, :W]
    clear = eye_frame(0)
    large = clear.copy()
    for cx in (W // 4, 3 * W // 4):
        large[(xx - cx) ** 2 + (yy - H // 2) ** 2 < 42 ** 2] = 20
    lines, code = _stream([clear] * 3 + [large] + [clear] * 3,
                          ("--queue-depth", "100", "--min-pupil-px", "20", "--max-pupil-px", "100"))
    assert code == 0 and _left_samples(lines)[3][4] == "0"
    assert _left_samples(lines)[-1][4] == "1"


def test_idle_preview_confirms_pupils_at_five_fps_despite_camera_sequence_gaps():
    frames = [eye_frame(0)] * 8
    stamps = [1_000_000_000 + i * 200_000_000 for i in range(len(frames))]
    seqs = [i * 6 for i in range(len(frames))]
    lines, code = _stream(frames, ("--queue-depth", "100", "--min-pupil-px", "20", "--max-pupil-px", "80"),
                          seqs=seqs, timestamps=stamps, preview=True)
    assert code == 0
    samples = _left_samples(lines)
    assert samples[0][4] == "0"
    assert all(p[4] == "1" for p in samples[1:])
    assert all(p[14] == "0" for p in samples)  # never interpolate idle preview
    strict, _ = _stream(frames, ("--queue-depth", "100"), seqs=seqs, timestamps=stamps)
    assert all(p[4] == "0" for p in _left_samples(strict))  # recording gap rules stay strict


@pytest.mark.skipif(not os.environ.get("PLR_TEST_VIDEO"), reason="set PLR_TEST_VIDEO to a local full-face video")
def test_full_face_sample_localization_and_measurement(tmp_path):
    video = Path(os.environ["PLR_TEST_VIDEO"])
    for swap in (False, True):
        s = settings()
        s["eye_localization"].update(mode="auto", swap_eyes=swap)
        out = tmp_path / str(swap)
        traces = purest.run_offline(video, out, s, frame_count=0)
        for eye in purest.EYES:
            trace = traces[eye]
            assert trace["analysed"] > 100
            assert trace["valid"].mean() > .75
            boxes = trace["eye_boxes"]
            valid = trace["valid"]
            assert (trace["center_x"][valid] >= boxes[valid, 0]).all()
            assert (trace["center_x"][valid] < boxes[valid, 0] + boxes[valid, 2]).all()
        left_x = np.nanmedian(traces["left"]["center_x"])
        right_x = np.nanmedian(traces["right"]["center_x"])
        assert (left_x < right_x) if swap else (left_x > right_x)


@pytest.mark.skipif(not os.environ.get("PLR_TEST_VIDEO"), reason="set PLR_TEST_VIDEO to a local full-face video")
def test_full_face_idle_preview_refreshes_boxes_before_expiry():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", os.environ["PLR_TEST_VIDEO"],
                                   "-frames:v", "120", "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)[::6]
    seqs = [i * 6 for i in range(len(frames))]
    stamps = [1_000_000_000 + i * 200_000_000 for i in range(len(frames))]
    lines, code = _stream(frames, ("--eye-mode", "auto", "--queue-depth", "100"),
                          seqs=seqs, timestamps=stamps, preview=True)
    assert code == 0
    for eye in ("left", "right"):
        samples = [line.split() for line in lines if line.startswith(f"P {eye} ")]
        assert sum(int(p[18]) > 0 for p in samples) > .8 * len(samples)
        assert sum(p[4] == "1" for p in samples) > .5 * len(samples)
