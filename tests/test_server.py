"""Recording -> saved-video analysis -> API/browser results regressions."""
from __future__ import annotations

import copy
import csv
import io
import json
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from plr import orchestrator, purest
from plr.preview import tracking_image
from plr.settings import DEFAULTS, SettingsStore


@pytest.fixture
def rig(tmp_path, monkeypatch):
    settings = SettingsStore(tmp_path / "settings.json")
    monkeypatch.setattr(server, "SETTINGS", settings)
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "STATE", {**copy.deepcopy(server.STATE), "stage": "idle", "error": None,
                                        "session_id": None, "session_dir": None})
    monkeypatch.setattr(server, "BACKGROUND", {"thread": None, "cancel": None, "info": None})
    monkeypatch.setattr(server, "SESSION_LOCK", threading.Lock())
    monkeypatch.setattr(server, "ABORT", threading.Event())
    live = SimpleNamespace(error=None, begin_session=lambda: True,
                           end_session=lambda: {"submitted": {}, "results": {}, "dropped": {"left": set(), "right": set()}},
                           status=lambda: {}, latest={"left": None, "right": None})
    monkeypatch.setattr(server, "RIG", {"live": live, "camera": object(), "endstops": None,
                                      "leds": SimpleNamespace(all_off=lambda: None)})
    path = tmp_path / "experiment"
    path.mkdir()
    paths = {"session_id": "experiment", "session_dir": path, "log": path / "session.log"}
    calls = []
    class Recorder:
        def __init__(self, *args): pass
        def run(self):
            calls.append("recorded")
            (path / "full_video.mp4").write_bytes(b"closed-test-video")
            (path / "frames.csv").write_text("frame,sensor_timestamp_ns,time_s\n0,1000,0.0\n")
            return {"timeline": {"warnings": [], "optical": {}}}
    monkeypatch.setattr(server, "SessionRecorder", Recorder)
    monkeypatch.setattr(server, "write_manifest", lambda *args: None)
    monkeypatch.setattr(purest, "live_traces", lambda *args: {"left": {"coverage": 1}, "right": {"coverage": 1}})
    def build(path, sid, traces, source, settings, participant=None, suffix="", **kwargs):
        (path / f"metrics{suffix}.json").write_text(json.dumps({"status": "ok", "session_id": sid, "analysis_source": source, "events": []}))
        (path / f"pupilogram{suffix}.json").write_text(json.dumps({"eyes": {}, "analysis_source": source}))
        return {"status": "ok", "coverage": {"left": 1, "right": 1}}
    monkeypatch.setattr(server, "build_results", build)
    def offline(path, sid, settings, participant, logger, progress=None, **kwargs):
        assert calls == ["recorded"] and server.STATE["stage"] == "analysis"
        assert (path / "full_video.mp4").read_bytes() == b"closed-test-video"
        calls.append("offline")
        progress(.5)
        assert server.STATE["stage"] == "analysis"
        return build(path, sid, {}, "offline", settings, suffix="_offline")
    monkeypatch.setattr(server, "offline_pass", offline)
    return settings.get(), paths, calls, live


def run(rig):
    settings, paths, _, _ = rig
    server.set_state(session_id=paths["session_id"], session_dir=str(paths["session_dir"]))
    server.SESSION_LOCK.acquire()
    server.run_session({}, paths, settings)
    assert not server.SESSION_LOCK.locked()


@pytest.mark.parametrize("report,enabled", [("auto", True), ("live", False), ("offline", True)])
def test_saved_video_always_runs_before_done_even_with_legacy_settings(rig, report, enabled):
    settings, paths, calls, _ = rig
    settings["purest"].update(report=report, offline=enabled)
    run(rig)
    assert calls == ["recorded", "offline"]
    assert server.STATE["stage"] == "done" and server.STATE["analysis_source"] == "offline"
    payload = server.app.test_client().get("/results").get_json()
    assert payload["analysis_source"] == "offline"
    assert json.loads((paths["session_dir"] / "metrics.json").read_text())["analysis_source"] == "offline"


@pytest.mark.parametrize("failure", ["begin", "end", "trace"])
def test_live_failure_cannot_prevent_saved_video_analysis(rig, monkeypatch, failure):
    def fail(*args): raise RuntimeError("live preview unavailable")
    if failure == "begin": rig[3].begin_session = fail
    elif failure == "end": rig[3].end_session = fail
    else: monkeypatch.setattr(purest, "live_traces", fail)
    run(rig)
    assert rig[2] == ["recorded", "offline"] and server.STATE["stage"] == "done"


def test_offline_failure_is_visible_instead_of_reporting_live_success(rig, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError("native executable incompatible: rebuild required")
    monkeypatch.setattr(server, "offline_pass", fail)
    run(rig)
    assert server.STATE["stage"] == "error" and server.STATE["error_stage"] == "analysis"
    assert "rebuild required" in server.STATE["error"]
    assert not (rig[1]["session_dir"] / "metrics.json").exists()


@pytest.mark.parametrize("report", ["live", "auto", "offline"])
def test_existing_settings_migrate_to_post_recording_analysis(tmp_path, report):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"purest": {"report": report, "offline": False}, "leds": {"left": {"power_pct": 73}}}))
    s = SettingsStore(path).get()
    assert s["purest"]["offline"] and s["purest"]["report"] == "offline"
    assert s["leds"]["left"]["power_pct"] == 73


def test_manual_stop_preserves_timing_files_for_analysis(tmp_path, monkeypatch):
    start = time.monotonic_ns() - 100_000_000
    class Camera:
        settings = {"fps": 30}
        def begin_recording(self, path):
            Path(path).write_bytes(b"partial-video")
            return {"started_monotonic_ns": start}
        def end_recording(self): return {"frame_timestamps_ns": [start, start+33_333_333, start+66_666_666]}
    def stop(*args): raise orchestrator.SessionAborted("manual stop")
    monkeypatch.setattr(orchestrator, "_sleep_until", stop)
    paths = {"session_dir": tmp_path, "video": tmp_path / "full_video.mp4", "events": tmp_path / "events.json"}
    recorder = orchestrator.SessionRecorder({"pre_roll_s": 0}, paths, Camera(), SimpleNamespace(all_off=lambda: None),
                                          None, {}, logging.getLogger("test-stop"), threading.Event())
    out = recorder.run()
    assert out["timeline"]["stopped_by_request"]
    with (tmp_path / "frames.csv").open() as f: assert len(list(csv.DictReader(f))) == 3
    assert json.loads(paths["events"].read_text())["stopped_by_request"]


def test_interrupted_flash_is_retained_in_timing(monkeypatch):
    def stamp(colors):
        now = time.monotonic_ns()
        return {"start_ns": now, "end_ns": now, "sides": {s: {"start_ns": now, "end_ns": now} for s in colors}}
    def stop(*args): raise orchestrator.SessionAborted("stop during flash")
    monkeypatch.setattr(orchestrator, "_sleep_until", stop)
    leds = SimpleNamespace(rgb_for=lambda *args: (255, 255, 255), set=stamp)
    recorder = orchestrator.SessionRecorder({"power": {"left": 100}, "enabled": {"left": True}}, {}, None,
                                          leds, None, {}, logging.getLogger("flash"), threading.Event())
    with pytest.raises(orchestrator.SessionAborted): recorder._flash(["left"], "#FFFFFF", 1000, {"left": 100}, {})
    assert len(recorder.commands) == 1 and recorder.commands[0]["side"] == "left"


def sample():
    return {"at": 10., "valid": True, "cx": 150., "cy": 140., "major": 40., "eye_box": [100, 100, 100, 80]}


def test_browser_preview_draws_fresh_boxes_and_pupils_only():
    grey = np.full((240, 320), 140, np.uint8)
    live = SimpleNamespace(latest={"left": sample(), "right": None}, error=None)
    image = tracking_image(grey, [640, 480], live, now=10.1)
    assert image.getpixel((50, 50)) == (65, 155, 255)
    assert image.getpixel((85, 70)) == (70, 255, 70)
    stale = tracking_image(grey, [640, 480], live, now=11.)
    assert stale.getpixel((50, 50)) == (140, 140, 140)


def test_browser_monitor_and_overlay_endpoints(rig):
    from plr.camera import _FrameSink
    sink = _FrameSink()
    sink.publish(np.full((240, 320), 140, np.uint8), 1000)
    server.RIG["camera"] = SimpleNamespace(sink=sink)
    s = sample(); s["at"] = time.monotonic()
    server.RIG["live"].latest["left"] = s
    client = server.app.test_client()
    page = client.get("/monitor")
    assert page.status_code == 200 and b'canvas id="left"' in page.data and b'canvas id="right"' in page.data
    overlay = client.get("/preview.jpg")
    raw = client.get("/preview.jpg?overlay=0")
    assert overlay.status_code == 200 and overlay.data != raw.data
    assert Image.open(io.BytesIO(overlay.data)).mode == "RGB"


def test_old_native_binary_requires_rebuild(tmp_path, monkeypatch):
    path = tmp_path / "old-tracker"
    path.write_text("#!/bin/sh\necho 2.2.0-softflow-lite-eyes\n")
    path.chmod(0o755)
    monkeypatch.setenv("PLR_PUREST_BIN", str(path))
    result = purest.binary_status()
    assert not result["ready"] and "scripts/build_native.sh" in result["error"]


def test_latest_preview_status_is_strict_json():
    engine = purest.LiveEngine(copy.deepcopy(DEFAULTS))
    s = sample(); s["major"] = float("nan")
    engine.latest["left"] = s
    result = engine.status()
    assert result["latest"]["left"]["major"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.skipif(not purest.binary_status()["ready"], reason="native Soft Flow Lite not built")
@pytest.mark.parametrize("manual_stop", [False, True])
def test_real_recording_handoff_runs_native_and_exposes_both_results(tmp_path, monkeypatch, manual_stop):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for mock camera recording")
    from plr.camera import MockCameraService
    settings = copy.deepcopy(DEFAULTS)
    settings["purest"].update(live=False, cpus=[], min_pupil_px=20, max_pupil_px=80)
    settings["eye_localization"]["mode"] = "fixed"
    camera = MockCameraService(settings["camera"])
    camera.start()
    led = SimpleNamespace(all_off=lambda: None)
    live = SimpleNamespace(error="live disabled", status=lambda: {})
    monkeypatch.setattr(server, "RIG", {"camera": camera, "leds": led, "live": live, "endstops": None})
    monkeypatch.setattr(server, "SESSION_LOCK", threading.Lock())
    monkeypatch.setattr(server, "ABORT", threading.Event())
    monkeypatch.setattr(server, "write_manifest", lambda *args: None)
    monkeypatch.setattr(server, "STATE", {**copy.deepcopy(server.STATE), "stage": "idle", "error": None,
                                        "session_id": "handoff", "session_dir": str(tmp_path)})
    paths = {"session_id": "handoff", "session_dir": tmp_path, "video": tmp_path / "full_video.mp4",
             "events": tmp_path / "events.json", "log": tmp_path / "session.log"}
    config = {"mode": "observe", "schedule": {"duration_ms": 800}, "pre_roll_s": 0, "post_roll_s": 0}
    timer = threading.Timer(.5, server.ABORT.set) if manual_stop else None
    try:
        server.SESSION_LOCK.acquire()
        if timer: timer.start()
        server.run_session(config, paths, settings)
        assert server.STATE["stage"] == "done", server.STATE
        payload = server.app.test_client().get("/results").get_json()
        assert payload["analysis_source"] == "offline"
        for eye in ("left", "right"):
            samples = payload["pupilogram"]["eyes"][eye]["samples"]
            assert len(samples) > 3 and any(s["valid"] and s["diameter_px"] > 0 for s in samples)
        if manual_stop:
            assert json.loads(paths["events"].read_text())["stopped_by_request"]
    finally:
        if timer: timer.cancel()
        camera.stop()
