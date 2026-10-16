"""Hardware-free tests. Run from the repo root:  python3 -m pytest tests -q"""

from __future__ import annotations

import copy
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plr import config_adapter  # noqa: E402
from plr.analysis.plr_metrics import calculate_plr_metrics  # noqa: E402
from plr.endstops import EndstopMonitor  # noqa: E402
from plr.leds import LedConfig  # noqa: E402
from plr.settings import DEFAULTS, SettingsStore, validate  # noqa: E402

def settings():
    return copy.deepcopy(DEFAULTS)


# ── settings ────────────────────────────────────────────────────────────────

def test_defaults_are_valid_and_match_wiring():
    s = validate(settings())
    assert s["leds"]["left"]["gpio"] == 18 and s["leds"]["right"]["gpio"] == 13
    assert s["endstops"]["left"]["gpio"] == 17 and s["endstops"]["right"]["gpio"] == 27
    assert (s["display"]["gpio_dc"], s["display"]["gpio_rst"]) == (24, 25)
    assert s["eye_roi"] == {"left": [0.0, 0.0, 0.5, 1.0], "right": [0.5, 0.0, 0.5, 1.0]}


def test_wiring_doc_matches_defaults():
    """docs/WIRING.md must list the same GPIOs as the defaults."""
    from plr.settings import PHYSICAL_PIN
    text = (Path(__file__).resolve().parents[1] / "docs" / "WIRING.md").read_text()
    s = settings()
    for gpio in (s["leds"]["left"]["gpio"], s["leds"]["right"]["gpio"], s["endstops"]["left"]["gpio"],
                 s["endstops"]["right"]["gpio"], s["display"]["gpio_dc"], s["display"]["gpio_rst"]):
        assert f"GPIO{gpio}" in text and f"| {PHYSICAL_PIN[gpio]} |" in text


@pytest.mark.parametrize("patch", [
    {"leds": {"left": {"gpio": 24}}},        # TFT DC
    {"leds": {"right": {"gpio": 10}}},       # SPI0 MOSI
    {"endstops": {"left": {"gpio": 8}}},     # SPI0 CE0 (TFT)
    {"eye_roi": {"left": [0.6, 0, 0.5, 1]}},  # outside the frame
    {"purest": {"report": "sometimes"}},
    {"purest": {"live": False, "offline": False}},
    {"purest": {"work_size": 1000}},
    {"purest": {"flow_frames": 6}},
    {"purest": {"reacquire_every": 0}},
    {"purest": {"work_size": 320.5}},
    {"purest": {"min_contrast": 0}},
    {"purest": {"max_gap_ms": float("nan")}},
    {"purest": {"glint_kernel": 8}},
    {"eye_localization": {"mode": "guess"}},
    {"eye_localization": {"interval": 1.5}},
    {"eye_localization": {"ttl_ms": float("nan")}},
    {"eye_localization": {"swap_eyes": "yes"}},
    {"endstops": {"right": {"gpio": 18}}},   # left LED
    {"leds": {"left": {"power_pct": 150}}},
])
def test_invalid_settings_rejected(tmp_path, patch):
    store = SettingsStore(tmp_path / "s.json")
    with pytest.raises(ValueError):
        store.update(patch)


def test_settings_persist(tmp_path):
    path = tmp_path / "s.json"
    SettingsStore(path).update({"leds": {"right": {"power_pct": 55}}})
    assert SettingsStore(path).get()["leds"]["right"]["power_pct"] == 55
    assert SettingsStore(path).get()["leds"]["left"]["power_pct"] == 100


# ── app request compatibility ──────────────────────────────────────────────

def test_old_app_dual_payload():
    payload = {
        "participant": {"name": "A"}, "controlMode": "dual", "pupilAnalyzer": "purest",
        "pupilPreprocess": "imagenet", "intensity": 70,
        "schedule": {"flashes": [{"hex": "#ff0000", "duration": 1, "intensity": 70}],
                     "initialBreak": 2, "gap": 1, "intensity": 70},
    }
    cfg = config_adapter.adapt(payload, settings())
    assert cfg["mode"] == "dual" and cfg["analyzer"] == "purest"
    flash = cfg["schedule"]["flashes"][0]
    assert flash["hex"] == "#FF0000" and flash["duration_ms"] == 1000
    assert flash["intensity"] == {"left": 70.0, "right": 70.0}
    assert cfg["schedule"]["gap_ms"] == 3000          # minimum gap enforced
    assert cfg["schedule"]["initial_delay_ms"] == 2000
    assert cfg["power"] == {"left": 100.0, "right": 100.0}


def test_old_app_sequential_payload():
    payload = {"controlMode": "right_to_left", "schedule": {
        "rounds": 2, "hex": "#FFFFFF", "duration": 0.5, "innerPause": 1, "gap": 4, "intensity": 50}}
    cfg = config_adapter.adapt(payload, settings())
    sched = cfg["schedule"]
    assert sched["order"] == ["right", "left"] and sched["active_side"] is None
    assert sched["rounds"] == 2 and sched["gap_ms"] == 4000
    assert sched["intensity"] == {"left": 50.0, "right": 50.0}


def test_legacy_single_eye_payload():
    cfg = config_adapter.adapt({"eye": "Right", "color": "All", "iterations": 2}, settings())
    assert cfg["mode"] == "right_to_left" and cfg["schedule"]["active_side"] == "right"
    assert cfg["schedule"]["rounds"] == 10 and len(cfg["schedule"]["colors_cycle"]) == 5


def test_new_per_eye_power_and_intensity():
    payload = {"controlMode": "dual",
               "leds": {"left": {"power": 80}, "right": {"power": 50, "intensity": 40}},
               "schedule": {"flashes": [{"hex": "#FFFFFF", "duration": 1, "leftIntensity": 90}], "intensity": 100}}
    cfg = config_adapter.adapt(payload, settings())
    assert cfg["power"] == {"left": 80.0, "right": 50.0}
    assert cfg["schedule"]["flashes"][0]["intensity"] == {"left": 90.0, "right": 40.0}


def test_any_requested_analyzer_runs_purest():
    """The unchanged app may still send pupilAnalyzer=resnet18; PuReST is used."""
    for requested in (None, "purest", "resnet18", "anything"):
        payload = {"controlMode": "observe"}
        if requested:
            payload["pupilAnalyzer"] = requested
        cfg = config_adapter.adapt(payload, settings())
        assert cfg["analyzer"] == "purest" and cfg["requested_analyzer"] == requested


@pytest.mark.parametrize("payload", [
    {"controlMode": "dual", "schedule": {"flashes": []}},
    {"controlMode": "dual", "schedule": {"flashes": [{"hex": "white", "duration": 1}]}},
    {"controlMode": "sideways"},
])
def test_bad_requests_rejected(payload):
    with pytest.raises(ValueError):
        config_adapter.adapt(payload, settings())


# ── LED brightness model ───────────────────────────────────────────────────

def test_brightness_is_power_times_intensity():
    cfg = LedConfig("left", 18, power_pct=50)
    assert cfg.scaled_rgb("#FFFFFF", 40) == (51, 51, 51)
    assert cfg.scaled_rgb("#FF8000", 100, power_pct=100) == (255, 128, 0)
    cfg.channel_gain = {"r": 0.5, "g": 1.0, "b": 1.0}
    assert cfg.scaled_rgb("#FFFFFF", 100, power_pct=100) == (128, 255, 255)


def test_ws2812_byte_order():
    assert LedConfig("left", 18, color_order="GRB").frame((1, 2, 3)) == bytearray([2, 1, 3])
    assert LedConfig("left", 18, color_order="RGB", pixel_count=2).frame((1, 2, 3)) == bytearray([1, 2, 3, 1, 2, 3])


# ── endstop pulse merging ──────────────────────────────────────────────────

def _monitor_with_edges(edges):
    monitor = EndstopMonitor(settings()["endstops"])
    monitor.sides = {"left": {"gpio": 17}, "right": {"gpio": 27}}
    for edge in edges:
        monitor._edges.append(edge)
    return monitor


def test_pwm_burst_merges_into_one_flash():
    ms = 1_000_000
    edges = []
    for k in range(50):                          # 1.2 kHz PWM for ~40 ms
        t = 100 * ms + k * 833_333
        edges += [("left", True, t), ("left", False, t + 200_000)]
    edges += [("left", True, 500 * ms), ("left", False, 600 * ms)]   # second flash
    pulses = _monitor_with_edges(edges).pulses("left", 0, 10**12)
    assert len(pulses) == 2
    assert pulses[0]["onset_ns"] == 100 * ms and pulses[0]["raw_pulses"] == 50
    assert pulses[1]["onset_ns"] == 500 * ms


def test_pulses_are_per_side():
    edges = [("left", True, 10), ("right", True, 20), ("left", False, 30), ("right", False, 40)]
    monitor = _monitor_with_edges(edges)
    assert monitor.pulses("left", 0, 100)[0]["onset_ns"] == 10
    assert monitor.pulses("right", 0, 100)[0]["onset_ns"] == 20


# ── timeline: command -> optical matching ─────────────────────────────────

class _FakeLeds:
    def all_off(self):
        pass


def test_optical_onset_used_and_fallback_flagged():
    from plr.orchestrator import SessionRecorder

    ms = 1_000_000
    t0 = 1_000 * ms
    frames = [t0 + i * 33_333_333 for i in range(300)]
    edges = [("left", True, t0 + 2_000 * ms + 300_000), ("left", False, t0 + 3_000 * ms)]
    monitor = _monitor_with_edges(edges)
    camera = type("Cam", (), {"settings": {"fps": 30}})()
    recorder = SessionRecorder({"mode": "dual"}, {}, camera, _FakeLeds(), monitor,
                               {"max_latency_ms": 30}, None, threading.Event())

    def command(side, on_ms):
        return {"side": side, "hex_color": "#FFFFFF", "intensity_pct": 100.0, "power_pct": 100.0,
                "rgb": [255, 255, 255], "enabled": True, "duration_ms": 1000, "mode": "dual",
                "cmd_on_start_ns": t0 + on_ms * ms, "cmd_on_end_ns": t0 + on_ms * ms + 50_000,
                "cmd_off_start_ns": t0 + (on_ms + 1000) * ms, "cmd_off_end_ns": t0 + (on_ms + 1000) * ms + 50_000}

    recorder.commands = [command("left", 2_000), command("right", 2_000)]
    timeline = recorder._build_timeline(frames, t0, t0 + 10_000 * ms)
    left = next(e for e in timeline["events"] if e["eye"] == "left")
    right = next(e for e in timeline["events"] if e["eye"] == "right")
    assert left["timestamp_source"] == "endstop_optical"
    assert left["onset_s"] == pytest.approx(2.0003, abs=1e-6)
    assert left["optical_latency_ms"] == pytest.approx(0.25, abs=1e-6)
    assert left["onset_frame"] == 61                     # first frame at/after onset
    assert right["optical_detected"] is False
    assert right["timestamp_source"].startswith("ws2812_command")
    assert any("not seen" in w for w in timeline["warnings"])


# ── metrics use measured frame times ───────────────────────────────────────

def test_metrics_with_sensor_times_and_dropped_frames():
    fps = 30.0
    frames = np.arange(120)
    times = frames / fps
    times[20:] += 0.1                                    # 3 frames dropped before the flash
    onset = 1.0
    diam = np.where(times < onset + 0.25, 40.0, 40.0 - 8.0 * (1 - np.exp(-(times - onset - 0.25) / 0.15)))
    with_times = calculate_plr_metrics(frames, diam, fps, onset, smoothing_window=1, times_s=times)
    assert with_times["mca_px"] == pytest.approx(8.0, abs=0.3)
    assert 230 <= with_times["latency_ms"] <= 300
    naive = calculate_plr_metrics(frames, diam, fps, onset, smoothing_window=1)
    assert abs(naive["latency_ms"] - with_times["latency_ms"]) > 50   # nominal fps is ~100 ms off


# ── TFT must not claim GPIO18 (left LED) ──────────────────────────────────

def test_display_never_claims_gpio18(monkeypatch):
    """luma.lcd defaults to a backlight on GPIO18; the rig passes a no-op one."""
    import types
    from plr.display import NoBacklight, TftDisplay

    captured = {}
    serial_mod = types.ModuleType("luma.core.interface.serial")
    serial_mod.spi = lambda **kw: captured.setdefault("spi", kw) or kw
    device_mod = types.ModuleType("luma.lcd.device")

    def st7789(serial, **kw):
        captured["st7789"] = kw
        return object()

    device_mod.st7789 = st7789
    for name in ("luma", "luma.core", "luma.core.interface", "luma.lcd"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "luma.core.interface.serial", serial_mod)
    monkeypatch.setitem(sys.modules, "luma.lcd.device", device_mod)

    s = settings()
    TftDisplay(s["display"], s["camera"], s["eye_roi"], camera=None, state_provider=dict)._make_device()
    assert isinstance(captured["st7789"]["backlight"], NoBacklight)
    assert "gpio_LIGHT" not in captured["st7789"]
    assert {captured["spi"]["gpio_DC"], captured["spi"]["gpio_RST"]} == {24, 25}
    assert 18 not in captured["spi"].values()


# ── camera frames handed to PuReST must not reference the camera buffer ────

def test_live_frames_are_copied_out_of_the_camera_buffer(monkeypatch):
    """A view escaping MappedArray made picamera2 fail (BufferError) and later SEGV."""
    import mmap
    import types
    from contextlib import contextmanager
    from plr.camera import CameraService

    w, h = 640, 480
    buf = mmap.mmap(-1, w * h * 3 // 2)       # YUV420 frame, like libcamera's dmabuf

    @contextmanager
    def MappedArray(request, stream):
        mapped = types.SimpleNamespace(array=np.frombuffer(buf, dtype=np.uint8).reshape(h * 3 // 2, w))
        try:
            yield mapped
        finally:
            mapped.array = None                # like picamera2: drop its own reference
            buf.close()                        # BufferError if the caller kept a view

    picamera2 = types.ModuleType("picamera2")
    picamera2.MappedArray = MappedArray
    monkeypatch.setitem(sys.modules, "picamera2", picamera2)

    got = []

    class Tap:
        def wants(self, ts):
            return True

        def submit(self, grey, seq, ts):
            got.append(grey)

    cam = CameraService(settings()["camera"])
    cam.frame_tap = Tap()
    cam._last_lores_ns = 2 ** 62               # skip the lores copy in this test
    request = types.SimpleNamespace(get_metadata=lambda: {"SensorTimestamp": 123})
    cam._on_frame(request)
    assert buf.closed                          # the camera buffer could be released
    assert len(got) == 1 and got[0].shape == (h, w) and got[0].flags.owndata


# ── WS2812 helper process ──────────────────────────────────────────────────

def _helper_leds(tmp_path, monkeypatch):
    from plr.leds import LedController
    log = tmp_path / "led.log"
    monkeypatch.setenv("PLR_LED_HELPER_FAKE", str(log))
    monkeypatch.delenv("PLR_HARDWARE", raising=False)
    leds = LedController(settings()["leds"])
    assert leds.ready and leds.status()["backend"] == "pi5_pio_helper", leds.error
    return leds, log


def test_led_helper_writes_both_leds_in_order_from_any_thread(tmp_path, monkeypatch):
    leds, log = _helper_leds(tmp_path, monkeypatch)
    try:
        log.write_text("")
        results = []
        worker = threading.Thread(target=lambda: results.append(leds.set({"left": (255, 0, 0), "right": (0, 0, 255)})))
        worker.start(); worker.join(10)
        env = results[0]
        assert log.read_text().split("\n")[:2] == ["18 00ff00", "13 0000ff"]   # GRB byte order
        l, r = env["sides"]["left"], env["sides"]["right"]
        assert env["start_ns"] <= l["start_ns"] <= l["end_ns"] <= r["start_ns"] <= r["end_ns"] <= env["end_ns"]
        assert leds.status()["leds"]["left"]["current_rgb"] == [255, 0, 0]
    finally:
        leds.close()


def test_led_helper_crash_does_not_take_the_server_down(tmp_path, monkeypatch):
    leds, log = _helper_leds(tmp_path, monkeypatch)
    try:
        leds.backend._proc.kill()                       # simulate the driver segfaulting
        leds.backend._proc.wait()
        leds.set({"left": (1, 2, 3)})                   # next write starts a fresh helper
        assert leds.status()["helper_restarts"] == 1
        assert "exited" in leds.status()["helper_last_crash"]
        assert log.read_text().strip().endswith("18 020103")
    finally:
        leds.close()
