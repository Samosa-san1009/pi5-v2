"""Persistent device settings.

Settings describe the *rig* (pins, LED power, camera, display, PuReST). They
live in one JSON file so they survive reboots and `git pull`, and can be
changed at runtime with `POST /settings` without touching the app.

Per-session stimulus choices (colour, duration, intensity) still come from the
app's `/session` request.

Path: $PLR_SETTINGS_PATH, default <repo>/device_settings.json (git-ignored).
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict = {
    "leds": {
        # Each eye has its own WS2812 on its own GPIO (BCM numbering).
        # power_pct scales everything sent to that LED: use it to balance the
        # two eyes so the same requested intensity gives the same light.
        "left": {
            "gpio": 18,               # physical pin 12
            "power_pct": 100.0,
            "enabled": True,
            "pixel_count": 1,
            "color_order": "GRB",
            "channel_gain": {"r": 1.0, "g": 1.0, "b": 1.0},
        },
        "right": {
            "gpio": 13,               # physical pin 33
            "power_pct": 100.0,
            "enabled": True,
            "pixel_count": 1,
            "color_order": "GRB",
            "channel_gain": {"r": 1.0, "g": 1.0, "b": 1.0},
        },
        # "auto" = Pi 5 PIO driver; "mock" for bench work without LEDs.
        "backend": "auto",
    },
    "endstops": {
        # Optical endstops with the IR emitter removed, one facing each WS2812.
        "left": {"gpio": 17, "enabled": True, "active": "auto"},    # physical pin 11
        "right": {"gpio": 27, "enabled": True, "active": "auto"},   # physical pin 13
        # The modules have their own pull-up resistors, so no internal bias.
        "bias": "none",
        # Below full brightness a WS2812 flickers (PWM), so one flash arrives
        # as many short pulses. Pulses closer than this are merged.
        "merge_gap_us": 3000,
        # Accept an optical onset at most this long after the LED command.
        "max_latency_ms": 30.0,
        # Fail the session if a flash is not seen optically (otherwise fall
        # back to the command time and flag the event).
        "require_optical": False,
    },
    "camera": {
        # "auto" = first camera libcamera lists. Set 0/1 if two are connected.
        "index": "auto",
        "resolution": [640, 480],
        "fps": 30,
        "lores": [320, 240],
        "bitrate": 2_000_000,
        # Extra libcamera controls, e.g. {"AeEnable": false,
        # "ExposureTime": 10000, "AnalogueGain": 4.0} for fixed IR exposure.
        "controls": {},
        "hflip": False,
        "vflip": False,
        # Restart the camera if no frame arrives for this long (idle only).
        "watchdog_s": 5.0,
    },
    "display": {
        # GMT020-02 ST7789 on SPI0 CE0. VCC/GND go straight to the Pi.
        "enabled": True,
        "spi_port": 0,
        "spi_device": 0,
        "gpio_dc": 24,                # physical pin 18
        "gpio_rst": 25,               # physical pin 22
        "width": 240,
        "height": 320,
        # 0-3. 1 = landscape so the 4:3 camera picture is not stretched.
        "rotate": 1,
        "bus_speed_hz": 40_000_000,
        "fps": 15,
        "overlay": True,
        # Draw both eye ROIs and the live PuReST pupil outline on the TFT.
        "show_rois": True,
        "show_pupils": True,
    },
    "purest": {
        # PuReST runs on both eyes at the same time, one core each:
        #   live     while recording, on every uncompressed camera frame
        #   offline  after recording, on every frame of the saved video
        "live": True,
        "offline": True,
        # Which result the app gets:
        #   "auto"    live result straight away if both eyes covered at least
        #             live_min_coverage of the frames, else wait for offline
        #   "live"    always the live result (offline still runs afterwards)
        #   "offline" always wait for the offline result
        "report": "auto",
        "live_min_coverage": 0.98,
        # Keep tracking at this rate between sessions, so the TFT shows the
        # pupils while you line the rig up. 0 = only while recording.
        "idle_preview_fps": 5,
        # Frames buffered per eye in the live engine before one is dropped.
        "queue_depth": 6,
        # Both eyes run on these cores; the camera and server use the others.
        "cpus": [2, 3],
        "live_nice": 0,
        "offline_nice": 10,
        "confidence_threshold": 0.70,
        # Pupil size limits in camera pixels (the unit the app shows). Leave
        # None at first. If a session shows the trace jumping to the iris
        # size, set max_pupil_px a little above the largest real pupil
        # (about 1.3x its dark-adapted diameter) and min_pupil_px around half
        # of the smallest constricted pupil.
        "min_pupil_px": None,
        "max_pupil_px": None,
    },
    # Where each eye is in the camera picture: normalised x, y, width, height.
    # "left" is the eye the left LED shines into. Check with /preview.jpg or
    # the TFT (ROI boxes) and swap/adjust if your camera is mounted differently.
    "eye_roi": {
        "left": [0.0, 0.0, 0.5, 1.0],
        "right": [0.5, 0.0, 0.5, 1.0],
    },
    "session": {
        "pre_roll_s": 1.0,
        "post_roll_s": 3.0,
        "min_gap_s": 3.0,
        "window_pre_s": 1.5,
        "window_post_s": 3.0,
    },
    "server": {"port": 5000},
}

# Physical header pins, for messages and docs only (BCM -> physical).
PHYSICAL_PIN = {2: 3, 3: 5, 4: 7, 5: 29, 6: 31, 7: 26, 8: 24, 9: 21, 10: 19, 11: 23, 12: 32, 13: 33,
                14: 8, 15: 10, 16: 36, 17: 11, 18: 12, 19: 35, 20: 38, 21: 40, 22: 15, 23: 16,
                24: 18, 25: 22, 26: 37, 27: 13}


def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "controls":
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _clamp(value, low, high, name):
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


def validate_roi(roi, name: str) -> list[float]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError(f"{name} must be [x, y, width, height] (normalised 0-1)")
    x, y, w, h = (float(v) for v in roi)
    if min(x, y) < 0 or w <= 0 or h <= 0 or x + w > 1.0001 or y + h > 1.0001:
        raise ValueError(f"{name} must stay inside the frame, got {list(roi)}")
    return [x, y, w, h]


def validate(settings: dict) -> dict:
    """Raise ValueError on settings that would misbehave on hardware."""
    used: dict[int, str] = {}

    def claim(gpio, owner):
        gpio = int(gpio)
        if not 2 <= gpio <= 27:
            raise ValueError(f"{owner}: GPIO{gpio} is not a usable header GPIO")
        if gpio in used:
            raise ValueError(f"GPIO{gpio} is used by both {used[gpio]} and {owner}")
        used[gpio] = owner

    display = settings["display"]
    if display.get("enabled"):
        # SPI0 pins belong to the display.
        for gpio, name in ((8, "SPI0 CE0"), (10, "SPI0 MOSI"), (11, "SPI0 SCLK"), (9, "SPI0 MISO")):
            used[gpio] = f"display {name}"
        for key in ("gpio_dc", "gpio_rst"):
            claim(display[key], f"display.{key}")
        _clamp(display["rotate"], 0, 3, "display.rotate")

    for side in ("left", "right"):
        led = settings["leds"][side]
        claim(led["gpio"], f"leds.{side}")
        _clamp(led["power_pct"], 0, 100, f"leds.{side}.power_pct")
        if int(led["pixel_count"]) < 1:
            raise ValueError(f"leds.{side}.pixel_count must be >= 1")
        if sorted(str(led["color_order"]).upper()) != ["B", "G", "R"]:
            raise ValueError(f"leds.{side}.color_order must be a permutation of RGB")
        for channel, gain in led.get("channel_gain", {}).items():
            _clamp(gain, 0, 1, f"leds.{side}.channel_gain.{channel}")
        stop = settings["endstops"][side]
        if stop.get("enabled", True):
            claim(stop["gpio"], f"endstops.{side}")
        if str(stop.get("active", "auto")).lower() not in {"auto", "low", "high"}:
            raise ValueError(f"endstops.{side}.active must be auto, low or high")
        settings["eye_roi"][side] = validate_roi(settings["eye_roi"][side], f"eye_roi.{side}")

    if str(settings["endstops"]["bias"]).lower() not in {"none", "pull_up", "pull_down"}:
        raise ValueError("endstops.bias must be none, pull_up or pull_down")
    width, height = settings["camera"]["resolution"]
    if int(width) % 2 or int(height) % 2:
        raise ValueError("camera.resolution must be even")
    _clamp(settings["camera"]["fps"], 1, 120, "camera.fps")

    purest = settings["purest"]
    if purest["report"] not in {"auto", "live", "offline"}:
        raise ValueError("purest.report must be auto, live or offline")
    if not (purest["live"] or purest["offline"]):
        raise ValueError("purest.live and purest.offline cannot both be off")
    _clamp(purest["live_min_coverage"], 0, 1, "purest.live_min_coverage")
    _clamp(purest["idle_preview_fps"], 0, 30, "purest.idle_preview_fps")
    _clamp(purest["confidence_threshold"], 0, 1, "purest.confidence_threshold")
    _clamp(purest["queue_depth"], 1, 120, "purest.queue_depth")
    for key in ("min_pupil_px", "max_pupil_px"):
        if purest.get(key) is not None:
            _clamp(purest[key], 2, 2000, f"purest.{key}")
    if purest.get("min_pupil_px") and purest.get("max_pupil_px") and purest["min_pupil_px"] >= purest["max_pupil_px"]:
        raise ValueError("purest.min_pupil_px must be smaller than purest.max_pupil_px")
    return settings


class SettingsStore:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path or os.environ.get("PLR_SETTINGS_PATH", REPO_ROOT / "device_settings.json"))
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict:
        data = copy.deepcopy(DEFAULTS)
        if self.path.is_file():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                stored = {k: v for k, v in stored.items() if k in DEFAULTS}   # drop retired sections
                data = validate(_deep_merge(data, stored))
            except Exception as exc:  # a broken file must not stop the rig booting
                print(f"  [settings] ignoring unusable {self.path}: {exc}")
                data = copy.deepcopy(DEFAULTS)
        return validate(data)

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def section(self, name: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._data[name])

    def update(self, patch: dict) -> dict:
        with self._lock:
            candidate = validate(_deep_merge(self._data, patch or {}))
            self._write(candidate)
            self._data = candidate
            return copy.deepcopy(candidate)

    def reset(self) -> dict:
        with self._lock:
            self._data = validate(copy.deepcopy(DEFAULTS))
            self._write(self._data)
            return copy.deepcopy(self._data)

    def _write(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
