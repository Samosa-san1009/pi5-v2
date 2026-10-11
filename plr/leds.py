"""Two independent WS2812 LEDs, one per eye, each on its own GPIO.

Brightness model (per colour channel, 0-255):

    byte = hex_channel x power_pct/100 x intensity_pct/100 x channel_gain

- hex_channel   colour requested by the app (#RRGGBB)
- power_pct     per-LED setting (device settings or the session request);
                use it to balance the two eyes
- intensity_pct per-flash stimulus intensity from the app
- channel_gain  optional per-channel calibration (0-1)

On a Pi 5 the LEDs are driven through the RP1 PIO block with Adafruit's
`adafruit_raspberry_pi5_neopixel_write`, which works on any GPIO and needs no
root. The command timestamps returned here are only the moment the data was
handed to the driver; the true light onset is measured by the endstops.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

SIDES = ("left", "right")


class LedHardwareUnavailable(RuntimeError):
    pass


def parse_hex(hex_color: str) -> tuple[int, int, int]:
    text = str(hex_color or "").strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected #RRGGBB colour, got {hex_color!r}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


@dataclass
class LedConfig:
    side: str
    gpio: int
    power_pct: float = 100.0
    enabled: bool = True
    pixel_count: int = 1
    color_order: str = "GRB"
    channel_gain: dict | None = None

    @classmethod
    def from_settings(cls, side: str, raw: dict) -> "LedConfig":
        return cls(
            side=side,
            gpio=int(raw["gpio"]),
            power_pct=float(raw.get("power_pct", 100.0)),
            enabled=bool(raw.get("enabled", True)),
            pixel_count=int(raw.get("pixel_count", 1)),
            color_order=str(raw.get("color_order", "GRB")).upper(),
            channel_gain=dict(raw.get("channel_gain") or {}),
        )

    def scaled_rgb(self, hex_color: str, intensity_pct: float, power_pct: float | None = None) -> tuple[int, int, int]:
        power = self.power_pct if power_pct is None else float(power_pct)
        scale = max(0.0, min(100.0, power)) / 100.0 * max(0.0, min(100.0, float(intensity_pct))) / 100.0
        gains = self.channel_gain or {}
        out = []
        for name, value in zip("rgb", parse_hex(hex_color)):
            gain = max(0.0, min(1.0, float(gains.get(name, 1.0))))
            out.append(int(round(value * scale * gain)))
        return tuple(out)  # type: ignore[return-value]

    def frame(self, rgb: tuple[int, int, int]) -> bytearray:
        lookup = dict(zip("RGB", rgb))
        pixel = bytes(lookup[c] for c in self.color_order)
        return bytearray(pixel * self.pixel_count)


class _PioBackend:
    """Raspberry Pi 5 PIO WS2812 output (any GPIO)."""

    name = "pi5_pio"

    def __init__(self):
        try:
            import board  # adafruit-blinka
            from adafruit_raspberry_pi5_neopixel_write import neopixel_write
        except Exception as exc:
            raise LedHardwareUnavailable(
                "Pi 5 WS2812 driver unavailable: install "
                "'Adafruit-Blinka-Raspberry-Pi5-Neopixel' and check /dev/pio0 "
                f"exists ({type(exc).__name__}: {exc})"
            ) from exc
        self._board = board
        self._write = neopixel_write
        self._pins: dict[int, object] = {}

    def write(self, gpio: int, data: bytearray):
        pin = self._pins.get(gpio)
        if pin is None:
            pin = getattr(self._board, f"D{gpio}")
            self._pins[gpio] = pin
        self._write(pin, data)


class _MockBackend:
    name = "mock"

    def __init__(self):
        self.listeners = []
        self.side_of_gpio: dict[int, str] = {}

    def write(self, gpio: int, data: bytearray):
        for listener in list(self.listeners):
            listener(gpio, bytes(data), time.monotonic_ns())


class LedController:
    """Thread-safe owner of both LEDs for the lifetime of the server."""

    def __init__(self, led_settings: dict, backend: str | None = None):
        requested = (backend or os.environ.get("PLR_LED_BACKEND") or led_settings.get("backend") or "auto").lower()
        if os.environ.get("PLR_HARDWARE", "").lower() == "mock":
            requested = "mock"
        self.error: str | None = None
        if requested == "mock":
            self.backend = _MockBackend()
        else:
            try:
                self.backend = _PioBackend()
            except LedHardwareUnavailable as exc:
                if requested == "auto" and os.environ.get("PLR_ALLOW_MOCK") == "1":
                    self.backend = _MockBackend()
                    self.error = str(exc)
                else:
                    self.backend = None
                    self.error = str(exc)
        self.configs = {side: LedConfig.from_settings(side, led_settings[side]) for side in SIDES}
        self._map_mock_pins()
        self._lock = threading.Lock()
        self._state = {side: (0, 0, 0) for side in SIDES}
        self.state_listeners = []  # callables(side, rgb, monotonic_ns) for the display
        if self.backend is not None:
            try:
                self.all_off()
            except Exception as exc:  # e.g. permission denied on /dev/pio0
                self.error = f"WS2812 write failed: {type(exc).__name__}: {exc}"
                self.backend = None

    def _map_mock_pins(self):
        if isinstance(self.backend, _MockBackend):
            self.backend.side_of_gpio = {cfg.gpio: side for side, cfg in self.configs.items()}

    # ── status ──────────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self.backend is not None

    @property
    def is_mock(self) -> bool:
        return isinstance(self.backend, _MockBackend)

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "backend": getattr(self.backend, "name", None),
            "mock": self.is_mock,
            "error": self.error,
            "leds": {
                side: {
                    "gpio": cfg.gpio,
                    "power_pct": cfg.power_pct,
                    "enabled": cfg.enabled,
                    "current_rgb": list(self._state[side]),
                }
                for side, cfg in self.configs.items()
            },
        }

    def reconfigure(self, led_settings: dict):
        with self._lock:
            old = self.configs
            self.configs = {side: LedConfig.from_settings(side, led_settings[side]) for side in SIDES}
            self._map_mock_pins()
            for side in SIDES:
                if old[side].gpio != self.configs[side].gpio and self.backend is not None:
                    self.backend.write(old[side].gpio, old[side].frame((0, 0, 0)))
        self.all_off()

    # ── output ──────────────────────────────────────────────────────────────
    def set(self, colors: dict[str, tuple[int, int, int]]) -> dict:
        """Set one or both LEDs to raw RGB bytes; return a command envelope.

        Both writes happen back to back inside one lock, so in dual mode the
        two LEDs change within a few tens of microseconds of each other.
        """
        if self.backend is None:
            raise LedHardwareUnavailable(self.error or "LED backend unavailable")
        with self._lock:
            start = time.monotonic_ns()
            per_side = {}
            for side, rgb in colors.items():
                cfg = self.configs[side]
                if not cfg.enabled:
                    rgb = (0, 0, 0)
                t0 = time.monotonic_ns()
                self.backend.write(cfg.gpio, cfg.frame(rgb))
                t1 = time.monotonic_ns()
                self._state[side] = tuple(rgb)
                per_side[side] = {"start_ns": t0, "end_ns": t1, "rgb": list(rgb)}
            end = time.monotonic_ns()
        for side, info in per_side.items():
            for listener in list(self.state_listeners):
                try:
                    listener(side, tuple(info["rgb"]), info["end_ns"])
                except Exception:
                    pass
        return {"start_ns": start, "end_ns": end, "sides": per_side}

    def all_off(self) -> dict | None:
        if self.backend is None:
            return None
        return self.set({side: (0, 0, 0) for side in SIDES})

    def rgb_for(self, side: str, hex_color: str, intensity_pct: float, power_pct: float | None = None):
        return self.configs[side].scaled_rgb(hex_color, intensity_pct, power_pct)

    def close(self):
        try:
            self.all_off()
        except Exception:
            pass
