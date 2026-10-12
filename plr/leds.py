"""Two independent WS2812 LEDs, one per eye, each on its own GPIO.

Brightness model (per colour channel, 0-255):

    byte = hex_channel x power_pct/100 x intensity_pct/100 x channel_gain

- hex_channel   colour requested by the app (#RRGGBB)
- power_pct     per-LED setting (device settings or the session request);
                use it to balance the two eyes
- intensity_pct per-flash stimulus intensity from the app
- channel_gain  optional per-channel calibration (0-1)

On a Pi 5 the LEDs are driven through the RP1 PIO block with Adafruit's
`adafruit_raspberry_pi5_neopixel_write`. That driver crashed (SEGV) when
called from the server's request threads while the camera was streaming, so
every write happens in a small helper process that owns both LEDs and does
all writes from one thread. If the driver ever crashes again only the helper
dies; the next write starts a new one. The command timestamps returned here
are CLOCK_MONOTONIC (system-wide, so valid across processes) around the
write; the true light onset is measured by the endstops.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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


# ── helper process (owns the PIO driver) ────────────────────────────────────
#
# stdin:  W <gpio>:<hex> [<gpio>:<hex> ...]      write those frames, in order
# stdout: READY | OK <t0>:<t1> [...] | ERR <message>

def _helper_main() -> int:
    import faulthandler

    faulthandler.enable()
    try:
        fake_log = os.environ.get("PLR_LED_HELPER_FAKE")      # tests only
        if fake_log:
            def write(gpio: int, data: bytearray):
                with open(fake_log, "a") as handle:
                    handle.write(f"{gpio} {data.hex()}\n")
        else:
            import board  # adafruit-blinka
            from adafruit_raspberry_pi5_neopixel_write import neopixel_write

            pins: dict[int, object] = {}

            def write(gpio: int, data: bytearray):
                pin = pins.get(gpio)
                if pin is None:
                    pin = pins[gpio] = getattr(board, f"D{gpio}")
                neopixel_write(pin, data)
    except Exception as exc:
        print(f"ERR {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("READY", flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] != "W":
            print("ERR unknown command", flush=True)
            continue
        stamps = []
        try:
            for item in parts[1:]:
                gpio, hexdata = item.split(":")
                t0 = time.monotonic_ns()
                write(int(gpio), bytearray.fromhex(hexdata))
                stamps.append(f"{t0}:{time.monotonic_ns()}")
            print("OK " + " ".join(stamps), flush=True)
        except Exception as exc:
            print(f"ERR {type(exc).__name__}: {exc}", flush=True)
    return 0


class _HelperBackend:
    """Pi 5 PIO WS2812 output, via the helper process."""

    name = "pi5_pio_helper"

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self.restarts = 0
        self.last_crash: str | None = None
        self._spawn()

    def _spawn(self):
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--helper"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = (proc.stdout.readline() or "").strip()
        if line != "READY":
            proc.kill()
            raise LedHardwareUnavailable(
                "Pi 5 WS2812 driver unavailable: install 'Adafruit-Blinka-Raspberry-Pi5-Neopixel' and "
                f"check /dev/pio0 exists ({line or 'helper exited during start-up'})")
        self._proc = proc

    def write_many(self, items: list[tuple[int, bytearray]]) -> list[tuple[int, int]]:
        message = "W " + " ".join(f"{gpio}:{bytes(data).hex()}" for gpio, data in items) + "\n"
        with self._lock:
            for _attempt in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    if self._proc is not None:
                        self.last_crash = f"helper exited with code {self._proc.poll()}"
                    self._spawn()
                    self.restarts += 1
                try:
                    self._proc.stdin.write(message)
                    self._proc.stdin.flush()
                    reply = (self._proc.stdout.readline() or "").strip()
                except (BrokenPipeError, OSError):
                    reply = ""
                if reply.startswith("OK"):
                    return [tuple(int(v) for v in pair.split(":")) for pair in reply.split()[1:]]
                if reply.startswith("ERR"):
                    raise RuntimeError(f"WS2812 write failed: {reply[4:]}")
                # The helper died mid-write: note it, retry once with a fresh helper.
                code = None
                if self._proc is not None:
                    try:
                        code = self._proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        code = "hung (killed)"
                self.last_crash = f"helper exited with code {code} during a write"
                self._proc = None
            raise LedHardwareUnavailable(f"WS2812 helper keeps crashing ({self.last_crash})")

    def write(self, gpio: int, data: bytearray):
        self.write_many([(gpio, data)])

    def close(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()


class _MockBackend:
    name = "mock"

    def __init__(self):
        self.listeners = []
        self.side_of_gpio: dict[int, str] = {}

    def write(self, gpio: int, data: bytearray):
        for listener in list(self.listeners):
            listener(gpio, bytes(data), time.monotonic_ns())

    def close(self):
        pass


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
                self.backend = _HelperBackend()
            except LedHardwareUnavailable as exc:
                if requested == "auto" and os.environ.get("PLR_ALLOW_MOCK") == "1":
                    self.backend = _MockBackend()
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
            "helper_restarts": getattr(self.backend, "restarts", 0),
            "helper_last_crash": getattr(self.backend, "last_crash", None),
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
            moved = [(None, old[side].gpio, old[side].frame((0, 0, 0))) for side in SIDES
                     if old[side].gpio != self.configs[side].gpio]
            if moved and self.backend is not None:
                self._write(moved)
        self.all_off()

    # ── output ──────────────────────────────────────────────────────────────
    def _write(self, items: list) -> list[tuple[int, int]]:
        """items: [(side, gpio, frame)]; returns [(t0_ns, t1_ns)] per item."""
        if hasattr(self.backend, "write_many"):
            return self.backend.write_many([(gpio, data) for _side, gpio, data in items])
        stamps = []
        for _side, gpio, data in items:
            t0 = time.monotonic_ns()
            self.backend.write(gpio, data)
            stamps.append((t0, time.monotonic_ns()))
        return stamps

    def set(self, colors: dict[str, tuple[int, int, int]]) -> dict:
        """Set one or both LEDs to raw RGB bytes; return a command envelope.

        Both frames go to the driver in one message and are written back to
        back, so in dual mode the two LEDs change within a fraction of a
        millisecond of each other.
        """
        if self.backend is None:
            raise LedHardwareUnavailable(self.error or "LED backend unavailable")
        with self._lock:
            items = []
            for side, rgb in colors.items():
                cfg = self.configs[side]
                if not cfg.enabled:
                    rgb = (0, 0, 0)
                items.append((side, cfg.gpio, cfg.frame(rgb), tuple(rgb)))
            start = time.monotonic_ns()
            stamps = self._write([(side, gpio, frame) for side, gpio, frame, _ in items])
            end = time.monotonic_ns()
            per_side = {}
            for (side, _gpio, _frame, rgb), (t0, t1) in zip(items, stamps):
                self._state[side] = rgb
                per_side[side] = {"start_ns": t0, "end_ns": t1, "rgb": list(rgb)}
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
        try:
            self.backend.close()
        except Exception:
            pass


if __name__ == "__main__" and "--helper" in sys.argv:
    sys.exit(_helper_main())
