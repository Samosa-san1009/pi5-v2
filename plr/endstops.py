"""Optical flash detection with the modified endstops.

Each endstop (IR emitter removed) faces its WS2812. Its phototransistor pulls
the signal line to its active level when the LED lights up. Edges are captured
with lgpio alerts, which use the kernel GPIO character device: every edge is
timestamped by the kernel at the moment it happens, so Python scheduling delay
does not affect the measured onset.

Polarity: with the LEDs off, the line sits at its idle level, and the active
level is the opposite. "auto" measures this at start-up.

WS2812 PWM: below full brightness the LED flickers at its internal PWM rate,
so one flash arrives as many short pulses. Pulses closer together than
`merge_gap_us` are merged into a single optical flash.
"""

from __future__ import annotations

import collections
import os
import threading
import time

from . import timebase

SIDES = ("left", "right")


class EndstopMonitor:
    def __init__(self, settings: dict, mock_led_backend=None):
        self.settings = settings
        self.merge_gap_ns = int(float(settings.get("merge_gap_us", 3000)) * 1000)
        self._edges: collections.deque = collections.deque(maxlen=500_000)
        self._lock = threading.Lock()
        self._handle = None
        self._callbacks = []
        self._chip = None
        self.clock: str | None = None
        self.error: str | None = None
        self.sides: dict[str, dict] = {}
        self.mock = mock_led_backend is not None or os.environ.get("PLR_HARDWARE", "").lower() == "mock"
        self._mock_backend = mock_led_backend
        self._mock_state: dict[str, tuple[int, float]] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self):
        enabled = {side: cfg for side in SIDES if (cfg := self.settings[side]).get("enabled", True)}
        if self.mock:
            self.clock = "monotonic"
            for side, cfg in enabled.items():
                self.sides[side] = {"gpio": int(cfg["gpio"]), "idle_level": 1, "active_level": 0, "mock": True}
            if self._mock_backend is not None:
                self._mock_backend.listeners.append(self._mock_led_write)
            return
        try:
            import lgpio
        except Exception as exc:
            self.error = f"lgpio unavailable ({exc}); install python3-lgpio"
            return
        self._lgpio = lgpio
        try:
            self._handle, self._chip = self._open_header_chip(lgpio)
            bias = {
                "none": lgpio.SET_PULL_NONE,
                "pull_up": lgpio.SET_PULL_UP,
                "pull_down": lgpio.SET_PULL_DOWN,
            }[str(self.settings.get("bias", "none")).lower()]
            for side, cfg in enabled.items():
                gpio = int(cfg["gpio"])
                lgpio.gpio_claim_alert(self._handle, gpio, lgpio.BOTH_EDGES, bias)
                time.sleep(0.01)
                idle = int(lgpio.gpio_read(self._handle, gpio))
                active = self._active_level(cfg.get("active", "auto"), idle)
                self.sides[side] = {"gpio": gpio, "idle_level": idle, "active_level": active}
                callback = lgpio.callback(self._handle, gpio, lgpio.BOTH_EDGES, self._make_callback(side))
                self._callbacks.append(callback)
        except Exception as exc:
            self.error = f"endstop setup failed: {type(exc).__name__}: {exc}"
            self.close()

    def close(self):
        for callback in self._callbacks:
            try:
                callback.cancel()
            except Exception:
                pass
        self._callbacks.clear()
        if self._handle is not None:
            for info in self.sides.values():
                try:
                    self._lgpio.gpio_free(self._handle, info["gpio"])
                except Exception:
                    pass
            try:
                self._lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
            self._handle = None

    @staticmethod
    def _open_header_chip(lgpio):
        # Recent Pi 5 kernels expose the header as gpiochip0, older ones as gpiochip4.
        for chip in (0, 4, 1, 2, 3):
            try:
                handle = lgpio.gpiochip_open(chip)
            except Exception:
                continue
            try:
                info = lgpio.gpio_get_chip_info(handle)
                text = " ".join(str(part) for part in info[1:]).lower()
                if "rp1" in text or "pinctrl-bcm2" in text:
                    return handle, chip
            except Exception:
                pass
            lgpio.gpiochip_close(handle)
        raise RuntimeError("could not find the 40-pin header gpiochip")

    @staticmethod
    def _active_level(setting, idle: int) -> int:
        setting = str(setting).lower()
        if setting == "low":
            return 0
        if setting == "high":
            return 1
        return 0 if idle == 1 else 1

    def _make_callback(self, side: str):
        def on_edge(chip, gpio, level, timestamp_ns):
            if level == 2:  # watchdog report, not an edge
                return
            if self.clock is None:
                self.clock = timebase.detect_clock(timestamp_ns)
            mono = timebase.to_monotonic_ns(timestamp_ns, self.clock)
            active = int(level) == self.sides[side]["active_level"]
            with self._lock:
                self._edges.append((side, active, mono))
        return on_edge

    # ── status ──────────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return bool(self.sides) and self.error is None

    def recheck_idle(self) -> dict:
        """Re-read the idle level (call with the LEDs off). Returns warnings."""
        warnings = {}
        if self.mock or self._handle is None:
            return warnings
        for side, info in self.sides.items():
            level = int(self._lgpio.gpio_read(self._handle, info["gpio"]))
            if level == info["active_level"]:
                warnings[side] = (
                    f"{side} endstop reads ACTIVE with LEDs off - ambient light "
                    "is reaching the sensor, or its polarity setting is wrong"
                )
        return warnings

    def status(self) -> dict:
        with self._lock:
            count = len(self._edges)
        return {
            "ready": self.ready,
            "mock": self.mock,
            "error": self.error,
            "gpiochip": self._chip,
            "timestamp_clock": self.clock,
            "edges_buffered": count,
            "sides": self.sides,
        }

    # ── queries ─────────────────────────────────────────────────────────────
    def edges_between(self, start_ns: int, end_ns: int) -> list[tuple[str, bool, int]]:
        with self._lock:
            return [edge for edge in self._edges if start_ns <= edge[2] <= end_ns]

    def pulses(self, side: str, start_ns: int, end_ns: int) -> list[dict]:
        """Merged optical flashes for one side within a time window."""
        raw = []
        onset = None
        for edge_side, active, ts in self.edges_between(start_ns, end_ns):
            if edge_side != side:
                continue
            if active and onset is None:
                onset = ts
            elif not active and onset is not None:
                raw.append((onset, ts))
                onset = None
        if onset is not None:
            raw.append((onset, end_ns))
        merged: list[dict] = []
        for on, off in raw:
            if merged and on - merged[-1]["offset_ns"] <= self.merge_gap_ns:
                merged[-1]["offset_ns"] = off
                merged[-1]["raw_pulses"] += 1
            else:
                merged.append({"onset_ns": on, "offset_ns": off, "raw_pulses": 1})
        return merged

    # ── mock optical model (bench testing without hardware) ────────────────
    def _mock_led_write(self, gpio: int, data: bytes, t_ns: int):
        side = getattr(self._mock_backend, "side_of_gpio", {}).get(gpio)
        if side not in self.sides:
            return
        peak = max(data) if data else 0
        previous = self._mock_state.get(side)
        if peak >= 6 and previous is None:
            self._mock_state[side] = (t_ns, peak)
        elif peak < 6 and previous is not None:
            on_ns, level = previous
            del self._mock_state[side]
            self._mock_generate(side, on_ns, t_ns, level)

    def _mock_generate(self, side: str, on_ns: int, off_ns: int, level: int):
        latency = 180_000 + (on_ns % 97) * 1_000          # 0.18-0.28 ms
        fall = 60_000
        edges = []
        if level >= 250:
            edges = [(on_ns + latency, True), (off_ns + fall, False)]
        else:
            period = 1_000_000_000 // 1200                # ~1.2 kHz WS2812 PWM
            high = max(20_000, period * level // 255)
            t = on_ns + latency
            while t < off_ns:
                edges.append((t, True))
                edges.append((min(t + high, off_ns + fall), False))
                t += period
        with self._lock:
            for ts, active in edges:
                self._edges.append((side, active, ts))
