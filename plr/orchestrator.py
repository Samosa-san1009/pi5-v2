"""Record one PLR session: video + LED flashes + optical flash timing.

Timeline sources (all kernel CLOCK_MONOTONIC, so directly comparable):
  - camera frames:  libcamera SensorTimestamp of every encoded frame
  - light onset:    endstop edge timestamps (kernel GPIO events)
  - LED command:    monotonic time around the WS2812 write (diagnostic)

Each flash event reports `onset_s` relative to the first video frame. When the
endstop saw the flash, onset_s is the measured optical onset
(timestamp_source = "endstop_optical"). If it did not (sensor disconnected,
intensity too low), onset_s falls back to the command time plus the median
measured command-to-light latency, and the event is flagged.
"""

from __future__ import annotations

import bisect
import csv
import json
import statistics
import threading
import time
from pathlib import Path

from . import timebase

SIDE_INDEX = {"left": 1, "right": 2}


class SessionAborted(RuntimeError):
    pass


class OpticalTimingError(RuntimeError):
    pass


def _sleep_until(deadline_ns: int, abort: threading.Event | None = None):
    """Sleep to a monotonic deadline; spin for the last 2 ms for precision."""
    while True:
        remaining = deadline_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        if abort is not None and abort.is_set():
            raise SessionAborted("session stopped by request")
        if remaining > 2_000_000:
            time.sleep(min((remaining - 2_000_000) / 1e9, 0.05))


class SessionRecorder:
    def __init__(self, config: dict, paths: dict, camera, leds, endstops, endstop_settings: dict, logger, abort: threading.Event):
        self.config = config
        self.paths = paths
        self.camera = camera
        self.leds = leds
        self.endstops = endstops
        self.endstop_settings = endstop_settings
        self.log = logger
        self.abort = abort
        self.commands: list[dict] = []

    # ── public ──────────────────────────────────────────────────────────────
    def run(self) -> dict:
        cfg = self.config
        warnings = []
        if self.endstops is not None and self.endstops.ready:
            for message in self.endstops.recheck_idle().values():
                warnings.append(message)
                self.log.warning(message)
        anchor_start = timebase.anchor()
        recording = self.camera.begin_recording(str(self.paths["video"]))
        rec_start_ns = recording["started_monotonic_ns"]
        self.log.info("Recording started -> %s (%s)", self.paths["video"], recording.get("output"))
        result = None
        try:
            _sleep_until(time.monotonic_ns() + int(cfg["pre_roll_s"] * 1e9), self.abort)
            _sleep_until(time.monotonic_ns() + int(cfg["schedule"].get("initial_delay_ms", 0) * 1e6), self.abort)
            mode = cfg["mode"]
            if mode == "dual":
                self._run_dual()
            elif mode in ("left_to_right", "right_to_left"):
                self._run_sequential()
            elif mode == "observe":
                _sleep_until(time.monotonic_ns() + int(cfg["schedule"]["duration_ms"] * 1e6), self.abort)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            _sleep_until(time.monotonic_ns() + int(cfg["post_roll_s"] * 1e9), self.abort)
        finally:
            try:
                self.leds.all_off()
            except Exception:
                self.log.exception("Could not switch LEDs off")
            result = self.camera.end_recording()
        rec_end_ns = time.monotonic_ns()
        self.log.info("Recording stopped: %d frames", len(result.get("frame_timestamps_ns") or []))

        frames = [ts for ts in (result.get("frame_timestamps_ns") or []) if ts is not None]
        timeline = self._build_timeline(frames, rec_start_ns, rec_end_ns)
        timeline["warnings"] = warnings + timeline["warnings"]
        timeline["clock_anchor_start"] = anchor_start
        timeline["clock_anchor_end"] = timebase.anchor()
        self._write_outputs(frames, timeline, rec_start_ns, rec_end_ns)
        return {"video": str(self.paths["video"]), "events": timeline["events"], "timeline": timeline}

    # ── schedules ───────────────────────────────────────────────────────────
    def _flash(self, sides: list[str], hex_color: str, duration_ms: int, intensity: dict, meta: dict):
        power = self.config["power"]
        enabled = self.config["enabled"]
        colors = {}
        for side in sides:
            rgb = self.leds.rgb_for(side, hex_color, intensity[side], power[side]) if enabled[side] else (0, 0, 0)
            colors[side] = rgb
        on = self.leds.set(colors)
        deadline = on["end_ns"] + int(duration_ms * 1e6)
        try:
            _sleep_until(deadline, self.abort)
        finally:
            off = self.leds.set({side: (0, 0, 0) for side in sides})
        for side in sides:
            self.commands.append({
                "side": side,
                "hex_color": hex_color,
                "intensity_pct": float(intensity[side]),
                "power_pct": float(power[side]),
                "rgb": list(colors[side]),
                "enabled": bool(enabled[side]),
                "cmd_on_start_ns": on["sides"][side]["start_ns"],
                "cmd_on_end_ns": on["sides"][side]["end_ns"],
                "cmd_off_start_ns": off["sides"][side]["start_ns"],
                "cmd_off_end_ns": off["sides"][side]["end_ns"],
                "duration_ms": duration_ms,
                **meta,
            })

    def _run_dual(self):
        sched = self.config["schedule"]
        for index, flash in enumerate(sched["flashes"]):
            self._flash(["left", "right"], flash["hex"], flash["duration_ms"], flash["intensity"],
                        {"mode": "dual", "flash_index": index})
            if index < len(sched["flashes"]) - 1:
                _sleep_until(time.monotonic_ns() + int(sched["gap_ms"] * 1e6), self.abort)

    def _run_sequential(self):
        sched = self.config["schedule"]
        colors = sched["colors_cycle"]
        active = sched.get("active_side")
        for round_index in range(1, sched["rounds"] + 1):
            color = colors[(round_index - 1) % len(colors)]
            for position, side in zip(("first", "second"), sched["order"]):
                if active in (None, side):
                    self._flash([side], color, sched["duration_ms"], sched["intensity"],
                                {"mode": self.config["mode"], "round": round_index, "flash_position": position})
                if position == "first" and active is None:
                    _sleep_until(time.monotonic_ns() + int(sched["inner_pause_ms"] * 1e6), self.abort)
            if round_index < sched["rounds"]:
                _sleep_until(time.monotonic_ns() + int(sched["gap_ms"] * 1e6), self.abort)

    # ── timeline ────────────────────────────────────────────────────────────
    def _build_timeline(self, frames: list[int], rec_start_ns: int, rec_end_ns: int) -> dict:
        warnings = []
        fps = float(self.camera.settings["fps"])
        if frames:
            t0 = frames[0]
            video_clock = "libcamera_sensor_timestamp"
        else:
            t0 = rec_start_ns
            video_clock = "recording_start_estimate"
            warnings.append("no per-frame timestamps; frame times estimated from fps")
        frame_period_ns = int(1e9 / fps)
        if len(frames) > 1:
            frame_period_ns = int(statistics.median(b - a for a, b in zip(frames, frames[1:])))

        max_latency_ns = int(float(self.endstop_settings.get("max_latency_ms", 30.0)) * 1e6)
        pulses = {}
        optical_ok = self.endstops is not None and self.endstops.ready
        for side in ("left", "right"):
            pulses[side] = self.endstops.pulses(side, rec_start_ns - 1_000_000_000, rec_end_ns) if optical_ok else []

        # Match each command to the first unclaimed optical pulse after it.
        claimed = {"left": set(), "right": set()}
        matches = []
        for cmd in sorted(self.commands, key=lambda c: c["cmd_on_start_ns"]):
            side = cmd["side"]
            hit = None
            if cmd["enabled"] and max(cmd["rgb"]) > 0:
                lo = cmd["cmd_on_start_ns"] - 1_000_000
                hi = cmd["cmd_on_end_ns"] + max_latency_ns
                for i, pulse in enumerate(pulses[side]):
                    if i in claimed[side]:
                        continue
                    if lo <= pulse["onset_ns"] <= hi:
                        hit = (i, pulse)
                        claimed[side].add(i)
                        break
            matches.append((cmd, hit))

        latencies = {"left": [], "right": []}
        for cmd, hit in matches:
            if hit:
                latencies[cmd["side"]].append(hit[1]["onset_ns"] - cmd["cmd_on_end_ns"])
        median_latency = {side: int(statistics.median(v)) if v else 0 for side, v in latencies.items()}

        def frame_of(ts: int) -> int:
            if frames:
                return min(len(frames) - 1, bisect.bisect_left(frames, ts))
            return max(0, int(round((ts - t0) / frame_period_ns)))

        events = []
        missing = 0
        for cmd, hit in matches:
            side = cmd["side"]
            if hit:
                pulse = hit[1]
                onset_ns, offset_ns = pulse["onset_ns"], pulse["offset_ns"]
                source = "endstop_optical"
                uncertainty_us = 50.0
                latency_ms = (onset_ns - cmd["cmd_on_end_ns"]) / 1e6
            else:
                onset_ns = cmd["cmd_on_end_ns"] + median_latency[side]
                offset_ns = cmd["cmd_off_end_ns"] + median_latency[side]
                source = "ws2812_command" + ("+median_latency" if median_latency[side] else "")
                spread = (cmd["cmd_on_end_ns"] - cmd["cmd_on_start_ns"]) / 1e3
                uncertainty_us = round(max(500.0, spread), 1)
                latency_ms = None
                if cmd["enabled"] and max(cmd["rgb"]) > 0:
                    missing += 1
            events.append({
                "led_index": SIDE_INDEX[side],
                "eye": side,
                "hex_color": cmd["hex_color"],
                "intensity_pct": cmd["intensity_pct"],
                "power_pct": cmd["power_pct"],
                "rgb_bytes": cmd["rgb"],
                "onset_s": round(max(0.0, (onset_ns - t0) / 1e9), 6),
                "offset_s": round(max(0.0, (offset_ns - t0) / 1e9), 6),
                "onset_frame": frame_of(onset_ns),
                "offset_frame": frame_of(offset_ns),
                "onset_monotonic_ns": onset_ns,
                "offset_monotonic_ns": offset_ns,
                "command_onset_s": round((cmd["cmd_on_end_ns"] - t0) / 1e9, 6),
                "optical_detected": hit is not None,
                "optical_latency_ms": round(latency_ms, 4) if latency_ms is not None else None,
                "optical_raw_pulses": hit[1]["raw_pulses"] if hit else 0,
                "measured_duration_ms": round((offset_ns - onset_ns) / 1e6, 3),
                "requested_duration_ms": cmd["duration_ms"],
                "timestamp_source": source,
                "video_clock_source": video_clock,
                "onset_uncertainty_us": uncertainty_us,
                "offset_uncertainty_us": uncertainty_us,
                **{k: cmd[k] for k in ("mode", "flash_index", "round", "flash_position") if k in cmd},
            })

        events.sort(key=lambda e: (e["onset_s"], e["led_index"]))
        for index, event in enumerate(events, start=1):
            event["event_id"] = f"flash_{index:03d}"

        # Light seen by a sensor while only the other eye was flashing.
        crosstalk = []
        for event in events:
            if event.get("mode") == "dual":
                continue
            other = "right" if event["eye"] == "left" else "left"
            for i, pulse in enumerate(pulses[other]):
                if i in claimed[other]:
                    continue
                if event["onset_monotonic_ns"] - 2_000_000 <= pulse["onset_ns"] <= event["offset_monotonic_ns"]:
                    crosstalk.append({"flash": event["event_id"], "seen_by": other})
        unmatched = {side: len(pulses[side]) - len(claimed[side]) for side in pulses}

        if missing:
            message = f"{missing} flash(es) were not seen by the endstops; their onset uses the LED command time"
            warnings.append(message)
            if self.endstop_settings.get("require_optical"):
                raise OpticalTimingError(message)
        if not optical_ok:
            warnings.append("endstops unavailable: flash times come from LED commands only")

        return {
            "events": events,
            "video": {
                "first_frame_monotonic_ns": t0,
                "frames": len(frames),
                "frame_period_ms": round(frame_period_ns / 1e6, 4),
                "clock": video_clock,
            },
            "optical": {
                "available": optical_ok,
                "median_latency_ms": {s: round(v / 1e6, 4) for s, v in median_latency.items()},
                "detected": sum(1 for _, hit in matches if hit),
                "commands": len(matches),
                "unmatched_pulses": unmatched,
                "crosstalk": crosstalk,
            },
            "warnings": warnings,
        }

    def _write_outputs(self, frames, timeline, rec_start_ns, rec_end_ns):
        session_dir = Path(self.paths["session_dir"])
        t0 = timeline["video"]["first_frame_monotonic_ns"]
        with open(session_dir / "frames.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "sensor_timestamp_ns", "time_s"])
            for index, ts in enumerate(frames):
                writer.writerow([index, ts, f"{(ts - t0) / 1e9:.6f}"])
        if self.endstops is not None and self.endstops.ready:
            with open(session_dir / "endstop_edges.csv", "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["side", "active", "monotonic_ns", "time_s"])
                for side, active, ts in self.endstops.edges_between(rec_start_ns - 1_000_000_000, rec_end_ns):
                    writer.writerow([side, int(active), ts, f"{(ts - t0) / 1e9:.6f}"])
        Path(self.paths["events"]).write_text(json.dumps(timeline, indent=2), encoding="utf-8")
