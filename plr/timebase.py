"""Clocks, and how every timestamp in a session relates to the others.

Measurement timing never uses wall-clock time. Camera frames (libcamera
SensorTimestamp) and endstop edges (kernel GPIO events) are both stamped by
the kernel on CLOCK_MONOTONIC, so they can be compared directly with
sub-millisecond precision. No external RTC module is needed for that.

Wall-clock time is only used to name and date sessions. It comes from NTP
when the Pi has network. Offline, the Pi 5's built-in RTC keeps the date if a
battery is plugged into its BAT (J5) connector.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# Anything earlier than this means the clock was never set.
_PLAUSIBLE_EPOCH_S = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
_CLOCKS = {
    "monotonic": time.CLOCK_MONOTONIC,
    "boottime": getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC),
    "realtime": time.CLOCK_REALTIME,
}
_ntp_cache: tuple[float, bool | None] = (0.0, None)


def now_ns(clock: str = "monotonic") -> int:
    return time.clock_gettime_ns(_CLOCKS[clock])


def anchor() -> dict:
    """Sample monotonic and wall-clock time as close together as possible."""
    best = None
    for _ in range(7):
        a = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        rt = time.clock_gettime_ns(time.CLOCK_REALTIME)
        b = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if best is None or (b - a) < best[0]:
            best = (b - a, (a + b) // 2, rt)
    span, mono, real = best
    return {
        "monotonic_ns": mono,
        "realtime_ns": real,
        "sample_span_ns": span,
        "realtime_iso": datetime.fromtimestamp(real / 1e9, tz=timezone.utc).isoformat(),
    }


def detect_clock(timestamp_ns: int) -> str:
    """Return which kernel clock a timestamp most likely came from."""
    readings = {name: time.clock_gettime_ns(cid) for name, cid in _CLOCKS.items()}
    return min(readings, key=lambda name: abs(readings[name] - int(timestamp_ns)))


def to_monotonic_ns(timestamp_ns: int, clock: str) -> int:
    """Convert a timestamp from `clock` into CLOCK_MONOTONIC."""
    if clock == "monotonic":
        return int(timestamp_ns)
    a = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    other = time.clock_gettime_ns(_CLOCKS[clock])
    b = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    return int(timestamp_ns) - other + (a + b) // 2


def ntp_synchronized() -> bool | None:
    global _ntp_cache
    checked_at, value = _ntp_cache
    if time.monotonic() - checked_at < 30:
        return value
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=2)
        value = out.stdout.strip() == "yes" if out.returncode == 0 else None
    except Exception:
        value = None
    _ntp_cache = (time.monotonic(), value)
    return value


def onboard_rtc() -> dict | None:
    """The Pi 5's own RTC (rpi-rtc), if the kernel exposes it."""
    for node in sorted(Path("/sys/class/rtc").glob("rtc*")):
        name = _read(node / "name") or ""
        if "rpi" in name.lower():
            info = {"device": f"/dev/{node.name}", "name": name}
            epoch = _read(node / "since_epoch")
            if epoch and epoch.isdigit():
                info["since_epoch"] = int(epoch)
            return info
    return None


def clock_status() -> dict:
    system_s = time.time()
    ntp = ntp_synchronized()
    plausible = system_s >= _PLAUSIBLE_EPOCH_S
    if ntp:
        source = "ntp"
    elif plausible:
        source = "rtc_or_unverified"
    else:
        source = "unset"
    return {
        "system_time": datetime.fromtimestamp(system_s).isoformat(timespec="seconds"),
        "system_time_plausible": plausible,
        "ntp_synchronized": ntp,
        "onboard_rtc": onboard_rtc(),
        "wall_clock_source": source,
    }


def session_datetime() -> tuple[datetime, str]:
    """Local wall-clock time for naming a session, and where it came from."""
    return datetime.now(), clock_status()["wall_clock_source"]


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except Exception:
        return None
