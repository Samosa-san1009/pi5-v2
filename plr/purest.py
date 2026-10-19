"""CPU-only PuReST Soft Flow Lite on both eyes, live and offline.

Both modes use the one native program, native/purest/build/plr_purest, which
runs a separate PuReST tracker per eye on its own thread (one CPU core each).

LiveEngine   A persistent `plr_purest --stream` process. While a session
             records, the camera hands it every uncompressed frame; results
             for both eyes come back asynchronously. Between
             sessions it is fed at a low rate so the TFT can show the pupils
             while the rig is lined up. It never blocks the camera: if it
             ever falls behind, frames are dropped and reported, and the
             offline pass covers them.

run_offline  After recording, `plr_purest --input full_video.mp4` analyses
             every frame of the saved video for both eyes.

Both return the same per-eye trace, matched to video frames.
"""

from __future__ import annotations

import collections
import csv
import math
import os
import re
import select
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "native" / "purest" / "build" / "plr_purest"
EYES = ("left", "right")
_HEADER = struct.Struct("<4sIQqII")      # magic, flags, seq, ts_ns, width, height
_BINARY_CACHE = {}


def binary_path() -> Path:
    return Path(os.environ.get("PLR_PUREST_BIN", DEFAULT_BINARY)).expanduser()


def binary_status() -> dict:
    path = binary_path()
    if not path.is_file():
        return {"ready": False, "binary": str(path), "error": "not built: run scripts/build_native.sh"}
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_mode)
    if key not in _BINARY_CACHE:
        result = {"ready": False, "binary": str(path)}
        try:
            probe = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=5)
            version = probe.stdout.strip()
            match = re.match(r"(\d+)\.(\d+)\.(\d+)-softflow-lite", version)
            if probe.returncode or not match or tuple(map(int, match.groups())) < (2, 3, 0):
                raise RuntimeError(f"outdated/incompatible native tracker ({version or probe.stderr.strip() or 'no version'})")
            result.update(ready=True, version=version)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            result["error"] = f"{exc}; run bash scripts/build_native.sh and restart plr.service"
        _BINARY_CACHE.clear()
        _BINARY_CACHE[key] = result
    return dict(_BINARY_CACHE[key])


def _engine_args(settings: dict) -> list[str]:
    purest = settings["purest"]
    roi = settings["eye_roi"]
    args = ["--roi-left", ",".join(f"{v:g}" for v in roi["left"]),
            "--roi-right", ",".join(f"{v:g}" for v in roi["right"])]
    if purest.get("min_pupil_px"):
        args += ["--min-pupil-px", str(purest["min_pupil_px"])]
    if purest.get("max_pupil_px"):
        args += ["--max-pupil-px", str(purest["max_pupil_px"])]
    for key, default in (("work_size", 320), ("flow_frames", 3), ("max_gap_ms", 150),
                         ("reacquire_every", 3), ("min_contrast", 12), ("glint_kernel", 11)):
        args += ["--" + key.replace("_", "-"), str(purest.get(key, default))]
    args += ["--confidence", str(purest.get("confidence_threshold", .70))]
    localization = settings.get("eye_localization", {"mode": "fixed"})
    args += ["--eye-mode", localization.get("mode", "auto"),
             "--eye-interval", str(localization.get("interval", 6)),
             "--eye-ttl-ms", str(localization.get("ttl_ms", 500)),
             "--cascade-dir", str(REPO_ROOT / "native" / "purest" / "models")]
    if localization.get("swap_eyes"):
        args += ["--swap-eyes"]
    return args


def _wrap_priority(cmd: list[str], cpus, nice: int) -> list[str]:
    """Pin the engine (and both its eye threads) to `cpus` and set its nice."""
    prefix = []
    if cpus and shutil.which("taskset"):
        available = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
        wanted = sorted(int(c) for c in cpus if not available or int(c) in available)
        if wanted:
            prefix += ["taskset", "-c", ",".join(str(c) for c in wanted)]
    if nice and shutil.which("nice"):
        prefix += ["nice", "-n", str(int(nice))]
    return prefix + cmd


def _num(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value


# ── live ────────────────────────────────────────────────────────────────────

class LiveEngine:
    """Persistent streaming PuReST process, shared by the camera and sessions."""

    def __init__(self, settings: dict, logger=None):
        self.settings = settings
        self.log = logger
        purest = settings["purest"]
        self.enabled = bool(purest.get("live", True))
        idle_fps = float(purest.get("idle_preview_fps", 0) or 0)
        self._idle_period_ns = int(1e9 / idle_fps) if idle_fps > 0 else None
        self._last_idle_ns = 0
        self._proc = None
        self._lock = threading.Lock()
        self._feed = collections.deque(maxlen=3)        # frames waiting for the pipe
        self._feed_event = threading.Event()
        self._stop = threading.Event()
        self._session = None                             # collector while recording
        self._reset_next = False
        self.latest = {eye: None for eye in EYES}        # newest result per eye (display)
        self.error: str | None = None
        self.restarts = 0
        self._last_start = 0.0
        self._stats = {eye: collections.deque(maxlen=300) for eye in EYES}
        self._localization_stats = collections.deque(maxlen=300)
        self._drops = {eye: 0 for eye in EYES}
        self._python_drops = 0
        self._feeder = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self):
        if not self.enabled:
            self.error = "disabled in settings (purest.live)"
            return
        status = binary_status()
        if not status["ready"]:
            self.error = status["error"]
            return
        self._spawn()
        self._feeder = threading.Thread(target=self._feed_loop, name="purest-feed", daemon=True)
        self._feeder.start()

    def _spawn(self):
        purest = self.settings["purest"]
        cmd = [str(binary_path()), "--stream", "--queue-depth", str(int(purest.get("queue_depth", 6)))]
        cmd += _engine_args(self.settings)
        if os.environ.get("PLR_PUREST_SIMULATE_LOAD_US"):          # tests only
            cmd += ["--simulate-load-us", os.environ["PLR_PUREST_SIMULATE_LOAD_US"]]
        cmd = _wrap_priority(cmd, purest.get("cpus"), purest.get("live_nice", 0))
        self._last_start = time.monotonic()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        readable, _, _ = select.select([proc.stdout], [], [], 8)
        line = proc.stdout.readline().decode(errors="replace").strip() if readable else ""
        if not line.startswith("READY"):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)
            err = proc.stderr.read().decode(errors="replace")[-600:]
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                pipe.close()
            raise RuntimeError(f"live PuReST did not start: {line or err or 'startup timed out'}")
        self._proc = proc
        self.error = None
        threading.Thread(target=self._read_loop, args=(proc,), name="purest-read", daemon=True).start()
        threading.Thread(target=self._stderr_loop, args=(proc,), name="purest-stderr", daemon=True).start()

    def _ensure_running(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        if time.monotonic() - self._last_start < 5.0:
            return False
        try:
            self._spawn()
            self.restarts += 1
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def stop(self):
        self._stop.set()
        self._feed_event.set()
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    @property
    def ready(self) -> bool:
        return self.enabled and self._proc is not None and self._proc.poll() is None

    # ── camera side (camera thread: must never block) ───────────────────────
    def wants(self, ts_ns: int) -> bool:
        if not self.enabled or self._stop.is_set():
            return False
        if self._session is not None:
            return True
        if self._idle_period_ns is None:
            return False
        if ts_ns - self._last_idle_ns >= self._idle_period_ns:
            self._last_idle_ns = ts_ns
            return True
        return False

    def submit(self, grey: np.ndarray, seq: int, ts_ns: int):
        session = self._session
        flags = 2 if session is None else 0
        with self._lock:
            if self._reset_next:
                flags |= 1
                self._reset_next = False
            if session is not None:
                session["submitted"][seq] = ts_ns
            if len(self._feed) == self._feed.maxlen:
                old = self._feed.popleft()
                self._python_drops += 1
                if old[3] & 1:
                    flags |= 1                      # keep the session-start reset
                if old[4] is not None:
                    for eye in EYES:
                        old[4]["dropped"][eye].add(old[1])
            self._feed.append((grey, seq, ts_ns, flags, session))
        self._feed_event.set()

    def _feed_loop(self):
        while not self._stop.is_set():
            self._feed_event.wait(0.5)
            self._feed_event.clear()
            while True:
                with self._lock:
                    if not self._feed:
                        break
                    grey, seq, ts_ns, flags, session = self._feed.popleft()
                if not self._ensure_running():
                    if session is not None:
                        for eye in EYES:
                            session["dropped"][eye].add(seq)
                    continue
                height, width = grey.shape[:2]
                try:
                    self._proc.stdin.write(_HEADER.pack(b"PLR1", flags, seq, ts_ns, width, height))
                    self._proc.stdin.write(memoryview(np.ascontiguousarray(grey, dtype=np.uint8)).cast("B"))
                except Exception as exc:
                    self.error = f"live PuReST pipe: {type(exc).__name__}: {exc}"
                    if session is not None:
                        for eye in EYES:
                            session["dropped"][eye].add(seq)

    def _read_loop(self, proc):
        for raw in proc.stdout:
            parts = raw.decode(errors="replace").split()
            if not parts:
                continue
            kind = parts[0]
            if kind == "P" and len(parts) in (13, 16, 22):
                eye, seq, ts = parts[1], int(parts[2]), int(parts[3])
                sample = {
                    "seq": seq, "ts": ts, "valid": parts[4] == "1",
                    "confidence": _num(parts[5]), "outline_confidence": _num(parts[6]),
                    "major": _num(parts[7]), "minor": _num(parts[8]),
                    "cx": _num(parts[9]), "cy": _num(parts[10]), "angle": _num(parts[11]),
                    "proc_us": int(parts[12]), "at": time.monotonic(),
                    "source": parts[13] if len(parts) >= 16 else "purest",
                    "estimated": len(parts) >= 16 and parts[14] == "1",
                    "visibility": parts[15] if len(parts) >= 16 else "unknown",
                    "eye_box": [int(v) for v in parts[16:20]] if len(parts) == 22 else None,
                    "localization": parts[20] if len(parts) == 22 else "fixed",
                    "localization_us": int(parts[21]) if len(parts) == 22 else 0,
                }
                self.latest[eye] = sample
                self._stats[eye].append(sample["proc_us"])
                if eye == "left":
                    self._localization_stats.append(sample["localization_us"])
                session = self._session
                if session is not None and seq in session["submitted"]:
                    session["results"][eye][seq] = sample
            elif kind == "D" and len(parts) == 4:
                eye, seq = parts[1], int(parts[2])
                self._drops[eye] += 1
                session = self._session
                if session is not None:
                    session["dropped"][eye].add(seq)
        if self._proc is proc and not self._stop.is_set():
            self.error = f"live PuReST exited with code {proc.wait()}"

    def _stderr_loop(self, proc):
        for raw in proc.stderr:
            line = raw.decode(errors="replace").strip()
            if line and self.log:
                self.log.warning("live PuReST: %s", line)

    # ── sessions ────────────────────────────────────────────────────────────
    def begin_session(self) -> bool:
        if not self.enabled:
            return False
        self._ensure_running()
        with self._lock:
            self._session = {"submitted": {}, "results": {e: {} for e in EYES},
                             "dropped": {e: set() for e in EYES}, "started": time.monotonic()}
            self._reset_next = True            # fresh trackers for every session
        return self.ready

    def end_session(self, timeout_s: float = 10.0) -> dict | None:
        """Stop collecting; wait for the frames already submitted; return them."""
        session = self._session
        if session is None:
            return None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                waiting = sum(1 for item in self._feed if item[4] is session)
            pending = waiting + sum(
                1 for seq in list(session["submitted"])
                for eye in EYES
                if seq not in session["results"][eye] and seq not in session["dropped"][eye])
            if pending == 0 or not self.ready:
                break
            time.sleep(0.02)
        with self._lock:
            self._session = None
        return {
            "submitted": dict(session["submitted"]),
            "results": {e: dict(session["results"][e]) for e in EYES},
            "dropped": {e: set(session["dropped"][e]) for e in EYES},
        }

    def status(self) -> dict:
        def summary(values):
            values = list(values)
            if not values:
                return None
            values.sort()
            return {"median_ms": round(values[len(values) // 2] / 1000, 2),
                    "p95_ms": round(values[int(len(values) * 0.95) - 1 if len(values) > 1 else 0] / 1000, 2)}

        def preview_sample(eye):
            sample = self.latest[eye]
            if sample is None:
                return None
            out = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                   for k, v in sample.items() if k != "at"}
            out["age_s"] = round(time.monotonic() - sample["at"], 3)
            return out

        return {
            "enabled": self.enabled,
            "variant": "softflow_lite",
            "eye_localization": self.settings.get("eye_localization", {"mode": "fixed"}),
            "eye_boxes": {e: (self.latest[e] or {}).get("eye_box") for e in EYES},
            "latest": {e: preview_sample(e) for e in EYES},
            "running": self.ready,
            "error": self.error,
            "restarts": self.restarts,
            "recording": self._session is not None,
            "processing_time": {eye: summary(self._stats[eye]) for eye in EYES},
            "localization_time": summary(self._localization_stats),
            "dropped_frames": {**self._drops, "before_engine": self._python_drops},
            "binary": str(binary_path()),
        }


def live_traces(collected: dict, frame_ts: list[int], threshold: float) -> dict:
    """Match live results to the recorded video frames by sensor timestamp."""
    index_of = {int(ts): i for i, ts in enumerate(frame_ts)}
    out = {}
    for eye in EYES:
        n = len(frame_ts)
        trace = _empty_trace(n)
        by_ts = {s["ts"]: s for s in collected["results"][eye].values()}
        matched = 0
        for ts, sample in by_ts.items():
            i = index_of.get(int(ts))
            if i is None:
                continue
            matched += 1
            _fill(trace, i, sample["valid"], sample["confidence"], sample["major"], sample["minor"],
                  sample["cx"], sample["cy"], threshold, sample.get("source", "purest"),
                  sample.get("estimated", False), sample.get("visibility", "unknown"))
            trace["eye_boxes"][i] = sample.get("eye_box") or [0, 0, 0, 0]
            trace["localization"][i] = sample.get("localization", "fixed")
        trace["analysed"] = matched
        trace["coverage"] = round(matched / n, 4) if n else 0.0
        out[eye] = trace
    return out


# ── offline ─────────────────────────────────────────────────────────────────

class OfflineCancelled(RuntimeError):
    pass


def run_offline(video: str | Path, work_dir: str | Path, settings: dict, frame_count: int,
                progress=None, cancel: threading.Event | None = None, logger=None) -> dict:
    """Analyse every frame of `video` for both eyes; return per-eye traces."""
    status = binary_status()
    if not status["ready"]:
        raise RuntimeError(status["error"])
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = {eye: work_dir / f"purest_{eye}.csv" for eye in EYES}
    purest = settings["purest"]
    cmd = [str(binary_path()), "--input", str(video), "--out-left", str(out["left"]),
           "--out-right", str(out["right"]), "--progress", "15"] + _engine_args(settings)
    cmd = _wrap_priority(cmd, purest.get("cpus"), purest.get("offline_nice", 10))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    total = frame_count or 0
    tail = []

    def watch_cancel():
        while proc.poll() is None:
            if cancel is not None and cancel.wait(0.25):
                proc.terminate()
                return

    if cancel is not None:
        threading.Thread(target=watch_cancel, daemon=True).start()
    for line in proc.stderr:
        line = line.strip()
        if line.startswith("total "):
            total = int(line.split()[1]) or total
        elif line.startswith("progress ") and progress and total:
            progress(min(1.0, int(line.split()[1]) / total))
        elif line:
            tail.append(line)
    code = proc.wait()
    if cancel is not None and cancel.is_set():
        raise OfflineCancelled("offline PuReST cancelled")
    if code != 0 or not all(p.is_file() for p in out.values()):
        raise RuntimeError(f"plr_purest exited with {code}: {' | '.join(tail[-5:])}")
    if logger:
        logger.info("offline PuReST: %s", tail[-1] if tail else "done")
    threshold = float(purest.get("confidence_threshold", 0.7))
    traces = {eye: read_csv_trace(out[eye], frame_count, threshold) for eye in EYES}
    for path in out.values():
        path.unlink(missing_ok=True)
    return traces


def read_csv_trace(path: Path, frame_count: int, threshold: float) -> dict:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    n = max(frame_count or 0, len(rows))
    trace = _empty_trace(n)
    for row in rows:
        i = int(row["Frame"])
        if i >= n:
            continue
        _fill(trace, i, row["Valid"] == "1", _num(row["Confidence"]), _num(row["Diameter_px"] or "nan"),
              _num(row["Minor_px"] or "nan"), _num(row["Center_x"] or "nan"), _num(row["Center_y"] or "nan"),
              threshold, row.get("Source", "purest"), row.get("Estimated") == "1",
              row.get("Visibility", "unknown"))
        trace["eye_boxes"][i] = [int(row.get(k, 0)) for k in ("Eye_x", "Eye_y", "Eye_width", "Eye_height")]
        trace["localization"][i] = row.get("Localization", "fixed")
    trace["analysed"] = len(rows)
    trace["coverage"] = round(min(len(rows), n) / n, 4) if n else 0.0
    return trace


# ── trace helpers ───────────────────────────────────────────────────────────

def _empty_trace(n: int) -> dict:
    nan = np.full(n, np.nan)
    return {"diameter_px": nan.copy(), "minor_px": nan.copy(), "confidence": nan.copy(),
            "center_x": nan.copy(), "center_y": nan.copy(), "valid": np.zeros(n, bool),
            "tracking_source": np.full(n, "missing", dtype=object),
            "estimated": np.zeros(n, bool), "estimate_px": nan.copy(),
            "visibility": np.full(n, "unknown", dtype=object),
            "eye_boxes": np.zeros((n, 4), dtype=int), "localization": np.full(n, "missing", dtype=object),
            "analysed": 0, "coverage": 0.0}


def _fill(trace, i, valid, confidence, major, minor, cx, cy, threshold,
          source="purest", estimated=False, visibility="unknown"):
    trace["tracking_source"][i] = source
    trace["estimated"][i] = estimated
    trace["visibility"][i] = visibility
    if estimated and math.isfinite(major):
        trace["estimate_px"][i] = major
    trace["confidence"][i] = confidence if math.isfinite(confidence) else 0.0
    ok = bool(valid) and not estimated and math.isfinite(major) and (confidence if math.isfinite(confidence) else 0.0) >= threshold
    trace["valid"][i] = ok
    if ok:
        trace["diameter_px"][i] = major
        trace["minor_px"][i] = minor
        trace["center_x"][i] = cx
        trace["center_y"][i] = cy
