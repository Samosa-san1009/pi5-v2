"""Always-on IR camera.

One Picamera2 instance opens when the server starts and stays open. Every
sensor frame produces two streams:

  main  (640x480) - H.264-encoded to full_video.mp4 while a session records,
        and (Y plane, uncompressed) handed to the live PuReST engine.
  lores (320x240) - Y plane copied at ~15 fps for the TFT, /preview.jpg and
        /stream.mjpg.

Both come out of the ISP for the same frame. The copies happen inside the
frame callback in well under a millisecond, and the live engine never blocks
the callback (it drops frames instead), so neither the display nor PuReST can
delay or alter the recorded video.

Every encoded frame's libcamera SensorTimestamp (kernel CLOCK_MONOTONIC) is
logged, so each video frame has an exact time that can be compared directly
with the endstop edges, and live PuReST results are matched to video frames
by that same timestamp.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import timebase


class _FrameSink:
    """Latest lores greyscale frame, shared with the display and HTTP preview."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._ts = 0

    def publish(self, frame: np.ndarray, ts_ns: int):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._ts = ts_ns
            self._cond.notify_all()

    def wait(self, after_seq: int = 0, timeout: float = 1.0):
        with self._cond:
            if self._seq <= after_seq:
                self._cond.wait(timeout)
            if self._frame is None or self._seq <= after_seq:
                return None
            return self._seq, self._frame, self._ts


class CameraService:
    def __init__(self, settings: dict):
        self.settings = settings
        self.sink = _FrameSink()
        # Live PuReST engine (or None). Must offer wants(ts_ns) and
        # submit(grey, seq, ts_ns); both are called from the camera thread.
        self.frame_tap = None
        self._lock = threading.RLock()
        self._picam = None
        self._encoder = None
        self._output = None
        self._recording_path: str | None = None
        self._frame_stamps: list = []
        self._frames_total = 0
        self._last_frame_mono = 0
        self._fps_window: list[int] = []
        self._last_lores_ns = 0
        self._sink_period_ns = int(1e9 / 15)
        self.sensor_clock: str | None = None
        self.error: str | None = None
        self.restarts = 0
        self.camera_info: dict = {}
        self._stop = threading.Event()
        self._watchdog = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self):
        try:
            self._open()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"  [camera] could not open camera, will keep retrying: {self.error}")
        if self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watchdog_loop, name="camera-watchdog", daemon=True)
            self._watchdog.start()

    def _open(self):
        from libcamera import Transform
        from picamera2 import Picamera2

        cfg = self.settings
        with self._lock:
            cameras = Picamera2.global_camera_info()
            if not cameras:
                raise RuntimeError("no camera detected (check the ribbon cable and 'rpicam-hello --list-cameras')")
            index = cfg.get("index", "auto")
            num = cameras[0]["Num"] if index in (None, "auto") else int(index)
            self.camera_info = {"selected": num, "available": cameras}
            picam = Picamera2(camera_num=num)
            controls = {"FrameRate": float(cfg["fps"])}
            controls.update(cfg.get("controls") or {})
            video_cfg = picam.create_video_configuration(
                main={"size": tuple(cfg["resolution"]), "format": "YUV420"},
                lores={"size": tuple(cfg["lores"]), "format": "YUV420"},
                controls=controls,
                transform=Transform(hflip=bool(cfg.get("hflip")), vflip=bool(cfg.get("vflip"))),
                buffer_count=6,
                encode="main",
                display=None,
            )
            picam.configure(video_cfg)
            picam.post_callback = self._on_frame
            picam.start()
            self._picam = picam
            self._last_frame_mono = time.monotonic_ns()
            self.error = None
            print(f"  [camera] open: camera {num} main={cfg['resolution']}@{cfg['fps']} lores={cfg['lores']}")

    def _close(self):
        with self._lock:
            picam, self._picam = self._picam, None
            if picam is None:
                return
            try:
                picam.post_callback = None
                picam.stop()
            except Exception:
                pass
            try:
                picam.close()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        try:
            if self.recording:
                self.end_recording()
        finally:
            self._close()

    def _watchdog_loop(self):
        limit_ns = int(float(self.settings.get("watchdog_s", 5.0)) * 1e9)
        while not self._stop.wait(1.0):
            with self._lock:
                stale = self._picam is not None and time.monotonic_ns() - self._last_frame_mono > limit_ns
                missing = self._picam is None
                recording = self._encoder is not None
            if recording:
                continue  # never restart mid-recording; the session reports it
            if stale or missing:
                if stale:
                    print("  [camera] no frames for a while; reopening camera")
                self._close()
                try:
                    self._open()
                    self.restarts += 1
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._stop.wait(5.0)

    # ── per-frame callback (camera thread; keep it short) ───────────────────
    def _note_frame(self, now: int):
        self._frames_total += 1
        self._last_frame_mono = now
        self._fps_window.append(now)
        if len(self._fps_window) > 60:
            del self._fps_window[:-60]

    def _on_frame(self, request):
        now = time.monotonic_ns()
        self._note_frame(now)
        try:
            ts = int(request.get_metadata()["SensorTimestamp"])
            if self.sensor_clock is None:
                self.sensor_clock = timebase.detect_clock(ts)
        except Exception:
            ts = now
        tap = self.frame_tap
        want_live = tap is not None and tap.wants(ts)
        want_lores = now - self._last_lores_ns >= self._sink_period_ns
        if not (want_live or want_lores):
            return
        try:
            from picamera2 import MappedArray

            if want_live:
                width, height = self.settings["resolution"]
                with MappedArray(request, "main") as mapped:
                    grey = np.ascontiguousarray(mapped.array[:height, :width])
                tap.submit(grey, self._frames_total, ts)
            if want_lores:
                self._last_lores_ns = now
                width, height = self.settings["lores"]
                with MappedArray(request, "lores") as mapped:
                    small = mapped.array[:height, :width].copy()
                self.sink.publish(small, ts)
        except Exception:
            pass

    # ── recording ───────────────────────────────────────────────────────────
    @property
    def recording(self) -> bool:
        return self._encoder is not None

    def begin_recording(self, path: str) -> dict:
        from picamera2.encoders import H264Encoder

        with self._lock:
            if self._picam is None:
                raise RuntimeError(f"camera is not running: {self.error or 'unknown error'}")
            if self._encoder is not None:
                raise RuntimeError("a recording is already in progress")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            encoder = H264Encoder(bitrate=int(self.settings.get("bitrate", 2_000_000)))
            if hasattr(encoder, "framerate"):
                encoder.framerate = int(self.settings["fps"])
            stamps: list = []
            original = encoder._encode

            def logged_encode(stream, request, _original=original, _stamps=stamps):
                try:
                    _stamps.append(int(request.get_metadata()["SensorTimestamp"]))
                except Exception:
                    _stamps.append(None)
                return _original(stream, request)

            encoder._encode = logged_encode
            output = self._make_output(path)
            self._picam.start_encoder(encoder, output, name="main")
            self._encoder, self._output = encoder, output
            self._frame_stamps = stamps
            self._recording_path = path
            return {"path": path, "started_monotonic_ns": time.monotonic_ns(), "output": type(output).__name__}

    @staticmethod
    def _make_output(path: str):
        try:
            from picamera2.outputs import PyavOutput

            return PyavOutput(path)  # keeps the encoder's sensor timestamps in the MP4
        except Exception:
            from picamera2.outputs import FfmpegOutput

            return FfmpegOutput(path)

    def end_recording(self) -> dict:
        with self._lock:
            encoder, self._encoder = self._encoder, None
            stamps = self._frame_stamps
            path = self._recording_path
            if encoder is None or self._picam is None:
                return {"path": path, "frame_timestamps_ns": list(stamps)}
            try:
                self._picam.stop_encoder(encoder)
            finally:
                self._output = None
        if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise RuntimeError(f"recording failed: no video written to {path}")
        return {"path": path, "frame_timestamps_ns": list(stamps), "sensor_clock": self.sensor_clock}

    # ── status ──────────────────────────────────────────────────────────────
    def measured_fps(self) -> float | None:
        window = list(self._fps_window)
        if len(window) < 2:
            return None
        return round((len(window) - 1) / ((window[-1] - window[0]) / 1e9), 2)

    def status(self) -> dict:
        age = (time.monotonic_ns() - self._last_frame_mono) / 1e9 if self._last_frame_mono else None
        return {
            "available": self._picam is not None,
            "running": self._picam is not None and age is not None and age < 2.0,
            "recording": self.recording,
            "frames_total": self._frames_total,
            "measured_fps": self.measured_fps(),
            "last_frame_age_s": round(age, 3) if age is not None else None,
            "sensor_timestamp_clock": self.sensor_clock,
            "restarts": self.restarts,
            "error": self.error,
            "camera": self.camera_info,
            "config": {k: self.settings[k] for k in ("resolution", "fps", "lores")},
        }


class MockCameraService(CameraService):
    """Synthetic two-eye IR video whose pupils react to the (mock) LEDs.

    Used with PLR_HARDWARE=mock to exercise the whole pipeline (LEDs,
    endstops, live + offline PuReST, the app contract) on any Linux PC.
    Needs only numpy and the ffmpeg command.
    """

    def __init__(self, settings: dict, leds=None):
        super().__init__(settings)
        self._thread = None
        self._writer = None
        self._rng = np.random.default_rng(1)
        self._light_events: list[tuple[int, str, float]] = []
        self._radius = {"left": 30.0, "right": 30.0}
        self.sensor_clock = "monotonic"
        width, height = settings["resolution"]
        self._yy, self._xx = np.mgrid[0:height, 0:width].astype(np.float32)
        if leds is not None:
            leds.state_listeners.append(self._on_led)

    def _on_led(self, side, rgb, t_ns):
        self._light_events.append((t_ns, side, max(rgb) / 255.0))

    def _open(self):
        self._picam = "mock"
        self.camera_info = {"selected": "mock", "available": []}
        self._last_frame_mono = time.monotonic_ns()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="mock-camera", daemon=True)
            self._thread.start()

    def _close(self):
        pass

    def _drive(self, side: str, now_ns: int) -> float:
        latency = 220_000_000
        other = "right" if side == "left" else "left"
        level = {"left": 0.0, "right": 0.0}
        for t, s, value in self._light_events:
            if t + latency <= now_ns:
                level[s] = value
        return max(level[side], 0.85 * level[other])     # consensual response

    def _disc(self, cx, cy, r, soft=1.5):
        d = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        return np.clip((r - d) / soft + 0.5, 0.0, 1.0)

    def _render(self, now_ns: int, dt: float) -> np.ndarray:
        width, height = self.settings["resolution"]
        img = np.full((height, width), 150.0, np.float32)
        cy = height / 2.0
        for side, cx in (("left", width / 4.0), ("right", 3 * width / 4.0)):
            target = 30.0 * (1.0 - 0.35 * self._drive(side, now_ns))
            tau = 0.15 if target < self._radius[side] else 0.8
            self._radius[side] += (target - self._radius[side]) * min(1.0, dt / tau)
            sclera = np.clip(1.0 - (((self._xx - cx) / 95.0) ** 2 + ((self._yy - cy) / 55.0) ** 2), 0, 1) > 0
            img[sclera] = 172.0
            img -= 22.0 * self._disc(cx, cy, 48.0, soft=6.0)                   # iris, soft edge
            pupil = self._disc(cx, cy, self._radius[side])
            img = img * (1 - pupil) + 20.0 * pupil                              # dark pupil
            glint = self._disc(cx + 10, cy - 12, 4.0)
            img = img * (1 - glint) + 250.0 * glint
        img += self._rng.normal(0, 3, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _run(self):
        fps = float(self.settings["fps"])
        period = 1.0 / fps
        next_t = time.monotonic()
        last_ns = time.monotonic_ns()
        lw, lh = self.settings["lores"]
        width, height = self.settings["resolution"]
        while not self._stop.is_set():
            now_ns = time.monotonic_ns()
            frame = self._render(now_ns, (now_ns - last_ns) / 1e9)
            last_ns = now_ns
            self._note_frame(now_ns)
            with self._lock:
                if self._writer is not None:
                    try:
                        self._writer.stdin.write(frame.tobytes())
                        self._frame_stamps.append(now_ns)
                    except Exception:
                        pass
            tap = self.frame_tap
            if tap is not None and tap.wants(now_ns):
                tap.submit(frame, self._frames_total, now_ns)
            if now_ns - self._last_lores_ns >= self._sink_period_ns:
                self._last_lores_ns = now_ns
                self.sink.publish(frame[:: height // lh, :: width // lw][:lh, :lw].copy(), now_ns)
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def begin_recording(self, path: str) -> dict:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("mock recording needs the ffmpeg command")
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("a recording is already in progress")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            width, height = self.settings["resolution"]
            cmd = [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                   "-s", f"{width}x{height}", "-r", str(self.settings["fps"]), "-i", "-",
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", path]
            self._writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self._frame_stamps = []
            self._recording_path = path
            self._encoder = "mock"
        return {"path": path, "started_monotonic_ns": time.monotonic_ns(), "output": "mock-ffmpeg"}

    def end_recording(self) -> dict:
        with self._lock:
            writer, self._writer = self._writer, None
            self._encoder = None
            stamps = list(self._frame_stamps)
        if writer is not None:
            writer.stdin.close()
            writer.wait(timeout=30)
        return {"path": self._recording_path, "frame_timestamps_ns": stamps, "sensor_clock": "monotonic"}


def create_camera(settings: dict, leds=None) -> CameraService:
    if os.environ.get("PLR_HARDWARE", "").lower() == "mock":
        return MockCameraService(settings, leds=leds)
    return CameraService(settings)
