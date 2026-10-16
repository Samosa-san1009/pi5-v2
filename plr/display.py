"""Live IR preview on the GMT020-02 ST7789 TFT (SPI0 CE0, DC=GPIO24, RST=GPIO25).

The display only reads the small frames the camera service already publishes,
so it can never open the camera itself or disturb the recorded stream.

Overlay (optional): stage, REC indicator, which LED is lit, the clock, the
Pi's address (type it into the app), both eye ROI boxes, and the live PuReST
pupil outline for each eye (green = tracked, red = lost).
"""

from __future__ import annotations

import math
import os
import socket
import threading
import time
from datetime import datetime


class NoBacklight:
    """Backlight 'controller' that touches no GPIO.

    luma.lcd drives a backlight pin on GPIO18 by default. This 7-pin panel has
    no backlight pin (the backlight is tied to VCC), and GPIO18 is the left
    WS2812's data pin, so luma must never claim it.
    """

    def __call__(self, is_enabled: bool):
        pass

    def cleanup(self):
        pass


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except Exception:
        return "no network"


class TftDisplay:
    def __init__(self, settings: dict, camera_settings: dict, eye_roi: dict, camera, state_provider,
                 leds=None, live=None):
        self.settings = settings
        self.camera_settings = camera_settings
        self.eye_roi = eye_roi
        self.camera = camera
        self.live = live
        self.state_provider = state_provider  # callable -> dict(stage=..., port=...)
        self.error: str | None = None
        self.frames_shown = 0
        self._device = None
        self._stop = threading.Event()
        self._thread = None
        self._led_rgb = {"left": (0, 0, 0), "right": (0, 0, 0)}
        self._ip = local_ip()
        self._ip_checked = time.monotonic()
        if leds is not None:
            leds.state_listeners.append(self._on_led)

    def _on_led(self, side, rgb, _t):
        self._led_rgb[side] = rgb

    def start(self):
        if not self.settings.get("enabled", True):
            self.error = "disabled in settings"
            return
        if os.environ.get("PLR_HARDWARE", "").lower() == "mock":
            self.error = "mock hardware: display not driven"
            return
        self._thread = threading.Thread(target=self._run, name="tft-display", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        return {"enabled": bool(self.settings.get("enabled", True)), "running": self._device is not None,
                "frames_shown": self.frames_shown, "error": self.error}

    def _make_device(self):
        from luma.core.interface.serial import spi
        from luma.lcd.device import st7789

        cfg = self.settings
        serial = spi(port=int(cfg["spi_port"]), device=int(cfg["spi_device"]), gpio_DC=int(cfg["gpio_dc"]),
                     gpio_RST=int(cfg["gpio_rst"]), bus_speed_hz=int(cfg["bus_speed_hz"]))
        # The backlight is tied to VCC: pass a no-op backlight so luma does not
        # claim its default backlight pin, GPIO18 (the left LED).
        return st7789(serial, width=int(cfg["width"]), height=int(cfg["height"]),
                      rotate=int(cfg["rotate"]), backlight=NoBacklight())

    def _run(self):
        from PIL import Image

        period = 1.0 / max(1.0, float(self.settings.get("fps", 15)))
        seq = 0
        backoff = 2.0
        while not self._stop.is_set():
            if self._device is None:
                try:
                    self._device = self._make_device()
                    self.error = None
                    backoff = 2.0
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._stop.wait(backoff)
                    backoff = min(30.0, backoff * 2)
                    continue
            item = self.camera.sink.wait(seq, timeout=1.0)
            if item is None:
                self._show_message("waiting for camera")
                continue
            seq, grey, _ts = item
            started = time.monotonic()
            try:
                img = Image.fromarray(grey, "L").convert("RGB")
                size = (self._device.width, self._device.height)
                if img.size != size:
                    img = img.resize(size)
                if self.settings.get("overlay", True):
                    self._draw_overlay(img)
                self._device.display(img)
                self.frames_shown += 1
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._device = None
                continue
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _show_message(self, text: str):
        if self._device is None:
            return
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (self._device.width, self._device.height), "black")
            draw = ImageDraw.Draw(img)
            draw.text((8, 8), text, fill="white")
            draw.text((8, 24), f"{self._ip}:{(self.state_provider() or {}).get('port', 5000)}", fill=(120, 220, 255))
            self._device.display(img)
        except Exception:
            self._device = None

    def _draw_overlay(self, img):
        from PIL import ImageDraw

        now = time.monotonic()
        if now - self._ip_checked > 30:
            self._ip, self._ip_checked = local_ip(), now
        draw = ImageDraw.Draw(img)
        width, height = img.size
        main_w, main_h = self.camera_settings["resolution"]
        sx, sy = width / float(main_w), height / float(main_h)

        if self.settings.get("show_rois", True):
            for eye, color in (("left", (37, 99, 235)), ("right", (220, 38, 38))):
                automatic = self.live is not None and self.live.settings.get("eye_localization", {}).get("mode") == "auto"
                if automatic:
                    sample = self.live.latest.get(eye)
                    box = (sample or {}).get("eye_box")
                    if not box or now - sample["at"] > .6 or box[2] <= 0:
                        continue
                    x, y, w, h = box
                    draw.rectangle((x * sx, y * sy, (x + w) * sx - 1, (y + h) * sy - 1), outline=color)
                else:
                    x, y, w, h = self.eye_roi[eye]
                    draw.rectangle((x * width, y * height, (x + w) * width - 1, (y + h) * height - 1), outline=color)
        if self.settings.get("show_pupils", True) and self.live is not None:
            for eye in ("left", "right"):
                sample = self.live.latest.get(eye)
                if not sample or now - sample["at"] > 0.6:
                    continue
                if sample["valid"] and math.isfinite(sample["cx"]):
                    cx, cy = sample["cx"] * sx, sample["cy"] * sy
                    r = max(2.0, 0.5 * sample["major"] * sx)
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 255, 0), width=2)
                else:
                    box = sample.get("eye_box")
                    text_x = (box[0] + box[2] / 2) * sx - 12 if box and box[2] else (6 if eye == "left" else width - 32)
                    draw.text((text_x, 18), "lost", fill=(255, 60, 60))

        state = self.state_provider() or {}
        stage = str(state.get("stage", "idle")).upper()
        draw.rectangle((0, 0, width, 14), fill=(0, 0, 0))
        draw.text((3, 2), stage, fill=(255, 255, 255))
        if self.camera.recording:
            draw.ellipse((width - 44, 3, width - 36, 11), fill=(255, 0, 0))
            draw.text((width - 32, 2), "REC", fill=(255, 0, 0))
        draw.text((width // 2 - 24, 2), datetime.now().strftime("%H:%M:%S"), fill=(180, 180, 180))
        for side, x in (("left", 6), ("right", width - 14)):
            rgb = self._led_rgb.get(side, (0, 0, 0))
            if max(rgb) > 0:
                draw.rectangle((x, 20, x + 8, 28), fill=tuple(min(255, c * 4) for c in rgb), outline=(255, 255, 255))
        draw.rectangle((0, height - 13, width, height), fill=(0, 0, 0))
        draw.text((3, height - 12), f"{self._ip}:{state.get('port', 5000)}", fill=(120, 220, 255))
