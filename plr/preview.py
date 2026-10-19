"""Lightweight browser tracking overlay. Pillow only; no OpenCV Python/model inference."""
from __future__ import annotations

import math
import time

from PIL import Image, ImageDraw


def tracking_image(grey, resolution, live=None, state=None, now=None):
    image = Image.fromarray(grey, "L").convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    sx, sy = width / resolution[0], height / resolution[1]
    now = time.monotonic() if now is None else now
    if live is not None:
        for eye, color in (("left", (65, 155, 255)), ("right", (255, 100, 100))):
            sample = live.latest.get(eye)
            if not sample or now - sample["at"] > .6:
                continue
            box = sample.get("eye_box")
            if box and box[2] > 0 and box[3] > 0:
                x, y, w, h = box
                draw.rectangle((x*sx, y*sy, (x+w)*sx, (y+h)*sy), outline=color, width=1)
                draw.text((x*sx, max(18, y*sy-13)), eye.upper(), fill=color)
            values = [sample.get(k) for k in ("cx", "cy", "major")]
            measured = (sample.get("valid") and not sample.get("estimated")
                        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values))
            if measured:
                cx, cy, diameter = values
                x, y = cx*sx, cy*sy
                rx, ry = max(2, diameter*sx/2), max(2, diameter*sy/2)
                draw.ellipse((x-rx, y-ry, x+rx, y+ry), outline=(70, 255, 70), width=2)
            elif box and box[2] > 0:
                draw.text((box[0]*sx, (box[1]+box[3])*sy+1), "lost", fill=(255, 100, 100))
    stage = str((state or {}).get("stage", "idle")).upper()
    draw.rectangle((0, 0, width, 15), fill=(10, 18, 28))
    draw.text((4, 1), f"Soft Flow Lite | {stage}", fill=(240, 245, 255))
    if live is not None and live.error:
        draw.rectangle((0, height-17, width, height), fill=(75, 15, 15))
        draw.text((4, height-15), "Tracker unavailable: check /purest", fill=(255, 220, 220))
    return image
