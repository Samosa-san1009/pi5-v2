"""Translate the app's /session request into a session config.

Works with the existing app unchanged:

    {
      "participant": {"name", "age", "sex"},
      "controlMode": "dual" | "left_to_right" | "right_to_left",
      "pupilAnalyzer": any value (accepted and recorded; PuReST is always used),
      "pupilPreprocess": any value (accepted and ignored),
      "intensity": 0-100,
      "schedule": {
          # dual
          "flashes": [{"hex": "#RRGGBB", "duration": s, "intensity": %}],
          "initialBreak": s, "gap": s, "intensity": %
          # sequential
          "rounds", "hex", "duration", "initialBreak", "innerPause", "gap", "intensity"
      }
    }

The older single-eye payload (eye/color/iterations/duration/delay) is also
accepted.

Optional extras (the app does not send them):

    "leds": {
        "left":  {"power": 0-100, "intensity": 0-100, "enabled": true},
        "right": {"power": 0-100, "intensity": 0-100, "enabled": true}
    }
    dual flashes / sequential schedules may carry "leftIntensity" / "rightIntensity".
    "controlMode": "observe" with schedule {"duration": s} records without flashing.
"""

from __future__ import annotations

SIDES = ("left", "right")
MODES = {"dual", "left_to_right", "right_to_left", "observe"}

COLOR_HEX = {
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "White": "#FFFFFF",
}
ALL_COLORS = ["Red", "Green", "Blue", "Yellow", "White"]


def _pct(value, default=100.0) -> float:
    if value is None:
        return float(default)
    return min(100.0, max(0.0, float(value)))


def _ms(seconds, minimum_ms=50, maximum_ms=None) -> int:
    value = max(minimum_ms, int(round(float(seconds) * 1000)))
    return min(value, maximum_ms) if maximum_ms is not None else value


def _hex(value) -> str:
    text = str(value or "").strip().upper()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError(f"invalid RGB hex colour: {value!r}")
    int(text[1:], 16)
    return text


def adapt(payload: dict, settings: dict) -> dict:
    payload = payload or {}
    mode = payload.get("controlMode") or payload.get("control_mode")
    session_cfg = settings["session"]
    min_gap_ms = int(float(session_cfg.get("min_gap_s", 3.0)) * 1000)

    if mode is None:
        schedule, mode = _legacy_schedule(payload, min_gap_ms)
    else:
        if mode not in MODES:
            raise ValueError(f"unsupported controlMode: {mode!r}")
        schedule = _explicit_schedule(payload, mode, min_gap_ms)

    led_req = payload.get("leds") or {}
    power, enabled = {}, {}
    for side in SIDES:
        side_req = led_req.get(side) or {}
        power[side] = _pct(side_req.get("power"), settings["leds"][side]["power_pct"])
        enabled[side] = bool(side_req.get("enabled", settings["leds"][side].get("enabled", True)))

    return {
        "mode": mode,
        "schedule": schedule,
        "power": power,
        "enabled": enabled,
        "analyzer": "purest",
        "requested_analyzer": payload.get("pupilAnalyzer", payload.get("analyzer")),
        "participant": payload.get("participant") or {},
        "pre_roll_s": float(payload.get("preRoll", session_cfg["pre_roll_s"])),
        "post_roll_s": float(payload.get("postRoll", session_cfg["post_roll_s"])),
        "request": payload,
    }


def _side_intensities(payload: dict, source: dict, shared: float) -> dict:
    led_req = payload.get("leds") or {}
    out = {}
    for side in SIDES:
        key = f"{side}Intensity"
        value = source.get(key)
        if value is None:
            value = (led_req.get(side) or {}).get("intensity")
        out[side] = _pct(value, shared)
    return out


def _explicit_schedule(payload: dict, mode: str, min_gap_ms: int) -> dict:
    incoming = payload.get("schedule") or {}
    shared = _pct(incoming.get("intensity", payload.get("intensity")), 100.0)
    initial_delay_ms = _ms(incoming.get("initialBreak", incoming.get("startBreak", 0.0)), 0, 120_000)

    if mode == "observe":
        return {"initial_delay_ms": initial_delay_ms, "duration_ms": _ms(incoming.get("duration", 10.0), 1000, 600_000)}

    gap_ms = _ms(incoming.get("gap", 3.0), min_gap_ms, 120_000)
    if mode == "dual":
        flashes_in = incoming.get("flashes") or []
        if not flashes_in:
            raise ValueError("dual mode requires at least one flash")
        flashes = []
        for flash in flashes_in:
            flash_shared = _pct(flash.get("intensity"), shared)
            flashes.append({
                "hex": _hex(flash.get("hex")),
                "duration_ms": _ms(flash.get("duration", 1.0)),
                "intensity": _side_intensities(payload, flash, flash_shared),
            })
        return {"initial_delay_ms": initial_delay_ms, "flashes": flashes, "gap_ms": gap_ms}

    order = ["left", "right"] if mode == "left_to_right" else ["right", "left"]
    return {
        "initial_delay_ms": initial_delay_ms,
        "rounds": max(1, int(incoming.get("rounds", 3))),
        "colors_cycle": [_hex(incoming.get("hex", "#FFFFFF"))],
        "duration_ms": _ms(incoming.get("duration", 1.0)),
        "inner_pause_ms": _ms(incoming.get("innerPause", 1.0), 50, 120_000),
        "gap_ms": gap_ms,
        "intensity": _side_intensities(payload, incoming, shared),
        "order": order,
        "active_side": None,
    }


def _legacy_schedule(payload: dict, min_gap_ms: int) -> tuple[dict, str]:
    eye = payload.get("eye", "Left")
    color = payload.get("color", "White")
    iterations = max(1, int(payload.get("iterations", 1)))
    duration_ms = _ms(payload.get("duration", 1.0))
    gap_ms = max(min_gap_ms, _ms(payload.get("delay", 1.0)))
    intensity = _pct(payload.get("intensity"), 100.0)
    colors = ALL_COLORS if color == "All" else [color]
    if any(c not in COLOR_HEX for c in colors):
        raise ValueError(f"unsupported colour: {color!r}")
    both = {"left": intensity, "right": intensity}

    if eye == "Both":
        flashes = [
            {"hex": COLOR_HEX[c], "duration_ms": duration_ms, "intensity": dict(both)}
            for c in colors for _ in range(iterations)
        ]
        return {"initial_delay_ms": 0, "flashes": flashes, "gap_ms": gap_ms}, "dual"

    mode = "right_to_left" if eye == "Right" else "left_to_right"
    side = "right" if eye == "Right" else "left"
    return {
        "initial_delay_ms": 0,
        "rounds": iterations * len(colors),
        "colors_cycle": [COLOR_HEX[c] for c in colors],
        "duration_ms": duration_ms,
        "inner_pause_ms": max(50, gap_ms // 2),
        "gap_ms": gap_ms,
        "intensity": both,
        "order": ["left", "right"] if side == "left" else ["right", "left"],
        "active_side": side,
    }, mode
