"""Mobile PLR server for Raspberry Pi 5.

Start:  .venv/bin/python server.py      (normally run by plr.service)

The existing app works unchanged: enter the Pi's address (shown on the TFT,
port 5000) in the app's "Pi server" field and start a session.

App endpoints (same request/response shapes as before):

    GET  /health   GET /status   POST /session   GET /results   GET /video

Extra endpoints:

    GET  /settings                 device settings (pins, LED power, camera, PuReST, eye ROIs)
    POST /settings                 partial update, e.g. {"eye_roi": {"left": [0, 0.1, 0.5, 0.8]}}
    POST /settings/reset
    GET  /purest                   live engine state, per-eye processing time, dropped frames
    POST /session/stop             abort the running session
    GET  /preview.jpg              latest IR frame (small stream)
    GET  /stream.mjpg              live IR view for a browser
    POST /leds/test                {"side": "left", "hex": "#FFFFFF", "intensity": 50, "duration": 0.2}
    POST /selftest                 flash each LED, report optical detection and latency
    GET  /time                     NTP / clock status
    GET  /sessions                 recorded sessions
    GET  /sessions/<id>/results    ?source=live|offline for that pass (default: what the app got)
    GET  /sessions/<id>/video
    POST /sessions/<id>/reanalyze  re-run offline PuReST on the saved video {"promote": false}
    POST /restart                  restart the service (applies camera/pin changes)
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import os
import re
import shutil
import signal
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plr import __version__, purest, timebase  # noqa: E402
from plr.analysis.results import build_results, load_frame_times  # noqa: E402
from plr.camera import create_camera  # noqa: E402
from plr.config_adapter import adapt  # noqa: E402
from plr.display import TftDisplay  # noqa: E402
from plr.endstops import EndstopMonitor  # noqa: E402
from plr.leds import LedController  # noqa: E402
from plr.orchestrator import SessionRecorder, _sleep_until  # noqa: E402
from plr.preview import tracking_image  # noqa: E402
from plr.settings import REPO_ROOT, SettingsStore  # noqa: E402

logging.basicConfig(
    level=getattr(logging, os.environ.get("PLR_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG = logging.getLogger("plr.server")
MOCK = os.environ.get("PLR_HARDWARE", "").lower() == "mock"
SESSIONS_ROOT = Path(os.environ.get("PLR_SESSIONS_ROOT", REPO_ROOT / "sessions"))

app = Flask(__name__)
SETTINGS = SettingsStore()
STATE_LOCK = threading.Lock()
SESSION_LOCK = threading.Lock()        # held while recording + producing the app's result
ABORT = threading.Event()
STATE: dict = {
    "stage": "idle", "started_at": None, "ended_at": None,
    "error": None, "error_type": None, "error_stage": None, "error_details": None, "error_log_path": None,
    "session_id": None, "session_dir": None, "log_path": None, "video_path": None,
    "pupilogram_path": None, "metrics_path": None, "progress": None, "session_status": None,
    "analyzer": "purest", "analysis_source": None, "live_coverage": None, "warnings": [],
}
BUSY = {"reason": None}                # what currently holds SESSION_LOCK
BACKGROUND: dict = {"thread": None, "cancel": None, "info": None}   # explicit saved-session reanalysis
RIG: dict = {}


# ── hardware ────────────────────────────────────────────────────────────────

def build_rig():
    settings = SETTINGS.get()
    leds = LedController(settings["leds"])
    mock_backend = leds.backend if leds.is_mock else None
    endstops = EndstopMonitor(settings["endstops"], mock_led_backend=mock_backend)
    endstops.start()
    live = purest.LiveEngine(settings, logger=LOG)
    try:
        live.start()
    except Exception as exc:
        live.error = f"{type(exc).__name__}: {exc}"
    camera = create_camera(settings["camera"], leds=leds)
    camera.frame_tap = live
    camera.start()
    display = TftDisplay(settings["display"], settings["camera"], settings["eye_roi"], camera,
                         state_provider=display_state, leds=leds, live=live)
    display.start()
    RIG.update(leds=leds, endstops=endstops, camera=camera, display=display, live=live)
    for name, part in (("LEDs", leds), ("endstops", endstops), ("live PuReST", live)):
        if getattr(part, "error", None):
            LOG.warning("%s: %s", name, part.error)


def rebuild_live_engine(settings: dict):
    """Apply new PuReST / eye-ROI settings (call with SESSION_LOCK held)."""
    old = RIG.get("live")
    live = purest.LiveEngine(settings, logger=LOG)
    try:
        live.start()
    except Exception as exc:
        live.error = f"{type(exc).__name__}: {exc}"
    RIG["camera"].frame_tap = live
    RIG["display"].live = live
    RIG["display"].eye_roi = settings["eye_roi"]
    RIG["live"] = live
    if old is not None:
        old.stop()


def display_state() -> dict:
    with STATE_LOCK:
        return {"stage": STATE["stage"], "port": SETTINGS.section("server")["port"]}


def shutdown(*_):
    try:
        ABORT.set()
        cancel_background()
        for name in ("leds", "display", "camera", "live", "endstops"):
            part = RIG.get(name)
            if part is None:
                continue
            (part.close if name in ("leds", "endstops") else part.stop)()
    except Exception:
        pass


def set_state(**changes):
    with STATE_LOCK:
        STATE.update(changes)


def state_copy() -> dict:
    with STATE_LOCK:
        out = json.loads(json.dumps(STATE, default=str))
    info = BACKGROUND.get("info")
    if info:
        out["background"] = dict(info)
    return out


# ── sessions ────────────────────────────────────────────────────────────────

def make_session_paths(participant: dict) -> dict:
    stamp, clock_source = timebase.session_datetime()
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(participant.get("name") or "anon")).strip("_") or "anon"
    session_id = f"{stamp.strftime('%Y%m%d_%H%M%S_%f')}_{name}"
    session_dir = SESSIONS_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    return {
        "session_id": session_id, "session_dir": session_dir, "wall_clock_source": clock_source,
        "video": session_dir / "full_video.mp4", "events": session_dir / "events.json",
        "pupilogram": session_dir / "pupilogram.json", "metrics": session_dir / "metrics.json",
        "log": session_dir / "session.log", "manifest": session_dir / "manifest.json",
    }


def session_logger(log_path: Path, session_id: str) -> logging.Logger:
    logger = logging.getLogger(f"plr.session.{session_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers = [handler]
    return logger


def close_logger(logger: logging.Logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def write_manifest(paths: dict, config: dict, settings: dict):
    manifest = {
        "session_id": paths["session_id"],
        "software_version": __version__,
        "mock_hardware": MOCK or RIG["leds"].is_mock,
        "wall_clock_source": paths["wall_clock_source"],
        "clock_status": timebase.clock_status(),
        "config": {k: v for k, v in config.items() if k != "request"},
        "request": config.get("request"),
        "settings": settings,
        "hardware": {
            "leds": RIG["leds"].status(),
            "endstops": RIG["endstops"].status(),
            "camera": RIG["camera"].status(),
            "purest_live": RIG["live"].status(),
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def copy_results(session_dir: Path, from_suffix: str, to_suffix: str = ""):
    extraction = session_dir / f"extraction{from_suffix}"
    if extraction.is_dir():
        destination = session_dir / f"extraction{to_suffix}"
        shutil.copytree(extraction, destination, dirs_exist_ok=True)
        for eye in ("left", "right"):
            name = f"{eye}_pupillogram.png"
            if not (extraction / name).is_file():
                (destination / name).unlink(missing_ok=True)
    for stem in ("pupilogram", "metrics"):
        src = session_dir / f"{stem}{from_suffix}.json"
        dst = session_dir / f"{stem}{to_suffix}.json"
        tmp = dst.with_suffix(".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


def offline_pass(session_dir: Path, session_id: str, settings: dict, participant, logger,
                 progress=None, cancel=None) -> dict:
    """Run offline PuReST on the saved video; write *_offline.json."""
    stamps, _ = load_frame_times(session_dir)
    started = time.monotonic()
    traces = purest.run_offline(session_dir / "full_video.mp4", session_dir / ".work", settings,
                                frame_count=len(stamps), progress=progress, cancel=cancel, logger=logger)
    seconds = round(time.monotonic() - started, 2)
    result = build_results(session_dir, session_id, traces, "offline", settings, participant,
                           suffix="_offline", engine={"offline_seconds": seconds, "frames": len(stamps)})
    logger.info("offline PuReST: %s in %.1fs, coverage %s", result["status"], seconds, result["coverage"])
    try:
        (session_dir / ".work").rmdir()
    except OSError:
        pass
    return result


def cancel_background(wait_s: float = 5.0):
    thread, cancel = BACKGROUND.get("thread"), BACKGROUND.get("cancel")
    if thread is not None and thread.is_alive():
        cancel.set()
        thread.join(wait_s)


def run_session(config: dict, paths: dict, settings: dict):
    logger = session_logger(paths["log"], paths["session_id"])
    session_dir, session_id = paths["session_dir"], paths["session_id"]
    participant = config.get("participant")
    purest_cfg = settings["purest"]
    live = RIG["live"]
    live_traces = None
    try:
        write_manifest(paths, config, settings)
        live_on = False
        if purest_cfg["live"]:
            try:
                live_on = live.begin_session()
            except Exception:
                logger.exception("Live preview failed to start; saved-video analysis will still run")
            if not live_on:
                logger.warning("Live preview unavailable (%s); the saved video will still be analyzed", live.error)
        set_state(stage="recording", started_at=time.time())
        recorder = SessionRecorder(config, paths, RIG["camera"], RIG["leds"], RIG["endstops"],
                                   settings["endstops"], logger, ABORT)
        try:
            timeline = recorder.run()["timeline"]
        finally:
            collected = None
            if purest_cfg["live"]:
                try:
                    collected = live.end_session()
                except Exception:
                    logger.exception("Live collection failed; continuing with saved-video analysis")
        set_state(stage="analysis", warnings=timeline["warnings"],
                  progress={"completed": 0, "total": 2, "item": "both eyes", "item_status": "live",
                            "fraction": 0.0})
        for message in timeline["warnings"]:
            logger.warning(message)
        logger.info("Optical timing: %s", json.dumps(timeline["optical"]))

        stamps, _ = load_frame_times(session_dir)
        if collected is not None and live_on and stamps:
            try:
                live_traces = purest.live_traces(collected, stamps, float(purest_cfg["confidence_threshold"]))
                engine = {**live.status(), "frames": len(stamps),
                          "dropped": {e: len(collected["dropped"][e]) for e in purest.EYES}}
                result = build_results(session_dir, session_id, live_traces, "live", settings, participant,
                                       suffix="_live", engine=engine)
                logger.info("Live preview pass: %s, coverage %s", result["status"], result["coverage"])
                set_state(live_coverage=result["coverage"])
            except Exception:
                logger.exception("Could not save live preview results; continuing with saved-video analysis")

        # Always analyze the finalized MP4, including manual Stop and legacy
        # live/auto settings. Publish DONE only once these results are saved.
        set_state(stage="analysis", progress={"completed": 0, "total": 2, "item": "both eyes",
                                              "item_status": "softflow_lite_offline", "fraction": 0.0})
        logger.info("Starting PuReST Soft Flow Lite on saved video: %s", session_dir / "full_video.mp4")
        result = offline_pass(session_dir, session_id, settings, participant, logger,
                              progress=lambda f: set_state(progress={
                                  "completed": 2 if f >= 1 else 0, "total": 2, "item": "both eyes",
                                  "item_status": "softflow_lite_offline", "fraction": round(f, 3)}))
        copy_results(session_dir, "_offline")
        set_state(stage="done", ended_at=time.time(), session_status=result["status"],
                  analysis_source="offline", progress=None)
        logger.info("Session complete from saved-video Soft Flow Lite: %s", result["status"])
    except Exception as exc:
        failed = state_copy()["stage"]
        logger.exception("Session failed at stage=%s", failed)
        traceback.print_exc()
        set_state(stage="error", ended_at=time.time(), error=str(exc) or type(exc).__name__,
                  error_type=type(exc).__name__, error_stage=failed,
                  error_details=getattr(exc, "details", None), error_log_path=str(paths["log"]))
    finally:
        try:
            RIG["leds"].all_off()
        except Exception:
            pass
        close_logger(logger)
        BUSY["reason"] = None
        SESSION_LOCK.release()


def session_dir_for(session_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError("invalid session id")
    path = SESSIONS_ROOT / session_id
    if not path.is_dir():
        raise FileNotFoundError(session_id)
    return path


def results_payload(session_dir: Path, source: str | None = None):
    suffix = f"_{source}" if source in ("live", "offline") else ""
    pupilogram_path = session_dir / f"pupilogram{suffix}.json"
    metrics_path = session_dir / f"metrics{suffix}.json"
    if not pupilogram_path.is_file() or not metrics_path.is_file():
        return None
    pupilogram = json.loads(pupilogram_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 6,
        "status": metrics["status"],
        "session_id": metrics["session_id"],
        "analysis_source": metrics.get("analysis_source"),
        "video": {"path": str(session_dir / "full_video.mp4"), "url": f"/sessions/{session_dir.name}/video"},
        "pupilogram": pupilogram,
        "metrics": metrics,
    }


# ── routes: app contract ────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
@app.route("/monitor", methods=["GET"])
def monitor():
    return send_file(REPO_ROOT / "plr" / "web" / "monitor.html", max_age=0)

@app.route("/health", methods=["GET"])
def health():
    leds, camera, endstops = RIG["leds"].status(), RIG["camera"].status(), RIG["endstops"].status()
    state = state_copy()
    error = None
    if state.get("error"):
        error = {k: state.get(k) for k in ("error", "error_type", "error_stage", "error_details", "error_log_path")}
    return jsonify({
        "ok": bool(leds["ready"] and camera["running"] and purest.binary_status()["ready"]),
        "version": __version__,
        "mock_hardware": MOCK,
        "stage": state["stage"],
        "error": error,
        "hardware": {
            "camera": camera,
            "leds": leds,
            "led_control": leds["ready"],
            "endstops": endstops,
            "display": RIG["display"].status(),
            "clock": timebase.clock_status(),
        },
        "purest": {"binary": purest.binary_status(), "live": RIG["live"].status()},
        "analyzers": {"purest": {"ready": purest.binary_status()["ready"],
                                 "description": "PuReST Soft Flow Lite, CPU-only C++, both eyes (live + offline)"}},
    })


@app.route("/session", methods=["POST"])
def start_session():
    if not SESSION_LOCK.acquire(blocking=False):
        return jsonify({"ok": False, "error": f"busy: {BUSY['reason'] or 'session running'}",
                        "stage": state_copy()["stage"]}), 409
    BUSY["reason"] = "session running"
    started = False
    try:
        settings = SETTINGS.get()
        try:
            config = adapt(request.get_json(silent=True) or {}, settings)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "error_type": type(exc).__name__,
                            "error_stage": "request_validation"}), 400
        leds = RIG["leds"]
        if not leds.ready:
            return jsonify({"ok": False, "stage": "hardware_initialization", "error": leds.error,
                            "hardware": {"leds": leds.status(), "led_control": False}}), 503
        if leds.is_mock and not (MOCK or os.environ.get("PLR_ALLOW_MOCK") == "1"):
            return jsonify({"ok": False, "error": "refusing a PLR session with simulated LEDs"}), 503
        if not RIG["camera"].status()["running"]:
            return jsonify({"ok": False, "stage": "hardware_initialization",
                            "error": f"camera not running: {RIG['camera'].error}"}), 503
        binary = purest.binary_status()
        if not binary["ready"]:
            return jsonify({"ok": False, "stage": "model_initialization",
                            "error": f"PuReST is not ready: {binary['error']}", "error_details": binary}), 503
        cancel_background()   # a previous session's offline pass must not steal the eye cores
        paths = make_session_paths(config.get("participant") or {})
        ABORT.clear()
        set_state(stage="starting", started_at=None, ended_at=None, error=None, error_type=None,
                  error_stage=None, error_details=None, error_log_path=None, progress=None,
                  session_status=None, warnings=[], analyzer="purest", analysis_source=None,
                  live_coverage=None, session_id=paths["session_id"], session_dir=str(paths["session_dir"]),
                  log_path=str(paths["log"]), video_path=str(paths["video"]),
                  pupilogram_path=str(paths["pupilogram"]), metrics_path=str(paths["metrics"]))
        threading.Thread(target=run_session, args=(config, paths, settings), daemon=True,
                         name=f"session-{paths['session_id']}").start()
        started = True
        return jsonify({"ok": True, "stage": "starting", "session_id": paths["session_id"],
                        "analyzer": "purest"}), 202
    finally:
        if not started:
            BUSY["reason"] = None
            SESSION_LOCK.release()


@app.route("/session/stop", methods=["POST"])
def stop_session():
    ABORT.set()
    return jsonify({"ok": True, "stage": state_copy()["stage"]})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(state_copy())


@app.route("/results", methods=["GET"])
def results():
    state = state_copy()
    if not state.get("session_dir"):
        return jsonify({"error": "not ready", "stage": state["stage"]}), 404
    payload = results_payload(Path(state["session_dir"]))
    if payload is None:
        return jsonify({"error": "not ready", "stage": state["stage"]}), 404
    payload["video"]["url"] = "/video"
    return jsonify(payload)


@app.route("/video", methods=["GET"])
def video():
    path = state_copy().get("video_path")
    if not path or not Path(path).is_file() or RIG["camera"].recording:
        return jsonify({"error": "not ready", "stage": state_copy()["stage"]}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=True)


# ── routes: settings, sessions ──────────────────────────────────────────────

@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify(SETTINGS.get())


@app.route("/settings", methods=["POST"])
def post_settings():
    patch = request.get_json(silent=True) or {}
    before = SETTINGS.get()
    try:
        after = SETTINGS.update(patch)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    applied, restart = [], []
    live_changed = (after["purest"] != before["purest"] or after["eye_roi"] != before["eye_roi"]
                    or after["eye_localization"] != before["eye_localization"])
    if after["leds"] != before["leds"] or live_changed:
        if SESSION_LOCK.acquire(blocking=False):
            try:
                if after["leds"] != before["leds"]:
                    RIG["leds"].reconfigure(after["leds"])
                    applied.append("leds")
                if live_changed:
                    rebuild_live_engine(after)
                    applied.append("purest/eye_roi/eye_localization")
            finally:
                SESSION_LOCK.release()
        else:
            restart.append("leds/purest (session running; save again after it ends)")
    for section in ("endstops", "camera", "display", "server"):
        if after[section] != before[section]:
            restart.append(section)
    if after["session"] != before["session"]:
        applied.append("session")
    return jsonify({"ok": True, "applied": applied, "restart_required": restart, "settings": after})


@app.route("/settings/reset", methods=["POST"])
def reset_settings():
    return jsonify({"ok": True, "settings": SETTINGS.reset(), "restart_required": ["all"]})


@app.route("/purest", methods=["GET"])
def purest_status():
    return jsonify({"binary": purest.binary_status(), "live": RIG["live"].status(),
                    "settings": SETTINGS.section("purest"), "eye_roi": SETTINGS.section("eye_roi"),
                    "eye_localization": SETTINGS.section("eye_localization"),
                    "background": BACKGROUND.get("info")})


@app.route("/sessions", methods=["GET"])
def list_sessions():
    out = []
    if SESSIONS_ROOT.is_dir():
        for path in sorted(SESSIONS_ROOT.iterdir(), reverse=True)[:200]:
            if not path.is_dir():
                continue
            entry = {"session_id": path.name, "has_video": (path / "full_video.mp4").is_file(),
                     "results": sorted(p.stem.replace("metrics", "").lstrip("_") or "default"
                                       for p in path.glob("metrics*.json"))}
            try:
                metrics = json.loads((path / "metrics.json").read_text())
                entry["status"] = metrics.get("status")
                entry["analysis_source"] = metrics.get("analysis_source")
            except Exception:
                pass
            out.append(entry)
    return jsonify({"sessions": out})


@app.route("/sessions/<session_id>/results", methods=["GET"])
def session_results(session_id):
    try:
        path = session_dir_for(session_id)
    except (ValueError, FileNotFoundError):
        return jsonify({"error": "unknown session"}), 404
    payload = results_payload(path, request.args.get("source"))
    if payload is None:
        return jsonify({"error": "not ready", "background": BACKGROUND.get("info")}), 404
    return jsonify(payload)


@app.route("/sessions/<session_id>/video", methods=["GET"])
def session_video(session_id):
    try:
        path = session_dir_for(session_id) / "full_video.mp4"
    except (ValueError, FileNotFoundError):
        return jsonify({"error": "unknown session"}), 404
    if not path.is_file():
        return jsonify({"error": "no video"}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=True)


@app.route("/sessions/<session_id>/extraction/<eye>/<artifact>", methods=["GET"])
def extraction_artifact(session_id, eye, artifact):
    source = request.args.get("source", "")
    if eye not in ("left", "right") or source not in ("", "live", "offline") or artifact not in (
            "smoothed.csv", "smoothed.json", "mca.csv", "mcv.csv", "pupillogram.png"):
        return jsonify({"error": "unknown extraction artifact"}), 404
    try:
        root = session_dir_for(session_id)
    except (ValueError, FileNotFoundError):
        return jsonify({"error": "unknown session"}), 404
    suffix = f"_{source}" if source else ""
    path = root / f"extraction{suffix}" / f"{eye}_{artifact}"
    if not path.is_file():
        return jsonify({"error": "extraction not available; analyze this saved session again"}), 404
    return send_file(path, max_age=0, as_attachment=not artifact.endswith(".png"))


@app.route("/sessions/<session_id>/reanalyze", methods=["POST"])
def reanalyze(session_id):
    body = request.get_json(silent=True) or {}
    promote = bool(body.get("promote", False))
    try:
        session_dir = session_dir_for(session_id)
    except (ValueError, FileNotFoundError):
        return jsonify({"ok": False, "error": "unknown session"}), 404
    if not (session_dir / "events.json").is_file() or not (session_dir / "full_video.mp4").is_file():
        return jsonify({"ok": False, "error": "session has no recording"}), 409
    if not SESSION_LOCK.acquire(blocking=False):
        return jsonify({"ok": False, "error": f"busy: {BUSY['reason'] or 'another operation is running'}"}), 409
    cancel_background()
    BUSY["reason"] = f"re-analysing {session_id}"
    settings = SETTINGS.get()
    info = {"session_id": session_id, "job": "reanalyze", "state": "running", "fraction": 0.0}
    BACKGROUND.update(thread=None, cancel=None, info=info)
    participant = None
    try:
        participant = json.loads((session_dir / "manifest.json").read_text()).get("config", {}).get("participant")
    except Exception:
        pass

    def work():
        logger = session_logger(session_dir / "session.log", f"{session_id}.reanalyze")
        try:
            result = offline_pass(session_dir, session_id, settings, participant, logger,
                                  progress=lambda f: info.update(fraction=round(f, 3)))
            if promote:
                copy_results(session_dir, "_offline")
            info.update(state="done", fraction=1.0, status=result["status"], promoted=promote)
        except Exception as exc:
            info.update(state="error", error=str(exc))
            logger.exception("re-analysis failed")
        finally:
            close_logger(logger)
            BUSY["reason"] = None
            SESSION_LOCK.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True, "results_url": f"/sessions/{session_id}/results?source=offline",
                    "promote": promote}), 202


# ── routes: live view, LEDs, diagnostics ────────────────────────────────────

def _jpeg(grey, overlay=True) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    image = (tracking_image(grey, SETTINGS.section("camera")["resolution"], RIG.get("live"), state_copy())
             if overlay else Image.fromarray(grey, "L"))
    image.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@app.route("/preview.jpg", methods=["GET"])
def preview():
    item = RIG["camera"].sink.wait(0, timeout=2.0)
    if item is None:
        return jsonify({"error": "no frame yet"}), 503
    return Response(_jpeg(item[1], request.args.get("overlay", "1") != "0"), mimetype="image/jpeg")


@app.route("/stream.mjpg", methods=["GET"])
def stream():
    fps = min(15.0, max(1.0, float(request.args.get("fps", 10))))
    overlay = request.args.get("overlay", "1") != "0"

    def frames():
        seq = 0
        while True:
            item = RIG["camera"].sink.wait(seq, timeout=2.0)
            if item is None:
                continue
            seq, grey, _ = item
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _jpeg(grey, overlay) + b"\r\n"
            time.sleep(1.0 / fps)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _flash_and_measure(side: str, hex_color: str, intensity: float, duration_s: float, power=None) -> dict:
    leds, endstops = RIG["leds"], RIG["endstops"]
    rgb = leds.rgb_for(side, hex_color, intensity, power)
    on = leds.set({side: rgb})
    _sleep_until(on["end_ns"] + int(duration_s * 1e9))
    off = leds.set({side: (0, 0, 0)})
    time.sleep(0.05)
    out = {"side": side, "rgb": list(rgb), "optical_detected": False}
    if endstops.ready:
        window_end = off["end_ns"] + 50_000_000
        max_latency = int(float(SETTINGS.section("endstops")["max_latency_ms"]) * 1e6)
        own = [p for p in endstops.pulses(side, on["start_ns"] - 1_000_000, window_end)
               if p["onset_ns"] <= on["end_ns"] + max_latency]
        other = "right" if side == "left" else "left"
        cross = endstops.pulses(other, on["start_ns"] - 1_000_000, window_end) if other in endstops.sides else []
        if own:
            pulse = own[0]
            out.update(optical_detected=True,
                       latency_ms=round((pulse["onset_ns"] - on["end_ns"]) / 1e6, 4),
                       measured_duration_ms=round((pulse["offset_ns"] - pulse["onset_ns"]) / 1e6, 3),
                       raw_pulses=pulse["raw_pulses"])
        out["crosstalk_pulses_on_other_sensor"] = len(cross)
    return out


@app.route("/leds/test", methods=["POST"])
def led_test():
    body = request.get_json(silent=True) or {}
    side = body.get("side", "left")
    if side not in ("left", "right", "both"):
        return jsonify({"ok": False, "error": "side must be left, right or both"}), 400
    if not RIG["leds"].ready:
        return jsonify({"ok": False, "error": RIG["leds"].error}), 503
    if not SESSION_LOCK.acquire(blocking=False):
        return jsonify({"ok": False, "error": f"busy: {BUSY['reason'] or 'another operation is running'}"}), 409
    try:
        sides = ["left", "right"] if side == "both" else [side]
        duration = min(5.0, max(0.02, float(body.get("duration", 0.2))))
        results = [_flash_and_measure(s, body.get("hex", "#FFFFFF"), float(body.get("intensity", 100)),
                                      duration, body.get("power")) for s in sides]
        return jsonify({"ok": True, "results": results})
    finally:
        SESSION_LOCK.release()


@app.route("/selftest", methods=["POST"])
def selftest():
    body = request.get_json(silent=True) or {}
    repeats = max(1, min(20, int(body.get("repeats", 5))))
    intensity = float(body.get("intensity", 100))
    if not RIG["leds"].ready:
        return jsonify({"ok": False, "error": RIG["leds"].error}), 503
    if not SESSION_LOCK.acquire(blocking=False):
        return jsonify({"ok": False, "error": f"busy: {BUSY['reason'] or 'another operation is running'}"}), 409
    try:
        report = {"endstops": RIG["endstops"].status(), "idle_warnings": RIG["endstops"].recheck_idle(), "sides": {}}
        for side in ("left", "right"):
            trials = []
            for _ in range(repeats):
                trials.append(_flash_and_measure(side, "#FFFFFF", intensity, 0.1))
                time.sleep(0.25)
            latencies = [t["latency_ms"] for t in trials if t.get("optical_detected")]
            report["sides"][side] = {
                "detected": f"{len(latencies)}/{repeats}",
                "latency_ms_median": round(statistics.median(latencies), 4) if latencies else None,
                "latency_ms_range": [min(latencies), max(latencies)] if latencies else None,
                "crosstalk_trials": sum(1 for t in trials if t.get("crosstalk_pulses_on_other_sensor")),
            }
        report["ok"] = all(v["detected"] == f"{repeats}/{repeats}" for v in report["sides"].values())
        return jsonify(report)
    finally:
        SESSION_LOCK.release()


@app.route("/time", methods=["GET"])
def time_status():
    return jsonify({**timebase.clock_status(), "anchor": timebase.anchor()})


@app.route("/restart", methods=["POST"])
def restart():
    if not SESSION_LOCK.acquire(blocking=False):
        return jsonify({"ok": False, "error": f"busy: {BUSY['reason'] or 'another operation is running'}"}), 409

    def later():
        time.sleep(0.5)
        shutdown()
        os._exit(0)  # systemd (Restart=always) starts a fresh process

    threading.Thread(target=later, daemon=True).start()
    return jsonify({"ok": True, "message": "restarting"})


@app.errorhandler(Exception)
def unexpected(exc):
    if isinstance(exc, HTTPException):
        return exc
    request_id = f"req_{time.time_ns()}"
    LOG.exception("Unhandled error request_id=%s %s %s", request_id, request.method, request.path)
    return jsonify({"ok": False, "error": str(exc) or "internal error", "error_type": type(exc).__name__,
                    "request_id": request_id}), 500


def main():
    import faulthandler

    faulthandler.enable()      # a native crash prints every thread's stack to the journal
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    build_rig()
    atexit.register(shutdown)
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), os._exit(0)))
    port = int(os.environ.get("PLR_PORT", SETTINGS.section("server")["port"]))
    LOG.info("Mobile PLR %s listening on 0.0.0.0:%d (mock=%s)", __version__, port, MOCK)
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
