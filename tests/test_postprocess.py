"""Extraction integration: timing, provenance, simultaneous LEDs and exports."""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plr import purest
from plr.analysis.postprocess import extract_eye, response_metrics, serializable
from plr.analysis.extraction.smoothen import preprocess
from plr.analysis.results import build_results
from plr.settings import DEFAULTS


def trace(n=240):
    t = purest._empty_trace(n)
    times = np.arange(n)/30
    diameter = 50-10*np.clip((times-2.2)/.6, 0, 1)
    diameter += 10*np.clip((times-3.3)/.5, 0, 1)
    t["diameter_px"] = diameter
    t["confidence"][:] = .9
    t["valid"][:] = True
    t["coverage"] = 1.
    t["tracking_source"][:] = "purest"
    return t, times


def event(led=1, onset=2, offset=5.5):
    return {"event_id": f"flash_{led}", "led_index": led, "onset_s": onset,
            "offset_s": offset, "hex_color": "#FFFFFF"}


def test_adapter_matches_extraction_numerically_and_retains_short_gap_provenance():
    raw, times = trace()
    raw["valid"][72:74] = False
    output, pairs = extract_eye(raw, times, [event()], "test", "left")
    reference_rows = [{"Frame": i, "timestamp_ms": t*1000, "diameter_px": d,
                       "Valid": "1" if raw["valid"][i] else "0"}
                      for i, (t, d) in enumerate(zip(times, raw["diameter_px"]))]
    reference, _ = preprocess(reference_rows)
    np.testing.assert_allclose([r["diameter_smoothed_px"] for r in output["samples"]],
                               [r["diameter_smoothed_px"] for r in reference], atol=1e-10)
    assert output["metadata"]["interpolated_frames"] == 2
    assert output["metadata"]["observed_frames"] == len(times)-2
    metrics = response_metrics(pairs[0], output, event())
    assert 8 < metrics["mca_px"] < 11
    assert metrics["mcv_px_s"] > 10
    assert "short_gap_interpolated" in metrics["quality_flags"]
    assert not raw["valid"][72]  # processing never rewrites detector evidence


def test_long_gaps_and_declared_blinks_are_not_filled():
    raw, times = trace()
    raw["valid"][95:103] = False
    raw["visibility"][70] = "absent"
    processed, _ = extract_eye(raw, times, [], "test", "left")
    assert all(r["diameter_smoothed_px"] is None for r in processed["samples"][95:103])
    assert processed["samples"][70]["blink_masked"] == 1
    assert processed["samples"][70]["is_interpolated"] == 0


def test_irregular_sensor_times_get_auditable_grid_without_double_counting():
    raw, times = trace()
    times[120:] += 2/30
    processed, pairs = extract_eye(raw, times, [event()], "test", "left")
    meta = processed["metadata"]
    assert meta["alignment"]["resampled"]
    assert meta["alignment"]["empty_grid_frames"] == 2
    frames = [r["source_frame"] for r in processed["samples"] if r["source_frame"] is not None]
    assert len(frames) == len(set(frames)) == len(times)
    assert meta["observed_frames"] == len(times)
    assert "sensor_times_resampled" in pairs[0]["mca"]["quality_flags"]
    assert processed["samples"][120]["is_observed"] == 0


def test_dual_led_flash_is_analyzed_and_flagged_without_false_previous_overlap():
    raw, times = trace()
    processed, pairs = extract_eye(raw, times, [event(1), event(2)], "test", "left")
    for pair in pairs:
        m = response_metrics(pair, processed, event())
        assert m["status"] == "ok" and m["mca_px"] > 8
        assert "bilateral_stimulation_not_isolated_consensual" in m["quality_flags"]
        assert "incomplete_or_invalid_stimulus_interval" not in m["quality_flags"]


def test_nonresponse_never_fabricates_zero_velocity():
    raw, times = trace()
    raw["diameter_px"][:] = 50
    processed, pairs = extract_eye(raw, times, [event()], "test", "left")
    m = response_metrics(pairs[0], processed, event())
    assert m["mca_status"] == "no_clear_constriction"
    assert m["mcv_status"] == "no_clear_constriction" and m["mcv_px_s"] is None
    json.dumps(serializable(m), allow_nan=False)


def test_insufficient_or_invalid_timestamps_keep_explicit_processing_error():
    raw, times = trace()
    times[3] = times[2]
    processed, pairs = extract_eye(raw, times, [event()], "test", "left")
    assert processed["status"] == "error" and not pairs
    m = response_metrics(None, processed, event())
    assert m["status"] == "error" and m["mca_px"] is None


@pytest.mark.parametrize("flashes", [[], [event(1), event(2)]])
def test_saved_results_preserve_raw_shape_and_export_processed_plots(tmp_path, flashes):
    raw, times = trace()
    with (tmp_path / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "sensor_timestamp_ns", "time_s"])
        for i, t in enumerate(times): writer.writerow([i, 10_000_000_000+round(t*1e9), t])
    (tmp_path / "events.json").write_text(json.dumps({"events": flashes, "warnings": []}))
    out = build_results(tmp_path, "test", {"left": raw, "right": copy.deepcopy(raw)}, "offline",
                        copy.deepcopy(DEFAULTS), suffix="_offline")
    assert out["status"] == "ok"
    p = json.loads((tmp_path / "pupilogram_offline.json").read_text())
    assert p["schema_version"] == 4
    for eye in ("left", "right"):
        assert set(p["eyes"][eye]["samples"][0]) == {"frame", "time_s", "diameter_px", "confidence", "valid"}
        assert p["eyes"][eye]["processed"]["metadata"]["observed_frames"] == len(times)
        for artifact in ("smoothed.csv", "smoothed.json", "mca.csv", "mcv.csv", "pupillogram.png"):
            assert (tmp_path / "extraction_offline" / f"{eye}_{artifact}").is_file()
    metrics = json.loads((tmp_path / "metrics_offline.json").read_text())
    assert len(metrics["events"]) == len(flashes)
    if flashes:
        assert metrics["events"][0]["direct"]["selection"]["mcv"]["mcv_peak_frame"] is not None
    import server
    server.copy_results(tmp_path, "_offline")
    assert (tmp_path / "extraction/left_pupillogram.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_artifact_api_uses_only_known_eyes_and_file_types(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    folder = tmp_path / "demo" / "extraction"
    folder.mkdir(parents=True)
    (folder / "left_mca.csv").write_text("mca_px\n10\n")
    client = server.app.test_client()
    assert client.get("/sessions/demo/extraction/left/mca.csv").status_code == 200
    assert client.get("/sessions/demo/extraction/left/mca.csv?source=../../").status_code == 404
    assert client.get("/sessions/demo/extraction/other/mca.csv").status_code == 404
    assert client.get("/sessions/demo/extraction/left/settings.json").status_code == 404
