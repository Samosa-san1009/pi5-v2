"""Numerical regression checks for artifact handling and metric definitions."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from plr.analysis.extraction.common import schedule
from plr.analysis.extraction.smoothen import discover, preprocess
from plr.analysis.extraction.extract_mca import compute_mca
from plr.analysis.extraction.extract_mcv import compute_mcv


def raw_rows(values):
    return [{"Frame": str(i), "timestamp_ms": str(i*1000/30), "diameter_px": str(value),
             "Valid": "1", "visibility_state": "observed", "result_source": "validated_purest"}
            for i, value in enumerate(values)]


def signal(values):
    n = len(values)
    return {"Frame": np.arange(n), "time_sec": np.arange(n)/30,
            "diameter_smoothed_px": np.asarray(values, dtype=float),
            "is_observed": np.ones(n), "is_inferred": np.zeros(n),
            "is_interpolated": np.zeros(n), "blink_masked": np.zeros(n),
            "filter_edge": np.zeros(n), "segment_id": np.zeros(n)}


META = {"participant_id": "TEST", "eye": "left", "fps": 30}


class PreprocessingTests(unittest.TestCase):
    def test_positive_rejected_diameter_is_not_a_measurement(self):
        rows = raw_rows(np.full(120, 40.))
        rows[50].update(Valid="0", diameter_px="1000")
        output, meta = preprocess(rows)
        self.assertTrue(np.isnan(output[50]["diameter_valid_px"]))
        self.assertEqual(output[50]["is_observed"], 0)
        self.assertEqual(output[50]["is_interpolated"], 1)
        self.assertAlmostEqual(output[50]["diameter_smoothed_px"], 40, places=8)

    def test_linear_interpolation_advances_across_gap(self):
        rows = raw_rows(np.full(120, 40.))
        rows[53]["diameter_px"] = "34"
        for i in (51, 52):
            rows[i]["Valid"] = "0"
        output, _ = preprocess(rows)
        self.assertAlmostEqual(output[51]["diameter_interpolated_px"], 38)
        self.assertAlmostEqual(output[52]["diameter_interpolated_px"], 36)

    def test_long_gap_remains_missing_and_segments_do_not_mix(self):
        rows = raw_rows(np.r_[np.full(60, 40.), np.full(60, 20.)])
        for i in range(55, 65):
            rows[i]["Valid"] = "0"
        output, _ = preprocess(rows)
        self.assertTrue(all(np.isnan(r["diameter_smoothed_px"]) for r in output[55:65]))
        self.assertAlmostEqual(output[50]["diameter_smoothed_px"], 40, places=8)
        self.assertAlmostEqual(output[70]["diameter_smoothed_px"], 20, places=8)

    def test_declared_blink_is_padded_and_not_reconstructed(self):
        rows = raw_rows(np.full(120, 40.))
        rows[60]["visibility_state"] = "absent"
        output, _ = preprocess(rows)
        self.assertEqual(output[58]["blink_masked"], 1)
        self.assertEqual(output[63]["blink_masked"], 1)
        self.assertTrue(np.isnan(output[60]["diameter_smoothed_px"]))

    def test_sampling_alignment_is_enforced(self):
        rows = raw_rows(np.full(120, 40.))
        rows[5]["Frame"] = "6"
        with self.assertRaises(ValueError):
            preprocess(rows)

    def test_optical_flow_does_not_count_as_observed(self):
        rows = raw_rows(np.full(120, 40.))
        rows[50]["result_source"] = "optical_flow_short_gap"
        output, _ = preprocess(rows)
        self.assertEqual(output[50]["is_observed"], 0)
        self.assertEqual(output[50]["is_inferred"], 1)


class MetricsTests(unittest.TestCase):
    def test_local_baseline_mca_and_constriction_only_velocity(self):
        # Baseline 50, true drop 10 over 15 frames, rapid later dilation to 70.
        values = np.full(180, 50.)
        values[67:83] = np.linspace(50, 40, 16)
        values[83:88] = np.linspace(42, 70, 5)
        values[88:] = 70
        data = signal(values)
        rows = compute_mca(data, META, [(1, "Red", 60, 170)])
        self.assertAlmostEqual(rows[0]["mca_px"], 10)
        self.assertEqual(rows[0]["minimum_frame"], 82)
        mcv = compute_mcv(data, META, rows)[0]
        self.assertAlmostEqual(mcv["mcv_px_per_sec"], 20, places=7)
        self.assertAlmostEqual(mcv["mcv_averaged_px_per_sec"], 20, places=7)
        self.assertLessEqual(mcv["averaged_end_frame"], 82)
        self.assertGreater(mcv["mcv_peak_frame"], 60)

    def test_monotonic_response_does_not_require_sign_changes(self):
        values = np.r_[np.full(60, 50.), np.linspace(50, 40, 100), np.full(20, 40.)]
        row = compute_mca(signal(values), META, [(1, "Red", 60, 175)])[0]
        self.assertAlmostEqual(row["mca_px"], 10)
        self.assertNotEqual(row["status"], "unmeasurable")
        self.assertIn("minimum_at_response_boundary", row["quality_flags"])

    def test_pre_stimulus_minimum_is_excluded(self):
        values = np.full(180, 50.)
        values[30] = 5
        row = compute_mca(signal(values), META, [(1, "Red", 60, 170)])[0]
        self.assertAlmostEqual(row["mca_px"], 0)
        self.assertEqual(row["status"], "no_clear_constriction")

    def test_missing_response_is_unmeasurable_instead_of_zero(self):
        data = signal(np.r_[np.full(60, 50.), np.linspace(50, 40, 100), np.full(20, 40.)])
        data["diameter_smoothed_px"][90:100] = np.nan
        data["is_observed"][90:100] = 0
        row = compute_mca(data, META, [(1, "Red", 60, 170)])[0]
        self.assertEqual(row["status"], "unmeasurable")
        self.assertTrue(np.isnan(row["mca_px"]))

    def test_no_response_has_no_fabricated_mcv(self):
        data = signal(np.full(180, 50.))
        mca = compute_mca(data, META, [(1, "No Light", 60, 170)])
        row = compute_mcv(data, META, mca)[0]
        self.assertEqual(row["status"], "no_clear_constriction")
        self.assertTrue(np.isnan(row["mcv_px_per_sec"]))

    def test_bad_timestamp_pair_does_not_fabricate_a_response(self):
        data = signal(np.full(300, 50.))
        rows = compute_mca(data, META, [(1, "Red", 60, 100), (2, "Green", 90, 170),
                                       (3, "Blue", 180, -1)])
        self.assertEqual(rows[0]["status"], "no_clear_constriction")
        self.assertEqual(rows[1]["status"], "unmeasurable")
        self.assertEqual(rows[2]["status"], "unmeasurable")

    def test_eight_onsets_and_offsets_and_duplicate_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table = root/"timestamps.csv"
            with table.open("w", newline="") as stream:
                csv.writer(stream).writerow(["TEST", *range(10, 170, 10)])
            rows = schedule("TEST", table)
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0], (1, "Red", 10, 20))
            self.assertEqual(rows[-1], (8, "Indigo", 150, 160))
            for name in ("TEST_left", "TEST_right", "TEST_left - Copy"):
                folder = root/"dataset_batch"/name
                folder.mkdir(parents=True)
                (folder/"detections.csv").write_text("Frame,Valid\n0,1\n")
            selected, excluded = discover(root)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(excluded), 1)


if __name__ == "__main__":
    unittest.main()
