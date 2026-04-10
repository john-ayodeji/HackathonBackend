"""
Unit tests for engine.py — PDI core clinical logic.

Run with:  python -m pytest test_engine.py -v
       or: python test_engine.py
"""

import math
import unittest

from engine import (
    linear_regression,
    project_vital,
    assess_risk,
    assess_vital,
    calc_pdi_score,
    generate_alert,
    run_full_assessment,
)


class TestLinearRegression(unittest.TestCase):
    """
    Verify the least-squares linear regression helper.

    linear_regression() fits a straight line y = slope*x + intercept to a
    list of evenly-spaced readings and returns the slope, intercept, and R².
    It is the numerical foundation of all trend-projection logic in the PDI
    engine: a rising slope flags a deteriorating vital sign before it crosses
    a clinical threshold.
    """

    def test_perfect_increasing_line(self):
        """Readings that lie exactly on a rising line → R² = 1, correct slope."""
        values = [60.0, 65.0, 70.0, 75.0, 80.0]
        result = linear_regression(values)

        self.assertAlmostEqual(result["slope"], 5.0, places=6)
        self.assertAlmostEqual(result["intercept"], 60.0, places=6)
        self.assertAlmostEqual(result["r2"], 1.0, places=6)

    def test_perfect_decreasing_line(self):
        """Readings on a falling line → negative slope, R² = 1."""
        values = [100.0, 98.0, 96.0, 94.0, 92.0]
        result = linear_regression(values)

        self.assertAlmostEqual(result["slope"], -2.0, places=6)
        self.assertAlmostEqual(result["r2"], 1.0, places=6)

    def test_flat_line(self):
        """Constant readings → slope = 0, R² = 0 (no variance to explain)."""
        values = [37.0, 37.0, 37.0, 37.0]
        result = linear_regression(values)

        self.assertAlmostEqual(result["slope"], 0.0, places=6)
        self.assertAlmostEqual(result["intercept"], 37.0, places=6)
        self.assertAlmostEqual(result["r2"], 0.0, places=6)

    def test_single_reading(self):
        """Only one reading → slope = 0, intercept = that reading, r2 = 0."""
        result = linear_regression([98.6])

        self.assertAlmostEqual(result["slope"], 0.0, places=6)
        self.assertAlmostEqual(result["intercept"], 98.6, places=6)
        self.assertAlmostEqual(result["r2"], 0.0, places=6)

    def test_empty_list(self):
        """Empty list → all zeros (no-op, no exception)."""
        result = linear_regression([])

        self.assertAlmostEqual(result["slope"], 0.0, places=6)
        self.assertAlmostEqual(result["intercept"], 0.0, places=6)
        self.assertAlmostEqual(result["r2"], 0.0, places=6)

    def test_two_readings(self):
        """Minimum non-trivial case: exactly two readings define a unique line."""
        values = [90.0, 100.0]
        result = linear_regression(values)

        self.assertAlmostEqual(result["slope"], 10.0, places=6)
        self.assertAlmostEqual(result["r2"], 1.0, places=6)

    def test_r2_between_zero_and_one(self):
        """Noisy readings → R² is in [0, 1]."""
        values = [60.0, 70.0, 65.0, 80.0, 75.0]
        result = linear_regression(values)

        self.assertGreaterEqual(result["r2"], 0.0)
        self.assertLessEqual(result["r2"], 1.0)

    def test_returns_required_keys(self):
        """Result dict always contains slope, intercept, and r2."""
        result = linear_regression([1.0, 2.0, 3.0])
        for key in ("slope", "intercept", "r2"):
            self.assertIn(key, result)


class TestProjectVital(unittest.TestCase):
    """
    Verify forward projection of a vital sign.

    project_vital() uses the linear trend to estimate where a vital will be
    in N hours.  Readings are assumed to be 2 hours apart (standard ICU
    observation frequency), so the per-index slope is divided by 2 to give a
    per-hour rate of change.
    """

    def test_projection_on_rising_trend(self):
        """
        HR readings: 80, 85, 90, 95 (one reading every 2 h, slope = +5/reading).
        Slope per hour should be +2.5 bpm/h.
        Projected 4 h ahead (2 more index steps) should be 95 + 2*5 = 105.
        """
        readings = [80.0, 85.0, 90.0, 95.0]
        result = project_vital(readings, hours_ahead=4.0)

        self.assertAlmostEqual(result["slope_per_hour"], 2.5, places=1)
        self.assertAlmostEqual(result["projected_value"], 105.0, places=1)
        self.assertAlmostEqual(result["r2"], 1.0, places=3)
        self.assertEqual(result["hours_ahead"], 4.0)

    def test_flat_trend_no_change(self):
        """Stable vital sign → projected value equals current value."""
        readings = [37.0, 37.0, 37.0, 37.0]
        result = project_vital(readings, hours_ahead=4.0)

        self.assertAlmostEqual(result["slope_per_hour"], 0.0, places=2)
        self.assertAlmostEqual(result["projected_value"], 37.0, places=1)

    def test_declining_spo2(self):
        """Falling SpO2: should project below current value."""
        readings = [98.0, 97.0, 96.0, 95.0]
        result = project_vital(readings, hours_ahead=4.0)

        self.assertLess(result["projected_value"], readings[-1])
        self.assertLess(result["slope_per_hour"], 0.0)

    def test_single_reading_no_crash(self):
        """Single reading: no trend available, projected = current value."""
        result = project_vital([120.0], hours_ahead=4.0)

        self.assertAlmostEqual(result["projected_value"], 120.0, places=1)
        self.assertAlmostEqual(result["slope_per_hour"], 0.0, places=2)


class TestAssessRisk(unittest.TestCase):
    """Spot-check the threshold-based single-value risk classifier."""

    def test_normal_hr_is_ok(self):
        self.assertEqual(assess_risk("hr", 75.0), "ok")

    def test_tachycardia_is_crit(self):
        self.assertEqual(assess_risk("hr", 115.0), "crit")

    def test_bradycardia_is_crit(self):
        self.assertEqual(assess_risk("hr", 50.0), "crit")

    def test_normal_spo2_is_ok(self):
        self.assertEqual(assess_risk("spo2", 98.0), "ok")

    def test_hypoxaemia_is_crit(self):
        self.assertEqual(assess_risk("spo2", 90.0), "crit")

    def test_normal_temp_is_ok(self):
        self.assertEqual(assess_risk("temp", 37.0), "ok")

    def test_fever_is_crit(self):
        self.assertEqual(assess_risk("temp", 38.5), "crit")


class TestAssessVital(unittest.TestCase):
    """Verify the per-vital full assessment (current + projected risk)."""

    def test_stable_hr_all_ok(self):
        readings = [75.0, 76.0, 75.0, 74.0]
        result = assess_vital("hr", readings)

        self.assertEqual(result["current_risk"], "ok")
        self.assertEqual(result["projected_risk"], "ok")
        self.assertEqual(result["worst_risk"], "ok")

    def test_rising_hr_flags_projected_crit(self):
        """
        HR starting normal but rising steeply: even if current is borderline,
        the projection 4 h ahead should be critical.
        """
        # 90 → 95 → 100 → 105 — already critical at 105+
        readings = [90.0, 95.0, 100.0, 105.0]
        result = assess_vital("hr", readings)

        self.assertEqual(result["worst_risk"], "crit")

    def test_empty_readings_returns_empty_dict(self):
        result = assess_vital("hr", [])
        self.assertEqual(result, {})

    def test_result_contains_expected_keys(self):
        readings = [80.0, 82.0, 84.0]
        result = assess_vital("hr", readings)

        for key in ("vital", "label", "unit", "current_value", "current_risk",
                    "projected_value", "projected_risk", "slope_per_hour", "r2", "worst_risk"):
            self.assertIn(key, result)


class TestCalcPdiScore(unittest.TestCase):
    """Verify the PDI composite score calculation."""

    def test_all_normal_vitals_score_near_zero(self):
        vital_data = {
            "hr":   [75.0, 76.0, 75.0],
            "rr":   [15.0, 15.0, 14.0],
            "spo2": [98.0, 98.0, 99.0],
            "temp": [37.0, 37.0, 36.9],
            "sbp":  [110.0, 112.0, 110.0],
            "dbp":  [70.0, 71.0, 70.0],
        }
        result = calc_pdi_score(vital_data)

        self.assertLessEqual(result["score"], 30)
        self.assertEqual(result["risk_level"], "ok")

    def test_critical_vitals_produce_high_score(self):
        vital_data = {
            "hr":   [110.0, 115.0, 120.0],
            "rr":   [22.0, 24.0, 26.0],
            "spo2": [88.0, 86.0, 84.0],
            "temp": [38.5, 38.8, 39.0],
        }
        result = calc_pdi_score(vital_data)

        self.assertGreaterEqual(result["score"], 30)

    def test_ai_boost_increases_score(self):
        vital_data = {"hr": [75.0, 76.0], "spo2": [98.0, 98.0]}
        base   = calc_pdi_score(vital_data, ai_weight=0.0)["score"]
        boosted = calc_pdi_score(vital_data, ai_weight=1.0)["score"]

        self.assertGreater(boosted, base)

    def test_score_capped_at_100(self):
        vital_data = {
            "hr":   [180.0, 185.0, 190.0],
            "rr":   [40.0, 42.0, 45.0],
            "spo2": [70.0, 68.0, 65.0],
            "temp": [40.0, 40.5, 41.0],
            "sbp":  [50.0, 48.0, 45.0],
            "dbp":  [30.0, 28.0, 25.0],
        }
        result = calc_pdi_score(vital_data, ai_weight=1.0)

        self.assertLessEqual(result["score"], 100)

    def test_missing_vitals_do_not_crash(self):
        """Partial vital data should still produce a valid score."""
        result = calc_pdi_score({"hr": [80.0, 82.0]})

        self.assertIn("score", result)
        self.assertIn("risk_level", result)

    def test_result_keys(self):
        result = calc_pdi_score({"hr": [75.0]})
        for key in ("score", "numeric_score", "ai_boost", "risk_level", "breakdown"):
            self.assertIn(key, result)


class TestGenerateAlert(unittest.TestCase):
    """Verify pre-emptive alert generation logic."""

    _OK_PDI  = {"score": 10, "risk_level": "ok"}
    _WARN_PDI = {"score": 40, "risk_level": "warn"}
    _CRIT_PDI = {"score": 70, "risk_level": "crit"}

    def _make_assessment(self, worst_risk: str) -> dict:
        return {
            "vital": "hr", "label": "Heart Rate", "unit": "bpm",
            "current_value": 80.0, "projected_value": 110.0,
            "current_risk": worst_risk, "projected_risk": worst_risk,
            "slope_per_hour": 2.5, "r2": 0.95, "worst_risk": worst_risk,
        }

    def test_no_alert_when_all_ok(self):
        result = generate_alert([self._make_assessment("ok")], self._OK_PDI)
        self.assertIsNone(result)

    def test_warn_vital_produces_warn_alert(self):
        result = generate_alert([self._make_assessment("warn")], self._OK_PDI)
        self.assertIsNotNone(result)
        self.assertEqual(result["level"], "warn")

    def test_crit_vital_produces_crit_alert(self):
        result = generate_alert([self._make_assessment("crit")], self._WARN_PDI)
        self.assertIsNotNone(result)
        self.assertEqual(result["level"], "crit")

    def test_alert_contains_expected_keys(self):
        result = generate_alert([self._make_assessment("warn")], self._WARN_PDI)
        for key in ("level", "pdi_score", "triggered_by", "actions"):
            self.assertIn(key, result)

    def test_triggered_by_shows_vital_details(self):
        result = generate_alert([self._make_assessment("crit")], self._CRIT_PDI)
        self.assertTrue(len(result["triggered_by"]) > 0)
        entry = result["triggered_by"][0]
        for key in ("vital", "current", "projected", "slope"):
            self.assertIn(key, entry)


class TestRunFullAssessment(unittest.TestCase):
    """Integration-level smoke tests for the top-level assessment entry point."""

    def test_returns_expected_structure(self):
        vital_data = {
            "hr":   [75.0, 76.0, 77.0],
            "spo2": [98.0, 97.0, 98.0],
            "rr":   [14.0, 15.0, 15.0],
            "temp": [37.0, 37.1, 37.0],
            "sbp":  [110.0, 112.0, 111.0],
            "dbp":  [70.0, 71.0, 70.0],
        }
        result = run_full_assessment(vital_data)

        self.assertIn("vitals", result)
        self.assertIn("pdi", result)
        self.assertIn("alert", result)

    def test_empty_vital_data_no_crash(self):
        result = run_full_assessment({})
        self.assertEqual(result["vitals"], [])
        self.assertIn("pdi", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
