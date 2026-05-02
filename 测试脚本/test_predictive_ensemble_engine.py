import unittest

import pandas as pd

import predictive_ensemble_engine as pee


class PredictiveEnsembleEngineTests(unittest.TestCase):
    def test_activity_proxy_uses_multiple_indicators_without_dropping_dates(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"] * 5,
                "report_uuid": [f"r{i}" for i in range(5)],
                "癌胚抗原 CEA": [2.5, 2.7, 3.0, 4.1, 5.8],
                "C反应蛋白 CRP": [2.0, 2.5, 3.0, 6.0, 11.0],
                "白蛋白 ALB": [44.0, 43.2, 42.5, 41.0, 39.0],
                "血红蛋白": [132.0, 130.0, 128.0, 124.0, 119.0],
            },
            index=pd.date_range("2025-01-01", periods=5, freq="90D"),
        )

        activity = pee.estimate_activity_proxy(df, horizon_days=(90,))

        self.assertEqual(activity["status"], "ok")
        self.assertEqual(activity["observations_used"], 5)
        self.assertIn("癌胚抗原 CEA", activity["indicators_used"])
        self.assertEqual(len(activity["future"]), 1)

    def test_conformal_calibration_produces_walk_forward_residuals(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"] * 8,
                "report_uuid": [f"r{i}" for i in range(8)],
                "癌胚抗原 CEA": [2.8, 2.9, 3.0, 3.2, 3.7, 4.5, 5.5, 6.8],
            },
            index=pd.date_range("2025-01-01", periods=8, freq="60D"),
        )

        calibration = pee.calibrate_indicator_intervals(df, "癌胚抗原 CEA")

        self.assertEqual(calibration["status"], "ok")
        self.assertGreater(calibration["residual_count"], 0)
        self.assertIsNotNone(calibration["absolute_residual_quantile"])

    def test_ensemble_report_contains_activity_and_calibrated_interval(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"] * 6,
                "report_uuid": [f"r{i}" for i in range(6)],
                "癌胚抗原 CEA": [2.8, 3.0, 3.3, 4.1, 5.4, 7.0],
                "C反应蛋白 CRP": [2.0, 2.1, 2.5, 4.0, 8.0, 12.0],
                "白蛋白 ALB": [44.0, 43.0, 42.0, 41.0, 40.0, 38.5],
            },
            index=pd.date_range("2025-01-01", periods=6, freq="75D"),
        )

        report = pee.analyze_predictive_ensemble(df, horizon_days=(75,))
        cea = report["indicators"]["癌胚抗原 CEA"]
        future = cea["future_predictions"][0]

        self.assertEqual(report["status"], "ok")
        self.assertEqual(cea["method"], "ensemble_state_space_conformal")
        self.assertIn("ensemble_prob_reliable_or_outside", future)
        self.assertIn("conformal_lower_95", future)
        self.assertIn("activity_proxy", report)


if __name__ == "__main__":
    unittest.main()
