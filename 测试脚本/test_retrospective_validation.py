import unittest

import retrospective_validation as rv


class RetrospectiveValidationTests(unittest.TestCase):
    def test_validation_report_contains_required_metrics(self):
        report = rv.run_validation()

        self.assertIn("alert_detection", report)
        self.assertIn("prediction_interval", report)
        self.assertIn("sample_size_stability", report)
        self.assertIn("sparse_longitudinal_engine", report)
        self.assertIn("predictive_ensemble_engine", report)

        alert = report["alert_detection"]
        self.assertIn("false_positive_rate", alert)
        self.assertIn("false_negative_rate", alert)
        self.assertGreaterEqual(alert["false_positive_rate"], 0.0)
        self.assertLessEqual(alert["false_positive_rate"], 1.0)

        interval = report["prediction_interval"]
        self.assertIn("prediction_interval_coverage", interval)
        self.assertGreaterEqual(interval["prediction_interval_coverage"], 0.0)
        self.assertLessEqual(interval["prediction_interval_coverage"], 1.0)

        sparse = report["sparse_longitudinal_engine"]["one_step_prediction"]
        self.assertIn("false_positive_rate", sparse)
        self.assertIn("false_negative_rate", sparse)
        self.assertIn("prediction_interval_coverage", sparse)
        self.assertIn("brier_score", sparse)

        ensemble = report["predictive_ensemble_engine"]["one_step_prediction"]
        self.assertIn("brier_score", ensemble)
        self.assertIn("calibration_curve", ensemble)
        self.assertIn("false_alarms_per_year", ensemble)
        self.assertIn("sensitivity_at_fixed_fpr", ensemble)

    def test_sample_size_stability_has_multiple_sizes(self):
        stability = rv.evaluate_sample_size_stability(sample_sizes=(8, 12), seeds=range(2))
        self.assertEqual(set(stability.keys()), {"8", "12"})
        self.assertIn("false_positive_rate_mean", stability["8"])
        self.assertIn("prediction_interval_coverage_std", stability["12"])

    def test_sparse_engine_stability_has_quarterly_sample_sizes(self):
        stability = rv.evaluate_sparse_engine_sample_size_stability(sample_sizes=(4, 8), seeds=range(2))
        self.assertEqual(set(stability.keys()), {"4", "8"})
        self.assertIn("samples_evaluated_mean", stability["4"])
        self.assertIn("mean_interval_width_mean", stability["8"])

    def test_ensemble_engine_stability_has_task_metrics(self):
        stability = rv.evaluate_ensemble_engine_sample_size_stability(sample_sizes=(4, 8), seeds=range(2))
        self.assertEqual(set(stability.keys()), {"4", "8"})
        self.assertIn("brier_score_mean", stability["4"])
        self.assertIn("sensitivity_at_fixed_fpr_mean", stability["8"])


if __name__ == "__main__":
    unittest.main()
