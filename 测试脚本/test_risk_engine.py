import unittest

import pandas as pd

import risk_engine
from _test_utils import make_patient_frame


class RiskEngineTests(unittest.TestCase):
    def test_select_mogp_indicators_prefers_tumor_marker_with_enough_data(self):
        df = make_patient_frame([2.0, 2.1, 2.2, 2.3, 2.4])
        selected, diagnostic = risk_engine.select_mogp_indicators(df, max_indicators=2)

        self.assertIn("癌胚抗原 CEA", selected)
        self.assertEqual(diagnostic["candidates"]["癌胚抗原 CEA"]["count"], 5)

    def test_observe_health_data_patterns_returns_structured_result(self):
        historical = make_patient_frame([2.0, 2.2, 2.3, 2.5, 7.2])
        latest = historical.iloc[-1]
        result = risk_engine.observe_health_data_patterns(
            latest_data=latest,
            historical_data=historical,
            ref_ranges={"癌胚抗原 CEA": (0.0, 5.0)},
            context={
                "current_phase_tag": "稳定监控期",
                "current_report_uuid": "r4",
                "patient_id": 1,
                "all_labels": {},
            },
        )

        self.assertIn("attention_level", result)
        self.assertIn("observations", result)
        self.assertIn("disclaimer", result)


if __name__ == "__main__":
    unittest.main()
