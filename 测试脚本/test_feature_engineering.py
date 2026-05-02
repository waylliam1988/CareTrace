# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

import unittest

import pandas as pd

import feature_engineering


class FeatureEngineeringTests(unittest.TestCase):
    def test_phase_boundary_rate_is_reset_and_transition_shock_is_kept(self):
        df = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
                "phase": ["稳定监控期", "稳定监控期", "强效治疗期"],
                "report_uuid": ["a", "b", "c"],
                "癌胚抗原 CEA": [2.0, 4.0, 10.0],
                "白细胞计数": [5.0, 5.2, 5.4],
            }
        ).set_index("report_date")

        result = feature_engineering.add_change_rate_features(
            df,
            {"癌胚抗原 CEA": (0.0, 5.0), "白细胞计数": (4.0, 10.0)},
        )

        boundary_date = pd.Timestamp("2026-01-15")
        self.assertIn("癌胚抗原 CEA_rate", result.columns)
        self.assertIn("癌胚抗原 CEA_transition_shock", result.columns)
        self.assertEqual(float(result.loc[boundary_date, "癌胚抗原 CEA_rate"]), 0.0)
        self.assertGreater(float(result.loc[boundary_date, "癌胚抗原 CEA_transition_shock"]), 0.0)

    def test_missing_original_values_are_preserved(self):
        df = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2026-01-01", "2026-01-08"]),
                "phase": ["稳定监控期", "稳定监控期"],
                "report_uuid": ["a", "b"],
                "癌胚抗原 CEA": [2.0, None],
            }
        ).set_index("report_date")

        result = feature_engineering.add_change_rate_features(df, {"癌胚抗原 CEA": (0.0, 5.0)})
        self.assertTrue(pd.isna(result.iloc[1]["癌胚抗原 CEA"]))
        self.assertFalse(pd.isna(result.iloc[1]["癌胚抗原 CEA_rate"]))


if __name__ == "__main__":
    unittest.main()
