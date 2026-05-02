# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

import unittest

import pandas as pd

import sparse_longitudinal_engine as sle


class SparseLongitudinalEngineTests(unittest.TestCase):
    def test_uses_all_sparse_points_and_predicts_future_interval(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"] * 4,
                "report_uuid": [f"r{i}" for i in range(4)],
                "癌胚抗原 CEA": [2.8, 3.2, 4.1, 5.6],
            },
            index=pd.to_datetime(["2025-01-01", "2025-04-02", "2025-07-01", "2025-10-02"]),
        )

        report = sle.analyze_sparse_trajectory(df, horizon_days=(90,))
        cea = report["indicators"]["癌胚抗原 CEA"]

        self.assertEqual(cea["data_quality"]["sample_count"], 4)
        self.assertTrue(cea["data_quality"]["uses_all_points"])
        self.assertEqual(len(cea["future_predictions"]), 1)
        prediction = cea["future_predictions"][0]
        self.assertLessEqual(prediction["lower_95"], prediction["median"])
        self.assertGreaterEqual(prediction["upper_95"], prediction["median"])

    def test_small_tumor_marker_change_below_rcv_is_not_reliable(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"] * 4,
                "report_uuid": [f"r{i}" for i in range(4)],
                "糖类抗原 19-9": [20.0, 21.0, 20.5, 22.0],
            },
            index=pd.date_range("2025-01-01", periods=4, freq="90D"),
        )

        result = sle.analyze_indicator(df, "糖类抗原 19-9")

        self.assertFalse(result["current_reliable_change"]["is_reliable"])
        self.assertIn(result["classification"], {"未见可靠变化", "可观察预测"})
        self.assertLess(
            max(p["prob_reliable_change"] for p in result["future_predictions"]),
            0.70,
        )

    def test_sustained_large_change_can_be_flagged_for_review(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"] * 5,
                "report_uuid": [f"r{i}" for i in range(5)],
                "糖类抗原 125": [18.0, 23.0, 31.0, 46.0, 68.0],
            },
            index=pd.date_range("2025-01-01", periods=5, freq="75D"),
        )

        result = sle.analyze_indicator(df, "糖类抗原 125")

        self.assertEqual(result["classification"], "需要复核的数据变化")
        self.assertGreaterEqual(result["priority_score"], 30.0)

    def test_one_point_is_reported_as_insufficient_without_failure(self):
        df = pd.DataFrame(
            {
                "phase": ["稳定监控期"],
                "report_uuid": ["r0"],
                "癌胚抗原 CEA": [3.0],
            },
            index=pd.to_datetime(["2025-01-01"]),
        )

        result = sle.analyze_indicator(df, "癌胚抗原 CEA")

        self.assertEqual(result["status"], "insufficient_points")
        self.assertEqual(result["classification"], "不足以判断")


if __name__ == "__main__":
    unittest.main()
