import unittest

import numpy as np

import config
import simulation_engine
from _test_utils import make_patient_frame


class SimulationEngineTests(unittest.TestCase):
    def test_treatment_schedule_uses_configured_dose_intensity(self):
        zero_dose = "0% 用药输入假设（仅模拟）"
        treatment_func = simulation_engine.build_treatment_schedule(
            [(7, "100% 标准剂量（说明书剂量）"), (7, zero_dose)]
        )

        self.assertEqual(treatment_func(0), config.DOSE_PRESETS["100% 标准剂量（说明书剂量）"])
        self.assertEqual(treatment_func(8), 0.0)
        self.assertEqual(treatment_func(20), 0.0)

    def test_norton_simon_simulation_returns_finite_trajectory(self):
        history = make_patient_frame([4.0, 4.2, 4.4, 4.6, 4.8])
        history["白蛋白 ALB"] = [42, 41, 40, 39, 39]
        history["C反应蛋白 CRP"] = [3, 4, 5, 8, 9]
        result = simulation_engine.run_adaptive_simulation(
            model_name="Norton-Simon (Gompertzian)",
            treatment_schedule=[(14, "100% 标准剂量（说明书剂量）")],
            initial_marker_value=4.8,
            patient_history=history,
            simulation_days=14,
            selected_marker="癌胚抗原 CEA",
        )

        self.assertTrue(result["success"])
        self.assertTrue(np.isfinite(result["total_burden"]).all())
        self.assertGreater(len(result["time"]), 0)
        self.assertIn("host_context", result)
        self.assertTrue(result["uncertainty"]["available"])
        self.assertEqual(len(result["uncertainty"]["p10"]), len(result["time"]))

    def test_host_context_derives_routine_blood_indices(self):
        history = make_patient_frame([4.0, 4.1, 4.2])
        history["白蛋白 ALB"] = [42.0, 40.0, 38.0]
        history["C反应蛋白 CRP"] = [2.0, 12.0, 15.0]

        context = simulation_engine.build_host_context(history)

        self.assertTrue(context["available"])
        self.assertAlmostEqual(context["indices"]["NLR"], history["中性粒细胞绝对数"].iloc[-1] / history["淋巴细胞绝对数"].iloc[-1])
        self.assertIn(context["indices"]["mGPS"], [0, 1, 2])
        self.assertGreaterEqual(context["scores"]["data_confidence"], 0.0)
        self.assertLessEqual(context["modifiers"]["trajectory_uncertainty"], 0.55)

    def test_evidence_annotations_are_exposed(self):
        annotations = simulation_engine.get_evidence_annotations()

        self.assertGreaterEqual(len(annotations), 4)
        self.assertTrue(all("doi" in note and "model_use" in note for note in annotations))


if __name__ == "__main__":
    unittest.main()
