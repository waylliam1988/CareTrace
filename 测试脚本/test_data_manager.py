import unittest

import pandas as pd

from _test_utils import isolated_config, make_items


class DataManagerTests(unittest.TestCase):
    def test_patient_and_lab_report_round_trip(self):
        import data_manager

        with isolated_config():
            self.assertTrue(data_manager.add_patient("测试患者"))
            patients = data_manager.get_patients()
            self.assertEqual(len(patients), 1)
            patient_id = int(patients.iloc[0]["id"])

            data_manager.save_lab_report(
                patient_id,
                "2026-01-01",
                "稳定监控期",
                make_items({"癌胚抗原 CEA": 3.2, "白细胞计数": 5.1}),
            )

            loaded = data_manager.load_patient_data(patient_id)
            self.assertEqual(len(loaded), 1)
            self.assertIn("癌胚抗原 CEA", loaded.columns)
            self.assertAlmostEqual(float(loaded.iloc[0]["癌胚抗原 CEA"]), 3.2)

    def test_save_or_merge_lab_report_overwrites_same_day_item(self):
        import data_manager

        with isolated_config():
            data_manager.add_patient("测试患者")
            patient_id = int(data_manager.get_patients().iloc[0]["id"])

            data_manager.save_or_merge_lab_report(
                patient_id,
                "2026-01-01",
                "稳定监控期",
                make_items({"癌胚抗原 CEA": 3.2}),
            )
            data_manager.save_or_merge_lab_report(
                patient_id,
                "2026-01-01",
                "稳定监控期",
                make_items({"癌胚抗原 CEA": 4.4, "白细胞计数": 6.0}),
            )

            loaded = data_manager.load_patient_data(patient_id)
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded.iloc[0]["癌胚抗原 CEA"]), 4.4)
            self.assertAlmostEqual(float(loaded.iloc[0]["白细胞计数"]), 6.0)

    def test_reference_ranges_fall_back_to_config(self):
        import data_manager

        with isolated_config():
            refs = data_manager.load_references()
            self.assertIn("癌胚抗原 CEA", refs.index)
            self.assertIn("upper_bound", refs.columns)


if __name__ == "__main__":
    unittest.main()
