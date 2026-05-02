import unittest

import ocr_importer


class OCRImporterTests(unittest.TestCase):
    def test_parse_blood_routine_screenshot_text(self):
        parsed = ocr_importer.parse_report_text_lines(
            [
                "申请时间 2026-04-16 14:50:54",
                "采样时间 2026-04-17 08:06:53",
                "报告时间2026-04-1708:33:12",
                "项目 结果 参考区间 单位",
                "★白细胞计数 7.12 3.5~9.5 10^9/L",
                "中性粒细胞% 70.90 40.00~75.00 %",
                "淋巴细胞绝对数 1.57 1.10~3.20 10^9/L",
                "★血小板计数 469 ↑ 125~350 10^9/L",
                "C-反应蛋白 51.52 ↑ 0~8.00 mg/L",
            ],
            source_name="blood.png",
        )

        values = {
            row["指标名称"]: row["检测值"]
            for _, row in parsed["items"].iterrows()
        }

        self.assertEqual(parsed["report_date"], "2026-04-17")
        self.assertEqual(parsed["date_source"], "采样时间")
        self.assertEqual(values["白细胞计数"], 7.12)
        self.assertEqual(values["血小板计数"], 469.0)
        self.assertEqual(values["C反应蛋白 CRP"], 51.52)

    def test_parse_biochemistry_and_tumor_marker_text(self):
        parsed = ocr_importer.parse_report_text_lines(
            [
                "报告时间 2026-04-17 11:25:56",
                "★总胆红素[TBIL] 17.3 0.0~23.0 umol/L",
                "★总胆固醇[CHOL] 7.51 ↑ ≤5.18 mmol/L",
                "★高密度脂蛋白胆固醇 1.68 ↑ 1.29~1.55 mmol/L",
                "★低密度脂蛋白胆固醇 5.43 ↑ ≤3.37 mmol/L",
                "载脂蛋白B[APOB] 1.54 ↑ 0.60~1.10 g/L",
                "碳酸氢盐[HCO3] 21.8 ↓ 22.0~29.0 mmol/L",
                "★天冬氨酸氨基转移酶23 13~35 IU/L",
                "[] 134.3↓ 137.0~147.0 mmol/L",
                "非高密度脂蛋白胆固醇5.83 mmol/L",
                "胃泌素释放肽前体 60.66 <68.3 pg/mL",
                "糖类抗原125 24.05 <35 U/mL",
                "糖类抗原19-9 40.73 ↑ <34 U/mL",
                "细胞角蛋白19片段 3.54 ↑ <3.30 ng/ml",
                "神经元特异性烯醇化酶 17.07 ↑ <16.30 ng/ml",
            ]
        )

        values = {
            row["指标名称"]: row["检测值"]
            for _, row in parsed["items"].iterrows()
        }

        self.assertEqual(parsed["report_date"], "2026-04-17")
        self.assertEqual(values["总胆红素 TBiL"], 17.3)
        self.assertEqual(values["总胆固醇 TC"], 7.51)
        self.assertEqual(values["载脂蛋白B APOB"], 1.54)
        self.assertEqual(values["碳酸氢盐 HCO3"], 21.8)
        self.assertEqual(values["谷草转氨酶 AST"], 23.0)
        self.assertEqual(values["钠 NA"], 134.3)
        self.assertEqual(values["非高密度脂蛋白胆固醇"], 5.83)
        self.assertEqual(values["胃泌素释放肽前体 ProGRP"], 60.66)
        self.assertEqual(values["糖类抗原 125"], 24.05)
        self.assertEqual(values["糖类抗原 19-9"], 40.73)
        self.assertEqual(values["细胞角蛋白19片段 CYFRA21-1"], 3.54)

    def test_parse_thyroid_text(self):
        parsed = ocr_importer.parse_report_text_lines(
            [
                "申请时间 2026-03-26 14:12:21",
                "采样时间 2026-04-17 09:10:12",
                "★促甲状腺激素(TSH) 5.052 0.56~5.91 mIU/L",
                "★游离T3（FT3） 3.45 ↓ 3.53~7.37 pmol/L",
                "★游离T4（FT4） 10.59 7.98~16.02 pmol/L",
            ]
        )

        values = {
            row["指标名称"]: row["检测值"]
            for _, row in parsed["items"].iterrows()
        }

        self.assertEqual(parsed["report_date"], "2026-04-17")
        self.assertEqual(values["促甲状腺激素"], 5.052)
        self.assertEqual(values["游离三碘甲状腺原氨酸"], 3.45)
        self.assertEqual(values["游离甲状腺素"], 10.59)


if __name__ == "__main__":
    unittest.main()
