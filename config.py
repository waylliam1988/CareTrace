# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

# config.py
import os
import re 

# --- 日志配置 (Logging Configuration) ---
LOG_FILE = "health_monitor.log"  # 日志文件名
LOG_LEVEL = "DEBUG"               # 日志级别 (可选: "DEBUG", "INFO", "WARNING", "ERROR")

# --- 文件与目录路径 (File and Directory Paths) ---

# 存储所有健康数据的SQLite数据库文件名称
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_FILE = os.path.join(BASE_DIR, "health_data.db")
# 存储所有病人训练好的模型文件（.pkl）和权重文件（.json）的目录名称
MODELS_DIR = os.path.join(BASE_DIR, "models")


# --- 特征工程动态阈值参数 ---
# 这些参数控制特征工程中动态阈值的计算逻辑

# 1. 用于动态阈值计算
# 在无法动态计算时，使用的后备缓冲区（
FALLBACK_BUFFER_FACTOR = 1.1
# 计算变异系数(CV)或阈值所需的最小样本数
MIN_SAMPLES_FOR_CV = 3
# CV 对缓冲区的影响乘数。越高，代表对历史波动越“宽容”
DYNAMIC_BUFFER_CV_MULTIPLIER = 2.0
# 在无法动态计算时，使用的后备增长率阈值 (50%)
FALLBACK_GROWTH_THRESHOLD = 0.5

# 2. 用于倍增/半衰期计算
# 避免除以零
EPSILON = 1e-9
# 合理的最小倍增时间（天）
MIN_PLAUSIBLE_DT = 7
# 合理的最小半衰期（天）
MIN_PLAUSIBLE_HT = 7
# 合理的最小恢复时间（天）
MIN_PLAUSIBLE_RT = 7


# --- 模型参数 (Model Parameters) ---

# 训练健康基线模型（Isolation Forest）所需的最少样本数量
# 低于此数量，系统将提示数据不足，无法训练模型。
MIN_SAMPLES_FOR_MODEL = 5

# MOGP 模型训练所需的最少稳定期数据点数
# 低于此值，将直接跳过该指标的趋势预测
MIN_GP_POINTS = 5

# 当检测到趋势突变时，进行 GP 预测所需的最少数据点数
# 如果数据点少于此值且有突变，将跳过预测以保证可靠性
MIN_POINTS_FOR_SPIKE_PREDICT = 8


# 个人基线监测器配置
# 基线偏离检测的Z-Score阈值 (基于修正Z-score)
# 注意：这些是相对固定的临床阈值，但Monitor内部会根据数据波动性微调
BASELINE_Z_SCORE_THRESHOLDS = {
    'high': 3.5,    # 显著偏离 - 对应高风险
    'medium': 2.5,  # 中度偏离 - 对应中风险
    'low': 1.5      # 轻度偏离 - 对应低风险 (提示关注)
}

# 时间衰减的默认半衰期（天），Monitor内部会动态调整
DEFAULT_HALF_LIFE_DAYS = 60.0

# 动态调整半衰期的边界 (天)
HALF_LIFE_BOUNDS = (30.0, 180.0)

# 动态调整Z-score阈值的CV阈值
BASELINE_CV_THRESHOLD_FOR_ADJUSTMENT = 0.3
BASELINE_Z_SCORE_ADJUSTMENT_FACTOR = 1.15 # 波动大时，阈值乘以该因子 (略微放宽)

# MOGP 预测窗口配置
MOGP_PREDICTION_WINDOW = {
    'min_days': 7,          # 最少预测7天
    'max_days': 14,         # 最多预测14天
    'ratio_of_interval': 0.5 # 预测窗口 = 中位采样间隔 * 50%
}

# MOGP 可信度评估阈值
MOGP_CONFIDENCE_THRESHOLDS = {
    'uncertainty_growth_high': 0.5, # 不确定性增长率 > 0.5 -> low confidence
    'error_ratio_high': 1.0       # 历史拟合误差 > 历史标准差 -> medium confidence
}


# --- 特征工程参数 (Feature Engineering Parameters) ---
# 用于识别所有衍生特征的正则表达式，以避免对它们进行重复计算
# 已生成的后缀包括: _rate, _rate_accel, _ewma, _norm_ratio, _norm_pos, _days_since_..., _is_changepoint

DERIVED_FEATURE_PATTERN = re.compile(
    r'_rate$|_rate_accel$|_ewma$|_norm_ratio$|_norm_pos$'
    r'|_days_since_first_high$|_days_since_first_low$|_days_since_cp$'
    r'|_is_changepoint$|_transition_shock$'
)

# ==============================================================================
# 【核心升级 V4.0 - 唯一事实来源 (Single Source of Truth)】
# LAB_REPORT_CONFIG 是系统中所有化验单知识的“唯一事实来源”。
# 结构:
# {
#     "报告单类型 (用于UI显示)": [
#         {
#             "name": "指标标准名",
#             "category": "指标的生物学分类 (用于内部逻辑)",
#             "aliases": ["别名1", "别名2"],
#             "lower": 下限, "upper": 上限, "unit": "单位"
#         },
#         ...
#     ]
# }
# ==============================================================================
LAB_REPORT_CONFIG = {
    "肿瘤标志物": [
        {"name": "癌胚抗原 CEA", "category": "TUMOR_MARKER", "aliases": ["CEA", "癌胚抗原", "a胚"], "lower": 0.0, "upper": 5.0, "unit": "ng/mL", "behavior": "high_is_bad"},
        {"name": "糖类抗原 19-9", "category": "TUMOR_MARKER", "aliases": ["CA199", "CA19-9", "CA 19-9"], "lower": 0.0, "upper": 37.0, "unit": "U/mL", "behavior": "high_is_bad"},
        {"name": "糖类抗原 125", "category": "TUMOR_MARKER", "aliases": ["CA125", "CA 125"], "lower": 0.0, "upper": 35.0, "unit": "U/mL", "behavior": "high_is_bad"},
        {"name": "糖类抗原 15-3", "category": "TUMOR_MARKER", "aliases": ["CA153", "CA 15-3"], "lower": 0.0, "upper": 28.0, "unit": "U/mL", "behavior": "high_is_bad"},
        {"name": "甲胎蛋白 AFP", "category": "TUMOR_MARKER", "aliases": ["AFP", "甲胎"], "lower": 0.0, "upper": 7.0, "unit": "ng/mL", "behavior": "high_is_bad"},
        {"name": "神经元特异性烯醇化酶 NSE", "category": "TUMOR_MARKER", "aliases": ["NSE"], "lower": 0.0, "upper": 16.3, "unit": "ng/mL", "behavior": "high_is_bad"},
        {"name": "胃泌素释放肽前体 ProGRP", "category": "TUMOR_MARKER", "aliases": ["ProGRP", "PROGRP", "胃泌素释放肽前体"], "lower": 0.0, "upper": 68.3, "unit": "pg/mL", "behavior": "high_is_bad"},
        {"name": "细胞角蛋白19片段 CYFRA21-1", "category": "TUMOR_MARKER", "aliases": ["CYFRA21-1", "CYFRA 21-1", "细胞角蛋白"], "lower": 0.0, "upper": 3.3, "unit": "ng/mL", "behavior": "high_is_bad"},
        {"name": "鳞状细胞癌抗原 SCCA", "category": "TUMOR_MARKER", "aliases": ["SCCA", "SCC", "鳞状细胞癌抗原", "鳞状细胞癌相关抗原"], "lower": 0.0, "upper": 1.5, "unit": "ng/mL", "behavior": "high_is_bad"},
        {"name": "前列腺特异性抗原 PSA", "category": "TUMOR_MARKER", "aliases": ["PSA", "tPSA", "fPSA", "前列腺特异性抗原"], "lower": 0.0, "upper": 4.0, "unit": "ng/mL", "behavior": "high_is_bad"},
        {"name": "人绒毛膜促性腺激素 HCG", "category": "TUMOR_MARKER", "aliases": ["HCG", "hCG", "β-HCG", "人绒毛膜促性腺激素"], "lower": 0.0, "upper": 5.0, "unit": "mIU/mL", "behavior": "high_is_bad"},
    ],
    "血常规": [
        {"name": "白细胞计数", "category": "BLOOD_ROUTINE", "aliases": ["WBC", "白细胞"], "lower": 4.0, "upper": 10.0, "unit": "10*9/L", "behavior": "bidirectional"},
        {"name": "中性粒细胞百分比", "category": "BLOOD_ROUTINE", "aliases": ["NEUT%", "中性%"], "lower": 50.0, "upper": 70.0, "unit": "%", "behavior": "bidirectional"},
        {"name": "淋巴细胞百分比", "category": "BLOOD_ROUTINE", "aliases": ["LYMPH%", "淋巴%"], "lower": 20.0, "upper": 40.0, "unit": "%", "behavior": "bidirectional"},
        {"name": "单核细胞百分比", "category": "BLOOD_ROUTINE", "aliases": ["MONO%", "单核%"], "lower": 3.0, "upper": 8.0, "unit": "%", "behavior": "high_is_bad"},
        {"name": "嗜酸性粒细胞百分比", "category": "BLOOD_ROUTINE", "aliases": ["EOS%", "嗜酸%"], "lower": 0.0, "upper": 5.0, "unit": "%", "behavior": "high_is_bad"},
        {"name": "嗜碱性粒细胞百分比", "category": "BLOOD_ROUTINE", "aliases": ["BASO%", "嗜碱%"], "lower": 0.0, "upper": 1.0, "unit": "%", "behavior": "high_is_bad"},
        {"name": "中性粒细胞绝对数", "category": "BLOOD_ROUTINE", "aliases": ["NEUT#", "中性粒细胞", "NE", "NEUT"], "lower": 1.5, "upper": 7.0, "unit": "10*9/L", "behavior": "bidirectional"},
        {"name": "淋巴细胞绝对数", "category": "BLOOD_ROUTINE", "aliases": ["LYMPH#", "淋巴细胞", "LY", "LYMPH"], "lower": 0.8, "upper": 4.0, "unit": "10*9/L", "behavior": "bidirectional"},
        {"name": "单核细胞绝对数", "category": "BLOOD_ROUTINE", "aliases": ["MONO#", "单核细胞", "MONO"], "lower": 0.12, "upper": 0.8, "unit": "10*9/L", "behavior": "high_is_bad"},
        {"name": "嗜酸性粒细胞绝对数", "category": "BLOOD_ROUTINE", "aliases": ["EOS#", "嗜酸细胞", "EOS"], "lower": 0.0, "upper": 0.5, "unit": "10*9/L", "behavior": "high_is_bad"},
        {"name": "嗜碱性粒细胞绝对数", "category": "BLOOD_ROUTINE", "aliases": ["BASO#", "嗜碱细胞", "BASO"], "lower": 0.0, "upper": 0.1, "unit": "10*9/L", "behavior": "high_is_bad"},
        {"name": "红细胞计数", "category": "BLOOD_ROUTINE", "aliases": ["RBC", "红细胞"], "lower": 3.5, "upper": 5.5, "unit": "10*12/L", "behavior": "bidirectional"},
        {"name": "血红蛋白", "category": "BLOOD_ROUTINE", "aliases": ["HGB", "HB", "血色素", "血红蛋白量"], "lower": 110.0, "upper": 150.0, "unit": "g/L", "behavior": "bidirectional"},
        {"name": "红细胞比积", "category": "BLOOD_ROUTINE", "aliases": ["HCT", "红细胞压积"], "lower": 35.0, "upper": 45.0, "unit": "%", "behavior": "bidirectional"},
        {"name": "平均红细胞体积", "category": "BLOOD_ROUTINE", "aliases": ["MCV"], "lower": 80.0, "upper": 100.0, "unit": "fL", "behavior": "bidirectional"},
        {"name": "平均血红蛋白含量", "category": "BLOOD_ROUTINE", "aliases": ["MCH", "平均红细胞血红蛋白含量", "平均红细胞血红蛋白含"], "lower": 27.3, "upper": 34.4, "unit": "pg", "behavior": "bidirectional"},
        {"name": "平均红细胞血红蛋白浓度", "category": "BLOOD_ROUTINE", "aliases": ["MCHC", "平均红细胞血红蛋白浓度", "平均红细胞血红蛋白浓"], "lower": 320.0, "upper": 360.0, "unit": "g/L", "behavior": "bidirectional"},
        {"name": "红细胞体积分布宽度-SD", "category": "BLOOD_ROUTINE", "aliases": ["RDW-SD"], "lower": 37.0, "upper": 54.0, "unit": "fL", "behavior": "high_is_bad"},
        {"name": "红细胞体积分布宽度-CV", "category": "BLOOD_ROUTINE", "aliases": ["RDW-CV"], "lower": 11.0, "upper": 16.0, "unit": "%", "behavior": "high_is_bad"},
        {"name": "血小板计数", "category": "BLOOD_ROUTINE", "aliases": ["PLT", "血小板"], "lower": 100.0, "upper": 300.0, "unit": "10*9/L", "behavior": "bidirectional"},
        {"name": "血小板比容", "category": "BLOOD_ROUTINE", "aliases": ["PCT"], "lower": 0.06, "upper": 0.28, "unit": "%", "behavior": "bidirectional"},
        {"name": "平均血小板体积", "category": "BLOOD_ROUTINE", "aliases": ["MPV"], "lower": 6.4, "upper": 12.1, "unit": "fL", "behavior": "bidirectional"},
        {"name": "血小板体积分布宽度", "category": "BLOOD_ROUTINE", "aliases": ["PDW"], "lower": 9.0, "upper": 17.0, "unit": "%", "behavior": "high_is_bad"},
        {"name": "大血小板比率", "category": "BLOOD_ROUTINE", "aliases": ["P-LCR", "PLCR", "大血小板比率"], "lower": 13.0, "upper": 43.0, "unit": "%", "behavior": "bidirectional"},
        {"name": "有核红细胞绝对计数", "category": "BLOOD_ROUTINE", "aliases": ["NRBC#", "有核红细胞绝对计数"], "lower": 0.0, "upper": 0.02, "unit": "10*9/L", "behavior": "high_is_bad"},
        {"name": "有核红细胞/白细胞", "category": "BLOOD_ROUTINE", "aliases": ["NRBC%", "有核红细胞/白细胞"], "lower": 0.0, "upper": 1.0, "unit": "%", "behavior": "high_is_bad"},
    ],
    "肝肾功能与电解质": [
        {"name": "总胆红素 TBiL", "category": "LIVER_KIDNEY", "aliases": ["TBIL", "总胆", "总胆红素"], "lower": 0.0, "upper": 23.0, "unit": "umol/L", "behavior": "high_is_bad"},
        {"name": "直接胆红素 DBiL", "category": "LIVER_KIDNEY", "aliases": ["DBIL", "直胆", "直接胆红素"], "lower": 0.0, "upper": 8.0, "unit": "umol/L", "behavior": "high_is_bad"},
        {"name": "间接胆红素 IBiL", "category": "LIVER_KIDNEY", "aliases": ["IBIL", "间胆", "间接胆红素"], "lower": 0.0, "upper": 20.0, "unit": "umol/L", "behavior": "high_is_bad"},
        {"name": "总蛋白 TP", "category": "LIVER_KIDNEY", "aliases": ["TP", "总蛋白"], "lower": 65.0, "upper": 85.0, "unit": "g/L", "behavior": "bidirectional"},
        {"name": "白蛋白 ALB", "category": "LIVER_KIDNEY", "aliases": ["ALB", "白蛋", "白蛋白"], "lower": 40.0, "upper": 55.0, "unit": "g/L", "behavior": "low_is_bad"},
        {"name": "球蛋白 GLOB", "category": "LIVER_KIDNEY", "aliases": ["GLOB", "球蛋", "球蛋白"], "lower": 20.0, "upper": 35.0, "unit": "g/L", "behavior": "bidirectional"},
        {"name": "白球比 A/G", "category": "LIVER_KIDNEY", "aliases": ["A/G"], "lower": 1.00, "upper": 2.50, "unit": "", "behavior": "low_is_bad"},
        {"name": "谷丙转氨酶 ALT", "category": "LIVER_KIDNEY", "aliases": ["ALT", "谷丙", "GPT", "丙氨酸氨基转移酶"], "lower": 7.0, "upper": 40.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "谷草转氨酶 AST", "category": "LIVER_KIDNEY", "aliases": ["AST", "谷草", "GOT", "天冬氨酸氨基转移酶"], "lower": 13.0, "upper": 35.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "AST/ALT", "category": "LIVER_KIDNEY", "aliases": [], "lower": 0.0, "upper": 2.0, "unit": "", "behavior": "neutral"},
        {"name": "γ-谷氨酰转移酶 GGT", "category": "LIVER_KIDNEY", "aliases": ["GGT", "γ-GT", "Y-谷氨酰转肽酶", "γ-谷氨酰转肽酶"], "lower": 7.0, "upper": 45.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "碱性磷酸酶 ALP", "category": "LIVER_KIDNEY", "aliases": ["ALP", "AKP", "碱性磷酸酶"], "lower": 50.0, "upper": 135.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "尿素 UREA", "category": "LIVER_KIDNEY", "aliases": ["UREA", "尿素氮"], "lower": 3.1, "upper": 7.4, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "肌酐 CREA", "category": "LIVER_KIDNEY", "aliases": ["CREA", "肌干", "肌酐"], "lower": 41.0, "upper": 81.0, "unit": "umol/L", "behavior": "high_is_bad"},
        {"name": "尿酸 URIC", "category": "LIVER_KIDNEY", "aliases": ["URIC", "UA", "尿酸"], "lower": 130.0, "upper": 430.0, "unit": "umol/L", "behavior": "high_is_bad"},
        {"name": "钾 K", "category": "ELECTROLYTE", "aliases": ["K", "钾"], "lower": 3.50, "upper": 5.50, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "钠 NA", "category": "ELECTROLYTE", "aliases": ["NA", "钠"], "lower": 135.0, "upper": 148.0, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "氯 CL", "category": "ELECTROLYTE", "aliases": ["CL", "氯"], "lower": 96.0, "upper": 112.0, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "碳酸氢盐 HCO3", "category": "ELECTROLYTE", "aliases": ["HCO3", "碳酸氢盐"], "lower": 22.0, "upper": 29.0, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "钙 CA", "category": "ELECTROLYTE", "aliases": ["CA", "钙"], "lower": 2.10, "upper": 2.70, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "镁", "category": "ELECTROLYTE", "aliases": ["Mg"], "lower": 0.75, "upper": 1.25, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "磷", "category": "ELECTROLYTE", "aliases": ["P", "PO4"], "lower": 0.81, "upper": 1.45, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "阴离子间隙 AG", "category": "ELECTROLYTE", "aliases": ["AG"], "lower": 8.0, "upper": 16.0, "unit": "mmol/L", "behavior": "high_is_bad"},
        {"name": "渗透压 OSM", "category": "ELECTROLYTE", "aliases": ["OSM"], "lower": 280.0, "upper": 310.0, "unit": "mOsm/kg", "behavior": "bidirectional"},
    ],
    "血脂、血糖与心肌酶": [
        {"name": "总胆固醇 TC", "category": "LIPID", "aliases": ["TC", "CHOL", "总固醇", "总胆固醇"], "lower": 3.60, "upper": 6.10, "unit": "mmol/L", "behavior": "high_is_bad"},
        {"name": "甘油三酯 TG", "category": "LIPID", "aliases": ["TG"], "lower": 0.90, "upper": 1.90, "unit": "mmol/L", "behavior": "high_is_bad"},
        {"name": "高密度脂蛋白胆固醇 HDL-C", "category": "LIPID", "aliases": ["HDL", "HDL-C", "高密度", "高密度脂蛋白胆固醇"], "lower": 0.90, "upper": 1.90, "unit": "mmol/L", "behavior": "low_is_bad"},
        {"name": "非高密度脂蛋白胆固醇", "category": "LIPID", "aliases": ["non-HDL", "nonHDL", "非高密度脂蛋白胆固醇"], "lower": None, "upper": None, "unit": "mmol/L", "behavior": "high_is_bad"},
        {"name": "低密度脂蛋白胆固醇 LDL-C", "category": "LIPID", "aliases": ["LDL", "LDL-C", "低密度", "低密度脂蛋白胆固醇"], "lower": 1.10, "upper": 3.50, "unit": "mmol/L", "behavior": "high_is_bad"},
        {"name": "载脂蛋白A1 APOA1", "category": "LIPID", "aliases": ["APOA1", "载脂蛋白A1", "载脂蛋白AⅠ"], "lower": 1.00, "upper": 1.60, "unit": "g/L", "behavior": "bidirectional"},
        {"name": "载脂蛋白B APOB", "category": "LIPID", "aliases": ["APOB", "载脂蛋白B"], "lower": 0.60, "upper": 1.10, "unit": "g/L", "behavior": "high_is_bad"},
        {"name": "葡萄糖 GLU", "category": "GLUCOSE", "aliases": ["GLU", "血糖"], "lower": 3.90, "upper": 6.10, "unit": "mmol/L", "behavior": "bidirectional"},
        {"name": "糖化血红蛋白 HbA1c", "category": "GLUCOSE", "aliases": ["HbA1c"], "lower": 4.0, "upper": 6.0, "unit": "%", "behavior": "high_is_bad"},
        {"name": "乳酸脱氢酶 LDH", "category": "ENZYME", "aliases": ["LDH", "乳酸脱氢酶"], "lower": 120.0, "upper": 250.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "肌酸激酶 CK", "category": "ENZYME", "aliases": ["CK", "肌酸激酶"], "lower": 22.0, "upper": 270.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "肌酸激酶同工酶 MB", "category": "ENZYME", "aliases": ["CK-MB", "CKMB", "肌酸激酶MB亚型"], "lower": 2.0, "upper": 25.0, "unit": "IU/L", "behavior": "high_is_bad"},
        {"name": "CKMB/CK", "category": "ENZYME", "aliases": [], "lower": None, "upper": None, "unit": "", "behavior": "neutral"},
    ],
    "炎症、凝血与内分泌": [
        {"name": "C反应蛋白 CRP", "category": "INFLAMMATION", "aliases": ["CRP", "C反应蛋白"], "lower": 0.0, "upper": 8.0, "unit": "mg/L", "behavior": "high_is_bad"},
        {"name": "超敏C反应蛋白", "category": "INFLAMMATION", "aliases": ["hs-CRP", "hsCRP"], "lower": 0.0, "upper": 3.0, "unit": "mg/L", "behavior": "high_is_bad"},
        {"name": "白介素-6", "category": "INFLAMMATION", "aliases": ["IL-6", "白介素6"], "lower": None, "upper": 7.0, "unit": "pg/mL", "behavior": "high_is_bad"},
        {"name": "铁蛋白", "category": "INFLAMMATION", "aliases": ["Ferritin"], "lower": 30.0, "upper": 400.0, "unit": "ng/mL", "behavior": "bidirectional"},
        {"name": "D-二聚体", "category": "COAGULATION", "aliases": ["D-Dimer"], "lower": None, "upper": 0.5, "unit": "mg/L FEU", "behavior": "high_is_bad"},
        {"name": "凝血酶原时间", "category": "COAGULATION", "aliases": ["PT"], "lower": 9.0, "upper": 13.0, "unit": "s", "behavior": "high_is_bad"},
        {"name": "活化部分凝血活酶时间", "category": "COAGULATION", "aliases": ["APTT"], "lower": 25.0, "upper": 35.0, "unit": "s", "behavior": "high_is_bad"},
        {"name": "纤维蛋白原", "category": "COAGULATION", "aliases": ["FIB"], "lower": 2.0, "upper": 4.0, "unit": "g/L", "behavior": "bidirectional"},
        {"name": "促甲状腺激素", "category": "THYROID", "aliases": ["TSH"], "lower": 0.27, "upper": 4.2, "unit": "μIU/mL", "behavior": "bidirectional"},
        {"name": "游离甲状腺素", "category": "THYROID", "aliases": ["FT4"], "lower": 12.0, "upper": 22.0, "unit": "pmol/L", "behavior": "bidirectional"},
        {"name": "游离三碘甲状腺原氨酸", "category": "THYROID", "aliases": ["FT3"], "lower": 3.1, "upper": 6.8, "unit": "pmol/L", "behavior": "bidirectional"},
        {"name": "前白蛋白", "category": "NUTRITION", "aliases": ["PA", "PAB"], "lower": 150, "upper": 400, "unit": "mg/L", "behavior": "low_is_bad"},
        {"name": "维生素D(25-羟)", "category": "VITAMIN", "aliases": ["Vit D", "VD"], "lower": 30.0, "upper": 100.0, "unit": "ng/mL", "behavior": "low_is_bad"},
        {"name": "维生素B12", "category": "VITAMIN", "aliases": ["Vit B12", "VB12"], "lower": 180.0, "upper": 914.0, "unit": "pg/mL", "behavior": "low_is_bad"},
        {"name": "叶酸", "category": "VITAMIN", "aliases": ["Folate"], "lower": 3.1, "upper": 20.5, "unit": "ng/mL", "behavior": "low_is_bad"},
    ],
    "铁代谢": [
        {"name": "血清铁 Iron", "category": "IRON_METABOLISM", "aliases": ["Iron", "血清铁"], "lower": 9.0, "upper": 32.0, "unit": "µmol/L", "behavior": "bidirectional"},
        {"name": "总铁结合力 TIBC", "category": "IRON_METABOLISM", "aliases": ["TIBC", "总铁结合力", "总铁"], "lower": 45.0, "upper": 81.0, "unit": "µmol/L", "behavior": "bidirectional"},
        {"name": "转铁蛋白饱和度 TSAT", "category": "IRON_METABOLISM", "aliases": ["TSAT", "转铁蛋白饱和度", "转铁饱和度"], "lower": 20.0, "upper": 50.0, "unit": "%", "behavior": "bidirectional"},
    ]
}


# ==============================================================================
# 临床综合指标配置（用于 risk_engine 自动生成观察规则）
# ==============================================================================
COMPOSITE_INDICATORS_CONFIG = {
    'NLR': {
        'full_name': '中性粒细胞/淋巴细胞比值',
        'category': 'INFLAMMATION',
        'threshold_high': 5.0,        # 高于此值提示炎症/免疫失衡
        'threshold_rate': 0.5,        # 变化率阈值（每天）
        'clinical_note': '该比值升高可能与炎症、应激、感染等多种因素有关',
        'reference': '文献常见参考值 < 5'
    },
    'PLR': {
        'full_name': '血小板/淋巴细胞比值',
        'category': 'INFLAMMATION',
        'threshold_high': 200.0,
        'threshold_rate': 10.0,
        'clinical_note': '该比值升高可能与凝血、炎症状态相关',
        'reference': '文献常见参考值 < 200'
    },
    'LMR': {
        'full_name': '淋巴细胞/单核细胞比值',
        'category': 'IMMUNE',
        'threshold_low': 2.0,         # 低于此值提示免疫功能下降
        'threshold_rate': 0.3,
        'clinical_note': '该比值降低可能提示免疫调节异常',
        'reference': '文献常见参考值 > 2'
    },
    'SII': {
        'full_name': '全身免疫炎症指数',
        'category': 'INFLAMMATION',
        'threshold_high': 900.0,      # 根据文献调整
        'threshold_rate': 50.0,
        'clinical_note': '该指数综合反映了血小板、中性粒细胞和淋巴细胞的相对关系',
        'reference': '文献常见参考值 < 900'
    },
    'SIRI': {
        'full_name': '全身炎症反应指数',
        'category': 'INFLAMMATION',
        'threshold_high': 2.5,        # 根据文献调整
        'threshold_rate': 0.5,
        'clinical_note': '该指数综合反映了中性粒细胞、单核细胞和淋巴细胞的关系',
        'reference': '文献常见参考值 < 2.5'
    }
}


# MOGP 预测窗口比例策略
MOGP_PREDICT_RATIO = {
    'conservative': 0.5,  # 综合指标：最多预测总时长的50%
    'aggressive': 1.0     # 肿瘤标志物：最多预测总时长的100%（更激进）
}

# MOGP 最小预测窗口（天）
MOGP_MIN_PREDICT_DAYS = 14  # 无论时间跨度多短，至少预测14天


# MOGP 指标选择优先级配置（用于 select_mogp_indicators）
MOGP_INDICATOR_PRIORITIES = {
    "肿瘤标志物": (1, 2),      # (优先级, 最大选择数量)
    "临床综合指标": (2, 2),
    "血常规": (3, 1),
    "肝肾功能与电解质": (4, 1),
}

# 临床综合指标列表（从 COMPOSITE_INDICATORS_CONFIG 自动生成）
COMPOSITE_INDICATORS = list(COMPOSITE_INDICATORS_CONFIG.keys())

# 营养指标配置（用于 _observe_nutrition_patterns）
NUTRITION_INDICATORS_CONFIG = {
    '白蛋白 ALB': {
        'threshold_low': 35.0,
        'rate_threshold': 0.2,
        'clinical_note': '白蛋白水平降低可能受饮食、吸收、肝功能等多种因素影响'
    },
    '总蛋白 TP': {
        'threshold_low': 60.0,
        'rate_threshold': 1.0,
        'clinical_note': '总蛋白水平降低可能与营养不良、蛋白丢失等因素有关'
    },
    '前白蛋白': {
        'threshold_low': 200.0,
        'rate_threshold': 10.0,
        'clinical_note': '前白蛋白是营养状态的敏感指标'
    }
}


# ==============================================================================
# --- 适应性治疗模拟配置 ---
# ==============================================================================

# 治疗剂量预设（用于 UI 和模型输入）
DOSE_PRESETS = {
    "150% 剂量假设": 1.5,
    "100% 标准剂量（说明书剂量）": 1.0,
    "75% 剂量假设": 0.75,
    "50% 剂量假设": 0.5,
    "25% 剂量假设": 0.25,
    "0% 用药输入假设（仅模拟）": 0.0
}

# 机理模型配置（对应 tumor_models.py）
MECHANISTIC_MODELS = {
    "经典竞争模型 (S-R)": {
        "description": "适用于化疗引起的竞争性耐药（如铂类、紫杉醇）",
        "key_features": ["敏感细胞 vs 抵抗细胞", "适应性成本", "表型可塑性"],
        "typical_use": "标准化疗方案"
    },
    "干细胞驱动模型 (B20)": {
        "description": "适用于肿瘤干细胞驱动的复发（如卵巢癌、胶质瘤）",
        "key_features": ["干细胞 vs 分化细胞", "不对称分裂"],
        "typical_use": "靶向治疗 + 化疗组合"
    },
    "Norton-Simon (Gompertzian)": {
        "description": "经典的肿瘤生长模型，适用于快速增殖肿瘤",
        "key_features": ["Gompertz生长", "Log-Kill假说"],
        "typical_use": "剂量密集化疗"
    },
    "三房室模型 (S-P-R)": {
        "description": "描述持留细胞（Persister）介导的耐药",
        "key_features": ["敏感-持留-抵抗三态", "可逆耐药"],
        "typical_use": "靶向治疗（如EGFR抑制剂）"
    },
    "免疫-肿瘤互作模型 (de Pillis 2005)": {
        "description": "考虑免疫系统与肿瘤的相互作用",
        "key_features": ["肿瘤-免疫动态", "免疫激活"],
        "typical_use": "免疫治疗 + 化疗组合"
    }
}

# 模拟默认参数
SIMULATION_DEFAULTS = {
    'simulation_days': 180,  # 默认模拟6个月
    'initial_burden_multiplier': 1.0,  # 初始肿瘤负荷倍数
    'num_scenarios': 3  # 最佳/最差/中位三种场景
}


# ========================================
# DTW 相似度阈值配置
# ========================================

# DTW 匹配的相似度阈值（0-1之间，越高越严格）
DTW_SIMILARITY_THRESHOLD = 0.3  # 30% - 用于判断是否生成观察项

# 高相似度阈值（用于主动学习请求）
DTW_HIGH_SIMILARITY_THRESHOLD = 0.35  # 35% - 触发历史标签请求

# UI 显示阈值（百分比形式，与 DTW_SIMILARITY_THRESHOLD 保持一致）
DTW_UI_DISPLAY_THRESHOLD = int(DTW_SIMILARITY_THRESHOLD * 100)  # 30
