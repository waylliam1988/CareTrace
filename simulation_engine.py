# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

# simulation_engine.py

"""
适应性治疗模拟引擎
基于 tumor_models.py 的机理模型进行前瞻性模拟
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import odeint
import logging

import config
import tumor_models

logger = logging.getLogger(__name__)


# ==============================================================================
# 论文注释与模型边界
# ==============================================================================

EVIDENCE_ANNOTATIONS = [
    {
        "key": "adaptive_therapy_evolution",
        "title": "适应性治疗的演化学基础",
        "citation": "Gatenby RA, Silva AS, Gillies RJ, Frieden BR. Adaptive Therapy. Cancer Research, 2009.",
        "doi": "10.1158/0008-5472.CAN-08-3658",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3728826/",
        "model_use": "支持本系统使用敏感/耐药亚群、竞争释放和适应性成本来做假设轨迹模拟。",
        "caution": "该思想不能从普通化验单直接推出个人给药方案；这里只用于解释模型结构。",
    },
    {
        "key": "preclinical_adaptive_control",
        "title": "通过保留敏感细胞延缓耐药的前临床证据",
        "citation": "Enriquez-Navas PM et al. Science Translational Medicine, 2016.",
        "doi": "10.1126/scitranslmed.aad7842",
        "url": "https://pubmed.ncbi.nlm.nih.gov/26912903/",
        "model_use": "支持在模型中显式保留适应性成本和敏感/耐药竞争，而不是只做最大杀伤外推。",
        "caution": "这是前临床模型，不能等同于具体病人的临床疗效。",
    },
    {
        "key": "psa_guided_trial",
        "title": "PSA 驱动的适应性阿比特龙临床试验",
        "citation": "Zhang J et al. Nature Communications, 2017.",
        "doi": "10.1038/s41467-017-01968-5",
        "url": "https://www.nature.com/articles/s41467-017-01968-5",
        "model_use": "支持使用纵向标志物轨迹来同步模拟周期，而不是使用固定日历周期。",
        "caution": "证据来自特定癌种、药物和严密临床监测环境；本系统不生成用药调整指令。",
    },
    {
        "key": "nlr_umbrella_review",
        "title": "NLR 与癌症预后的伞状综述",
        "citation": "Cupp MA et al. BMC Medicine, 2020.",
        "doi": "10.1186/s12916-020-01817-1",
        "url": "https://link.springer.com/article/10.1186/s12916-020-01817-1",
        "model_use": "支持把中性粒/淋巴比值作为廉价、可及的炎症状态代理变量。",
        "caution": "NLR 是相关性指标，受感染、应激、用药等影响，不能独立判断肿瘤变化。",
    },
    {
        "key": "pni_meta_analysis",
        "title": "PNI 营养免疫指数的癌症预后综述",
        "citation": "Sun K et al. Journal of Cancer Research and Clinical Oncology, 2014.",
        "doi": "10.1007/s00432-014-1714-3",
        "url": "https://pubmed.ncbi.nlm.nih.gov/24878931/",
        "model_use": "支持把白蛋白和淋巴细胞组合成营养/免疫储备观察指标。",
        "caution": "PNI 不是治疗反应指标，只能作为身体状态的间接观察。",
    },
    {
        "key": "siri_meta_analysis",
        "title": "SIRI 全身炎症反应指数的癌症预后荟萃分析",
        "citation": "Zhou Q et al. Dose-Response, 2021.",
        "doi": "10.1177/15593258211064744",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8689621/",
        "model_use": "支持用中性粒、单核和淋巴细胞构造炎症负荷代理变量。",
        "caution": "SIRI 反映全身炎症关联，不是个人疗效因果证据。",
    },
]


def get_evidence_annotations() -> List[Dict[str, str]]:
    """返回模拟页展示用的论文注释。"""
    return EVIDENCE_ANNOTATIONS


def convert_to_json_serializable(obj):
    """
    递归转换对象为 JSON 可序列化的类型
    
    处理：
    - NumPy 数值类型 (np.float64, np.int64, np.bool_)
    - NumPy 数组 (ndarray)
    - 嵌套字典和列表
    - 元组（转换为列表）
    
    :param obj: 任意 Python 对象
    :return: JSON 可序列化的对象
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.api.types.is_scalar(obj) and pd.isna(obj):
        return None  # NaN/NaT 转换为 null
    else:
        return obj


def _latest_numeric(patient_history: Optional[pd.DataFrame], column: str) -> Optional[float]:
    """读取某一列最新的有效数值。"""
    if patient_history is None or column not in patient_history.columns:
        return None

    values = pd.to_numeric(patient_history[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(denominator) < 1e-9:
        return None
    return float(numerator / denominator)


def _linear_high_score(value: Optional[float], threshold: float, severe_threshold: float) -> Optional[float]:
    if value is None:
        return None
    if value <= threshold:
        return 0.0
    if value >= severe_threshold:
        return 1.0
    return float((value - threshold) / (severe_threshold - threshold))


def _linear_low_score(value: Optional[float], threshold: float, severe_threshold: float) -> Optional[float]:
    if value is None:
        return None
    if value >= threshold:
        return 0.0
    if value <= severe_threshold:
        return 1.0
    return float((threshold - value) / (threshold - severe_threshold))


def _mean_available(values: List[Optional[float]], default: float) -> float:
    clean_values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean_values:
        return float(default)
    return float(np.clip(np.mean(clean_values), 0.0, 1.0))


def _derive_abs_from_percent(
    patient_history: Optional[pd.DataFrame],
    abs_column: str,
    percent_column: str,
) -> Optional[float]:
    direct_value = _latest_numeric(patient_history, abs_column)
    if direct_value is not None:
        return direct_value

    wbc = _latest_numeric(patient_history, "白细胞计数")
    pct = _latest_numeric(patient_history, percent_column)
    if wbc is None or pct is None:
        return None
    return float(wbc * pct / 100.0)


def build_host_context(patient_history: Optional[pd.DataFrame]) -> Dict:
    """
    从血常规、CRP、白蛋白等基础报告中提取“宿主状态”代理变量。

    论文注释：
    - NLR 来自 routine blood cell counts，Cupp et al. 2020 指出其在多癌种预后中有
      强/高度提示性证据，但也强调异质性和临床效用仍需验证。
    - PNI = 白蛋白(g/L) + 5 * 淋巴细胞绝对数(10^9/L)，用于营养/免疫储备观察；
      Sun et al. 2014 将其作为癌症预后相关的简单指标进行系统综述。
    - SIRI = 中性粒 * 单核 / 淋巴，Zhou et al. 2021 将其视为廉价的炎症反应指标。

    这些指标只用于：
    1. 给机理模型增加保守的宿主状态约束；
    2. 决定模拟不确定性区间宽窄；
    3. 生成可与医生讨论的观察点。
    它们不能推导个人用药剂量，也不能替代临床判断。
    """
    if patient_history is None or patient_history.empty:
        return {
            "available": False,
            "data_points": 0,
            "indices": {},
            "scores": {
                "inflammation_load": 0.5,
                "immune_reserve": 0.5,
                "nutrition_reserve": 0.5,
                "data_confidence": 0.0,
            },
            "modifiers": {
                "effective_immune_multiplier": 1.0,
                "trajectory_uncertainty": 0.45,
            },
            "observations": ["基础化验数据不足，模拟使用文献默认假设，区间会相对更宽。"],
        }

    neut = _derive_abs_from_percent(patient_history, "中性粒细胞绝对数", "中性粒细胞百分比")
    lymph = _derive_abs_from_percent(patient_history, "淋巴细胞绝对数", "淋巴细胞百分比")
    mono = _derive_abs_from_percent(patient_history, "单核细胞绝对数", "单核细胞百分比")
    platelet = _latest_numeric(patient_history, "血小板计数")
    albumin = _latest_numeric(patient_history, "白蛋白 ALB")
    crp = _latest_numeric(patient_history, "C反应蛋白 CRP")
    if crp is None:
        crp = _latest_numeric(patient_history, "超敏C反应蛋白")

    nlr = _safe_ratio(neut, lymph)
    plr = _safe_ratio(platelet, lymph)
    lmr = _safe_ratio(lymph, mono)
    sii = None
    if platelet is not None and neut is not None and lymph is not None and lymph > 1e-9:
        sii = float(platelet * neut / lymph)
    siri = None
    if neut is not None and mono is not None and lymph is not None and lymph > 1e-9:
        siri = float(neut * mono / lymph)
    pni = None
    if albumin is not None and lymph is not None:
        pni = float(albumin + 5.0 * lymph)

    mgps = None
    if crp is not None:
        if crp > 10 and albumin is not None and albumin < 35:
            mgps = 2
        elif crp > 10:
            mgps = 1
        else:
            mgps = 0

    indices = {
        "NLR": nlr,
        "PLR": plr,
        "LMR": lmr,
        "SII": sii,
        "SIRI": siri,
        "PNI": pni,
        "mGPS": mgps,
        "淋巴细胞绝对数": lymph,
        "白蛋白 ALB": albumin,
        "C反应蛋白 CRP": crp,
    }
    available_count = sum(v is not None for v in indices.values())

    inflammation_load = _mean_available(
        [
            _linear_high_score(nlr, 5.0, 8.0),
            _linear_high_score(plr, 200.0, 320.0),
            _linear_high_score(sii, 900.0, 1600.0),
            _linear_high_score(siri, 2.5, 5.0),
            _linear_high_score(crp, 10.0, 30.0),
        ],
        default=0.35,
    )

    immune_risk = _mean_available(
        [
            _linear_low_score(lymph, 1.0, 0.5),
            _linear_low_score(lmr, 2.0, 1.0),
            _linear_low_score(pni, 45.0, 35.0),
        ],
        default=0.5,
    )
    immune_reserve = float(np.clip(1.0 - immune_risk, 0.0, 1.0))

    nutrition_risk = _mean_available(
        [
            _linear_low_score(albumin, 35.0, 28.0),
            _linear_low_score(pni, 45.0, 35.0),
        ],
        default=0.5,
    )
    nutrition_reserve = float(np.clip(1.0 - nutrition_risk, 0.0, 1.0))

    sample_confidence = float(np.clip(len(patient_history) / 8.0, 0.0, 1.0))
    lab_confidence = float(np.clip(available_count / 7.0, 0.0, 1.0))
    data_confidence = float(np.clip(0.55 * sample_confidence + 0.45 * lab_confidence, 0.0, 1.0))

    # 保守调节：基础血检只影响免疫代理和不确定性，不直接输出剂量。
    effective_immune_multiplier = float(
        np.clip(0.65 + 0.55 * immune_reserve - 0.20 * inflammation_load, 0.35, 1.20)
    )
    trajectory_uncertainty = float(
        np.clip(0.12 + 0.25 * (1.0 - data_confidence) + 0.18 * inflammation_load + 0.10 * (1.0 - immune_reserve), 0.12, 0.55)
    )

    observations = []
    if nlr is not None:
        if nlr >= 5.0:
            observations.append(f"NLR={nlr:.2f}，炎症/应激代理指标偏高；请结合感染、近期治疗和医生判断。")
        else:
            observations.append(f"NLR={nlr:.2f}，当前未触发高炎症代理阈值。")
    if siri is not None and siri >= 2.5:
        observations.append(f"SIRI={siri:.2f}，提示全身炎症反应代理指标偏高。")
    if pni is not None:
        if pni < 45.0:
            observations.append(f"PNI={pni:.1f}，营养/免疫储备代理指标偏低，可作为复诊沟通点。")
        else:
            observations.append(f"PNI={pni:.1f}，营养/免疫储备代理指标未触发偏低阈值。")
    if mgps is not None and mgps > 0:
        observations.append(f"mGPS={mgps}，CRP/白蛋白组合提示炎症或营养状态需要结合医生评估。")
    if not observations:
        observations.append("可用基础化验不足，宿主状态约束较弱，模拟区间会相对更宽。")

    return {
        "available": available_count > 0,
        "data_points": int(len(patient_history)),
        "indices": indices,
        "scores": {
            "inflammation_load": inflammation_load,
            "immune_reserve": immune_reserve,
            "nutrition_reserve": nutrition_reserve,
            "data_confidence": data_confidence,
        },
        "modifiers": {
            "effective_immune_multiplier": effective_immune_multiplier,
            "trajectory_uncertainty": trajectory_uncertainty,
        },
        "observations": observations,
    }


# ==============================================================================
# 1. 治疗方案构建器
# ==============================================================================

def build_treatment_schedule(
    phases: List[Tuple[int, str]]  # [(天数, 剂量名称), ...]
) -> callable:
    """
    将用户友好的治疗方案转换为模型需要的函数
    
    :param phases: 例如 [(28, "100% 标准剂量"), (14, "0% 用药输入假设（仅模拟）"), ...]
    :return: treatment_func(t) -> intensity (0-1.5)
    """
    # 转换为累积时间和强度
    cumulative_schedule = []
    cumulative_days = 0
    
    for days, dose_name in phases:
        intensity = config.DOSE_PRESETS.get(dose_name, 0.0)
        cumulative_schedule.append((cumulative_days, cumulative_days + days, intensity))
        cumulative_days += days
    
    def treatment_func(t):
        """返回时间 t 的治疗强度"""
        for start, end, intensity in cumulative_schedule:
            if start <= t < end:
                return intensity
        return 0.0  # 超出计划时间后按 0% 用药输入处理
    
    return treatment_func


# ==============================================================================
# 2. 初始状态估算器
# ==============================================================================

def estimate_initial_state(
    model_name: str,
    latest_marker_value: float,
    patient_history: pd.DataFrame = None,
    host_context: Optional[Dict] = None
) -> Dict[str, float]:
    """
    根据最新肿瘤标志物值估算模型初始状态
    
    策略：
    - 对于S-R模型：假设90%敏感，10%抵抗（治疗初期）
    - 对于干细胞模型：假设5%干细胞，95%分化细胞
    - 对于Norton-Simon：直接使用标志物值
    """
    model_config = tumor_models.MODEL_FACTORY.get(model_name)
    
    if not model_config:
        raise ValueError(f"未知模型: {model_name}")
    
    states = model_config.get('states', [])
    
    # --- 根据模型类型设置初始比例 ---
    if model_name == "经典竞争模型 (S-R)":
        # 检查患者是否有治疗史
        if patient_history is not None and len(patient_history) > 5:
            # 有治疗史：假设已发展出一定耐药性
            s_fraction = 0.7  # 70% 敏感
        else:
            # 无治疗史：初治
            s_fraction = 0.9  # 90% 敏感
        
        return {
            'S': latest_marker_value * s_fraction,
            'R': latest_marker_value * (1 - s_fraction)
        }
    
    elif model_name == "干细胞驱动模型 (B20)":
        return {
            'Stem': latest_marker_value * 0.05,  # 5% 干细胞
            'Differentiated': latest_marker_value * 0.95
        }
    
    elif model_name == "Norton-Simon (Gompertzian)":
        return {
            'Tumor Volume': latest_marker_value
        }
    
    elif model_name == "三房室模型 (S-P-R)":
        return {
            'S': latest_marker_value * 0.8,
            'P': latest_marker_value * 0.1,
            'R': latest_marker_value * 0.1
        }
    
    elif model_name == "免疫-肿瘤互作模型 (de Pillis 2005)":
        # 【修复】免疫细胞的单位应该是细胞数/L，不需要乘以1e9
        # 正常淋巴细胞绝对数范围：1.1-3.2 × 10^9/L
        immune_level = 1.5  # 默认正常值（1.5×10^9/L）
        
        if patient_history is not None and '淋巴细胞绝对数' in patient_history.columns:
            latest_lymph = patient_history['淋巴细胞绝对数'].dropna()
            if len(latest_lymph) > 0:
                immune_level = latest_lymph.iloc[-1]

        if host_context is not None:
            lymph_proxy = host_context.get("indices", {}).get("淋巴细胞绝对数")
            if lymph_proxy is not None:
                immune_level = float(lymph_proxy)
            immune_level *= host_context.get("modifiers", {}).get("effective_immune_multiplier", 1.0)
            immune_level = max(0.05, immune_level)
        
        # 【新增2】估算累积药物暴露
        cumulative_exposure = 0.0
        
        if patient_history is not None and 'phase' in patient_history.columns:
            # 统计历史数据中"强效治疗期"的总天数
            treatment_days = patient_history[
                patient_history['phase'] == '强效治疗期'
            ]
            
            if len(treatment_days) > 0:
                # 使用索引计算时间跨度
                first_treatment = treatment_days.index.min()
                last_treatment = treatment_days.index.max()
                cumulative_exposure = (last_treatment - first_treatment).days
                
        return {
            'Tumor': latest_marker_value,
            'Immune': immune_level,
            'CumulativeExposure': cumulative_exposure  # ✅ 新增
        }
    
    else:
        raise ValueError(f"未实现的模型初始化: {model_name}")


def _total_burden_from_solution(model_name: str, solution: np.ndarray) -> np.ndarray:
    """把 ODE 状态矩阵转换为外部可观察的总负荷。"""
    if model_name == "免疫-肿瘤互作模型 (de Pillis 2005)":
        return solution[:, 0]
    return solution.sum(axis=1)


def _solve_trajectory(
    model_name: str,
    model_config: Dict,
    y0: List[float],
    time_points: np.ndarray,
    params: Tuple[float, ...],
    treatment_func,
    immune_func=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """运行一次 ODE，并返回状态矩阵和总负荷。"""
    ode_func = model_config["ode_func"]

    def wrapped_ode(y, t, p):
        kwargs = {"treatment_func": treatment_func}
        if immune_func is not None:
            kwargs["immune_func"] = immune_func
        return ode_func(y, t, p, **kwargs)

    solution = odeint(
        wrapped_ode,
        y0,
        time_points,
        args=(params,),
        rtol=1e-6,
        atol=1e-8,
    )
    return solution, _total_burden_from_solution(model_name, solution)


def _bounded_params(model_name: str, params: np.ndarray, latest_marker: float) -> np.ndarray:
    """把随机扰动后的参数限制在数值稳定且生物学上可解释的范围。"""
    bounded = np.asarray(params, dtype=float).copy()
    bounded = np.nan_to_num(bounded, nan=1e-6, posinf=1e6, neginf=1e-6)
    bounded = np.maximum(bounded, 1e-9)

    if model_name in ["经典竞争模型 (S-R)", "三房室模型 (S-P-R)"]:
        k_index = 2
        bounded[k_index] = max(bounded[k_index], latest_marker * 1.05, 1.0)
    elif model_name == "免疫-肿瘤互作模型 (de Pillis 2005)":
        bounded[1] = min(max(bounded[1], 1e-6), 1.0)

    return bounded


def _build_uncertainty_envelope(
    model_name: str,
    model_config: Dict,
    y0: List[float],
    time_points: np.ndarray,
    params: Tuple[float, ...],
    treatment_func,
    immune_func,
    latest_marker: float,
    host_context: Dict,
    n_samples: int = 41,
) -> Dict:
    """
    用参数扰动生成可解释的不确定性区间。

    论文注释：
    Zhang et al. 2017 强调适应性治疗应与个人纵向标志物动态同步；在只有少量
    基础化验和肿瘤标志物时，模型不可辨识性很强。因此这里不只给一条曲线，
    而是把基础化验缺失、炎症负荷和免疫/营养储备折算为更宽或更窄的轨迹区间。
    """
    uncertainty_scale = host_context.get("modifiers", {}).get("trajectory_uncertainty", 0.35)
    rng = np.random.default_rng(20260501)
    trajectories = []

    for _ in range(n_samples):
        noise = rng.normal(loc=0.0, scale=uncertainty_scale, size=len(params))
        sampled_params = np.asarray(params, dtype=float) * np.exp(noise)
        sampled_params = _bounded_params(model_name, sampled_params, latest_marker)

        try:
            _, burden = _solve_trajectory(
                model_name,
                model_config,
                y0,
                time_points,
                tuple(float(x) for x in sampled_params),
                treatment_func,
                immune_func=immune_func,
            )
            if np.isfinite(burden).all():
                trajectories.append(burden)
        except Exception as exc:
            logger.debug(f"不确定性采样失败，已跳过该参数组合: {exc}")

    if not trajectories:
        return {
            "available": False,
            "reason": "未能生成稳定的参数扰动轨迹",
        }

    trajectory_matrix = np.vstack(trajectories)
    return {
        "available": True,
        "sample_count": int(len(trajectories)),
        "p10": np.percentile(trajectory_matrix, 10, axis=0),
        "p50": np.percentile(trajectory_matrix, 50, axis=0),
        "p90": np.percentile(trajectory_matrix, 90, axis=0),
        "uncertainty_scale": float(uncertainty_scale),
    }


# ==============================================================================
# 3. 核心模拟函数
# ==============================================================================

def run_adaptive_simulation(
    model_name: str,
    treatment_schedule: List[Tuple[int, str]],
    initial_marker_value: float,
    patient_history: pd.DataFrame = None,
    simulation_days: int = 180,
    selected_marker: str = None,
    patient_id: int = None
) -> Dict:
    """
    运行单次适应性治疗模拟
    
    【V2更新】: 支持使用个性化校准参数
    
    :param selected_marker: 用于校准的标志物名称（如"CEA"）
    :param patient_id: 患者ID（用于加载校准参数）
    """
    logger.info(
        f"开始模拟 - 模型: {model_name}, 初始值: {initial_marker_value:.2f}, "
        f"时长: {simulation_days}天"
    )
    
    # 1. 获取模型配置
    model_config = tumor_models.MODEL_FACTORY[model_name]
    param_names = model_config['params']

    # 1.5 基础化验驱动的宿主状态约束
    host_context = build_host_context(patient_history)
    
    # 2. 设置默认参数（使用文献中值或 PyMC 先验的中心值）
    default_params = get_adaptive_params(
        model_name, 
        patient_history, 
        initial_marker_value,
        selected_marker=selected_marker,  # 【传递新参数】
        patient_id=patient_id,             # 【传递新参数】
        host_context=host_context
    )
    
    # 3. 估算初始状态
    initial_state_dict = estimate_initial_state(
        model_name, initial_marker_value, patient_history, host_context=host_context
    )
    y0 = [initial_state_dict[state] for state in model_config['states']]
    
    # 4. 构建治疗函数
    treatment_func = build_treatment_schedule(treatment_schedule)
    
    # 5. 构建免疫函数（如果模型需要）
    if 'immune_func' in model_config['ode_func'].__code__.co_varnames:
        immune_func = _build_immune_func(patient_history, host_context=host_context)
    else:
        immune_func = None
    
    # 6. 运行 ODE 求解
    time_points = np.arange(0, simulation_days + 1)
    
    try:
        solution, total_burden = _solve_trajectory(
            model_name,
            model_config,
            y0,
            time_points,
            default_params,
            treatment_func,
            immune_func=immune_func,
        )

        uncertainty = _build_uncertainty_envelope(
            model_name,
            model_config,
            y0,
            time_points,
            default_params,
            treatment_func,
            immune_func,
            initial_marker_value,
            host_context,
        )
        
        # 8. 记录治疗强度
        treatment_intensity = np.array([treatment_func(t) for t in time_points])
        
        return {
            'success': True,
            'time': time_points,
            'states': solution,
            'state_names': model_config['states'],
            'total_burden': total_burden,
            'treatment_intensity': treatment_intensity,
            'initial_state': initial_state_dict,
            'parameters': dict(zip(param_names, default_params)),
            'host_context': host_context,
            'uncertainty': uncertainty,
            'evidence_annotations': get_evidence_annotations(),
        }
        
    except Exception as e:
        logger.error(f"模拟失败: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


# ==============================================================================
# 4. 辅助函数
# ==============================================================================

def _get_default_params(model_name: str) -> tuple:
    """
    为每个模型返回文献默认参数
    （这些值来自 tumor_models.py 中的 PyMC 先验均值）
    """
    if model_name == "经典竞争模型 (S-R)":
        return (
            0.03,    # r_s ← 【修改】降低敏感细胞生长率（从0.08→0.03）
            0.015,   # r_r ← 【修改】同步降低抵抗细胞生长率
            100.0,   # K
            1.2,     # alpha_rs ← 【修改】提高竞争压力
            1.0,     # alpha_sr
            0.20,    # d_s ← 【修改】提高药物杀伤率（从0.15→0.20）
            2.0,     # cost_factor ← 【关键修改】适应性成本翻倍（从1.1→2.0）
            0.01,    # k_sr
            0.01,    # k_rs
            5e-6     # c ← 【修改】免疫杀伤率提高50倍
        )
    
    elif model_name == "干细胞驱动模型 (B20)":
        return (
            0.2,     # p_s (对称分裂概率)
            0.1,     # delta_d (药物杀伤率)
            1e-7     # c (免疫杀伤)
        )
    
    elif model_name == "Norton-Simon (Gompertzian)":
        return (
            0.05,    # rho0 (初始比生长率)
            0.001,   # alpha (衰减常数)
            0.1      # kill_rate
        )
    
    elif model_name == "三房室模型 (S-P-R)":
        return (
            0.03,    # r_s ← 【修改1】与S-R模型保持一致
            0.015,   # r_r ← 【修改2】与S-R模型保持一致
            100.0,   # K
            0.20,    # d_s ← 【修改3】提高药物杀伤率（与S-R一致）
            
            # === 【核心修复】持留细胞的动态参数 ===
            
            0.005,   # k_sp ← 【关键修改4】S→P转换率降低75%（从0.02→0.005）
                    #        生物学依据：只有5%的S细胞会进入持留状态（文献值：1-10%）
            
            0.05,    # k_ps ← 【关键修改5】P→S逆转率提高150%（从0.02→0.05）
                    #        生物学依据：无用药压力后P细胞应快速恢复，半衰期应<7天
            
            0.002,   # k_pr ← 【关键修改6】P→R突变率降低80%（从0.01→0.002）
                    #        生物学依据：表型可塑性不应快速固化为遗传耐药
            
            0.03     # delta_p ← 【关键修改7】P细胞死亡率提高50%（从0.02→0.03）
                    #          生物学依据：持留状态是高度应激状态，代谢成本高
        )
    
    elif model_name == "免疫-肿瘤互作模型 (de Pillis 2005)":
        return (
            0.06,     # a (肿瘤生长率) ← 【修改】从0.04提高到0.06
                    #   理由：需要更强的生长驱动力
            
            0.01,     # b (1/K) -> K = 100
            
            0.12,     # c (免疫基础杀伤率) ← 【修改】从0.08提高到0.12
                    #   配合 c_effective = c / 10.0
            
            0.15,     # s (免疫来源) ← 【修改】从0.2降低到0.15
            
            0.5,      # g (免疫招募率) ← 【修改】从0.6降低到0.5
            
            20.0,     # h (半饱和常数)
            
            0.10,     # d (免疫死亡率) ← 【修改】从0.08提高到0.10
                    #   理由：加速免疫衰减
            
            0.025     # p (免疫耗竭率) ← 【修改】从0.020提高到0.025
                    #   配合代码中的 exhaustion_reduction_factor=1.5
        )
    
    else:
        raise ValueError(f"未定义的模型参数: {model_name}")


def get_adaptive_params(
    model_name: str,
    patient_history: pd.DataFrame,
    latest_marker: float,
    selected_marker: str = None,  # 【已有】
    patient_id: int = None,        # 【新增】
    host_context: Optional[Dict] = None
) -> tuple:
    """
    【V2 更新】自适应参数获取（支持加载校准参数）
    
    优先级：
    1. 如果存在校准参数 → 使用校准参数
    2. 否则根据患者历史调整默认参数
    
    :param model_name: 模型名称
    :param patient_history: 患者历史数据
    :param latest_marker: 最新标志物值
    :param selected_marker: 用户选择的标志物名称
    :param patient_id: 患者ID（用于加载校准参数）
    """
    # === 【新增】尝试加载校准参数 ===
    if patient_id is not None and selected_marker is not None:
        # 【关键修复】需要导入 data_manager
        try:
            import data_manager
            
            calibrated = data_manager.load_calibrated_params(
                patient_id, 
                model_name, 
                selected_marker
            )
            
            if calibrated and calibrated.get('is_reliable', False):
                logger.info(
                    f"✅ 使用校准参数（验证误差: {calibrated['error']:.1%}）"
                )
                return calibrated['params']
            
            elif calibrated:
                logger.warning(
                    f"⚠️ 发现校准参数但不可靠（误差 {calibrated['error']:.1%} > 30%），"
                    f"使用默认参数"
                )
        
        except Exception as e:
            logger.debug(f"加载校准参数失败，使用默认参数: {e}")
    
    # === 【原有逻辑】使用默认参数并根据历史调整 ===
    base_params = list(_get_default_params(model_name))
    
    if patient_history is None or len(patient_history) < 2:
        return _apply_host_context_to_params(model_name, tuple(base_params), host_context)
    
    # 1. 计算治疗强度历史
    treatment_phases = patient_history['phase'].value_counts()
    treatment_ratio = treatment_phases.get('强效治疗期', 0) / len(patient_history)
    
    # 2. 计算标志物趋势（传递用户选择）
    marker_trend = _calculate_marker_trend(
        patient_history, 
        selected_marker=selected_marker
    )
    
    # 3. 根据模型类型调整参数
    if model_name == "经典竞争模型 (S-R)":
        # 治疗经验丰富 → 提高耐药性
        if treatment_ratio > 0.5:
            base_params[1] *= 1.3  # r_r (耐药细胞生长率)
            base_params[6] *= 1.2  # cost_factor (降低适应性成本)
        
        # 标志物持续上升 → 提高生长率
        if marker_trend > 0.1:  # 每月上升10%
            base_params[0] *= 1.2  # r_s
            base_params[1] *= 1.2  # r_r
    
    elif model_name == "免疫-肿瘤互作模型 (de Pillis 2005)":
        # 淋巴细胞持续偏低 → 降低免疫效应
        if '淋巴细胞绝对数' in patient_history.columns:
            recent_lymph = patient_history['淋巴细胞绝对数'].tail(3).mean()
            if recent_lymph < 1.0:  # 低于正常下限
                base_params[2] *= 0.5  # c (免疫杀伤率)
    
    return _apply_host_context_to_params(model_name, tuple(base_params), host_context)


def _apply_host_context_to_params(
    model_name: str,
    params: Tuple[float, ...],
    host_context: Optional[Dict],
) -> Tuple[float, ...]:
    """
    将基础化验提取出的宿主状态保守映射到模型参数。

    这里有意只做小幅调整：
    - 免疫相关模型：用淋巴细胞、NLR、PNI 等代理变量调节免疫杀伤项；
    - 高炎症负荷：只轻微增加生长/不确定性倾向，不把它解释成肿瘤进展；
    - Norton-Simon/S-P-R 等不含免疫项的模型主要通过不确定性区间体现宿主状态。
    """
    if not host_context:
        return tuple(params)

    adjusted = list(params)
    modifiers = host_context.get("modifiers", {})
    scores = host_context.get("scores", {})
    immune_multiplier = modifiers.get("effective_immune_multiplier", 1.0)
    inflammation_load = scores.get("inflammation_load", 0.0)

    if model_name == "经典竞争模型 (S-R)":
        adjusted[0] *= 1.0 + 0.06 * inflammation_load
        adjusted[1] *= 1.0 + 0.04 * inflammation_load
        adjusted[9] *= immune_multiplier

    elif model_name == "干细胞驱动模型 (B20)":
        adjusted[2] *= immune_multiplier

    elif model_name == "免疫-肿瘤互作模型 (de Pillis 2005)":
        adjusted[2] *= immune_multiplier
        adjusted[6] *= 1.0 + 0.08 * inflammation_load

    return tuple(float(max(v, 1e-9)) for v in adjusted)


def _calculate_marker_trend(patient_history: pd.DataFrame, selected_marker: str = None) -> float:
    """
    计算肿瘤标志物的月增长率
    
    核心改进：
    - 只使用 **最新阶段** 的数据（治疗期用治疗期数据，稳定期用稳定期数据）
    - 避免跨阶段边界的"污染"（如混入治疗前的高值）
    - 反映患者 **当前状态** 下的真实趋势
    
    与特征工程的区别：
    - 特征工程需要"纯净的稳定期数据"（用于异常检测模型训练）
    - 模拟引擎需要"当前阶段的真实趋势"（用于动态参数调整）
    
    :param patient_history: 患者历史数据（需包含 'phase' 列）
    :param selected_marker: 用户选择的标志物名称（优先使用）
    :return: 月增长率（浮点数，正数=上升，负数=下降）
    """
    # === 1. 数据完整性检查 ===
    if patient_history.empty:
        logger.warning("趋势计算失败：数据为空")
        return 0.0
    
    if 'phase' not in patient_history.columns:
        logger.warning("趋势计算失败：缺少 'phase' 列，使用后备计算（全部数据）")
        recent_data = patient_history.tail(3)
    else:
        # 只使用最新阶段的数据 
        latest_phase = patient_history.iloc[-1]['phase']
        phase_data = patient_history[patient_history['phase'] == latest_phase]
        recent_data = phase_data.tail(3)
        
        logger.debug(
            f"趋势计算：使用最新阶段 '{latest_phase}' 的最后 {len(recent_data)} 个数据点 "
            f"(Phase-Aware策略，避免跨界污染)"
        )
    
    # === 3. 数据充足性检查 ===
    if len(recent_data) < 2:
        logger.debug(f"趋势计算：同一阶段数据点不足 (<2)，返回 0.0")
        return 0.0
    
    days_span = (recent_data.index[-1] - recent_data.index[0]).days
    
    if days_span == 0:
        logger.debug("趋势计算：同一阶段数据点时间跨度为0，返回 0.0")
        return 0.0
    
    # === 4. 【保持原有逻辑】选择标志物列 ===
    if selected_marker and selected_marker in recent_data.columns:
        marker_col = selected_marker
        logger.debug(f"✅ 趋势计算使用用户选择的标志物: {marker_col}")
    else:
        # 后备方案：从 config 获取肿瘤标志物列表
        tumor_markers = []
        for template_name, items in config.LAB_REPORT_CONFIG.items():
            if template_name == "肿瘤标志物":
                tumor_markers = [item['name'] for item in items]
                for item in items:
                    tumor_markers.extend(item.get('aliases', []))
                break
        
        marker_col = None
        for col in recent_data.columns:
            if col in tumor_markers:
                marker_col = col
                break
        
        if marker_col is None:
            exclude_cols = ['report_uuid', 'phase', 'user_label', 'id', 'patient_id']
            numeric_cols = [
                col for col in recent_data.columns 
                if col not in exclude_cols 
                and recent_data[col].dtype in ['float64', 'int64', 'float32', 'int32']
            ]
            
            if not numeric_cols:
                logger.warning("⚠️ 无数值列可用于趋势计算")
                return 0.0
            
            marker_col = numeric_cols[0]
        
        logger.debug(f"趋势计算使用指标: {marker_col}")
    
    # === 5. 计算月增长率 ===
    try:
        marker_values = pd.to_numeric(recent_data[marker_col], errors='coerce')
        
        if marker_values.isna().all():
            logger.warning(f"⚠️ 标志物 '{marker_col}' 的所有值都无法转换为数值")
            return 0.0
        
        marker_values = marker_values.dropna()
        
        if len(marker_values) < 2:
            return 0.0
        
        value_change = marker_values.iloc[-1] - marker_values.iloc[0]
        initial_value = marker_values.iloc[0]
        
        if abs(initial_value) < 1e-9:
            return 0.0
        
        monthly_rate = (value_change / initial_value) * (30 / days_span)
        
        logger.debug(
            f"✅ 月增长率 (阶段 '{latest_phase}', 跨度 {days_span}天) = {monthly_rate:.4f}"
        )
        return monthly_rate
        
    except Exception as e:
        logger.error(f"❌ 计算趋势失败: {e}", exc_info=True)
        return 0.0


def _build_immune_func(patient_history: pd.DataFrame, host_context: Optional[Dict] = None) -> callable:
    """
    构建免疫函数（基于患者淋巴细胞历史数据）
    """
    immune_multiplier = 1.0
    if host_context is not None:
        immune_multiplier = host_context.get("modifiers", {}).get("effective_immune_multiplier", 1.0)

    if patient_history is None or '淋巴细胞绝对数' not in patient_history.columns:
        # 默认：假设免疫水平恒定（正常值1.5×10^9/L）
        return lambda t: max(0.05, 1.5 * immune_multiplier)
    
    lymph_history = patient_history['淋巴细胞绝对数'].dropna()
    
    if len(lymph_history) < 2:
        latest_value = lymph_history.iloc[-1] if len(lymph_history) > 0 else 1.5
        return lambda t: max(0.05, float(latest_value) * immune_multiplier)
    
    # 简化版：假设免疫水平维持在最近的平均值
    recent_mean = lymph_history.tail(3).mean()
    
    return lambda t: max(0.05, float(recent_mean) * immune_multiplier)


# ==============================================================================
# 5. 比较分析函数
# ==============================================================================

def compare_treatment_strategies(
    model_name: str,
    strategies: Dict[str, List[Tuple[int, str]]],
    initial_marker_value: float,
    patient_history: pd.DataFrame = None
) -> Dict:
    """
    比较多个治疗策略
    
    :param strategies: {"策略A": schedule_A, "策略B": schedule_B, ...}
    :return: 比较结果字典
    """
    logger.info(f"开始比较 {len(strategies)} 个治疗策略...")
    
    results = {}
    
    for strategy_name, schedule in strategies.items():
        logger.debug(f"  模拟策略: {strategy_name}")
        
        sim_result = run_adaptive_simulation(
            model_name, schedule, initial_marker_value, patient_history
        )
        
        if sim_result['success']:
            # 计算关键指标
            final_burden = sim_result['total_burden'][-1]
            min_burden = sim_result['total_burden'].min()
            time_to_min = sim_result['time'][np.argmin(sim_result['total_burden'])]
            
            results[strategy_name] = {
                'simulation': sim_result,
                'final_burden': final_burden,
                'min_burden': min_burden,
                'time_to_nadir': time_to_min,
                'reduction_rate': (initial_marker_value - final_burden) / initial_marker_value
            }
    
    return results

# ==============================================================================
# 6. 模型校准函数（基于 scipy.optimize，适用于小样本数据）
# ==============================================================================

def calibrate_model_with_scipy(
    model_name: str,
    patient_history: pd.DataFrame,
    selected_marker: str
) -> Dict:
    """
    【V3 - scipy 优化版】用历史数据校准模型参数，最新点验证
    
    核心改进：
    - 使用 scipy.optimize.differential_evolution（全局优化）
    - 利用所有历史数据（不只是最后几个点）
    - 计算时间从 8小时 → 10-30秒
    - 数值稳定性更好（自动处理边界约束）
    
    工作原理：
    1. 将历史数据分为"训练集"（前N-1个点）和"验证集"（最后1个点）
    2. 用训练集拟合模型参数（最小化预测误差）
    3. 用验证集检验预测精度
    
    :param model_name: 模型名称（从 tumor_models.MODEL_FACTORY 中选择）
    :param patient_history: 患者历史数据（包含 phase 列和 DatetimeIndex）
    :param selected_marker: 用于校准的标志物名称
    :return: {
        'success': bool,
        'calibrated_params': tuple,  # 校准后的参数
        'validation_error': float,   # 验证误差（百分比）
        'predicted_value': float,    # 预测值
        'actual_value': float,       # 实际值
        'is_reliable': bool,         # 是否可靠（误差<30%）
        'validation_date': str,
        'optimization_info': dict    # 优化过程信息
    }
    """
    logger.info(f"开始校准模型（scipy优化）: {model_name}, 标志物: {selected_marker}")
    
    # === 1. 数据完整性检查 ===
    if len(patient_history) < 3:
        return {
            'success': False,
            'error': '数据不足（需要至少3个数据点进行校准）'
        }
    
    if selected_marker not in patient_history.columns:
        return {
            'success': False,
            'error': f'标志物 {selected_marker} 不在数据中'
        }
    
    marker_data = patient_history[selected_marker].dropna()
    if len(marker_data) < 3:
        return {
            'success': False,
            'error': f'标志物 {selected_marker} 的有效数据不足（需要至少3个非空值）'
        }
    
    logger.info(f"数据检查通过：共 {len(marker_data)} 个有效数据点")
    
    # === 2. 分割数据（时间旅行验证）===
    history_for_fit = patient_history.iloc[:-1].copy()  # 前N-1个点用于训练
    holdout_point = patient_history.iloc[-1]            # 最后1个点用于验证
    
    logger.info(
        f"数据分割：训练集 {len(history_for_fit)} 个点，"
        f"验证集 1 个点（{holdout_point.name.strftime('%Y-%m-%d')}）"
    )
    
    # === 3. 推断治疗函数 ===
    treatment_func = _infer_treatment_from_history(history_for_fit)
    
    # === 4. 准备训练数据 ===
    valid_indices = history_for_fit[selected_marker].notna()
    time_points = (
        history_for_fit.index[valid_indices] - history_for_fit.index[0]
    ).days.values
    marker_values = history_for_fit[selected_marker][valid_indices].values
    
    if len(time_points) < 2:
        return {
            'success': False,
            'error': '训练数据不足（去除空值后少于2个点）'
        }
    
    logger.debug(f"训练数据：{len(time_points)} 个时间点，范围 {time_points[0]} 至 {time_points[-1]} 天")
    
    # === 5. 获取模型配置 ===
    model_config = tumor_models.MODEL_FACTORY.get(model_name)
    if not model_config:
        return {
            'success': False,
            'error': f'未知模型: {model_name}'
        }
    
    # 使用模型配置中的参数列表长度作为基准
    expected_n_params = len(model_config['params'])
    logger.debug(f"模型 '{model_name}' 期望 {expected_n_params} 个参数: {model_config['params']}")
    
    try:

        # === 6. 定义目标函数（最小化残差平方和）===
        def objective_function(params):
            """计算参数 params 下的预测误差（相对RSS）"""
            try:
                # 确保传入的参数数量正确
                if len(params) != expected_n_params:
                    logger.error(f"❌ 参数数量不匹配: 期望 {expected_n_params}，实际 {len(params)}")
                    return 1e10
                
                # 6.1 获取初始状态（基于第一个数据点）
                initial_state_dict = estimate_initial_state(
                    model_name,
                    marker_values[0],
                    history_for_fit
                )
                y0 = [initial_state_dict[state] for state in model_config['states']]
                
                # 6.2 构建免疫函数（如果需要）
                immune_func = None
                if 'immune_func' in model_config['ode_func'].__code__.co_varnames:
                    immune_func = _build_immune_func(history_for_fit)
                
                # 6.3 创建包装函数统一处理参数传递
                def wrapped_ode(y, t, p):
                    """包装 ODE 函数以正确传递关键字参数"""
                    kwargs = {'treatment_func': treatment_func}
                    if immune_func is not None:
                        kwargs['immune_func'] = immune_func
                    return model_config['ode_func'](y, t, p, **kwargs)
                
                # 6.4 运行 ODE 求解器
                solution = odeint(
                    wrapped_ode,
                    y0,
                    time_points,
                    args=(params,),
                    tfirst=False
                )
                
                # 6.5 计算预测值（肿瘤总负荷）
                predicted_values = solution[:, :].sum(axis=1)
                
                # 6.6 计算相对误差（避免除零）
                epsilon = 1e-6
                residuals = (predicted_values - marker_values) / (marker_values + epsilon)
                
                # 6.7 计算RSS（残差平方和）
                rss = np.sum(residuals ** 2)
                
                # 6.8 添加正则化项（防止参数过大导致数值不稳定）
                regularization = 0.01 * np.sum(np.array(params) ** 2)
                
                total_cost = rss + regularization
                
                return total_cost
                
            except Exception as e:
                # 如果该参数组合导致ODE求解失败，返回极大值
                logger.debug(f"⚠️ 优化迭代失败（参数组合不合理）: {e}")
                return 1e10
        
        # === 7. 设置参数边界（基于生物学合理性）===
        default_params = _get_default_params(model_name)
        
        # 确保默认参数数量与模型配置匹配
        if len(default_params) != expected_n_params:
            logger.error(
                f"❌ _get_default_params 返回的参数数量（{len(default_params)}）"
                f"与模型配置不匹配（期望 {expected_n_params}）\n"
                f"  模型期望: {model_config['params']}\n"
                f"  默认参数: {default_params}"
            )
            return {
                'success': False,
                'error': f'内部错误：默认参数数量不匹配（{len(default_params)} vs {expected_n_params}）'
            }
        
        bounds = []
        for param_name, default_value in zip(model_config['params'], default_params):
            # 根据参数名称智能设置边界
            if 'growth' in param_name.lower() or param_name in ['r_s', 'r_r', 'a']:
                # 生长率：[0.001, 2倍默认值]
                bounds.append((0.001, max(default_value * 2, 1.0)))
            
            elif 'death' in param_name.lower() or param_name in ['d_s', 'd', 'delta_d', 'delta_p']:
                # 死亡率：[0, 2倍默认值]
                bounds.append((0.0, max(default_value * 2, 1.0)))
            
            elif param_name in ['K'] or 'carry' in param_name.lower():
                # 承载力：[当前最大值, 10倍最大值]
                bounds.append((marker_values.max(), marker_values.max() * 10))
            
            elif 'alpha' in param_name.lower() or 'beta' in param_name.lower():
                # 竞争/转换系数：[0, 10倍默认值]
                bounds.append((0.0, max(default_value * 10, 1.0)))
            
            elif param_name in ['c']:  # ✅ 【新增】免疫杀伤参数的特殊处理
                # 免疫杀伤率通常非常小：[1e-10, 1e-4]
                bounds.append((1e-10, 1e-4))
            
            else:
                # 其他参数：[0.1倍, 10倍默认值]
                bounds.append((
                    max(default_value * 0.1, 1e-6),
                    default_value * 10
                ))
        
        # ✅ 【最终检查】确保边界数量与参数数量匹配
        if len(bounds) != expected_n_params:
            logger.error(f"❌ 边界数量（{len(bounds)}）与参数数量不匹配（{expected_n_params}）")
            return {
                'success': False,
                'error': '内部错误：参数边界配置失败'
            }
        
        logger.debug(f"参数边界设置：\n{list(zip(model_config['params'], bounds))}")
        logger.debug(f"初始猜测（文献默认值）：{default_params}")
        
        # === 8. 运行全局优化 ===
        logger.info("开始全局参数优化（差分进化算法）...")
        
        result_de = differential_evolution(
            objective_function,
            bounds=bounds,
            seed=42,                    # 固定随机种子（可复现）
            maxiter=100,                # 最大迭代次数（12个点不需要太多）
            popsize=15,                 # 种群大小（默认15）
            atol=1e-3,                  # 收敛容忍度
            tol=0.01,                   # 相对容忍度
            workers=1,                  # 单线程（避免并发问题）
            polish=True,                # 最后用局部优化精细化
            updating='deferred'         # 延迟更新（更稳定）
        )
        
        logger.info(
            f"全局优化完成：\n"
            f"  - 最终成本函数值: {result_de.fun:.4f}\n"
            f"  - 迭代次数: {result_de.nit}\n"
            f"  - 是否收敛: {result_de.success}"
        )
        
        # === 9. 局部精细化（可选）===
        logger.info("开始局部精细化优化（L-BFGS-B）...")
        
        result_local = minimize(
            objective_function,
            x0=result_de.x,             # 从全局最优解开始
            method='L-BFGS-B',          # 支持边界约束的梯度方法
            bounds=bounds,
            options={
                'maxiter': 50,
                'ftol': 1e-6
            }
        )
        
        # === 10. 选择最优结果 ===
        if result_local.fun < result_de.fun:
            result = result_local
            method_used = "L-BFGS-B（局部优化）"
            logger.info("局部优化找到了更好的解，采用局部优化结果")
        else:
            result = result_de
            method_used = "差分进化（全局搜索）"
            logger.info("全局优化的解更优，采用全局优化结果")
        
        if not result.success:
            logger.warning(f"⚠️ 优化未完全收敛: {result.message}，但仍使用当前最佳结果")
        
        calibrated_params = tuple(result.x)
        
        logger.info(
            f"✅ 参数优化完成\n"
            f"  - 方法: {method_used}\n"
            f"  - 最终成本: {result.fun:.4f}\n"
            f"  - 校准参数: {calibrated_params}"
        )
        
        # === 11. 验证预测（模拟到最新点）===
        logger.info("开始验证预测精度...")

        last_fit_date = history_for_fit.index[-1]
        holdout_date = holdout_point.name
        validation_days = (holdout_date - last_fit_date).days

        if validation_days <= 0:
            return {
                'success': False,
                'error': '验证点时间错误（不在未来），请检查数据顺序'
            }

        logger.debug(
            f"验证跨度：从 {last_fit_date.strftime('%Y-%m-%d')} "
            f"到 {holdout_date.strftime('%Y-%m-%d')}，共 {validation_days} 天"
        )

        # 获取验证前的状态
        last_marker_value = history_for_fit[selected_marker].dropna().iloc[-1]

        initial_state_dict = estimate_initial_state(
            model_name,
            last_marker_value,
            history_for_fit
        )
        y0 = [initial_state_dict[state] for state in model_config['states']]

        # 准备 ODE 求解
        time_span = np.arange(0, validation_days + 1)

        # ✅ 【修复1】确保 immune_func 在包装函数外部定义
        immune_func = None
        if 'immune_func' in model_config['ode_func'].__code__.co_varnames:
            immune_func = _build_immune_func(history_for_fit)
            logger.debug("✅ 构建免疫函数用于验证")

        # ✅ 【修复2】创建包装函数（immune_func 在外部作用域中）
        def wrapped_ode_for_validation(y, t, p):
            """包装 ODE 函数以正确传递关键字参数"""
            kwargs = {'treatment_func': treatment_func}
            if immune_func is not None:
                kwargs['immune_func'] = immune_func
            return model_config['ode_func'](y, t, p, **kwargs)

        # 使用包装函数调用 odeint
        try:
            solution = odeint(
                wrapped_ode_for_validation,
                y0,
                time_span,
                args=(calibrated_params,),
                tfirst=False
            )
            
            predicted_value = solution[-1, :].sum()  # 最后一天的总负荷
            
        except Exception as e:
            logger.error(f"❌ 验证 ODE 求解失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'验证阶段 ODE 求解失败: {str(e)}'
            }
        
        # === 12. 计算验证误差 ===
        actual_value = holdout_point[selected_marker]
        
        if pd.isna(actual_value):
            return {
                'success': False,
                'error': '验证点的实际值缺失（NaN）'
            }
        
        error = abs(predicted_value - actual_value) / actual_value
        
        logger.info(
            f"✅ 校准验证完成：\n"
            f"  - 预测值: {predicted_value:.2f}\n"
            f"  - 实际值: {actual_value:.2f}\n"
            f"  - 误差: {error:.1%}\n"
            f"  - 验证日期: {holdout_date.strftime('%Y-%m-%d')}"
        )
        
        # === 13. 返回结果 ===
        logger.info("✅ 校准成功完成")

        # 添加 is_reliable 判断逻辑
        is_reliable = error < 0.2  # 误差小于20%认为可靠

        result_dict = {
            'success': True,
            'calibrated_params': calibrated_params,
            'validation_error': error,
            'predicted_value': predicted_value,
            'actual_value': actual_value,
            'validation_date': holdout_date.strftime('%Y-%m-%d'),
            'is_reliable': is_reliable,
            'method': method_used,
            'final_cost': result.fun,
            'optimization_info': {
                'method': method_used,
                'final_cost': float(result.fun),
                'iterations': int(result.nit) if hasattr(result, 'nit') else None,
                'convergence': bool(result.success)
            }
        }

        # 递归转换所有 NumPy 类型为 Python 原生类型
        return convert_to_json_serializable(result_dict)

    except Exception as e:
        logger.error(f"❌ 校准过程失败: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'校准异常: {str(e)}'
        }


def _infer_treatment_from_history(patient_history: pd.DataFrame) -> callable:
    """
    从患者历史的 phase 列推断治疗强度函数
    
    策略：
    - '强效治疗期' → 强度 1.0（主要治疗活动期代理）
    - '稳定监控期' → 强度 0.1（维持或无主要治疗活动的代理）
    
    :param patient_history: 患者历史数据（需包含 'phase' 列）
    :return: 治疗强度函数 treatment_func(t)
    """
    if 'phase' not in patient_history.columns:
        logger.warning("历史数据缺少 'phase' 列，使用默认治疗函数（恒定强度0.5）")
        return lambda t: 0.5
    
    # 创建时间-强度映射表
    phase_timeline = []
    for date, row in patient_history.iterrows():
        days_from_start = (date - patient_history.index[0]).days
        intensity = 1.0 if row['phase'] == '强效治疗期' else 0.1
        phase_timeline.append((days_from_start, intensity))
    
    logger.debug(f"治疗时间线（共 {len(phase_timeline)} 个阶段）: {phase_timeline}")
    
    def treatment_func(t):
        """
        根据时间 t（天）返回对应的治疗强度
        
        采用"右查找"策略：t 时刻的强度 = 最近的历史记录的强度
        """
        # 如果 t 在第一个记录之前，使用第一个记录的强度
        if t < phase_timeline[0][0]:
            return phase_timeline[0][1]
        
        # 如果 t 在最后一个记录之后，使用最后一个记录的强度
        if t >= phase_timeline[-1][0]:
            return phase_timeline[-1][1]
        
        # 否则，找到 t 所在的区间
        for i in range(len(phase_timeline) - 1):
            if phase_timeline[i][0] <= t < phase_timeline[i+1][0]:
                return phase_timeline[i][1]
        
        # 兜底（理论上不会到这里）
        return phase_timeline[-1][1]
    
    return treatment_func
