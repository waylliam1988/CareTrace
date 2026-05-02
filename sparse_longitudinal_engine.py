# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

"""
Sparse longitudinal prediction for very small personal lab histories.

This module is intentionally separate from the Streamlit state and from the
existing risk engine.  It predicts measurable future laboratory trajectories
and "reliable change" probabilities; it does not diagnose recurrence,
progression, or treatment response.

Method notes / paper annotations:
- Dynamic prediction models in oncology increasingly use repeated measures,
  with joint models and AI becoming more common, but small-sample longitudinal
  oncology prediction remains a known limitation:
  Zhou Q et al. npj Precision Oncology, 2025.
  https://www.nature.com/articles/s41698-025-01162-7
- Bayesian hierarchical models are appropriate when combining weak population
  knowledge with sparse individual measurements and reporting posterior
  intervals:
  Zeger SL et al. PCORI / NCBI Bookshelf, 2020.
  https://www.ncbi.nlm.nih.gov/books/NBK594756/
- Tumor marker interpretation should account for analytical and biological
  variation.  CA19-9, CEA, and AFP RCV values are based on Erden et al.;
  CA15-3 and CA125 uncertainty is informed by Guillaume et al. and conservative
  patient-monitoring practice.
  https://doi.org/10.1080/00365510701601699
  https://pubmed.ncbi.nlm.nih.gov/37043610/
- Tumor marker platforms are not fully harmonized, especially CA19-9, so
  cross-platform continuity should widen uncertainty:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11222321/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config


META_COLUMNS = {
    "id",
    "patient_id",
    "report_uuid",
    "phase",
    "created_at",
    "updated_at",
}

HORIZON_DAYS = (30, 60, 90)

# Reference change values.  These are deliberately conservative because this
# app may not know the assay platform and patient disease states can add
# additional variability beyond healthy-volunteer estimates.
TUMOR_MARKER_RCV: dict[str, float] = {
    "糖类抗原 19-9": 0.6471,
    "癌胚抗原 CEA": 0.7257,
    "甲胎蛋白 AFP": 0.6262,
    "糖类抗原 15-3": 0.35,
    "糖类抗原 125": 0.40,
    "神经元特异性烯醇化酶 NSE": 0.30,
    "胃泌素释放肽前体 ProGRP": 0.30,
    "细胞角蛋白19片段 CYFRA21-1": 0.30,
    "鳞状细胞癌抗原 SCCA": 0.30,
}

TUMOR_MARKERS = {
    item["name"] for item in config.LAB_REPORT_CONFIG.get("肿瘤标志物", [])
}


@dataclass(frozen=True)
class IndicatorMeta:
    name: str
    category: str
    lower: float | None
    upper: float | None
    unit: str
    behavior: str


def analyze_sparse_trajectory(
    patient_df: pd.DataFrame,
    lab_context: dict[str, Any] | None = None,
    horizon_days: Iterable[int] = HORIZON_DAYS,
    max_indicators: int | None = None,
) -> dict[str, Any]:
    """
    Analyze sparse longitudinal lab trajectories.

    The engine uses every available point for each indicator.  No moving
    average, rolling window, or point dropping is used.
    """

    if patient_df is None or patient_df.empty:
        return {
            "status": "no_data",
            "summary": "没有可分析的数据。",
            "indicators": {},
            "top_observations": [],
            "method": "sparse_bayesian_shrinkage",
        }

    df = _prepare_patient_frame(patient_df)
    metadata = _lab_metadata()
    candidates = _select_candidate_indicators(df, metadata)

    results: dict[str, dict[str, Any]] = {}
    for indicator in candidates:
        result = analyze_indicator(
            df,
            indicator,
            metadata[indicator],
            lab_context=lab_context,
            horizon_days=horizon_days,
        )
        if result["status"] != "no_numeric_data":
            results[indicator] = result

    ranked = sorted(
        results.values(),
        key=lambda item: (
            item.get("priority_score", 0.0),
            item.get("data_quality", {}).get("sample_count", 0),
        ),
        reverse=True,
    )
    if max_indicators is not None:
        ranked = ranked[:max_indicators]
        results = {item["indicator"]: item for item in ranked}

    return {
        "status": "ok" if results else "insufficient_data",
        "summary": _summarize_results(ranked),
        "indicators": results,
        "top_observations": [_format_observation(item) for item in ranked[:5]],
        "method": "sparse_bayesian_shrinkage",
        "horizon_days": list(horizon_days),
    }


def analyze_indicator(
    patient_df: pd.DataFrame,
    indicator: str,
    meta: IndicatorMeta | None = None,
    lab_context: dict[str, Any] | None = None,
    horizon_days: Iterable[int] = HORIZON_DAYS,
) -> dict[str, Any]:
    """Analyze one indicator with robust baseline and shrinkage trend."""

    if meta is None:
        meta = _lab_metadata().get(
            indicator,
            IndicatorMeta(indicator, "UNKNOWN", None, None, "", "bidirectional"),
        )

    series = pd.to_numeric(patient_df.get(indicator), errors="coerce").dropna()
    series = series[np.isfinite(series.astype(float))]
    if series.empty:
        return {"indicator": indicator, "status": "no_numeric_data"}

    series = series.sort_index()
    values = series.astype(float).to_numpy()
    dates = pd.to_datetime(series.index)
    sample_count = int(len(values))

    baseline = _robust_baseline(values, meta)
    platform = _platform_continuity(patient_df.loc[series.index], lab_context)

    if sample_count < 2 or dates.nunique() < 2:
        return _insufficient_indicator_result(indicator, meta, values, dates, baseline, platform)

    transform = _choose_transform(values, meta)
    y = transform_values(values, transform)
    x_years = (dates - dates.min()).days.to_numpy(dtype=float) / 365.25

    slope_model = _fit_shrunken_trend(
        x_years,
        y,
        meta=meta,
        platform_factor=platform["uncertainty_factor"],
    )

    latest_value = float(values[-1])
    latest_y = float(y[-1])
    rcv = _rcv_for_indicator(indicator, meta, baseline, latest_value)
    future = []
    for day in horizon_days:
        pred = _predict_at_horizon(
            latest_y=latest_y,
            horizon_day=int(day),
            slope_model=slope_model,
            transform=transform,
            meta=meta,
            rcv=rcv,
            latest_value=latest_value,
        )
        future.append(pred)

    current_change = _current_reliable_change(values, transform, rcv, meta, baseline)
    trend_probability = _trend_probability(slope_model["slope"], slope_model["slope_sd"], meta)
    classification, priority = _classify_indicator(
        sample_count=sample_count,
        time_span_days=int((dates.max() - dates.min()).days),
        future=future,
        current_change=current_change,
        trend_probability=trend_probability,
        platform=platform,
    )

    return {
        "indicator": indicator,
        "status": "ok",
        "category": meta.category,
        "unit": meta.unit,
        "behavior": meta.behavior,
        "latest": {
            "date": dates[-1].date().isoformat(),
            "value": latest_value,
        },
        "baseline": baseline,
        "rcv": rcv,
        "trend": {
            "posterior_slope_per_year": slope_model["slope"],
            "posterior_slope_sd": slope_model["slope_sd"],
            "shrinkage": slope_model["shrinkage"],
            "direction_probability": trend_probability,
        },
        "future_predictions": future,
        "current_reliable_change": current_change,
        "platform_continuity": platform,
        "data_quality": {
            "sample_count": sample_count,
            "time_span_days": int((dates.max() - dates.min()).days),
            "median_interval_days": _median_interval_days(dates),
            "uses_all_points": True,
        },
        "classification": classification,
        "priority_score": priority,
        "explanation": _indicator_explanation(indicator, classification, future, current_change, trend_probability),
    }


def transform_values(values: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    if transform["kind"] == "log":
        offset = transform["offset"]
        return np.log(np.maximum(values + offset, 1e-9))
    return values.astype(float)


def inverse_transform(value: float, transform: dict[str, Any]) -> float:
    if transform["kind"] == "log":
        return max(0.0, math.exp(value) - transform["offset"])
    return float(value)


def _prepare_patient_frame(patient_df: pd.DataFrame) -> pd.DataFrame:
    df = patient_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df


def _lab_metadata() -> dict[str, IndicatorMeta]:
    metadata = {}
    for items in config.LAB_REPORT_CONFIG.values():
        for item in items:
            metadata[item["name"]] = IndicatorMeta(
                name=item["name"],
                category=item.get("category", "UNKNOWN"),
                lower=_none_or_float(item.get("lower")),
                upper=_none_or_float(item.get("upper")),
                unit=item.get("unit", ""),
                behavior=item.get("behavior", "bidirectional"),
            )
    return metadata


def _select_candidate_indicators(df: pd.DataFrame, metadata: dict[str, IndicatorMeta]) -> list[str]:
    candidates = []
    for column in df.columns:
        if column in META_COLUMNS or column not in metadata:
            continue
        if config.DERIVED_FEATURE_PATTERN.search(str(column)):
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 1:
            candidates.append(column)

    def priority(name: str) -> tuple[int, str]:
        meta = metadata[name]
        if name in TUMOR_MARKERS:
            return (0, name)
        if meta.category in {"INFLAMMATION", "NUTRITION"}:
            return (1, name)
        if meta.category == "BLOOD_ROUTINE":
            return (2, name)
        return (3, name)

    return sorted(candidates, key=priority)


def _robust_baseline(values: np.ndarray, meta: IndicatorMeta) -> dict[str, float | None]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)) * 1.4826)
    if not np.isfinite(mad) or mad <= 1e-9:
        q75, q25 = np.percentile(clean, [75, 25])
        mad = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    ref_scale = _reference_scale(meta, median)
    if not np.isfinite(mad) or mad <= 1e-9:
        mad = ref_scale
    return {
        "median": median,
        "mad": float(max(mad, 1e-6)),
        "reference_scale": ref_scale,
        "lower": meta.lower,
        "upper": meta.upper,
    }


def _choose_transform(values: np.ndarray, meta: IndicatorMeta) -> dict[str, Any]:
    if meta.behavior == "high_is_bad" and np.nanmin(values) >= 0:
        return {"kind": "log", "offset": max(1e-6, _reference_scale(meta, float(np.nanmedian(values))) * 0.01)}
    return {"kind": "identity"}


def _fit_shrunken_trend(
    x_years: np.ndarray,
    y: np.ndarray,
    meta: IndicatorMeta,
    platform_factor: float = 1.0,
) -> dict[str, float]:
    pair_slopes = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            dx = x_years[j] - x_years[i]
            if abs(dx) > 1e-9:
                pair_slopes.append((y[j] - y[i]) / dx)

    if not pair_slopes:
        return {"slope": 0.0, "slope_sd": 1.0, "shrinkage": 0.0, "observed_slope": 0.0}

    observed_slope = float(np.median(pair_slopes))
    slope_mad = float(np.median(np.abs(np.asarray(pair_slopes) - observed_slope)) * 1.4826)

    residual_scale = _robust_residual_scale(x_years, y, observed_slope)
    time_span = max(float(np.max(x_years) - np.min(x_years)), 1.0 / 365.25)
    measurement_slope_sd = residual_scale / max(time_span, 1e-3)

    if not np.isfinite(slope_mad) or slope_mad <= 1e-9:
        slope_mad = measurement_slope_sd

    likelihood_sd = max(
        slope_mad / math.sqrt(max(1, len(pair_slopes))),
        measurement_slope_sd / math.sqrt(max(1, len(y) - 1)),
        0.05,
    ) * platform_factor

    prior_sd = _prior_slope_sd(meta)
    prior_var = prior_sd**2
    likelihood_var = likelihood_sd**2
    shrinkage = prior_var / (prior_var + likelihood_var)
    posterior_slope = shrinkage * observed_slope
    posterior_sd = math.sqrt(1.0 / (1.0 / prior_var + 1.0 / likelihood_var))

    return {
        "slope": float(posterior_slope),
        "slope_sd": float(max(posterior_sd, 1e-6)),
        "shrinkage": float(shrinkage),
        "observed_slope": float(observed_slope),
        "residual_scale": float(max(residual_scale, 1e-6)),
    }


def _predict_at_horizon(
    latest_y: float,
    horizon_day: int,
    slope_model: dict[str, float],
    transform: dict[str, Any],
    meta: IndicatorMeta,
    rcv: dict[str, Any],
    latest_value: float,
) -> dict[str, Any]:
    horizon_years = horizon_day / 365.25
    mean_y = latest_y + slope_model["slope"] * horizon_years
    process_noise = _process_noise_sd(meta, transform) * math.sqrt(max(1.0, horizon_day / 90.0))
    sd_y = math.sqrt(
        (slope_model["slope_sd"] * horizon_years) ** 2
        + slope_model.get("residual_scale", 0.05) ** 2
        + process_noise ** 2
    )

    lower_y = mean_y - 1.96 * sd_y
    upper_y = mean_y + 1.96 * sd_y
    median_value = inverse_transform(mean_y, transform)
    lower_value = inverse_transform(lower_y, transform)
    upper_value = inverse_transform(upper_y, transform)

    reliable_prob = _prob_reliable_change(
        latest_y,
        mean_y,
        sd_y,
        transform,
        rcv,
        meta,
        latest_value,
    )
    abnormal_prob = _prob_abnormal(mean_y, sd_y, transform, meta)

    return {
        "horizon_days": int(horizon_day),
        "median": float(median_value),
        "lower_95": float(min(lower_value, upper_value)),
        "upper_95": float(max(lower_value, upper_value)),
        "prob_reliable_change": float(reliable_prob),
        "prob_outside_reference": float(abnormal_prob),
    }


def _prob_reliable_change(
    latest_y: float,
    mean_y: float,
    sd_y: float,
    transform: dict[str, Any],
    rcv: dict[str, Any],
    meta: IndicatorMeta,
    latest_value: float,
) -> float:
    normal = NormalDist()
    if sd_y <= 1e-9:
        return 0.0

    if rcv["type"] == "relative" and transform["kind"] == "log":
        threshold = math.log1p(float(rcv["value"]))
        if meta.behavior == "low_is_bad":
            z = ((latest_y - threshold) - mean_y) / sd_y
            return normal.cdf(z)
        z = (mean_y - (latest_y + threshold)) / sd_y
        return normal.cdf(z)

    threshold_abs = float(rcv["absolute_value"])
    if meta.behavior == "low_is_bad":
        threshold_y = latest_value - threshold_abs
        z = (threshold_y - inverse_transform(mean_y, {"kind": "identity"})) / max(sd_y, 1e-9)
        return normal.cdf(z)
    if meta.behavior == "bidirectional":
        high_z = (mean_y - (latest_y + threshold_abs)) / sd_y
        low_z = ((latest_y - threshold_abs) - mean_y) / sd_y
        return min(1.0, normal.cdf(high_z) + normal.cdf(low_z))
    z = (mean_y - (latest_y + threshold_abs)) / sd_y
    return normal.cdf(z)


def _prob_abnormal(mean_y: float, sd_y: float, transform: dict[str, Any], meta: IndicatorMeta) -> float:
    if sd_y <= 1e-9:
        return 0.0
    normal = NormalDist()

    def transformed_threshold(value: float) -> float:
        return transform_values(np.asarray([value], dtype=float), transform)[0]

    probability = 0.0
    if meta.upper is not None and meta.behavior in {"high_is_bad", "bidirectional"}:
        upper_y = transformed_threshold(meta.upper)
        probability += normal.cdf((mean_y - upper_y) / sd_y)
    if meta.lower is not None and meta.behavior in {"low_is_bad", "bidirectional"}:
        lower_y = transformed_threshold(meta.lower)
        probability += normal.cdf((lower_y - mean_y) / sd_y)
    return float(min(1.0, max(0.0, probability)))


def _current_reliable_change(
    values: np.ndarray,
    transform: dict[str, Any],
    rcv: dict[str, Any],
    meta: IndicatorMeta,
    baseline: dict[str, float | None],
) -> dict[str, Any]:
    if len(values) < 2:
        return {"available": False}

    latest = float(values[-1])
    baseline_value = float(baseline["median"])

    if rcv["type"] == "relative" and baseline_value > 0:
        pct_change = (latest - baseline_value) / baseline_value
        threshold = float(rcv["value"])
        reliable = abs(pct_change) >= threshold
        direction = "上升" if pct_change > 0 else "下降"
        magnitude = abs(pct_change)
    else:
        threshold = float(rcv["absolute_value"])
        delta = latest - baseline_value
        reliable = abs(delta) >= threshold
        direction = "上升" if delta > 0 else "下降"
        magnitude = abs(delta)

    return {
        "available": True,
        "baseline_value": baseline_value,
        "latest_value": latest,
        "direction": direction,
        "magnitude": float(magnitude),
        "threshold": float(threshold),
        "is_reliable": bool(reliable),
    }


def _rcv_for_indicator(
    indicator: str,
    meta: IndicatorMeta,
    baseline: dict[str, float | None],
    latest_value: float,
) -> dict[str, Any]:
    if indicator in TUMOR_MARKER_RCV:
        return {
            "type": "relative",
            "value": TUMOR_MARKER_RCV[indicator],
            "absolute_value": max(abs(latest_value) * TUMOR_MARKER_RCV[indicator], 1e-6),
            "source": "tumor_marker_rcv_literature_or_conservative_default",
        }

    scale = max(float(baseline["mad"]), _reference_scale(meta, latest_value) * 0.5, 1e-6)
    return {
        "type": "absolute",
        "value": None,
        "absolute_value": scale,
        "source": "personal_mad_and_reference_scale",
    }


def _trend_probability(slope: float, slope_sd: float, meta: IndicatorMeta) -> float:
    if slope_sd <= 1e-9:
        return 1.0 if slope > 0 else 0.0
    normal = NormalDist()
    if meta.behavior == "low_is_bad":
        return normal.cdf((-slope) / slope_sd)
    if meta.behavior == "bidirectional":
        return max(normal.cdf(slope / slope_sd), normal.cdf((-slope) / slope_sd))
    return normal.cdf(slope / slope_sd)


def _classify_indicator(
    sample_count: int,
    time_span_days: int,
    future: list[dict[str, Any]],
    current_change: dict[str, Any],
    trend_probability: float,
    platform: dict[str, Any],
) -> tuple[str, float]:
    max_reliable_prob = max((item["prob_reliable_change"] for item in future), default=0.0)
    max_abnormal_prob = max((item["prob_outside_reference"] for item in future), default=0.0)
    current_reliable = bool(current_change.get("is_reliable"))

    priority = (
        max_reliable_prob * 45.0
        + max_abnormal_prob * 35.0
        + max(0.0, trend_probability - 0.5) * 25.0
        + (20.0 if current_reliable else 0.0)
    )
    if platform["continuity"] == "unknown":
        priority *= 0.90
    elif platform["continuity"] == "mixed_or_changed":
        priority *= 0.75

    if sample_count < 3 or time_span_days < 21:
        return "不足以判断", min(priority, 25.0)
    if max_reliable_prob >= 0.70 or max_abnormal_prob >= 0.70 or current_reliable:
        return "需要复核的数据变化", priority
    if max_reliable_prob >= 0.35 or max_abnormal_prob >= 0.35 or trend_probability >= 0.75:
        return "可观察预测", priority
    return "未见可靠变化", priority


def _indicator_explanation(
    indicator: str,
    classification: str,
    future: list[dict[str, Any]],
    current_change: dict[str, Any],
    trend_probability: float,
) -> str:
    if not future:
        return "数据点不足，暂不生成预测。"
    horizon = max(future, key=lambda item: item["prob_reliable_change"])
    return (
        f"{indicator}：{classification}。"
        f"{horizon['horizon_days']} 天预测中位值 {horizon['median']:.2f}，"
        f"95%区间 {horizon['lower_95']:.2f}-{horizon['upper_95']:.2f}；"
        f"超过可靠变化阈值概率 {horizon['prob_reliable_change']:.0%}，"
        f"方向性趋势概率 {trend_probability:.0%}。"
    )


def _format_observation(result: dict[str, Any]) -> dict[str, Any]:
    future = result.get("future_predictions", [])
    horizon = max(future, key=lambda item: item["prob_reliable_change"]) if future else None
    current_change = result.get("current_reliable_change", {})
    if current_change.get("is_reliable"):
        threshold = current_change.get("threshold", 0.0)
        magnitude = current_change.get("magnitude", 0.0)
        if result.get("rcv", {}).get("type") == "relative":
            observation = (
                f"{result['indicator']} 较个人中位基线{current_change.get('direction', '变化')} "
                f"{magnitude:.0%}，超过可靠变化阈值 {threshold:.0%}"
            )
        else:
            observation = (
                f"{result['indicator']} 较个人中位基线{current_change.get('direction', '变化')} "
                f"{magnitude:.2f} {result.get('unit', '')}，超过可靠变化阈值 {threshold:.2f}"
            )
    elif horizon:
        observation = (
            f"{result['indicator']} {horizon['horizon_days']}天预测中位值 "
            f"{horizon['median']:.2f} {result.get('unit', '')}，"
            f"可靠变化概率 {horizon['prob_reliable_change']:.0%}"
        )
    else:
        observation = f"{result['indicator']} 数据点不足，暂不生成预测"

    classification = result.get("classification", "不足以判断")
    if classification == "需要复核的数据变化":
        level = "medium"
        note = "这是基于稀疏纵向数据的概率提示；请结合症状、影像和医生意见决定是否复查。"
    elif classification == "可观察预测":
        level = "low"
        note = "观察到一定方向性，但样本很少，适合继续记录并等待下一次检测确认。"
    else:
        level = "info"
        note = "当前数据不足以支持可靠变化判断，继续保留后续检测点会提高稳定性。"

    return {
        "indicator": result["indicator"],
        "classification": classification,
        "attention_level": level,
        "observation": observation,
        "note": note,
        "priority_score": result.get("priority_score", 0.0),
    }


def _summarize_results(ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return "暂无足够的纵向数据生成小样本预测。"
    review_count = sum(1 for item in ranked if item.get("classification") == "需要复核的数据变化")
    observable_count = sum(1 for item in ranked if item.get("classification") == "可观察预测")
    return f"小样本纵向预测完成：{review_count} 项需要复核的数据变化，{observable_count} 项可观察预测。"


def _platform_continuity(df: pd.DataFrame, lab_context: dict[str, Any] | None) -> dict[str, Any]:
    context = lab_context or {}
    lab_column = context.get("lab_column")
    platform_column = context.get("platform_column")
    candidate_columns = [lab_column, platform_column, "lab", "laboratory", "hospital", "platform", "assay_platform"]
    present = [col for col in candidate_columns if col and col in df.columns]
    if not present:
        return {
            "continuity": "unknown",
            "uncertainty_factor": 1.20,
            "note": "未记录实验室/检测平台，已适度放宽不确定区间。",
        }
    values = df[present].astype(str).agg("|".join, axis=1).dropna().unique()
    if len(values) <= 1:
        return {
            "continuity": "same_platform",
            "uncertainty_factor": 1.0,
            "note": "检测平台记录连续。",
        }
    return {
        "continuity": "mixed_or_changed",
        "uncertainty_factor": 1.60,
        "note": "检测平台或实验室记录不一致，预测不确定性已扩大。",
    }


def _insufficient_indicator_result(
    indicator: str,
    meta: IndicatorMeta,
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    baseline: dict[str, float | None],
    platform: dict[str, Any],
) -> dict[str, Any]:
    return {
        "indicator": indicator,
        "status": "insufficient_points",
        "category": meta.category,
        "unit": meta.unit,
        "behavior": meta.behavior,
        "latest": {
            "date": dates[-1].date().isoformat(),
            "value": float(values[-1]),
        },
        "baseline": baseline,
        "future_predictions": [],
        "platform_continuity": platform,
        "data_quality": {
            "sample_count": int(len(values)),
            "time_span_days": int((dates.max() - dates.min()).days) if len(dates) else 0,
            "median_interval_days": _median_interval_days(dates),
            "uses_all_points": True,
        },
        "classification": "不足以判断",
        "priority_score": 0.0,
        "explanation": "数据点不足，暂不生成预测。",
    }


def _robust_residual_scale(x: np.ndarray, y: np.ndarray, slope: float) -> float:
    intercept = float(np.median(y - slope * x))
    residuals = y - (intercept + slope * x)
    scale = float(np.median(np.abs(residuals - np.median(residuals))) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.05
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 0.05
    return scale


def _process_noise_sd(meta: IndicatorMeta, transform: dict[str, Any]) -> float:
    """
    Minimum quarter-scale process noise.

    Sparse follow-up data cannot rule out clinically meaningful changes between
    visits.  This term prevents 3-4 stable points from producing unrealistically
    narrow future intervals.
    """

    if transform["kind"] == "log" and meta.name in TUMOR_MARKER_RCV:
        return max(0.16, math.log1p(TUMOR_MARKER_RCV[meta.name]) / 2.0)
    if transform["kind"] == "log" and meta.name in TUMOR_MARKERS:
        return 0.18
    if meta.category in {"INFLAMMATION", "ENZYME"}:
        return 0.12
    if meta.category in {"BLOOD_ROUTINE", "ELECTROLYTE"}:
        return 0.06
    return 0.08


def _prior_slope_sd(meta: IndicatorMeta) -> float:
    if meta.name in TUMOR_MARKERS:
        return 0.75
    if meta.category in {"INFLAMMATION", "ENZYME"}:
        return 0.90
    if meta.category in {"BLOOD_ROUTINE", "ELECTROLYTE"}:
        return 0.45
    return 0.60


def _reference_scale(meta: IndicatorMeta, fallback_value: float) -> float:
    if meta.lower is not None and meta.upper is not None and meta.upper > meta.lower:
        return max((meta.upper - meta.lower) / 4.0, 1e-6)
    if meta.upper is not None and meta.upper > 0:
        return max(meta.upper * 0.20, 1e-6)
    return max(abs(fallback_value) * 0.10, 1.0)


def _median_interval_days(dates: pd.DatetimeIndex) -> float | None:
    unique_dates = pd.Series(pd.to_datetime(dates).unique()).sort_values()
    if len(unique_dates) < 2:
        return None
    intervals = unique_dates.diff().dt.days.dropna()
    if intervals.empty:
        return None
    return float(intervals.median())


def _none_or_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
