"""
Ensemble prediction layer for sparse longitudinal laboratory data.

This module uses only laboratory time series.  It does not require imaging
reports or clinician-confirmed labels.  The target is a measurable future data
event, such as a future lab value crossing a reference limit or a reliable
change threshold; it is not a diagnosis of progression.

Implemented components:
- Linear-Gaussian state-space model for a latent "data activity proxy".
- Multi-indicator dynamic factor extracted from tumor markers, CRP, albumin,
  blood counts, and related laboratory signals.
- Conformal-style interval calibration using walk-forward residuals.
- Lightweight ensemble of sparse Bayesian trend and latent activity proxy.
- Task-oriented outputs for retrospective validation.

Method notes:
- The state-space layer is a pragmatic Kalman-filter approximation.  It treats
  the patient's underlying data activity as latent and laboratory values as
  noisy observations.
- The conformal calibration layer estimates empirical one-step residuals from
  previous predictions and widens future intervals when past residuals require
  it.  With very few points it falls back to conservative model intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config
import sparse_longitudinal_engine


HORIZON_DAYS = (30, 60, 90)
CONFORMAL_ALPHA = 0.05
ACTIVITY_HIGH_THRESHOLD = 1.0

META_COLUMNS = {
    "id",
    "patient_id",
    "report_uuid",
    "phase",
    "created_at",
    "updated_at",
}


@dataclass(frozen=True)
class IndicatorMeta:
    name: str
    category: str
    lower: float | None
    upper: float | None
    unit: str
    behavior: str


def analyze_predictive_ensemble(
    patient_df: pd.DataFrame,
    horizon_days: Iterable[int] = HORIZON_DAYS,
    mogp_results: dict[str, Any] | None = None,
    max_indicators: int | None = None,
) -> dict[str, Any]:
    """
    Build an ensemble prediction report from laboratory-only data.

    The function keeps every non-missing point used by the underlying models.
    It adds calibrated intervals and a cross-indicator activity proxy to the
    existing sparse longitudinal predictions.
    """

    df = _prepare_frame(patient_df)
    if df.empty:
        return {
            "status": "no_data",
            "method": "laboratory_ensemble_state_space_conformal",
            "summary": "没有可分析的数据。",
            "activity_proxy": {"status": "no_data"},
            "indicators": {},
            "top_observations": [],
        }

    horizons = tuple(int(day) for day in horizon_days)
    sparse_report = sparse_longitudinal_engine.analyze_sparse_trajectory(
        df,
        horizon_days=horizons,
        max_indicators=None,
    )
    activity_proxy = estimate_activity_proxy(df, horizon_days=horizons)

    indicator_reports: dict[str, dict[str, Any]] = {}
    for indicator, sparse_result in sparse_report.get("indicators", {}).items():
        if sparse_result.get("status") != "ok":
            indicator_reports[indicator] = sparse_result
            continue

        calibration = calibrate_indicator_intervals(df, indicator)
        ensemble_result = _merge_indicator_prediction(
            sparse_result=sparse_result,
            activity_proxy=activity_proxy,
            calibration=calibration,
            mogp_result=(mogp_results or {}).get(indicator),
        )
        indicator_reports[indicator] = ensemble_result

    ranked = sorted(
        [
            item for item in indicator_reports.values()
            if item.get("status") == "ok"
        ],
        key=lambda item: item.get("priority_score", 0.0),
        reverse=True,
    )
    if max_indicators is not None:
        ranked = ranked[:max_indicators]
        indicator_reports = {item["indicator"]: item for item in ranked}

    return {
        "status": "ok" if indicator_reports else "insufficient_data",
        "method": "laboratory_ensemble_state_space_conformal",
        "summary": _summarize_ensemble(ranked),
        "activity_proxy": activity_proxy,
        "indicators": indicator_reports,
        "top_observations": [_format_ensemble_observation(item) for item in ranked[:5]],
        "horizon_days": list(horizons),
    }


def estimate_activity_proxy(
    patient_df: pd.DataFrame,
    horizon_days: Iterable[int] = HORIZON_DAYS,
) -> dict[str, Any]:
    """
    Estimate a latent laboratory activity proxy from multiple indicators.

    The proxy is not a disease state.  It is a cross-indicator summary of
    whether laboratory data are moving away from the patient's own baseline in
    clinically relevant directions.
    """

    df = _prepare_frame(patient_df)
    metadata = _metadata()
    score_frame, contribution_table = _build_activity_score_frame(df, metadata)
    if score_frame.empty:
        return {
            "status": "insufficient_data",
            "message": "没有足够的可标准化指标构建活动代理因子。",
            "observations_used": 0,
        }

    dates = pd.to_datetime(score_frame.index)
    observations = score_frame["activity_observation"].to_numpy(dtype=float)
    obs_vars = np.square(score_frame["observation_sd"].to_numpy(dtype=float))

    filtered_mean, filtered_var = _kalman_filter_activity(dates, observations, obs_vars)
    latest_mean = float(filtered_mean[-1])
    latest_sd = float(math.sqrt(max(filtered_var[-1], 1e-9)))
    future = _predict_activity_future(dates[-1], latest_mean, latest_sd, horizon_days)

    return {
        "status": "ok",
        "dates": [date.date().isoformat() for date in dates],
        "observed_activity": observations.tolist(),
        "filtered_activity": filtered_mean.tolist(),
        "filtered_sd": np.sqrt(np.maximum(filtered_var, 1e-9)).tolist(),
        "latest": {
            "date": dates[-1].date().isoformat(),
            "mean": latest_mean,
            "sd": latest_sd,
            "prob_high": _normal_tail_probability(latest_mean, latest_sd, ACTIVITY_HIGH_THRESHOLD),
        },
        "future": future,
        "contributions": contribution_table,
        "observations_used": int(len(score_frame)),
        "indicators_used": sorted({item["indicator"] for item in contribution_table}),
    }


def calibrate_indicator_intervals(
    patient_df: pd.DataFrame,
    indicator: str,
    min_history: int = 3,
    alpha: float = CONFORMAL_ALPHA,
) -> dict[str, Any]:
    """
    Estimate empirical one-step residuals for conformal-style interval widening.

    This does not claim finite-sample validity for a single patient's tiny time
    series.  It is a transparent calibration guardrail: if earlier predictions
    missed by more than the model interval, future intervals are widened.
    """

    df = _prepare_frame(patient_df)
    series = pd.to_numeric(df.get(indicator), errors="coerce").dropna().sort_index()
    if len(series) <= min_history:
        return {
            "status": "insufficient_history",
            "residual_count": 0,
            "absolute_residual_quantile": None,
            "empirical_coverage": None,
            "interval_expansion_source": "model_only",
        }

    residuals: list[float] = []
    covered: list[bool] = []
    probability_records: list[dict[str, Any]] = []

    for i in range(min_history, len(series)):
        current_date = series.index[i]
        previous_date = series.index[i - 1]
        history = df[df.index < current_date]
        if history.empty:
            continue

        horizon = max(1, int((current_date - previous_date).days))
        prediction = sparse_longitudinal_engine.analyze_indicator(
            history,
            indicator,
            horizon_days=(horizon,),
        )
        if prediction.get("status") != "ok" or not prediction.get("future_predictions"):
            continue

        future = prediction["future_predictions"][0]
        actual_value = float(series.iloc[i])
        residual = abs(actual_value - float(future["median"]))
        residuals.append(residual)
        covered.append(float(future["lower_95"]) <= actual_value <= float(future["upper_95"]))
        probability_records.append(
            {
                "probability": max(
                    float(future.get("prob_reliable_change", 0.0)),
                    float(future.get("prob_outside_reference", 0.0)),
                ),
                "actual": bool(_is_future_data_event(actual_value, indicator)),
                "date": current_date,
                "history_date": previous_date,
                "actual_value": actual_value,
            }
        )

    if not residuals:
        return {
            "status": "insufficient_history",
            "residual_count": 0,
            "absolute_residual_quantile": None,
            "empirical_coverage": None,
            "interval_expansion_source": "model_only",
        }

    residual_arr = np.asarray(residuals, dtype=float)
    quantile_level = min(1.0, math.ceil((len(residual_arr) + 1) * (1.0 - alpha)) / len(residual_arr))
    residual_quantile = float(np.quantile(residual_arr, quantile_level, method="higher"))

    return {
        "status": "ok",
        "residual_count": int(len(residual_arr)),
        "absolute_residual_quantile": residual_quantile,
        "empirical_coverage": float(np.mean(covered)),
        "interval_expansion_source": "walk_forward_residuals",
        "probability_records": probability_records,
    }


def apply_conformal_interval(
    prediction: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of a sparse prediction with calibrated interval fields."""

    adjusted = dict(prediction)
    median = float(prediction["median"])
    lower = float(prediction["lower_95"])
    upper = float(prediction["upper_95"])
    original_half_width = max((upper - lower) / 2.0, 0.0)

    q = calibration.get("absolute_residual_quantile")
    if q is None or not np.isfinite(q):
        calibrated_half_width = original_half_width
        source = "model_only"
    else:
        calibrated_half_width = max(original_half_width, float(q))
        source = calibration.get("interval_expansion_source", "walk_forward_residuals")

    adjusted["conformal_lower_95"] = float(max(0.0, median - calibrated_half_width))
    adjusted["conformal_upper_95"] = float(median + calibrated_half_width)
    adjusted["conformal_half_width"] = float(calibrated_half_width)
    adjusted["conformal_source"] = source
    return adjusted


def _merge_indicator_prediction(
    sparse_result: dict[str, Any],
    activity_proxy: dict[str, Any],
    calibration: dict[str, Any],
    mogp_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indicator = sparse_result["indicator"]
    sample_count = sparse_result.get("data_quality", {}).get("sample_count", 0)
    current_change = sparse_result.get("current_reliable_change", {})
    activity_by_horizon = {
        item["horizon_days"]: item for item in activity_proxy.get("future", [])
    } if activity_proxy.get("status") == "ok" else {}

    future_predictions = []
    max_ensemble_probability = 0.0
    max_abnormal_probability = 0.0

    for future in sparse_result.get("future_predictions", []):
        adjusted = apply_conformal_interval(future, calibration)
        horizon = int(future["horizon_days"])
        activity_future = activity_by_horizon.get(horizon, {})
        activity_probability = float(activity_future.get("prob_high", 0.0))
        sparse_probability = max(
            float(future.get("prob_reliable_change", 0.0)),
            float(future.get("prob_outside_reference", 0.0)),
        )
        mogp_probability = _mogp_probability_proxy(mogp_result, horizon)
        ensemble_probability = _combine_probabilities(
            sparse_probability=sparse_probability,
            activity_probability=activity_probability,
            mogp_probability=mogp_probability,
            sample_count=int(sample_count),
            calibration=calibration,
            category=sparse_result.get("category", ""),
        )
        adjusted["activity_proxy_prob_high"] = activity_probability
        adjusted["sparse_event_probability"] = sparse_probability
        adjusted["mogp_probability_proxy"] = mogp_probability
        adjusted["ensemble_prob_reliable_or_outside"] = ensemble_probability
        future_predictions.append(adjusted)
        max_ensemble_probability = max(max_ensemble_probability, ensemble_probability)
        max_abnormal_probability = max(
            max_abnormal_probability,
            float(future.get("prob_outside_reference", 0.0)),
        )

    classification, priority = _classify_ensemble(
        sparse_result=sparse_result,
        max_ensemble_probability=max_ensemble_probability,
        max_abnormal_probability=max_abnormal_probability,
        activity_proxy=activity_proxy,
    )

    result = dict(sparse_result)
    result["method"] = "ensemble_state_space_conformal"
    result["future_predictions"] = future_predictions
    result["conformal_calibration"] = {
        key: value for key, value in calibration.items()
        if key != "probability_records"
    }
    result["activity_proxy_latest"] = activity_proxy.get("latest")
    result["classification"] = classification
    result["priority_score"] = priority
    result["ensemble_summary"] = {
        "max_ensemble_probability": max_ensemble_probability,
        "max_abnormal_probability": max_abnormal_probability,
        "current_reliable_change": bool(current_change.get("is_reliable")),
    }
    return result


def _combine_probabilities(
    sparse_probability: float,
    activity_probability: float,
    mogp_probability: float | None,
    sample_count: int,
    calibration: dict[str, Any],
    category: str,
) -> float:
    coverage = calibration.get("empirical_coverage")
    if coverage is None:
        calibration_weight = 0.85
    else:
        calibration_weight = max(0.55, 1.0 - abs(float(coverage) - 0.95))

    sparse_weight = (0.55 + min(sample_count, 10) * 0.03) * calibration_weight
    activity_weight = 0.30 if category in {"TUMOR_MARKER", "INFLAMMATION", "NUTRITION"} else 0.18

    weighted_sum = sparse_weight * sparse_probability + activity_weight * activity_probability
    total_weight = sparse_weight + activity_weight

    if mogp_probability is not None:
        weighted_sum += 0.20 * mogp_probability
        total_weight += 0.20

    if total_weight <= 0:
        return 0.0
    return float(min(1.0, max(0.0, weighted_sum / total_weight)))


def _classify_ensemble(
    sparse_result: dict[str, Any],
    max_ensemble_probability: float,
    max_abnormal_probability: float,
    activity_proxy: dict[str, Any],
) -> tuple[str, float]:
    sample_count = sparse_result.get("data_quality", {}).get("sample_count", 0)
    current_reliable = bool(sparse_result.get("current_reliable_change", {}).get("is_reliable"))
    activity_latest = activity_proxy.get("latest", {}) if activity_proxy.get("status") == "ok" else {}
    activity_prob = float(activity_latest.get("prob_high", 0.0))

    priority = (
        max_ensemble_probability * 55.0
        + max_abnormal_probability * 25.0
        + activity_prob * 15.0
        + (20.0 if current_reliable else 0.0)
    )

    if sample_count < 3:
        return "不足以判断", min(priority, 20.0)
    if current_reliable or max_ensemble_probability >= 0.70 or max_abnormal_probability >= 0.75:
        return "需要复核的数据变化", priority
    if max_ensemble_probability >= 0.35 or activity_prob >= 0.65:
        return "可观察预测", priority
    return "未见可靠变化", priority


def _format_ensemble_observation(result: dict[str, Any]) -> dict[str, Any]:
    current_change = result.get("current_reliable_change", {})
    if current_change.get("is_reliable"):
        observation = sparse_longitudinal_engine._format_observation(result)["observation"]
    else:
        future = result.get("future_predictions", [])
        best = max(
            future,
            key=lambda item: item.get("ensemble_prob_reliable_or_outside", 0.0),
        ) if future else None
        if best:
            observation = (
                f"{result['indicator']} {best['horizon_days']}天集成预测中位值 "
                f"{best['median']:.2f} {result.get('unit', '')}，"
                f"数据事件概率 {best.get('ensemble_prob_reliable_or_outside', 0.0):.0%}，"
                f"校准区间 {best.get('conformal_lower_95', best['lower_95']):.2f}-"
                f"{best.get('conformal_upper_95', best['upper_95']):.2f}"
            )
        else:
            observation = f"{result['indicator']} 数据点不足，暂不生成集成预测。"

    classification = result.get("classification", "不足以判断")
    if classification == "需要复核的数据变化":
        level = "medium"
        note = "这是化验数据代理模型的复核提示，不代表诊断；请结合症状、复查和医生意见。"
    elif classification == "可观察预测":
        level = "low"
        note = "观察到一定方向性，当前更适合作为下次复查时的对照材料。"
    else:
        level = "info"
        note = "当前化验数据不足以支持可靠变化判断。"

    return {
        "indicator": result["indicator"],
        "classification": classification,
        "attention_level": level,
        "observation": observation,
        "note": note,
        "priority_score": result.get("priority_score", 0.0),
    }


def _build_activity_score_frame(
    df: pd.DataFrame,
    metadata: dict[str, IndicatorMeta],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    transformed_by_indicator: dict[str, pd.Series] = {}
    contribution_table: list[dict[str, Any]] = []

    for indicator, meta in metadata.items():
        if indicator not in df.columns:
            continue
        series = pd.to_numeric(df[indicator], errors="coerce").dropna().sort_index()
        if len(series) < 2:
            continue

        transformed = _activity_transform(series.astype(float), meta)
        center = float(np.median(transformed))
        scale = _robust_scale(transformed)
        if not np.isfinite(scale) or scale <= 1e-9:
            continue

        z_series = pd.Series((transformed - center) / scale, index=series.index)
        if meta.behavior == "bidirectional":
            z_series = z_series.abs()
        weight = _activity_weight(meta)
        if weight <= 0:
            continue

        transformed_by_indicator[indicator] = z_series
        contribution_table.extend(
            {
                "date": idx.date().isoformat() if hasattr(idx, "date") else str(idx),
                "indicator": indicator,
                "score": float(score),
                "weight": float(weight),
            }
            for idx, score in z_series.items()
            if np.isfinite(score)
        )

    if not transformed_by_indicator:
        return pd.DataFrame(), []

    rows = []
    all_dates = sorted(set().union(*(series.index for series in transformed_by_indicator.values())))
    for date in all_dates:
        weighted_scores = []
        weights = []
        for indicator, z_series in transformed_by_indicator.items():
            if date not in z_series.index:
                continue
            score = float(z_series.loc[date])
            if not np.isfinite(score):
                continue
            weight = _activity_weight(metadata[indicator])
            weighted_scores.append(score * weight)
            weights.append(weight)
        if not weights:
            continue
        total_weight = float(np.sum(weights))
        activity_obs = float(np.sum(weighted_scores) / total_weight)
        obs_sd = float(math.sqrt(1.0 / max(total_weight, 1e-6) + 0.05))
        rows.append(
            {
                "date": date,
                "activity_observation": activity_obs,
                "observation_sd": obs_sd,
                "indicator_count": len(weights),
            }
        )

    if not rows:
        return pd.DataFrame(), contribution_table

    score_frame = pd.DataFrame(rows).set_index("date").sort_index()
    score_frame.index = pd.to_datetime(score_frame.index)
    return score_frame, contribution_table


def _kalman_filter_activity(
    dates: pd.DatetimeIndex,
    observations: np.ndarray,
    obs_vars: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    process_var_per_90_days = 0.30**2
    mean = 0.0
    var = 1.5
    filtered_mean = []
    filtered_var = []
    previous_date = dates[0]

    for date, obs, obs_var in zip(dates, observations, obs_vars):
        dt_days = max(1.0, float((date - previous_date).days))
        var = var + process_var_per_90_days * max(0.25, dt_days / 90.0)
        obs_var = max(float(obs_var), 1e-4)
        gain = var / (var + obs_var)
        mean = mean + gain * (float(obs) - mean)
        var = max((1.0 - gain) * var, 1e-6)
        filtered_mean.append(mean)
        filtered_var.append(var)
        previous_date = date

    return np.asarray(filtered_mean, dtype=float), np.asarray(filtered_var, dtype=float)


def _predict_activity_future(
    latest_date: pd.Timestamp,
    latest_mean: float,
    latest_sd: float,
    horizon_days: Iterable[int],
) -> list[dict[str, Any]]:
    process_var_per_90_days = 0.30**2
    future = []
    for day in horizon_days:
        variance = latest_sd**2 + process_var_per_90_days * max(0.25, int(day) / 90.0)
        sd = math.sqrt(max(variance, 1e-9))
        future.append(
            {
                "horizon_days": int(day),
                "date": (latest_date + pd.Timedelta(days=int(day))).date().isoformat(),
                "mean": float(latest_mean),
                "sd": float(sd),
                "lower_95": float(latest_mean - 1.96 * sd),
                "upper_95": float(latest_mean + 1.96 * sd),
                "prob_high": _normal_tail_probability(latest_mean, sd, ACTIVITY_HIGH_THRESHOLD),
            }
        )
    return future


def _activity_transform(series: pd.Series, meta: IndicatorMeta) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    offset = max(1e-6, _reference_scale(meta, float(np.nanmedian(values))) * 0.01)
    if meta.behavior == "high_is_bad":
        anchor = meta.upper if meta.upper is not None and meta.upper > 0 else float(np.nanmedian(values))
        return np.log((values + offset) / (anchor + offset))
    if meta.behavior == "low_is_bad":
        anchor = meta.lower if meta.lower is not None and meta.lower > 0 else float(np.nanmedian(values))
        return np.log((anchor + offset) / (values + offset))
    return values.astype(float)


def _activity_weight(meta: IndicatorMeta) -> float:
    if meta.category == "TUMOR_MARKER":
        return 1.40
    if meta.category in {"INFLAMMATION", "NUTRITION", "COMPOSITE"}:
        return 1.00
    if meta.category == "BLOOD_ROUTINE":
        return 0.70
    if meta.category in {"LIVER_KIDNEY", "ENZYME", "LIPID", "GLUCOSE"}:
        return 0.55
    if meta.category == "ELECTROLYTE":
        return 0.30
    return 0.40


def _metadata() -> dict[str, IndicatorMeta]:
    metadata: dict[str, IndicatorMeta] = {}
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

    for name, item in config.COMPOSITE_INDICATORS_CONFIG.items():
        behavior = "low_is_bad" if "threshold_low" in item else "high_is_bad"
        metadata[name] = IndicatorMeta(
            name=name,
            category=item.get("category", "COMPOSITE"),
            lower=_none_or_float(item.get("threshold_low")),
            upper=_none_or_float(item.get("threshold_high")),
            unit="",
            behavior=behavior,
        )
    return metadata


def _is_future_data_event(actual_value: float, indicator: str) -> bool:
    meta = _metadata().get(indicator)
    if meta is None:
        return False
    if meta.behavior == "low_is_bad" and meta.lower is not None:
        return actual_value < meta.lower
    if meta.upper is not None and meta.behavior in {"high_is_bad", "bidirectional"}:
        return actual_value > meta.upper
    if meta.lower is not None and meta.behavior == "bidirectional":
        return actual_value < meta.lower
    return False


def _mogp_probability_proxy(mogp_result: dict[str, Any] | None, horizon: int) -> float | None:
    if not mogp_result:
        return None
    means = mogp_result.get("predicted_mean")
    lowers = mogp_result.get("confidence_lower")
    uppers = mogp_result.get("confidence_upper")
    if means is None or lowers is None or uppers is None:
        return None
    try:
        width = abs(float(uppers[-1]) - float(lowers[-1]))
        mean_shift = abs(float(means[-1]) - float(means[0])) if len(means) > 1 else 0.0
    except Exception:
        return None
    if width <= 1e-9:
        return None
    return float(min(1.0, mean_shift / width))


def _normal_tail_probability(mean: float, sd: float, threshold: float) -> float:
    if sd <= 1e-9:
        return 1.0 if mean > threshold else 0.0
    return NormalDist().cdf((mean - threshold) / sd)


def _robust_scale(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)) * 1.4826)
    if np.isfinite(mad) and mad > 1e-9:
        return mad
    q75, q25 = np.percentile(clean, [75, 25])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    if np.isfinite(iqr_scale) and iqr_scale > 1e-9:
        return iqr_scale
    return float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0


def _reference_scale(meta: IndicatorMeta, fallback_value: float) -> float:
    if meta.lower is not None and meta.upper is not None and meta.upper > meta.lower:
        return max((meta.upper - meta.lower) / 4.0, 1e-6)
    if meta.upper is not None and meta.upper > 0:
        return max(meta.upper * 0.20, 1e-6)
    if meta.lower is not None and meta.lower > 0:
        return max(meta.lower * 0.20, 1e-6)
    return max(abs(fallback_value) * 0.10, 1.0)


def _prepare_frame(patient_df: pd.DataFrame) -> pd.DataFrame:
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()
    df = patient_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df


def _none_or_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _summarize_ensemble(ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return "暂无足够的化验纵向数据生成集成预测。"
    review = sum(1 for item in ranked if item.get("classification") == "需要复核的数据变化")
    observable = sum(1 for item in ranked if item.get("classification") == "可观察预测")
    return f"化验数据集成预测完成：{review} 项需要复核的数据变化，{observable} 项可观察预测。"
