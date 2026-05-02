# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

"""
Retrospective validation harness.

Run:
    python .\测试脚本\retrospective_validation.py

The default dataset is synthetic so the script is repeatable and does not touch
real patient data.  The functions are written so a real historical DataFrame can
be plugged in later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baseline_monitor  # noqa: E402
import sparse_longitudinal_engine  # noqa: E402
import predictive_ensemble_engine  # noqa: E402


INDICATOR = "癌胚抗原 CEA"


def generate_synthetic_history(n: int = 36, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="14D")
    values = 3.0 + rng.normal(0, 0.18, size=n)

    anomaly_mask = np.zeros(n, dtype=bool)
    if n >= 12:
        anomaly_indices = [int(n * 0.70), int(n * 0.86)]
        for idx in anomaly_indices:
            values[idx] = 8.0 + rng.normal(0, 0.25)
            anomaly_mask[idx] = True

    df = pd.DataFrame(
        {
            "phase": ["稳定监控期"] * n,
            "report_uuid": [f"synthetic-{i}" for i in range(n)],
            INDICATOR: values,
        },
        index=dates,
    )
    labels = pd.Series(anomaly_mask, index=dates, name="actual_anomaly")
    return df, labels


def evaluate_alert_detection(
    df: pd.DataFrame,
    labels: pd.Series,
    indicator: str = INDICATOR,
    min_history: int = 4,
) -> dict:
    monitor = baseline_monitor.PersonalBaselineMonitor()
    ref_ranges = {indicator: (0.0, 5.0)}

    predictions = []
    actuals = []
    for i in range(min_history, len(df)):
        history_with_current = df.iloc[: i + 1]
        result = monitor.assess_current_value(history_with_current, indicator, ref_ranges)
        predicted = result.get("level") in {"medium", "high"}
        predictions.append(predicted)
        actuals.append(bool(labels.iloc[i]))

    predictions_arr = np.array(predictions, dtype=bool)
    actuals_arr = np.array(actuals, dtype=bool)

    negative_count = max(1, int((~actuals_arr).sum()))
    positive_count = max(1, int(actuals_arr.sum()))
    false_positive_rate = float((predictions_arr & ~actuals_arr).sum() / negative_count)
    false_negative_rate = float((~predictions_arr & actuals_arr).sum() / positive_count)

    return {
        "samples_evaluated": int(len(actuals_arr)),
        "actual_positive_count": int(actuals_arr.sum()),
        "predicted_positive_count": int(predictions_arr.sum()),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def evaluate_prediction_interval_coverage(
    df: pd.DataFrame,
    indicator: str = INDICATOR,
    min_history: int = 8,
    window: int = 8,
    z_value: float = 1.96,
) -> dict:
    values = df[indicator].astype(float)
    covered = []
    interval_widths = []

    for i in range(min_history, len(values)):
        train = values.iloc[max(0, i - window):i]
        mean = float(train.mean())
        std = float(train.std(ddof=1))
        if not np.isfinite(std) or std < 1e-6:
            std = 1e-6

        lower = mean - z_value * std
        upper = mean + z_value * std
        actual = float(values.iloc[i])
        covered.append(lower <= actual <= upper)
        interval_widths.append(upper - lower)

    return {
        "samples_evaluated": int(len(covered)),
        "prediction_interval_coverage": float(np.mean(covered)) if covered else 0.0,
        "mean_interval_width": float(np.mean(interval_widths)) if interval_widths else 0.0,
    }


def evaluate_sample_size_stability(sample_sizes=(8, 12, 20, 36), seeds=range(5)) -> dict:
    stability = {}
    for size in sample_sizes:
        fp_rates = []
        fn_rates = []
        coverages = []
        for seed in seeds:
            df, labels = generate_synthetic_history(n=size, seed=seed)
            alert_metrics = evaluate_alert_detection(df, labels, min_history=min(4, max(3, size // 3)))
            coverage_metrics = evaluate_prediction_interval_coverage(
                df,
                min_history=min(8, max(4, size // 2)),
                window=min(8, max(3, size // 2)),
            )
            fp_rates.append(alert_metrics["false_positive_rate"])
            fn_rates.append(alert_metrics["false_negative_rate"])
            coverages.append(coverage_metrics["prediction_interval_coverage"])

        stability[str(size)] = {
            "false_positive_rate_mean": float(np.mean(fp_rates)),
            "false_positive_rate_std": float(np.std(fp_rates)),
            "false_negative_rate_mean": float(np.mean(fn_rates)),
            "false_negative_rate_std": float(np.std(fn_rates)),
            "prediction_interval_coverage_mean": float(np.mean(coverages)),
            "prediction_interval_coverage_std": float(np.std(coverages)),
        }
    return stability


def generate_sparse_synthetic_history(n: int = 16, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """
    生成接近真实随访频率的稀疏数据：约每 90 天一次。

    标签不是“疾病诊断”，而是“后续出现了超过参考上限/可靠变化阈值的数据事件”，
    用于回测 sparse_longitudinal_engine 对可测量化验变化的提前提示能力。
    """

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="90D")
    values = 3.0 + rng.normal(0, 0.12, size=n)

    event_start = max(3, int(n * 0.70))
    if n >= 6:
        for idx in range(event_start, n):
            values[idx] += (idx - event_start + 1) * (1.1 + rng.normal(0, 0.05))

    labels = pd.Series(values > 5.0, index=dates, name="actual_data_event")
    df = pd.DataFrame(
        {
            "phase": ["稳定监控期"] * n,
            "report_uuid": [f"sparse-{seed}-{i}" for i in range(n)],
            INDICATOR: values,
        },
        index=dates,
    )
    return df, labels


def evaluate_sparse_engine_one_step(
    df: pd.DataFrame,
    labels: pd.Series,
    indicator: str = INDICATOR,
    min_history: int = 3,
    probability_threshold: float = 0.55,
) -> dict:
    predictions = []
    actuals = []
    covered = []
    interval_widths = []
    probability_records = []

    for i in range(min_history, len(df)):
        history = df.iloc[:i]
        actual_date = df.index[i]
        previous_date = history.index[-1]
        horizon_days = max(1, int((actual_date - previous_date).days))
        result = sparse_longitudinal_engine.analyze_indicator(
            history,
            indicator,
            horizon_days=(horizon_days,),
        )
        if result.get("status") != "ok" or not result.get("future_predictions"):
            continue

        pred = result["future_predictions"][0]
        probability = max(
            float(pred["prob_reliable_change"]),
            float(pred["prob_outside_reference"]),
        )
        predicted_event = (
            probability >= probability_threshold
        )
        actual_value = float(df[indicator].iloc[i])
        predictions.append(predicted_event)
        actuals.append(bool(labels.iloc[i]))
        covered.append(pred["lower_95"] <= actual_value <= pred["upper_95"])
        interval_widths.append(pred["upper_95"] - pred["lower_95"])
        probability_records.append(
            {
                "probability": probability,
                "actual": bool(labels.iloc[i]),
                "date": actual_date,
                "history_date": previous_date,
            }
        )

    predictions_arr = np.array(predictions, dtype=bool)
    actuals_arr = np.array(actuals, dtype=bool)

    negative_count = max(1, int((~actuals_arr).sum()))
    positive_count = max(1, int(actuals_arr.sum()))

    metrics = {
        "samples_evaluated": int(len(actuals_arr)),
        "actual_positive_count": int(actuals_arr.sum()),
        "predicted_positive_count": int(predictions_arr.sum()),
        "false_positive_rate": float((predictions_arr & ~actuals_arr).sum() / negative_count),
        "false_negative_rate": float((~predictions_arr & actuals_arr).sum() / positive_count),
        "prediction_interval_coverage": float(np.mean(covered)) if covered else 0.0,
        "mean_interval_width": float(np.mean(interval_widths)) if interval_widths else 0.0,
    }
    metrics.update(_probability_task_metrics(probability_records, probability_threshold))
    return metrics


def evaluate_sparse_engine_sample_size_stability(sample_sizes=(4, 6, 8, 12, 16), seeds=range(8)) -> dict:
    stability = {}
    for size in sample_sizes:
        fp_rates = []
        fn_rates = []
        coverages = []
        widths = []
        evaluated = []
        for seed in seeds:
            df, labels = generate_sparse_synthetic_history(n=size, seed=seed)
            metrics = evaluate_sparse_engine_one_step(
                df,
                labels,
                min_history=min(3, max(2, size // 2)),
            )
            fp_rates.append(metrics["false_positive_rate"])
            fn_rates.append(metrics["false_negative_rate"])
            coverages.append(metrics["prediction_interval_coverage"])
            widths.append(metrics["mean_interval_width"])
            evaluated.append(metrics["samples_evaluated"])

        stability[str(size)] = {
            "samples_evaluated_mean": float(np.mean(evaluated)),
            "false_positive_rate_mean": float(np.mean(fp_rates)),
            "false_positive_rate_std": float(np.std(fp_rates)),
            "false_negative_rate_mean": float(np.mean(fn_rates)),
            "false_negative_rate_std": float(np.std(fn_rates)),
            "prediction_interval_coverage_mean": float(np.mean(coverages)),
            "prediction_interval_coverage_std": float(np.std(coverages)),
            "mean_interval_width_mean": float(np.mean(widths)),
        }
    return stability


def run_sparse_engine_validation() -> dict:
    df, labels = generate_sparse_synthetic_history()
    return {
        "dataset": {
            "type": "synthetic_sparse_quarterly",
            "records": int(len(df)),
            "indicator": INDICATOR,
            "event_definition": "future lab value exceeds reference upper limit",
        },
        "one_step_prediction": evaluate_sparse_engine_one_step(df, labels),
        "sample_size_stability": evaluate_sparse_engine_sample_size_stability(),
    }


def evaluate_ensemble_engine_one_step(
    df: pd.DataFrame,
    labels: pd.Series,
    indicator: str = INDICATOR,
    min_history: int = 3,
    probability_threshold: float = 0.55,
) -> dict:
    covered = []
    interval_widths = []
    probability_records = []

    for i in range(min_history, len(df)):
        history = df.iloc[:i]
        actual_date = df.index[i]
        previous_date = history.index[-1]
        horizon_days = max(1, int((actual_date - previous_date).days))
        report = predictive_ensemble_engine.analyze_predictive_ensemble(
            history,
            horizon_days=(horizon_days,),
        )
        result = report.get("indicators", {}).get(indicator)
        if not result or result.get("status") != "ok" or not result.get("future_predictions"):
            continue

        pred = result["future_predictions"][0]
        probability = float(pred.get("ensemble_prob_reliable_or_outside", 0.0))
        actual_value = float(df[indicator].iloc[i])
        lower = float(pred.get("conformal_lower_95", pred["lower_95"]))
        upper = float(pred.get("conformal_upper_95", pred["upper_95"]))

        covered.append(lower <= actual_value <= upper)
        interval_widths.append(upper - lower)
        probability_records.append(
            {
                "probability": probability,
                "actual": bool(labels.iloc[i]),
                "date": actual_date,
                "history_date": previous_date,
            }
        )

    metrics = _probability_task_metrics(probability_records, probability_threshold)
    metrics["prediction_interval_coverage"] = float(np.mean(covered)) if covered else 0.0
    metrics["mean_interval_width"] = float(np.mean(interval_widths)) if interval_widths else 0.0
    return metrics


def evaluate_ensemble_engine_sample_size_stability(sample_sizes=(4, 6, 8, 12, 16), seeds=range(8)) -> dict:
    stability = {}
    for size in sample_sizes:
        fp_rates = []
        fn_rates = []
        briers = []
        coverages = []
        false_alarms = []
        sensitivities = []
        for seed in seeds:
            df, labels = generate_sparse_synthetic_history(n=size, seed=seed)
            metrics = evaluate_ensemble_engine_one_step(
                df,
                labels,
                min_history=min(3, max(2, size // 2)),
            )
            fp_rates.append(metrics["false_positive_rate"])
            fn_rates.append(metrics["false_negative_rate"])
            briers.append(metrics["brier_score"])
            coverages.append(metrics["prediction_interval_coverage"])
            false_alarms.append(metrics["false_alarms_per_year"])
            sensitivities.append(metrics["sensitivity_at_fixed_fpr"])

        stability[str(size)] = {
            "false_positive_rate_mean": float(np.mean(fp_rates)),
            "false_positive_rate_std": float(np.std(fp_rates)),
            "false_negative_rate_mean": float(np.mean(fn_rates)),
            "false_negative_rate_std": float(np.std(fn_rates)),
            "brier_score_mean": float(np.mean(briers)),
            "prediction_interval_coverage_mean": float(np.mean(coverages)),
            "false_alarms_per_year_mean": float(np.mean(false_alarms)),
            "sensitivity_at_fixed_fpr_mean": float(np.mean(sensitivities)),
        }
    return stability


def run_ensemble_engine_validation() -> dict:
    df, labels = generate_sparse_synthetic_history()
    return {
        "dataset": {
            "type": "synthetic_sparse_quarterly",
            "records": int(len(df)),
            "indicator": INDICATOR,
            "event_definition": "future lab value exceeds reference upper limit",
        },
        "one_step_prediction": evaluate_ensemble_engine_one_step(df, labels),
        "sample_size_stability": evaluate_ensemble_engine_sample_size_stability(),
    }


def _probability_task_metrics(records: list[dict], probability_threshold: float, fixed_fpr: float = 0.10) -> dict:
    if not records:
        return {
            "samples_evaluated": 0,
            "actual_positive_count": 0,
            "predicted_positive_count": 0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "brier_score": 0.0,
            "false_alarms_per_year": 0.0,
            "mean_lead_time_days": None,
            "sensitivity_at_fixed_fpr": 0.0,
            "fixed_fpr": fixed_fpr,
            "calibration_curve": [],
        }

    probabilities = np.asarray([item["probability"] for item in records], dtype=float)
    actuals = np.asarray([item["actual"] for item in records], dtype=bool)
    predictions = probabilities >= probability_threshold

    negative_count = int((~actuals).sum())
    positive_count = int(actuals.sum())
    false_positive_count = int((predictions & ~actuals).sum())
    false_negative_count = int((~predictions & actuals).sum())
    false_positive_rate = false_positive_count / negative_count if negative_count else 0.0
    false_negative_rate = false_negative_count / positive_count if positive_count else 0.0

    dates = pd.to_datetime([item["date"] for item in records])
    duration_years = max(1.0 / 365.25, (dates.max() - dates.min()).days / 365.25)
    false_alarms_per_year = false_positive_count / duration_years

    lead_times = [
        max(0, int((pd.Timestamp(item["date"]) - pd.Timestamp(item["history_date"])).days))
        for item, predicted in zip(records, predictions)
        if predicted and item["actual"]
    ]

    return {
        "samples_evaluated": int(len(records)),
        "actual_positive_count": positive_count,
        "predicted_positive_count": int(predictions.sum()),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "brier_score": float(np.mean((probabilities - actuals.astype(float)) ** 2)),
        "false_alarms_per_year": float(false_alarms_per_year),
        "mean_lead_time_days": float(np.mean(lead_times)) if lead_times else None,
        "sensitivity_at_fixed_fpr": float(_sensitivity_at_fixed_fpr(probabilities, actuals, fixed_fpr)),
        "fixed_fpr": fixed_fpr,
        "calibration_curve": _calibration_curve(probabilities, actuals),
    }


def _calibration_curve(probabilities: np.ndarray, actuals: np.ndarray, bins: int = 5) -> list[dict]:
    curve = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        if high >= 1.0:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            curve.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "count": 0,
                "mean_predicted": None,
                "observed_rate": None,
            })
            continue
        curve.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "count": int(mask.sum()),
            "mean_predicted": float(probabilities[mask].mean()),
            "observed_rate": float(actuals[mask].mean()),
        })
    return curve


def _sensitivity_at_fixed_fpr(probabilities: np.ndarray, actuals: np.ndarray, fixed_fpr: float) -> float:
    positive_count = int(actuals.sum())
    negative_count = int((~actuals).sum())
    if positive_count == 0:
        return 0.0
    thresholds = sorted(set(probabilities.tolist() + [0.0, 1.0]), reverse=True)
    best_sensitivity = 0.0
    for threshold in thresholds:
        predictions = probabilities >= threshold
        fp = int((predictions & ~actuals).sum())
        tp = int((predictions & actuals).sum())
        fpr = fp / negative_count if negative_count else 0.0
        if fpr <= fixed_fpr:
            best_sensitivity = max(best_sensitivity, tp / positive_count)
    return float(best_sensitivity)


def run_validation() -> dict:
    df, labels = generate_synthetic_history()
    return {
        "dataset": {
            "type": "synthetic",
            "records": int(len(df)),
            "indicator": INDICATOR,
        },
        "alert_detection": evaluate_alert_detection(df, labels),
        "prediction_interval": evaluate_prediction_interval_coverage(df),
        "sample_size_stability": evaluate_sample_size_stability(),
        "sparse_longitudinal_engine": run_sparse_engine_validation(),
        "predictive_ensemble_engine": run_ensemble_engine_validation(),
    }


def main() -> None:
    print(json.dumps(run_validation(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
