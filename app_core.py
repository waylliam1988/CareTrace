"""
Core workflow helpers used by the Streamlit page layer.

These functions coordinate controller calls without depending on Streamlit UI
objects.  The page can still own rendering and widgets, while tests can execute
these workflows with a DictStateStore.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import risk_engine


def restore_mogp_for_patient(health_controller, patient_id: int, patient_data_raw: pd.DataFrame, state) -> dict[str, Any]:
    """
    Restore MOGP predictions from state or database and refresh diagnostics.

    Returns a small status dictionary.  If ``force_refresh`` is true, the caller
    should ask the controller to rebuild processed data so restored MOGP features
    are merged into the feature frame.
    """

    load_key = f"mogp_loaded_from_db_{patient_id}"
    state.setdefault(load_key, False)

    if not state.get(load_key):
        mogp_cache = health_controller.get_mogp_predictions()
        state.set(load_key, True)

        if mogp_cache:
            _refresh_mogp_diagnostic(patient_data_raw, state)
            return {
                "restored": True,
                "force_refresh": True,
                "message": "从数据库恢复 MOGP 结果",
            }

        state.set("mogp_results", {})
        state.set("mogp_diagnostic_info", {})
        return {
            "restored": False,
            "force_refresh": False,
            "message": "数据库中暂无 MOGP 结果",
        }

    if state.get("mogp_results") is None:
        mogp_cache = health_controller.get_mogp_predictions()
        if mogp_cache:
            _refresh_mogp_diagnostic(patient_data_raw, state)
            return {
                "restored": True,
                "force_refresh": True,
                "message": "从数据库恢复丢失的 MOGP 状态",
            }

        state.set("mogp_results", {})
        state.set("mogp_diagnostic_info", {})
        return {
            "restored": False,
            "force_refresh": False,
            "message": "状态和数据库均无 MOGP 结果",
        }

    return {
        "restored": False,
        "force_refresh": False,
        "message": "MOGP 状态已存在",
    }


def train_models(health_controller) -> dict[str, Any]:
    """Run model training through the controller."""

    return health_controller.train_models()


def assess_current_risk(health_controller) -> dict[str, Any]:
    """Run current risk/data observation through the controller."""

    return health_controller.assess_current_risk()


def _refresh_mogp_diagnostic(patient_data_raw: pd.DataFrame, state) -> None:
    if patient_data_raw.empty or "phase" not in patient_data_raw.columns:
        state.set("mogp_diagnostic_info", {})
        return

    surveillance_df_raw = patient_data_raw[
        patient_data_raw["phase"] == "稳定监控期"
    ].copy()

    if surveillance_df_raw.empty:
        state.set("mogp_diagnostic_info", {})
        return

    _, diagnostic_info = risk_engine.select_mogp_indicators(
        surveillance_df_raw,
        max_indicators=4,
    )
    state.set("mogp_diagnostic_info", diagnostic_info)
