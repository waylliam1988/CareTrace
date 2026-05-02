"""
State storage adapters used by the page layer and the core controller.

The controller should not depend directly on Streamlit's session_state.  This
module keeps that dependency at the boundary so tests and batch scripts can use
a plain dictionary-backed store.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


class DictStateStore:
    """Small dictionary-backed state store for tests and non-Streamlit scripts."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self._data.setdefault(key, default)

    def update(self, values: dict[str, Any]) -> None:
        self._data.update(values)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._data.pop(key, default)

    def contains(self, key: str) -> bool:
        return key in self._data

    def clear_mogp(self) -> None:
        for key in MOGP_STATE_KEYS:
            self.set(key, None)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


class StreamlitStateStore:
    """Adapter around st.session_state, imported lazily to keep core code testable."""

    @property
    def _state(self) -> MutableMapping[str, Any]:
        import streamlit as st

        return st.session_state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        state = self._state
        if key not in state:
            state[key] = default
        return state[key]

    def update(self, values: dict[str, Any]) -> None:
        self._state.update(values)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._state.pop(key, default)

    def contains(self, key: str) -> bool:
        return key in self._state

    def clear_mogp(self) -> None:
        for key in MOGP_STATE_KEYS:
            self.set(key, None)


MOGP_STATE_KEYS = (
    "mogp_results",
    "mogp_last_updated",
    "mogp_target_indicators",
    "mogp_diagnostic_info",
    "processed_mogp_version",
)


def initialize_app_state(state: DictStateStore | StreamlitStateStore) -> None:
    """Initialize keys shared by the page and controller layers."""

    defaults = {
        "data_changed": False,
        "cache_version": 0,
        "mogp_results": None,
        "mogp_last_updated": None,
        "mogp_target_indicators": None,
        "mogp_diagnostic_info": None,
        "boost_df": None,
        "dynamic_params": None,
        "processed_mogp_version": None,
        "staged_items": {},
        "editor_reset_counter": 0,
    }
    for key, value in defaults.items():
        state.setdefault(key, value)
