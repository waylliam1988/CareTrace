# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

import unittest

import app_state
from _test_utils import isolated_config


class ControllerStateTests(unittest.TestCase):
    def test_controller_clears_mogp_state_without_streamlit_session(self):
        import controller

        with isolated_config():
            state = app_state.DictStateStore(
                {
                    "mogp_results": {"x": {}},
                    "mogp_last_updated": "now",
                    "mogp_target_indicators": ["x"],
                    "mogp_diagnostic_info": {"selected": ["x"]},
                    "processed_mogp_version": "now",
                }
            )
            health_controller = controller.HealthController(1, state=state)
            health_controller.clear_cache()

            self.assertIsNone(state.get("mogp_results"))
            self.assertIsNone(state.get("mogp_last_updated"))
            self.assertIsNone(state.get("processed_mogp_version"))


if __name__ == "__main__":
    unittest.main()
