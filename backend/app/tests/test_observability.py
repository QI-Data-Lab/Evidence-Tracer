from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core import observability


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        observability.reset_observability_for_tests()

    def tearDown(self) -> None:
        observability.reset_observability_for_tests()

    def test_disabled_settings_skip_registration(self) -> None:
        settings = SimpleNamespace(
            phoenix_enabled=False,
            phoenix_project_name="evidence-tracer",
            phoenix_collector_endpoint="http://127.0.0.1:6006/v1/traces",
        )

        with patch("app.core.observability.register") as register:
            observability.init_observability(settings)

        register.assert_not_called()

    def test_registers_phoenix_once(self) -> None:
        settings = SimpleNamespace(
            phoenix_enabled=True,
            phoenix_project_name="evidence-tracer",
            phoenix_collector_endpoint="http://127.0.0.1:6006/v1/traces",
        )

        with patch("app.core.observability.register", return_value=Mock()) as register:
            observability.init_observability(settings)
            observability.init_observability(settings)

        register.assert_called_once_with(
            project_name="evidence-tracer",
            endpoint="http://127.0.0.1:6006/v1/traces",
            protocol="http/protobuf",
            auto_instrument=True,
        )

    def test_registration_failure_is_logged_and_swallowed(self) -> None:
        settings = SimpleNamespace(
            phoenix_enabled=True,
            phoenix_project_name="evidence-tracer",
            phoenix_collector_endpoint="http://127.0.0.1:6006/v1/traces",
        )

        with patch("app.core.observability.register", side_effect=RuntimeError("boom")):
            with self.assertLogs("app.core.observability", level="WARNING") as logs:
                observability.init_observability(settings)

        self.assertIn("Phoenix tracing could not be initialized", "\n".join(logs.output))


class MainObservabilityTests(unittest.TestCase):
    def test_main_initializes_observability_with_settings(self) -> None:
        sys.modules.pop("app.main", None)
        self.addCleanup(sys.modules.pop, "app.main", None)

        with patch("app.core.observability.init_observability") as init_observability:
            main = importlib.import_module("app.main")

        init_observability.assert_called_once_with(main.settings)
