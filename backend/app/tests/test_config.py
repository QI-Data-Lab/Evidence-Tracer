from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.core.config import get_settings


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

    def test_phoenix_defaults_target_local_server(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()

        self.assertTrue(settings.phoenix_enabled)
        self.assertEqual(settings.phoenix_project_name, "evidence-tracer")
        self.assertEqual(settings.phoenix_collector_endpoint, "http://127.0.0.1:6006/v1/traces")

    def test_phoenix_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PHOENIX_ENABLED": "false",
                "PHOENIX_PROJECT_NAME": "local-debug",
                "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:7007/v1/traces",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertFalse(settings.phoenix_enabled)
        self.assertEqual(settings.phoenix_project_name, "local-debug")
        self.assertEqual(settings.phoenix_collector_endpoint, "http://127.0.0.1:7007/v1/traces")
