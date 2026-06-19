from __future__ import annotations

import logging
from typing import Any

try:
    from phoenix.otel import register
except ImportError:
    register = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_initialized = False


def init_observability(settings: Any) -> None:
    global _initialized
    if _initialized or not getattr(settings, "phoenix_enabled", False):
        return
    if register is None:
        logger.warning("Phoenix tracing could not be initialized: phoenix.otel is not installed.")
        return
    try:
        register(
            project_name=str(settings.phoenix_project_name),
            endpoint=str(settings.phoenix_collector_endpoint),
            protocol="http/protobuf",
            auto_instrument=True,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary around optional telemetry
        logger.warning("Phoenix tracing could not be initialized: %s", exc)
        return
    _initialized = True


def reset_observability_for_tests() -> None:
    global _initialized
    _initialized = False
