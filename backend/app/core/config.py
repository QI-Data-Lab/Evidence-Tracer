from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend_dir: Path
    data_dir: Path
    preview_dir: Path
    db_path: Path
    api_title: str
    api_version: str
    ollama_base_url: str
    ollama_model: str
    ollama_num_predict: int
    agent_max_graph_steps: int
    agent_max_tool_calls: int
    agent_max_evidence: int
    agent_max_tasks: int
    phoenix_enabled: bool
    phoenix_project_name: str
    phoenix_collector_endpoint: str


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip() or default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend_dir = Path(__file__).resolve().parents[2]
    data_dir = backend_dir / "data"
    preview_dir = data_dir / "page_previews"
    db_path = data_dir / "catalog.sqlite3"

    data_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        backend_dir=backend_dir,
        data_dir=data_dir,
        preview_dir=preview_dir,
        db_path=db_path,
        api_title="Evidence Tracer Backend",
        api_version="0.1.0",
        ollama_base_url="http://127.0.0.1:8880",
        ollama_model="qwen3:latest",
        ollama_num_predict=1024,
        agent_max_graph_steps=800,
        agent_max_tool_calls=250,
        agent_max_evidence=250,
        agent_max_tasks=250,
        phoenix_enabled=_env_bool("PHOENIX_ENABLED", True),
        phoenix_project_name=_env_str("PHOENIX_PROJECT_NAME", "evidence-tracer"),
        phoenix_collector_endpoint=_env_str("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"),
    )
