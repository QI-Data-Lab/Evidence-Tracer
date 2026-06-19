from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.database import connection, init_database
from app.services.evidence_service import EvidenceService
from app.utils.common import json_dumps


OLLAMA_BASE_URL = "http://127.0.0.1:8880"
OLLAMA_MODEL = "qwen3:latest"


class RealOllamaAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "catalog.sqlite3"
        self.settings = SimpleNamespace(
            db_path=self.db_path,
            data_dir=self.temp_path / "data",
            preview_dir=self.temp_path / "data" / "page_previews",
            ollama_base_url=OLLAMA_BASE_URL,
            ollama_model=OLLAMA_MODEL,
            agent_max_graph_steps=10,
            agent_max_tool_calls=10,
            agent_max_evidence=8,
            agent_max_tasks=8,
        )
        self.settings.data_dir.mkdir(parents=True)
        self.settings.preview_dir.mkdir(parents=True)

        self.database_patch = patch("app.core.database.get_settings", return_value=self.settings)
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        init_database()
        self._seed_catalog()

    def _seed_catalog(self) -> None:
        root = str(self.temp_path / "docs")
        now = "2026-05-08T00:00:00+00:00"
        with connection() as conn:
            conn.execute("INSERT INTO settings(key, value, updated_at) VALUES('root_path', ?, ?)", (root, now))
            conn.execute(
                """
                INSERT INTO documents(
                    id, root_path, relative_path, source_path, file_name, status, page_count,
                    file_size, modified_time, fingerprint, summary_json, metadata_json,
                    error_message, last_scanned_at, last_processed_at, created_at, updated_at
                )
                VALUES(1, ?, 'CASTOR.pdf', ?, 'CASTOR.pdf', 'done', 2,
                    100, 1.0, 'castor', '{}', '{}', NULL, ?, ?, ?, ?)
                """,
                (root, f"{root}/CASTOR.pdf", now, now, now, now),
            )
            conn.execute("INSERT INTO document_pages(document_id, page_number, width, height, rotation) VALUES(1, 1, 600, 800, 0)")
            conn.execute("INSERT INTO document_pages(document_id, page_number, width, height, rotation) VALUES(1, 2, 600, 800, 0)")
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(1, 1, '#/texts/0', NULL, 0, 1, 'section_header', 'TextItem',
                    '2.1.1 Brennelemente', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["2.1.1 Brennelemente"]),
                    json_dumps([1]),
                    json_dumps({"l": 20, "t": 40, "r": 260, "b": 65}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(2, 1, '#/texts/1', '#/texts/0', 1, NULL, 'text', 'TextItem',
                    'Der Behälter umfasst bis zu 17 Brennelemente.', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["2.1.1 Brennelemente"]),
                    json_dumps([1]),
                    json_dumps({"l": 20, "t": 80, "r": 520, "b": 120}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(10, 1, 'hierarchical', 0, 'text',
                    'Der Behälter umfasst bis zu 17 Brennelemente.',
                    '2.1.1 Brennelemente\n\nDer Behälter umfasst bis zu 17 Brennelemente.',
                    ?, ?, ?, '{}')
                """,
                (json_dumps(["2.1.1 Brennelemente"]), json_dumps([1]), json_dumps(["#/texts/1"])),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(10, 1, 1, ?)",
                (json_dumps({"l": 20, "t": 80, "r": 520, "b": 120}),),
            )

    def test_ollama_preflight_requires_qwen3_latest(self) -> None:
        try:
            with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            self.fail(f"Ollama is not reachable at {OLLAMA_BASE_URL}: {exc}")

        model_names = {model.get("name") for model in payload.get("models", [])}
        self.assertIn(OLLAMA_MODEL, model_names, f"Run `ollama pull {OLLAMA_MODEL}` before testing.")

    def test_evidence_run_uses_real_langgraph_ollama_agent(self) -> None:
        service = EvidenceService(settings=self.settings)
        run = service.run_agentic_retrieval(query="how many brennelemente", document_ids=[1], max_tasks=6)

        self.assertEqual(run["agent_backend"], "langgraph_ollama")
        self.assertTrue(run["tasks"], "Planner should create real tasks.")
        self.assertIn("planner", {event["node"] for event in run["trace_events"]})
        self.assertIn("actor", {event["node"] for event in run["trace_events"]})
        self.assertIn("curator", {event["node"] for event in run["trace_events"]})
        self.assertIn("doc:1:page:1:chunk:10", {evidence["source_key"] for evidence in run["evidence"]})

        trace_text = Path(run["artifacts"]["trace_text_path"]).read_text(encoding="utf-8")
        self.assertIn("Agent Backend: langgraph_ollama", trace_text)
        self.assertIn("Step-by-Step:", trace_text)
        self.assertIn("Model input:", trace_text)
        self.assertTrue(Path(run["artifacts"]["model_inputs_text_path"]).exists())
        self.assertTrue(Path(run["artifacts"]["model_outputs_text_path"]).exists())


if __name__ == "__main__":
    unittest.main()
