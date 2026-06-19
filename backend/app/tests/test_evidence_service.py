from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import connection, init_database
from app.utils.common import json_dumps


class EvidenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "catalog.sqlite3"
        self.settings = SimpleNamespace(
            db_path=self.db_path,
            data_dir=self.temp_path / "data",
            preview_dir=self.temp_path / "data" / "page_previews",
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
        now = "2026-05-07T00:00:00+00:00"
        with connection() as conn:
            conn.execute(
                "INSERT INTO settings(key, value, updated_at) VALUES('root_path', ?, ?)",
                (root, now),
            )
            conn.execute(
                """
                INSERT INTO documents(
                    id, root_path, relative_path, source_path, file_name, status, page_count,
                    file_size, modified_time, fingerprint, summary_json, metadata_json,
                    error_message, last_scanned_at, last_processed_at, created_at, updated_at
                )
                VALUES(1, ?, 'SPEC-A.pdf', ?, 'SPEC-A.pdf', 'done', 2,
                    100, 1.0, 'a', '{}', '{}', NULL, ?, ?, ?, ?)
                """,
                (root, f"{root}/SPEC-A.pdf", now, now, now, now),
            )
            conn.execute(
                """
                INSERT INTO documents(
                    id, root_path, relative_path, source_path, file_name, status, page_count,
                    file_size, modified_time, fingerprint, summary_json, metadata_json,
                    error_message, last_scanned_at, last_processed_at, created_at, updated_at
                )
                VALUES(2, ?, 'SPEC-B.pdf', ?, 'SPEC-B.pdf', 'done', 1,
                    100, 1.0, 'b', '{}', '{}', NULL, ?, ?, ?, ?)
                """,
                (root, f"{root}/SPEC-B.pdf", now, now, now, now),
            )
            conn.execute(
                "INSERT INTO document_pages(document_id, page_number, width, height, rotation) VALUES(1, 1, 600, 800, 0)"
            )
            conn.execute(
                "INSERT INTO document_pages(document_id, page_number, width, height, rotation) VALUES(1, 2, 600, 800, 0)"
            )
            conn.execute(
                "INSERT INTO document_pages(document_id, page_number, width, height, rotation) VALUES(2, 1, 600, 800, 0)"
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(1, 1, '#/texts/0', NULL, 0, 1, 'section_header', 'TextItem',
                    '1 Scope', NULL, ?, ?, '[]', ?, '{}')
                """,
                (json_dumps(["1 Scope"]), json_dumps([1]), json_dumps({"l": 10, "t": 20, "r": 200, "b": 40})),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(2, 1, '#/texts/1', NULL, 1, 2, 'section_header', 'TextItem',
                    '3.2 Thermal Stress', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["3.2 Thermal Stress"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 80, "r": 260, "b": 105}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(3, 1, '#/texts/2', NULL, 2, 2, 'section_header', 'TextItem',
                    '4 Brennelemente', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Brennelemente"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 300, "r": 260, "b": 325}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(4, 1, '#/texts/3', '#/texts/2', 3, NULL, 'text', 'TextItem',
                    'Die Anzahl betraegt 12.', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Brennelemente"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 330, "r": 520, "b": 360}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(5, 1, '#/texts/4', NULL, 4, NULL, 'text', 'TextItem',
                    '[3] SPEC-B.pdf Validation Limits.', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["Literaturverzeichnis"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 520, "r": 520, "b": 550}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(6, 1, '#/tables/0', '#/texts/2', 5, NULL, 'table', 'TableItem',
                    NULL, NULL, ?, ?, '[]', ?, ?)
                """,
                (
                    json_dumps(["4 Brennelemente"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 370, "r": 520, "b": 440}),
                    json_dumps({"table_text": "Parameter | Wert\nAnzahl der Brennelemente | max. 12"}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(10, 1, 'hierarchical', 0, 'text', 'Scope overview.',
                    '1 Scope\n\nScope overview.', ?, ?, ?, '{}')
                """,
                (json_dumps(["1 Scope"]), json_dumps([1]), json_dumps(["#/texts/0"])),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(11, 1, 'hierarchical', 1, 'text',
                    'Thermal stress shall be checked using SPEC-B.pdf validation limits.',
                    '3.2 Thermal Stress\n\nThermal stress shall be checked using SPEC-B.pdf validation limits.',
                    ?, ?, ?, '{}')
                """,
                (json_dumps(["3.2 Thermal Stress"]), json_dumps([2]), json_dumps(["#/texts/1"])),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(12, 1, 'hierarchical', 2, 'text',
                    'Die Anzahl betraegt 12.',
                    'Die Anzahl betraegt 12.',
                    ?, ?, ?, '{}')
                """,
                (json_dumps(["4 Brennelemente"]), json_dumps([2]), json_dumps(["#/texts/2"])),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(13, 1, 'hierarchical', 3, 'text',
                    'm2001257',
                    '4 Brennelemente\n\nm2001257',
                    ?, ?, ?, '{}')
                """,
                (json_dumps(["4 Brennelemente"]), json_dumps([2]), json_dumps(["#/texts/2"])),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(20, 2, 'hierarchical', 0, 'text',
                    'SPEC-B contains the referenced validation limits for thermal cycling.',
                    'Validation Limits\n\nSPEC-B contains the referenced validation limits for thermal cycling.',
                    ?, ?, '[]', '{}')
                """,
                (json_dumps(["Validation Limits"]), json_dumps([1])),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(10, 1, 1, ?)",
                (json_dumps({"l": 10, "t": 50, "r": 300, "b": 130}),),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(11, 1, 2, ?)",
                (json_dumps({"l": 20, "t": 120, "r": 520, "b": 260}),),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(12, 1, 2, ?)",
                (json_dumps({"l": 20, "t": 330, "r": 520, "b": 390}),),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(13, 1, 2, ?)",
                (json_dumps({"l": 20, "t": 400, "r": 520, "b": 430}),),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(20, 2, 1, ?)",
                (json_dumps({"l": 40, "t": 90, "r": 500, "b": 210}),),
            )

    def test_readable_toc_returns_navigable_section_entries(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        toc = service.get_readable_toc(1)

        self.assertEqual([entry["title"] for entry in toc["entries"]], ["1 Scope", "3.2 Thermal Stress", "4 Brennelemente"])
        self.assertEqual(toc["entries"][1]["source_locator"]["page_number"], 2)
        self.assertEqual(toc["entries"][1]["source_key"], "doc:1:page:2:item:2")

    def test_section_navigation_returns_ordered_chunks_with_source_keys(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        section = service.navigate_to_section(document_id=1, section_query="3.2")

        self.assertEqual(section["matched_section"]["title"], "3.2 Thermal Stress")
        self.assertEqual([chunk["source_key"] for chunk in section["chunks"]], ["doc:1:page:2:chunk:11"])

    def test_section_navigation_returns_merged_body_text_and_ordered_items(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        section = service.navigate_to_section(document_id=1, section_query="4 Brennelemente")

        self.assertEqual(section["matched_section"]["title"], "4 Brennelemente")
        self.assertIn("Die Anzahl betraegt 12.", section["merged_text"])
        self.assertIn("Anzahl der Brennelemente | max. 12", section["merged_text"])
        self.assertEqual(
            [item["source_key"] for item in section["ordered_items"]],
            ["doc:1:page:2:item:4", "doc:1:page:2:item:6"],
        )

    def test_reference_resolver_maps_numbered_citation_to_loaded_document(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.resolve_citation(document_id=1, citation="[3]")

        self.assertEqual(result["citation"], "[3]")
        self.assertEqual(result["matched_document"]["id"], 2)
        self.assertEqual(result["reference"]["source_key"], "doc:1:page:2:item:5")
        self.assertIn("SPEC-B.pdf", result["reference"]["display_text"])

    def test_list_tables_figures_returns_structured_items_for_section(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.list_tables_figures(document_id=1, section_query="4 Brennelemente")

        self.assertEqual([item["source_key"] for item in result["items"]], ["doc:1:page:2:item:6"])
        self.assertEqual(result["items"][0]["result_type"], "item")
        self.assertEqual(result["items"][0]["label"], "table")
        self.assertIn("max. 12", result["items"][0]["display_text"])

    def test_list_tables_figures_resolves_section_referenced_table_from_toc_page(self) -> None:
        self._seed_misgrouped_table_case()
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.list_tables_figures(document_id=1, section_query="4.2 Nachzerfallsleistung")

        self.assertEqual([item["source_key"] for item in result["items"]], ["doc:1:page:2:item:31"])
        self.assertIn("Tab.6 Maximale Nachzerfallsleistung", result["items"][0]["display_text"])
        self.assertIn("5,597", result["items"][0]["display_text"])

    def test_section_navigation_includes_section_referenced_table_from_toc_page(self) -> None:
        self._seed_misgrouped_table_case()
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.navigate_to_section(document_id=1, section_query="4.2 Nachzerfallsleistung")

        self.assertIn("doc:1:page:2:item:31", [item["source_key"] for item in result["ordered_items"]])
        self.assertIn("Tab.6 Maximale Nachzerfallsleistung", result["merged_text"])
        self.assertIn("5,597", result["merged_text"])

    def test_section_navigation_includes_misnested_numbered_subsection_body(self) -> None:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(40, 1, '#/texts/section-63', NULL, 20, 1, 'section_header', 'TextItem',
                    '6.3 Berechnungsergebnisse', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["6 Temperaturen", "6.3 Berechnungsergebnisse"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 80, "r": 420, "b": 105}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(41, 1, '#/texts/section-631', NULL, 21, 1, 'section_header', 'TextItem',
                    '6.3.1 Axialmodell', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["6 Temperaturen", "6.3.1 Axialmodell"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 120, "r": 420, "b": 145}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(42, 1, '#/texts/section-631-body', NULL, 22, NULL, 'text', 'TextItem',
                    'Die maximale Temperatur betraegt 252 C.', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["6 Temperaturen", "6.3.1 Axialmodell"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 150, "r": 520, "b": 180}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(42, 1, 'hierarchical', 4, 'text',
                    'Die maximale Temperatur betraegt 252 C.',
                    '6.3.1 Axialmodell\n\nDie maximale Temperatur betraegt 252 C.',
                    ?, ?, ?, '{}')
                """,
                (
                    json_dumps(["6 Temperaturen", "6.3.1 Axialmodell"]),
                    json_dumps([2]),
                    json_dumps(["#/texts/section-631-body"]),
                ),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(42, 1, 2, ?)",
                (json_dumps({"l": 20, "t": 150, "r": 520, "b": 180}),),
            )

        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.navigate_to_section(document_id=1, section_query="6.3 Berechnungsergebnisse")

        self.assertIn("Die maximale Temperatur betraegt 252 C.", result["merged_text"])
        self.assertEqual([chunk["source_key"] for chunk in result["chunks"]], ["doc:1:page:2:chunk:42"])

    def test_table_caption_navigation_does_not_pull_ambiguous_toc_section_chunks(self) -> None:
        self._seed_misgrouped_table_case()
        self._seed_ambiguous_reference_section()
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.navigate_to_section(
            document_id=1,
            section_query="Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter",
        )

        self.assertIsNone(result["matched_section"])
        self.assertIn("doc:1:page:2:item:31", [item["source_key"] for item in result["ordered_items"]])
        self.assertEqual(result["chunks"], [])
        self.assertNotIn("Aktivitätsinventar", result["merged_text"])

    def test_section_navigation_keeps_caption_enrichment_for_direct_table_items(self) -> None:
        with connection() as conn:
            conn.execute(
                """
                UPDATE document_items
                SET text = 'Die Anzahl betraegt 12 (siehe Tabelle 1).'
                WHERE id = 4
                """
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(7, 1, '#/texts/table-caption', NULL, 6, 3, 'section_header', 'TextItem',
                    'Tabelle 1: Brennelementanzahl', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Brennelemente", "Tabelle 1: Brennelementanzahl"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 445, "r": 520, "b": 465}),
                ),
            )

        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        table = service.list_tables_figures(document_id=1, section_query="4 Brennelemente")["items"][0]
        section = service.navigate_to_section(document_id=1, section_query="4 Brennelemente")
        section_table = next(item for item in section["ordered_items"] if item["source_key"] == "doc:1:page:2:item:6")

        self.assertIn("Tabelle 1: Brennelementanzahl", table["display_text"])
        self.assertIn("Tabelle 1: Brennelementanzahl", section_table["display_text"])
        self.assertIn("Anzahl der Brennelemente | max. 12", section_table["display_text"])

    def test_inspect_item_returns_table_payload_and_locator(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        result = service.inspect_item(document_id=1, item_id=6)

        self.assertEqual(result["item"]["source_key"], "doc:1:page:2:item:6")
        self.assertEqual(result["item"]["payload"]["table_text"], "Parameter | Wert\nAnzahl der Brennelemente | max. 12")
        self.assertEqual(result["item"]["source_locator"]["item_id"], 6)

    def test_keyword_search_returns_source_linked_results(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        results = service.search_keywords(query="thermal stress", limit=3)

        self.assertEqual(results["results"][0]["source_key"], "doc:1:page:2:chunk:11")
        self.assertIn("keyword:thermal", results["results"][0]["match_reasons"])
        self.assertEqual(results["results"][0]["source_locator"]["chunk_id"], 11)

    def test_keyword_search_handles_german_inflection_without_stopword_noise(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        results = service.search_keywords(query="Stimmt Brennelementen", document_ids=[1], limit=3)

        self.assertEqual(results["results"][0]["source_key"], "doc:1:page:2:chunk:12")
        self.assertIn("keyword:brennelementen", results["results"][0]["match_reasons"])

    def test_keyword_search_ranks_body_hits_above_header_only_hits(self) -> None:
        with connection() as conn:
            conn.execute(
                """
                UPDATE document_chunks
                SET text = 'Thermal Stress',
                    contextualized_text = 'Contents\n\nThermal Stress',
                    section_path_json = ?
                WHERE id = 10
                """,
                (json_dumps(["Contents"]),),
            )
            conn.execute(
                """
                UPDATE document_chunks
                SET text = 'Thermal stress thermal stress limits govern the calculation.',
                    contextualized_text = '3.2 Thermal Stress\n\nThermal stress thermal stress limits govern the calculation.'
                WHERE id = 11
                """
            )

        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        results = service.search_keywords(query="thermal stress", document_ids=[1], limit=3)

        self.assertEqual(results["results"][0]["source_key"], "doc:1:page:2:chunk:11")
        self.assertTrue(results["coverage"]["search_index_used"])

    def test_evidence_routes_expose_run_and_toc(self) -> None:
        with patch("app.services.evidence_service.get_settings", return_value=self.settings):
            from app.api.routes.evidence import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            toc_response = client.get("/evidence/documents/1/toc")
            self.assertEqual(toc_response.status_code, 200)
            self.assertEqual(toc_response.json()["entries"][1]["title"], "3.2 Thermal Stress")

    def test_stream_events_are_encoded_as_single_ndjson_lines(self) -> None:
        from app.services.evidence_service import encode_stream_event

        line = encode_stream_event({"type": "status", "label": "Curator reviewing 2 candidate(s)..."})

        self.assertEqual(line, '{"type": "status", "label": "Curator reviewing 2 candidate(s)..."}\n')

    def test_write_artifacts_creates_readable_step_trace_and_model_trace_files(self) -> None:
        from app.services.evidence_service import EvidenceService

        service = EvidenceService(settings=self.settings)
        artifacts = service._write_artifacts(
            run_id="run-1",
            query="compare heat values",
            trace_lines=[],
            tool_result_lines=["Tool Results For Query: compare heat values", "Run: run-1", "", "[S002] T1 search_keywords", "Results: 1"],
            model_input_lines=["Model Inputs For Query: compare heat values", "Run: run-1", "", "[S001] planner PlannerDecision input"],
            model_output_lines=["Model Outputs For Query: compare heat values", "Run: run-1", "", "[S001] planner PlannerDecision output"],
            trace_graph={"nodes": [], "edges": []},
            trace_events=[
                {
                    "step_id": "S001",
                    "node": "planner",
                    "message": "No rationale provided by model.",
                    "data": {
                        "tasks": [
                            {"type": "search_keywords", "reason": "No reason provided by model.", "params": {"query": "heat"}},
                        ]
                    },
                },
                {
                    "step_id": "S002",
                    "node": "actor",
                    "message": "Run the search.",
                    "data": {
                        "task": {"id": "T1", "type": "search_keywords", "status": "done", "observation_count": 1},
                        "tool": {"tool_name": "search_keywords", "tool_args": {"query": "heat"}},
                    },
                },
                {
                    "step_id": "S003",
                    "node": "curator",
                    "message": "Accept direct evidence and request citation follow-up.",
                    "data": {"accepted_source_keys": ["doc:1:page:2:chunk:11"], "planner_request": "Resolve [3]."},
                },
            ],
            tasks=[
                {
                    "id": "T1",
                    "type": "search_keywords",
                    "status": "done",
                    "reason": "No reason provided by model.",
                    "params": {"query": "heat"},
                },
            ],
            evidence=[
                {
                    "id": "E1",
                    "source_key": "doc:1:page:2:chunk:11",
                    "found_by_tasks": ["T1"],
                }
            ],
            stop_reason="complete",
        )

        trace_text = Path(artifacts["trace_text_path"]).read_text(encoding="utf-8")
        self.assertIn("Task List:", trace_text)
        self.assertIn("T1 search_keywords [done]", trace_text)
        self.assertNotIn("No rationale provided by model", trace_text)
        self.assertNotIn("No reason provided by model", trace_text)
        self.assertIn("[S001] planner", trace_text)
        self.assertIn("Model input: run-1-model-inputs.txt#S001", trace_text)
        self.assertIn("Tool results: run-1-tool-results.txt#S002", trace_text)
        self.assertIn("Curator selected evidence: doc:1:page:2:chunk:11", trace_text)
        self.assertIn("Planner request: Resolve [3].", trace_text)
        self.assertTrue(Path(artifacts["model_inputs_text_path"]).exists())
        self.assertTrue(Path(artifacts["model_outputs_text_path"]).exists())

    def _seed_misgrouped_table_case(self) -> None:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(28, 1, '#/tables/toc', NULL, 20, NULL, 'document_index', 'TableItem',
                    NULL, NULL, ?, ?, '[]', NULL, ?)
                """,
                (
                    json_dumps(["Tabellenverzeichnis"]),
                    json_dumps([1]),
                    json_dumps(
                        {
                            "table_cells": [
                                {"text": "Tab.6", "row": 0, "col": 0},
                                {"text": "Maximale Nachzerfallsleistung und Referenzwert für den Behälter", "row": 0, "col": 1},
                                {"text": "2", "row": 0, "col": 2},
                            ],
                            "table_text": "Tab.6 | Maximale Nachzerfallsleistung und Referenzwert für den Behälter | 2",
                        }
                    ),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(29, 1, '#/texts/section-42', NULL, 21, 2, 'section_header', 'TextItem',
                    '4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Ergebnisse", "4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 80, "r": 420, "b": 105}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(30, 1, '#/texts/section-42-body', NULL, 22, NULL, 'text', 'TextItem',
                    'Die maximale Nachzerfallsleistung ist in Tab. 6 dargestellt.', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Ergebnisse", "4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert"]),
                    json_dumps([2]),
                    json_dumps({"l": 20, "t": 120, "r": 520, "b": 150}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(31, 1, '#/tables/tab-6', NULL, 40, NULL, 'table', 'TableItem',
                    NULL, NULL, ?, ?, '[]', ?, ?)
                """,
                (
                    json_dumps(["4 Ergebnisse", "4.5 Spaltgasinventar", "Brennelementpositionen im Tragkorb"]),
                    json_dumps([2]),
                    json_dumps({"l": 40, "t": 300, "r": 560, "b": 460}),
                    json_dumps({"table_text": "Referenzwert | Summe im Behälter, kW\nmaximal | 5,597"}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(32, 1, '#/texts/tab-6-caption', NULL, 42, 3, 'section_header', 'TextItem',
                    'Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter', NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Ergebnisse", "4.5 Spaltgasinventar", "Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter"]),
                    json_dumps([2]),
                    json_dumps({"l": 40, "t": 470, "r": 560, "b": 490}),
                ),
            )

    def _seed_ambiguous_reference_section(self) -> None:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO document_items(
                    id, document_id, self_ref, parent_ref, position, level, label, item_type,
                    text, orig_text, section_path_json, pages_json, provenance_json, bbox_json, payload_json
                )
                VALUES(33, 1, '#/texts/activity-section', NULL, 18, 2, 'section_header', 'TextItem',
                    '4.4 Aktivitätsinventar der Referenzbeladungen und Referenzwert für den Behälter',
                    NULL, ?, ?, '[]', ?, '{}')
                """,
                (
                    json_dumps(["4 Ergebnisse", "4.4 Aktivitätsinventar der Referenzbeladungen und Referenzwert für den Behälter"]),
                    json_dumps([1]),
                    json_dumps({"l": 20, "t": 560, "r": 520, "b": 590}),
                ),
            )
            conn.execute(
                """
                INSERT INTO document_chunks(
                    id, document_id, chunker, chunk_index, chunk_type, text, contextualized_text,
                    section_path_json, pages_json, item_refs_json, payload_json
                )
                VALUES(34, 1, 'hierarchical', 4, 'text',
                    'Das Aktivitätsinventar ist ein anderer Referenzwert.',
                    '4.4 Aktivitätsinventar der Referenzbeladungen und Referenzwert für den Behälter\n\nDas Aktivitätsinventar ist ein anderer Referenzwert.',
                    ?, ?, ?, '{}')
                """,
                (
                    json_dumps(["4 Ergebnisse", "4.4 Aktivitätsinventar der Referenzbeladungen und Referenzwert für den Behälter"]),
                    json_dumps([1]),
                    json_dumps(["#/texts/activity-section"]),
                ),
            )
            conn.execute(
                "INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json) VALUES(34, 1, 1, ?)",
                (json_dumps({"l": 20, "t": 600, "r": 520, "b": 650}),),
            )


if __name__ == "__main__":
    unittest.main()
