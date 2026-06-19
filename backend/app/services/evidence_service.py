from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.database import connection
from app.utils.bbox import normalize_bbox
from app.utils.common import json_loads, utc_now_iso
from app.utils.docling_records import is_asset_caption_text


WORD_RE = re.compile(r"\w[\w_.-]*", flags=re.UNICODE)
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "berein",
    "by",
    "der",
    "die",
    "das",
    "for",
    "how",
    "in",
    "ist",
    "many",
    "mit",
    "of",
    "stimmt",
    "the",
    "to",
    "und",
    "uberein",
    "ueberein",
    "von",
    "was",
    "what",
    "wie",
}
DEFAULT_MODEL_REASON = "No reason provided by model."
DEFAULT_MODEL_RATIONALE = "No rationale provided by model."


def encode_stream_event(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class AgentTask:
    id: str
    task_type: str
    reason: str
    params: dict[str, Any]


class EvidenceService:
    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()

    def get_readable_toc(self, document_id: int) -> dict[str, Any]:
        document = self._get_document(document_id)
        entries = []
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT i.*, p.width, p.height
                FROM document_items i
                LEFT JOIN document_pages p
                    ON p.document_id = i.document_id
                    AND p.page_number = CAST(json_extract(i.pages_json, '$[0]') AS INTEGER)
                WHERE i.document_id = ? AND i.label = 'section_header'
                ORDER BY i.position
                """,
                (document_id,),
            ).fetchall()

        for row in rows:
            pages = json_loads(row["pages_json"], [])
            page_number = int(pages[0]) if pages else None
            title = self._clean_text(row["text"] or row["orig_text"] or "")
            if is_asset_caption_text(title):
                continue
            bbox = json_loads(row["bbox_json"], None)
            locator = self._source_locator(
                document=document,
                page_number=page_number,
                item_id=int(row["id"]),
                item_ref=row["self_ref"],
                bbox=bbox,
                page_width=float(row["width"]) if row["width"] is not None else None,
                page_height=float(row["height"]) if row["height"] is not None else None,
            )
            entries.append(
                {
                    "title": title,
                    "level": row["level"],
                    "page_number": page_number,
                    "source_item_id": int(row["id"]),
                    "source_ref": row["self_ref"],
                    "section_path": json_loads(row["section_path_json"], []),
                    "source_key": self._source_key(locator),
                    "source_locator": locator,
                }
            )

        return {
            "document": self._compact_document(document),
            "entries": entries,
            "readable_text": "\n".join(
                f"{'  ' * max(0, int(entry['level'] or 1) - 1)}- p.{entry['page_number'] or '?'} {entry['title']}"
                for entry in entries
            ),
            "warnings": [] if entries else ["No section-header TOC entries found."],
        }

    def navigate_to_section(self, *, document_id: int, section_query: str) -> dict[str, Any]:
        document = self._get_document(document_id)
        reference_keys = self._structured_reference_keys(section_query)
        if reference_keys:
            structured_items = self._structured_items_for_references(document=document, reference_keys=reference_keys)
            if structured_items:
                return {
                    "document": self._compact_document(document),
                    "matched_section": None,
                    "merged_text": "\n\n".join(item["display_text"] for item in structured_items if item.get("display_text")),
                    "ordered_items": structured_items,
                    "chunks": [],
                    "warnings": [],
                }

        toc = self.get_readable_toc(document_id)
        matched = self._best_toc_match(toc["entries"], section_query)
        if matched is None:
            return {
                "document": toc["document"],
                "matched_section": None,
                "merged_text": "",
                "ordered_items": [],
                "chunks": [],
                "warnings": [f"No section matched: {section_query}"],
            }

        section_title = matched["title"]
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*, cp.page_number, cp.bbox_json, p.width, p.height
                FROM document_chunks c
                LEFT JOIN chunk_pages cp ON cp.chunk_id = c.id
                LEFT JOIN document_pages p
                    ON p.document_id = c.document_id
                    AND p.page_number = cp.page_number
                WHERE c.document_id = ?
                ORDER BY c.chunk_index, cp.page_number
                """,
                (document_id,),
            ).fetchall()

        target_section_path = matched.get("section_path") or [section_title]
        ordered_items = self._section_items(
            document=document,
            target_section_path=target_section_path,
        )
        if not ordered_items:
            ordered_items = self._section_items_by_position(document=document, matched_section=matched)
        structured_items = self.list_tables_figures(document_id=document_id, section_query=section_query).get("items", [])
        structured_by_source_key = {item["source_key"]: item for item in structured_items}
        merged_items = []
        ordered_source_keys: set[str] = set()
        for item in ordered_items:
            item = structured_by_source_key.get(item["source_key"], item)
            if item["source_key"] in ordered_source_keys:
                continue
            ordered_source_keys.add(item["source_key"])
            merged_items.append(item)
        for item in structured_items:
            if item["source_key"] in ordered_source_keys:
                continue
            ordered_source_keys.add(item["source_key"])
            merged_items.append(item)
        ordered_items = merged_items
        merged_text = "\n\n".join(item["display_text"] for item in ordered_items if item.get("display_text"))
        ordered_item_refs = {
            item["source_locator"].get("item_ref")
            for item in ordered_items
            if item.get("source_locator", {}).get("item_ref")
        }
        chunks = []
        for row in rows:
            section_path = json_loads(row["section_path_json"], [])
            chunk_item_refs = set(json_loads(row["item_refs_json"], []))
            if not self._chunk_matches_section(
                section_path=section_path,
                target_section_path=target_section_path,
                section_title=section_title,
                section_query=section_query,
                chunk_item_refs=chunk_item_refs,
                ordered_item_refs=ordered_item_refs,
            ):
                continue
            chunks.append(self._chunk_result(row=row, document=document, query_terms=[]))

        return {
            "document": toc["document"],
            "matched_section": matched,
            "merged_text": merged_text,
            "ordered_items": ordered_items,
            "chunks": chunks,
            "warnings": [] if chunks else ["Section matched, but no chunks were found under it."],
        }

    def search_keywords(
        self,
        *,
        query: str,
        document_ids: list[int] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        terms = self._important_terms(query) or self._terms(query)
        documents = self._documents_by_id()
        clauses = ["d.status = 'done'"]
        params: list[Any] = []
        if document_ids:
            clauses.append(f"c.document_id IN ({', '.join('?' for _ in document_ids)})")
            params.extend(document_ids)

        with connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.*,
                    d.file_name,
                    d.relative_path,
                    cp.page_number,
                    cp.bbox_json,
                    p.width,
                    p.height
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                LEFT JOIN chunk_pages cp ON cp.chunk_id = c.id
                LEFT JOIN document_pages p
                    ON p.document_id = c.document_id
                    AND p.page_number = cp.page_number
                WHERE {' AND '.join(clauses)}
                ORDER BY c.document_id, c.chunk_index, cp.page_number
                """,
                params,
            ).fetchall()
            fts_ranking = self._rank_chunk_rows_with_fts(conn=conn, rows=rows, terms=terms)

        scored_results = []
        for row in rows:
            document = documents.get(int(row["document_id"]))
            if document is None:
                continue
            result = self._chunk_result(row=row, document=document, query_terms=terms)
            if terms and result["score"] == 0:
                continue
            scored_results.append(result)

        results = self._rank_keyword_results(
            rows=rows,
            documents=documents,
            query_terms=terms,
            scored_results=scored_results,
            fts_ranking=fts_ranking,
        )
        return {
            "query": query,
            "mode": "keyword",
            "coverage": {
                "searched_documents": len({result["document"]["id"] for result in results}),
                "search_index_used": bool(fts_ranking),
            },
            "results": results[:limit],
        }

    def list_inventory(self, *, document_ids: list[int] | None = None) -> dict[str, Any]:
        documents = [
            self._compact_document(document)
            for document in self._candidate_documents(set(document_ids or []))
        ]
        return {"documents": documents}

    def inspect_chunk(self, *, document_id: int, chunk_id: int) -> dict[str, Any]:
        document = self._get_document(document_id)
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*, cp.page_number, cp.bbox_json, p.width, p.height
                FROM document_chunks c
                LEFT JOIN chunk_pages cp ON cp.chunk_id = c.id
                LEFT JOIN document_pages p
                    ON p.document_id = c.document_id
                    AND p.page_number = cp.page_number
                WHERE c.document_id = ? AND c.id = ?
                ORDER BY cp.page_number
                """,
                (document_id, chunk_id),
            ).fetchall()
        return {"results": [self._chunk_result(row=row, document=document, query_terms=[]) for row in rows]}

    def inspect_page(self, *, document_id: int, page_number: int) -> dict[str, Any]:
        document = self._get_document(document_id)
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*, cp.page_number, cp.bbox_json, p.width, p.height
                FROM document_chunks c
                JOIN chunk_pages cp ON cp.chunk_id = c.id
                LEFT JOIN document_pages p
                    ON p.document_id = c.document_id
                    AND p.page_number = cp.page_number
                WHERE c.document_id = ? AND cp.page_number = ?
                ORDER BY c.chunk_index
                """,
                (document_id, page_number),
            ).fetchall()
        return {"results": [self._chunk_result(row=row, document=document, query_terms=[]) for row in rows]}

    def resolve_citation(self, *, document_id: int, citation: str) -> dict[str, Any]:
        document = self._get_document(document_id)
        references = self._citation_items(document=document, citation=citation)
        ranked = []
        for reference in references:
            candidates = self._matched_documents_from_text(
                reference["display_text"],
                current_document_id=document_id,
            )
            best_score = int(candidates[0]["score"]) if candidates else 0
            section_bonus = 5 if self._looks_like_reference_section(reference.get("section_path", [])) else 0
            ranked.append((best_score + section_bonus, reference, candidates))
        ranked.sort(key=lambda item: (-item[0], item[1]["source_locator"].get("page_number") or 0))
        best_reference = ranked[0][1] if ranked else None
        matched_document = None
        candidates = []
        if ranked:
            candidates = ranked[0][2]
            if candidates:
                matched_document = candidates[0]["document"]
        return {
            "document": self._compact_document(document),
            "citation": citation,
            "reference": best_reference,
            "references": references,
            "matched_document": matched_document,
            "candidate_documents": candidates,
            "warnings": [] if best_reference else [f"No bibliography entry matched citation {citation!r}."],
        }

    def list_tables_figures(
        self,
        *,
        document_id: int,
        section_query: str | None = None,
        page_number: int | None = None,
    ) -> dict[str, Any]:
        document = self._get_document(document_id)
        reference_keys = self._structured_reference_keys(section_query or "")
        if reference_keys:
            items = self._structured_items_for_references(
                document=document,
                reference_keys=reference_keys,
                page_number=page_number,
            )
            if items:
                return {
                    "document": self._compact_document(document),
                    "matched_section": None,
                    "items": items,
                    "warnings": [],
                }

        matched_section = None
        target_section_path: list[str] | None = None
        if section_query:
            toc = self.get_readable_toc(document_id)
            matched_section = self._best_toc_match(toc["entries"], section_query)
            if matched_section is None:
                return {
                    "document": self._compact_document(document),
                    "matched_section": None,
                    "items": [],
                    "warnings": [f"No section matched: {section_query}"],
                }
            target_section_path = matched_section.get("section_path") or [matched_section["title"]]

        items = self._document_items(document=document)
        referenced_keys: set[str] = set()
        toc_target_pages: set[int] = set()
        if target_section_path is not None:
            section_items = self._section_items(document=document, target_section_path=target_section_path)
            section_text = "\n".join(
                [
                    section_query or "",
                    matched_section.get("title", "") if matched_section else "",
                    *(item.get("display_text") or item.get("text") or "" for item in section_items),
                ]
            )
            referenced_keys = self._structured_reference_keys(section_text)
            if referenced_keys:
                toc_target_pages = {
                    int(entry["page_number"])
                    for entry in self._document_index_entries(document_id)
                    if entry["reference_key"] in referenced_keys and entry.get("page_number") is not None
                }

        filtered = []
        seen_source_keys: set[str] = set()
        for item in items:
            if not self._is_structured_evidence_item(item):
                continue
            if page_number is not None and item["source_locator"].get("page_number") != page_number:
                continue
            direct_section_match = (
                target_section_path is None
                or item.get("section_path", [])[: len(target_section_path)] == target_section_path
            )
            referenced_table_match = (
                target_section_path is not None
                and self._is_asset_item(item)
                and self._item_matches_structured_reference(
                    item=item,
                    all_items=items,
                    reference_keys=referenced_keys,
                    toc_target_pages=toc_target_pages,
                )
            )
            if not direct_section_match and not referenced_table_match:
                continue
            if item["source_key"] in seen_source_keys:
                continue
            seen_source_keys.add(item["source_key"])
            if self._is_asset_item(item) and (direct_section_match or referenced_table_match):
                item = self._with_nearby_caption_text(
                    item=item,
                    all_items=items,
                    reference_keys=referenced_keys,
                    require_reference_match=referenced_table_match and not direct_section_match,
                )
            filtered.append(item)

        warnings = [] if filtered else ["No table or figure items matched the requested scope."]
        return {
            "document": self._compact_document(document),
            "matched_section": matched_section,
            "items": filtered,
            "warnings": warnings,
        }

    def _structured_items_for_references(
        self,
        *,
        document: dict[str, Any],
        reference_keys: set[str],
        page_number: int | None = None,
    ) -> list[dict[str, Any]]:
        items = self._document_items(document=document)
        toc_target_pages = {
            int(entry["page_number"])
            for entry in self._document_index_entries(int(document["id"]))
            if entry["reference_key"] in reference_keys and entry.get("page_number") is not None
        }
        filtered = []
        seen_source_keys: set[str] = set()
        for item in items:
            if not self._is_structured_evidence_item(item):
                continue
            if page_number is not None and item["source_locator"].get("page_number") != page_number:
                continue
            if not self._item_matches_structured_reference(
                item=item,
                all_items=items,
                reference_keys=reference_keys,
                toc_target_pages=toc_target_pages,
            ):
                continue
            if item["source_key"] in seen_source_keys:
                continue
            seen_source_keys.add(item["source_key"])
            if self._is_asset_item(item):
                item = self._with_nearby_caption_text(
                    item=item,
                    all_items=items,
                    reference_keys=reference_keys,
                    require_reference_match=False,
                )
            filtered.append(item)
        return filtered

    def inspect_item(
        self,
        *,
        document_id: int,
        item_id: int | None = None,
        item_ref: str | None = None,
    ) -> dict[str, Any]:
        if item_id is None and item_ref is None:
            raise ValueError("inspect_item requires item_id or item_ref.")
        document = self._get_document(document_id)
        clauses = ["i.document_id = ?"]
        params: list[Any] = [document_id]
        if item_id is not None:
            clauses.append("i.id = ?")
            params.append(item_id)
        if item_ref is not None:
            clauses.append("i.self_ref = ?")
            params.append(item_ref)
        with connection() as conn:
            row = conn.execute(
                f"""
                SELECT i.*, p.width, p.height
                FROM document_items i
                LEFT JOIN document_pages p
                    ON p.document_id = i.document_id
                    AND p.page_number = CAST(json_extract(i.pages_json, '$[0]') AS INTEGER)
                WHERE {' AND '.join(clauses)}
                ORDER BY i.position
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown item in document {document_id}: {item_id or item_ref}")
        return {"item": self._item_result(row=row, document=document)}

    def run_agentic_retrieval(
        self,
        *,
        query: str,
        document_ids: list[int] | None = None,
        max_tasks: int = 32,
    ) -> dict[str, Any]:
        run_id = self._new_run_id()
        agent_state = self._run_agentic_retrieval_state(
            run_id=run_id,
            query=query,
            document_ids=document_ids,
            max_tasks=max_tasks,
        )
        return self._finalize_agentic_retrieval(run_id=run_id, query=query, agent_state=agent_state)

    def run_agentic_retrieval_stream(
        self,
        *,
        query: str,
        document_ids: list[int] | None = None,
        max_tasks: int = 32,
    ) -> Iterator[str]:
        run_id = self._new_run_id()
        events: queue.Queue[dict[str, Any] | object] = queue.Queue()
        stream_done = object()

        def emit(event: dict[str, Any]) -> None:
            events.put(event)

        def run_worker() -> None:
            try:
                agent_state = self._run_agentic_retrieval_state(
                    run_id=run_id,
                    query=query,
                    document_ids=document_ids,
                    max_tasks=max_tasks,
                    emit=emit,
                )
                result = self._finalize_agentic_retrieval(run_id=run_id, query=query, agent_state=agent_state)
                events.put({"type": "run_completed", "run_id": run_id, "result": result})
            except Exception as exc:
                events.put({"type": "run_error", "run_id": run_id, "detail": str(exc)})
            finally:
                events.put(stream_done)

        threading.Thread(target=run_worker, daemon=True).start()
        while True:
            event = events.get()
            if event is stream_done:
                break
            yield encode_stream_event(event)

    def _new_run_id(self) -> str:
        return f"{utc_now_iso().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"

    def _run_agentic_retrieval_state(
        self,
        *,
        run_id: str,
        query: str,
        document_ids: list[int] | None,
        max_tasks: int,
        emit: Any = None,
    ) -> dict[str, Any]:
        from app.services.evidence_agent import EvidenceAgentRunner

        return EvidenceAgentRunner(service=self, settings=self.settings, emit=emit).run(
            run_id=run_id,
            query=query,
            document_ids=document_ids,
            max_tasks=max_tasks,
        )

    def _finalize_agentic_retrieval(self, *, run_id: str, query: str, agent_state: dict[str, Any]) -> dict[str, Any]:
        evidence = list(agent_state.get("evidence_by_key", {}).values())
        analysis = self._analysis_from_evidence(evidence)
        stop_reason = agent_state.get("stop_reason") or "complete"
        trace_events = agent_state.get("trace_events", [])
        tasks = [*agent_state.get("completed_tasks", []), *agent_state.get("task_queue", [])]
        graph_nodes = agent_state.get("graph_nodes", {})
        graph_edges = agent_state.get("graph_edges", [])
        tool_result_lines = agent_state.get("tool_result_lines", [])
        model_input_lines = agent_state.get("model_input_lines", [])
        model_output_lines = agent_state.get("model_output_lines", [])

        artifacts = self._write_artifacts(
            run_id=run_id,
            query=query,
            trace_lines=[],
            tool_result_lines=tool_result_lines,
            model_input_lines=model_input_lines,
            model_output_lines=model_output_lines,
            trace_graph={"nodes": list(graph_nodes.values()), "edges": graph_edges},
            trace_events=trace_events,
            tasks=tasks,
            evidence=evidence,
            analysis=analysis,
            stop_reason=stop_reason,
        )

        return {
            "agent_backend": "langgraph_ollama",
            "run_id": run_id,
            "query": query,
            "stop_reason": stop_reason,
            "tasks": tasks,
            "evidence": evidence,
            "analysis": analysis,
            "trace_events": trace_events,
            "trace_graph": {"nodes": list(graph_nodes.values()), "edges": graph_edges},
            "artifacts": artifacts,
        }

    def _empty_evidence_analysis(self) -> dict[str, Any]:
        return {
            "for_items": [],
            "against_items": [],
            "neutral_items": [],
            "supporting_chunk_count": 0,
            "refuting_chunk_count": 0,
            "neutral_chunk_count": 0,
            "warning": None,
        }

    def _analysis_from_evidence(self, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        analysis = self._empty_evidence_analysis()
        for item in evidence:
            source_key = str(item.get("source_key") or "")
            if not source_key:
                continue
            stance = str(item.get("stance") or "neutral")
            target_key = {
                "for": "for_items",
                "against": "against_items",
                "neutral": "neutral_items",
            }.get(stance, "neutral_items")
            analysis[target_key].append(
                {
                    "summary": item.get("relevance") or item.get("stance_rationale") or source_key,
                    "source_keys": [source_key],
                    "chunk_count": 1,
                }
            )
        analysis["supporting_chunk_count"] = len(analysis["for_items"])
        analysis["refuting_chunk_count"] = len(analysis["against_items"])
        analysis["neutral_chunk_count"] = len(analysis["neutral_items"])
        return analysis

    def _write_artifacts(
        self,
        *,
        run_id: str,
        query: str,
        trace_lines: list[str],
        tool_result_lines: list[str],
        model_input_lines: list[str],
        model_output_lines: list[str],
        trace_graph: dict[str, Any],
        trace_events: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        stop_reason: str,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        analysis = analysis or self._empty_evidence_analysis()
        trace_dir = Path(self.settings.data_dir) / "retrieval_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_text_path = trace_dir / f"{run_id}.txt"
        trace_json_path = trace_dir / f"{run_id}.json"
        tool_results_text_path = trace_dir / f"{run_id}-tool-results.txt"
        model_inputs_text_path = trace_dir / f"{run_id}-model-inputs.txt"
        model_outputs_text_path = trace_dir / f"{run_id}-model-outputs.txt"
        query_log_path = Path(self.settings.data_dir) / "retrieval_query_results.jsonl"

        if not trace_lines:
            trace_lines = self._format_readable_trace(
                run_id=run_id,
                query=query,
                stop_reason=stop_reason,
                tasks=tasks,
                evidence=evidence,
                analysis=analysis,
                trace_events=trace_events,
                tool_results_text_path=tool_results_text_path,
                model_inputs_text_path=model_inputs_text_path,
                model_outputs_text_path=model_outputs_text_path,
            )
        trace_text_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
        tool_results_text_path.write_text("\n".join(tool_result_lines) + "\n", encoding="utf-8")
        model_inputs_text_path.write_text("\n".join(model_input_lines) + "\n", encoding="utf-8")
        model_outputs_text_path.write_text("\n".join(model_output_lines) + "\n", encoding="utf-8")
        trace_json_path.write_text(
            json.dumps(
                {
                    "agent_backend": "langgraph_ollama",
                    "run_id": run_id,
                    "query": query,
                    "stop_reason": stop_reason,
                    "trace_events": trace_events,
                    "tasks": [{key: value for key, value in task.items() if key != "signature"} for task in tasks],
                    "evidence": evidence,
                    "analysis": analysis,
                    "trace_graph": trace_graph,
                    "trace_files": {
                        "readable": str(trace_text_path),
                        "tool_results": str(tool_results_text_path),
                        "model_inputs": str(model_inputs_text_path),
                        "model_outputs": str(model_outputs_text_path),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with query_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "query": query,
                        "found_sources": [item["source_key"] for item in evidence],
                        "supporting_chunk_count": analysis.get("supporting_chunk_count", 0),
                        "refuting_chunk_count": analysis.get("refuting_chunk_count", 0),
                        "neutral_chunk_count": analysis.get("neutral_chunk_count", 0),
                        "stop_reason": stop_reason,
                        "trace_text_path": str(trace_text_path),
                        "trace_json_path": str(trace_json_path),
                        "tool_results_text_path": str(tool_results_text_path),
                        "model_inputs_text_path": str(model_inputs_text_path),
                        "model_outputs_text_path": str(model_outputs_text_path),
                    }
                )
                + "\n"
            )

        return {
            "trace_text_path": str(trace_text_path),
            "trace_json_path": str(trace_json_path),
            "tool_results_text_path": str(tool_results_text_path),
            "model_inputs_text_path": str(model_inputs_text_path),
            "model_outputs_text_path": str(model_outputs_text_path),
            "query_log_path": str(query_log_path),
        }

    def _format_readable_trace(
        self,
        *,
        run_id: str,
        query: str,
        stop_reason: str,
        tasks: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        analysis: dict[str, Any],
        trace_events: list[dict[str, Any]],
        tool_results_text_path: Path,
        model_inputs_text_path: Path,
        model_outputs_text_path: Path,
    ) -> list[str]:
        lines = [
            f"Query: {query}",
            f"Run: {run_id}",
            "Agent Backend: langgraph_ollama",
            f"Stop Reason: {stop_reason}",
            "",
            "Trace Files:",
            f"- Model inputs: {model_inputs_text_path.name}",
            f"- Model outputs: {model_outputs_text_path.name}",
            f"- Tool results: {tool_results_text_path.name}",
            "",
            "Task List:",
        ]
        if tasks:
            for task in tasks:
                reason = str(task.get("reason") or "")
                reason_suffix = f" {reason}" if reason and reason != DEFAULT_MODEL_REASON else ""
                lines.append(
                    f"- {task.get('id', 'T?')} {task.get('type', '-')} "
                    f"[{task.get('status', '-')}]{reason_suffix}"
                )
                params = task.get("params") or {}
                if params:
                    lines.append(f"  params: {json.dumps(params, ensure_ascii=False, sort_keys=True)}")
        else:
            lines.append("- none")

        lines.extend(["", "Step-by-Step:"])
        if trace_events:
            for event in trace_events:
                lines.extend(
                    self._format_trace_event(
                        event=event,
                        tool_results_name=tool_results_text_path.name,
                        model_inputs_name=model_inputs_text_path.name,
                        model_outputs_name=model_outputs_text_path.name,
                    )
                )
        else:
            lines.append("- no trace events")

        lines.extend(["", "Evidence:"])
        if evidence:
            for item in evidence:
                found_by = ", ".join(item.get("found_by_tasks", [])) or "-"
                lines.append(f"- {item.get('id', 'E?')} {item.get('source_key', '-')} from {found_by}")
        else:
            lines.append("- none")

        lines.extend(["", "Evidence Analysis:"])
        lines.append(f"- Supporting chunks: {analysis.get('supporting_chunk_count', 0)}")
        lines.append(f"- Refuting chunks: {analysis.get('refuting_chunk_count', 0)}")
        lines.append(f"- Neutral chunks: {analysis.get('neutral_chunk_count', 0)}")
        for heading, key in (("For", "for_items"), ("Against", "against_items"), ("Neutral", "neutral_items")):
            items = analysis.get(key) if isinstance(analysis.get(key), list) else []
            lines.append(f"{heading}:")
            if not items:
                lines.append("- none")
                continue
            for item in items:
                source_keys = ", ".join(item.get("source_keys", [])) or "-"
                lines.append(f"- ({item.get('chunk_count', 0)}) {item.get('summary', '')} [{source_keys}]")
        return lines

    def _format_trace_event(
        self,
        *,
        event: dict[str, Any],
        tool_results_name: str,
        model_inputs_name: str,
        model_outputs_name: str,
    ) -> list[str]:
        step_id = str(event.get("step_id") or "S???")
        node = str(event.get("node") or "-")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        lines = [f"[{step_id}] {node}"]
        message = str(event.get("message") or "")
        if message and message != DEFAULT_MODEL_RATIONALE:
            lines.append(f"Decision: {message}")
        if node in {"planner", "actor", "curator", "evaluator", "analyst"}:
            lines.append(f"Model input: {model_inputs_name}#{step_id}")
            lines.append(f"Model output: {model_outputs_name}#{step_id}")

        if node == "planner":
            structured_requests = data.get("structured_requests") or []
            if structured_requests:
                lines.append("Curator instruction to planner:")
                for request in structured_requests:
                    lines.append(f"- {request}")
            self._append_task_lines(lines, "Added tasks", data.get("added_tasks") or data.get("tasks") or [])
        elif node == "actor":
            task = data.get("task") if isinstance(data.get("task"), dict) else {}
            tool = data.get("tool") if isinstance(data.get("tool"), dict) else {}
            if task:
                lines.append(
                    f"Task: {task.get('id', 'T?')} {task.get('type', '-')} "
                    f"[{task.get('status', '-')}] observations:{task.get('observation_count', 0)}"
                )
            if tool:
                lines.append(
                    f"Tool: {tool.get('tool_name', '-')} "
                    f"{json.dumps(tool.get('tool_args') or {}, ensure_ascii=False, sort_keys=True)}"
                )
            lines.append(f"Tool results: {tool_results_name}#{step_id}")
            source_keys = [key for key in data.get("observation_source_keys", []) if key]
            if source_keys:
                lines.append("Observed sources: " + ", ".join(source_keys))
        elif node == "curator":
            accepted = data.get("accepted_source_keys") or []
            lines.append("Curator selected evidence: " + (", ".join(accepted) if accepted else "none"))
            planner_request = data.get("planner_request")
            if planner_request:
                lines.append(f"Planner request: {planner_request}")
            planner_requests = [request for request in data.get("planner_requests", []) if request != planner_request]
            if planner_requests:
                lines.append("Derived planner requests:")
                for request in planner_requests:
                    lines.append(f"- {request}")
            self._append_task_lines(lines, "Curator-added tasks", data.get("added_tasks") or [])
        elif node == "analyst":
            analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
            lines.append(f"Supporting chunks: {analysis.get('supporting_chunk_count', 0)}")
            lines.append(f"Refuting chunks: {analysis.get('refuting_chunk_count', 0)}")
            warning = analysis.get("warning")
            if warning:
                lines.append(f"Warning: {warning}")
        elif node == "evaluator":
            stance = data.get("stance") if isinstance(data.get("stance"), dict) else {}
            lines.append(f"Evidence: {data.get('source_key') or '-'}")
            lines.append(f"Stance: {stance.get('stance') or '-'}")
            rationale = stance.get("rationale")
            if rationale and rationale != DEFAULT_MODEL_RATIONALE:
                lines.append(f"Rationale: {rationale}")
        elif node == "actor_tool_error":
            lines.append(f"Tool error: {event.get('message') or '-'}")

        lines.append("")
        return lines

    def _append_task_lines(self, lines: list[str], heading: str, tasks: list[dict[str, Any]]) -> None:
        if not tasks:
            return
        lines.append(f"{heading}:")
        for task in tasks:
            if hasattr(task, "model_dump"):
                task = task.model_dump()
            lines.append(
                f"- {task.get('id', 'new')} {task.get('type', '-')} "
                f"{json.dumps(task.get('params') or {}, ensure_ascii=False, sort_keys=True)}"
            )
            reason = task.get("reason")
            if reason and reason != DEFAULT_MODEL_REASON:
                lines.append(f"  reason: {reason}")

    def _append_tool_results(
        self,
        *,
        lines: list[str],
        task: AgentTask,
        tool_name: str,
        results: list[dict[str, Any]],
    ) -> None:
        lines.append(f"{task.id} {tool_name}")
        self._append_task_reason(lines, task)
        lines.append(f"Results: {len(results)}")
        if not results:
            lines.extend(["- no results", ""])
            return
        for index, result in enumerate(results, start=1):
            locator = result["source_locator"]
            section = " / ".join(result.get("section_path", [])) or "-"
            text = self._clean_text(result.get("raw_text") or result.get("text", ""))
            lines.extend(
                [
                    f"{index}. {result['source_key']}",
                    f"   document: {result['document']['relative_path']}",
                    f"   page: {locator.get('page_number') or '?'}",
                    f"   section: {section}",
                    f"   score: {result.get('score', 0)}",
                    f"   reasons: {', '.join(result.get('match_reasons', [])) or '-'}",
                    f"   text: {text[:500] or '(empty text)'}",
                ]
            )
        lines.append("")

    def _append_section_results(self, *, lines: list[str], task: AgentTask, section: dict[str, Any]) -> None:
        chunks = section["chunks"]
        structured_items = [
            item for item in section.get("ordered_items", [])
            if self._is_structured_evidence_item(item)
        ]
        matched = section.get("matched_section") or {}
        lines.append(f"{task.id} navigate_to_section")
        self._append_task_reason(lines, task)
        lines.extend(
            [
                f"Document: {section['document']['relative_path']}",
                f"Matched section: {matched.get('title', '-')}",
                f"Merged body text: {self._clean_text(section.get('merged_text', ''))[:1000] or '(empty)'}",
                f"Ordered body items: {len(section.get('ordered_items', []))}",
                f"Chunk results: {len(chunks)}",
                f"Structured item results: {len(structured_items)}",
            ]
        )
        if not chunks and not structured_items:
            lines.extend(["- no chunks or structured items", ""])
            return
        for index, result in enumerate(chunks, start=1):
            locator = result["source_locator"]
            section_path = " / ".join(result.get("section_path", [])) or "-"
            raw_text = self._clean_text(result.get("raw_text") or "")
            contextualized_text = self._clean_text(result.get("text") or "")
            lines.extend(
                [
                    f"{index}. {result['source_key']}",
                    f"   document: {result['document']['relative_path']}",
                    f"   page: {locator.get('page_number') or '?'}",
                    f"   section: {section_path}",
                    f"   raw_text: {raw_text[:500] or '(empty raw text)'}",
                    f"   contextualized_text: {contextualized_text[:500] or '(empty text)'}",
                ]
            )
        for index, result in enumerate(structured_items, start=1):
            locator = result["source_locator"]
            section_path = " / ".join(result.get("section_path", [])) or "-"
            display_text = self._clean_text(result.get("display_text") or "")
            lines.extend(
                [
                    f"item {index}. {result['source_key']}",
                    f"   label: {result.get('label', '-')}",
                    f"   document: {result['document']['relative_path']}",
                    f"   page: {locator.get('page_number') or '?'}",
                    f"   section: {section_path}",
                    f"   text: {display_text[:500] or '(empty text)'}",
                ]
            )
        lines.append("")

    def _append_toc_results(self, *, lines: list[str], task: AgentTask, toc: dict[str, Any]) -> None:
        entries = toc["entries"]
        lines.append(f"{task.id} get_readable_toc")
        self._append_task_reason(lines, task)
        lines.extend([f"Document: {toc['document']['relative_path']}", f"Results: {len(entries)}"])
        if not entries:
            lines.extend(["- no TOC entries", ""])
            return
        for index, entry in enumerate(entries, start=1):
            lines.append(f"{index}. p.{entry['page_number'] or '?'} {entry['title']} [{entry['source_key']}]")
        lines.append("")

    def _append_task_reason(self, lines: list[str], task: AgentTask) -> None:
        if task.reason and task.reason != DEFAULT_MODEL_REASON:
            lines.append(f"Reason: {task.reason}")

    def _get_document(self, document_id: int) -> dict[str, Any]:
        documents = self._documents_by_id()
        if document_id not in documents:
            raise ValueError(f"Unknown document: {document_id}")
        return documents[document_id]

    def _documents_by_id(self) -> dict[int, dict[str, Any]]:
        with connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY relative_path").fetchall()
        return {int(row["id"]): dict(row) for row in rows}

    def _candidate_documents(self, document_scope: set[int]) -> list[dict[str, Any]]:
        documents = [
            document
            for document in self._documents_by_id().values()
            if document["status"] == "done" and (not document_scope or int(document["id"]) in document_scope)
        ]
        return sorted(documents, key=lambda document: str(document["relative_path"]))

    def _compact_document(self, document: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(document["id"]),
            "relative_path": document["relative_path"],
            "file_name": document["file_name"],
            "status": document["status"],
            "page_count": document["page_count"],
        }

    def _section_items(self, *, document: dict[str, Any], target_section_path: list[str]) -> list[dict[str, Any]]:
        items = []
        for item in self._document_items(document=document):
            if item["label"] == "section_header":
                continue
            section_path = item.get("section_path", [])
            if section_path[: len(target_section_path)] != target_section_path:
                continue
            items.append(item)
        return items

    def _section_items_by_position(self, *, document: dict[str, Any], matched_section: dict[str, Any]) -> list[dict[str, Any]]:
        source_item_id = matched_section.get("source_item_id")
        if source_item_id is None:
            return []
        items = self._document_items(document=document)
        header = next(
            (item for item in items if item["source_locator"].get("item_id") == int(source_item_id)),
            None,
        )
        if header is None:
            return []
        target_number = self._leading_section_number(matched_section.get("title", ""))
        body = []
        for item in items:
            if item["position"] <= header["position"]:
                continue
            if item["label"] == "section_header":
                if self._is_position_section_boundary(item.get("display_text", ""), target_number):
                    break
                continue
            body.append(item)
        return body

    def _chunk_matches_section(
        self,
        *,
        section_path: list[str],
        target_section_path: list[str],
        section_title: str,
        section_query: str,
        chunk_item_refs: set[str],
        ordered_item_refs: set[str],
    ) -> bool:
        if section_path[: len(target_section_path)] == target_section_path:
            return True
        if section_title in section_path:
            return True
        if section_query.lower() in " ".join(section_path).lower():
            return True
        return bool(ordered_item_refs and chunk_item_refs.intersection(ordered_item_refs))

    def _is_position_section_boundary(self, title: str, target_number: str | None) -> bool:
        if not target_number:
            return bool(title.strip())
        candidate_number = self._leading_section_number(title)
        if candidate_number is None:
            return title.lower().strip().startswith(("literatur", "references", "anhang", "appendix"))
        if candidate_number == target_number or candidate_number.startswith(f"{target_number}."):
            return False
        candidate_parts = self._section_number_parts(candidate_number)
        target_parts = self._section_number_parts(target_number)
        if not candidate_parts or not target_parts:
            return False
        return candidate_parts[0] > target_parts[0] or candidate_parts[0] == target_parts[0]

    def _leading_section_number(self, title: str) -> str | None:
        match = re.match(r"\s*(\d+(?:\s*\.\s*\d+)*)", title)
        if match is None:
            return None
        return re.sub(r"\s*\.\s*", ".", match.group(1)).strip(".")

    def _section_number_parts(self, section_number: str) -> list[int]:
        return [int(part) for part in section_number.split(".") if part.isdigit()]

    def _document_items(self, *, document: dict[str, Any]) -> list[dict[str, Any]]:
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT i.*, p.width, p.height
                FROM document_items i
                LEFT JOIN document_pages p
                    ON p.document_id = i.document_id
                    AND p.page_number = CAST(json_extract(i.pages_json, '$[0]') AS INTEGER)
                WHERE i.document_id = ?
                ORDER BY i.position
                """,
                (int(document["id"]),),
            ).fetchall()
        return [self._item_result(row=row, document=document) for row in rows]

    def _item_result(self, *, row: Any, document: dict[str, Any]) -> dict[str, Any]:
        pages = json_loads(row["pages_json"], [])
        page_number = int(pages[0]) if pages else None
        bbox = json_loads(row["bbox_json"], None)
        payload = json_loads(row["payload_json"], {})
        display_text = self._item_display_text(row=row, payload=payload)
        locator = self._source_locator(
            document=document,
            page_number=page_number,
            item_id=int(row["id"]),
            item_ref=row["self_ref"],
            bbox=bbox,
            page_width=float(row["width"]) if row["width"] is not None else None,
            page_height=float(row["height"]) if row["height"] is not None else None,
        )
        return {
            "result_type": "item",
            "source_key": self._source_key(locator),
            "source_locator": locator,
            "document": self._compact_document(document),
            "label": row["label"],
            "item_type": row["item_type"],
            "section_path": json_loads(row["section_path_json"], []),
            "position": int(row["position"]),
            "text": display_text,
            "raw_text": display_text,
            "display_text": display_text,
            "payload": payload,
        }

    def _chunk_result(self, *, row: Any, document: dict[str, Any], query_terms: list[str]) -> dict[str, Any]:
        section_path = json_loads(row["section_path_json"], [])
        page_number = int(row["page_number"]) if row["page_number"] is not None else self._first_page(row["pages_json"])
        bbox = json_loads(row["bbox_json"], None)
        locator = self._source_locator(
            document=document,
            page_number=page_number,
            chunk_id=int(row["id"]),
            chunk_index=int(row["chunk_index"]),
            bbox=bbox,
            page_width=float(row["width"]) if row["width"] is not None else None,
            page_height=float(row["height"]) if row["height"] is not None else None,
        )
        text = row["contextualized_text"] or row["text"] or ""
        searchable = "\n".join(
            [
                document["file_name"],
                document["relative_path"],
                " ".join(section_path),
                text,
            ]
        )
        score = self._score(query_terms, searchable)
        return {
            "result_type": "chunk",
            "score": score,
            "match_reasons": [f"keyword:{term}" for term in query_terms if self._term_matches_text(term, searchable.lower())],
            "source_key": self._source_key(locator),
            "source_locator": locator,
            "document": self._compact_document(document),
            "chunk_index": int(row["chunk_index"]),
            "section_path": section_path,
            "text": text,
            "raw_text": row["text"] or "",
        }

    def _rank_keyword_results(
        self,
        *,
        rows: list[Any],
        documents: dict[int, dict[str, Any]],
        query_terms: list[str],
        scored_results: list[dict[str, Any]],
        fts_ranking: list[tuple[int, float]],
    ) -> list[dict[str, Any]]:
        fallback_results = sorted(
            scored_results,
            key=lambda result: (-result["score"], result["document"]["relative_path"], result["chunk_index"]),
        )
        if not fts_ranking:
            return fallback_results

        results = []
        seen_source_keys: set[str] = set()
        for row_position, rank in fts_ranking:
            row = rows[row_position]
            document = documents.get(int(row["document_id"]))
            if document is None:
                continue
            result = self._chunk_result(row=row, document=document, query_terms=query_terms)
            if query_terms and result["score"] == 0:
                continue
            result["search_rank"] = rank
            if result["source_key"] in seen_source_keys:
                continue
            seen_source_keys.add(result["source_key"])
            results.append(result)

        for result in fallback_results:
            if result["source_key"] in seen_source_keys:
                continue
            seen_source_keys.add(result["source_key"])
            results.append(result)
        return results

    def _rank_chunk_rows_with_fts(
        self,
        *,
        conn: sqlite3.Connection,
        rows: list[Any],
        terms: list[str],
    ) -> list[tuple[int, float]]:
        match_query = self._fts_match_query(terms)
        if not match_query or not rows:
            return []
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE temp.keyword_search_fts
                USING fts5(row_position UNINDEXED, section, text, raw_text, tokenize='unicode61')
                """
            )
            conn.executemany(
                """
                INSERT INTO temp.keyword_search_fts(row_position, section, text, raw_text)
                VALUES(?, ?, ?, ?)
                """,
                [
                    (
                        index,
                        " / ".join(json_loads(row["section_path_json"], [])),
                        row["contextualized_text"] or row["text"] or "",
                        row["text"] or "",
                    )
                    for index, row in enumerate(rows)
                ],
            )
            ranked_rows = conn.execute(
                """
                SELECT row_position, bm25(keyword_search_fts, 0.2, 1.0, 2.0) AS rank
                FROM temp.keyword_search_fts
                WHERE keyword_search_fts MATCH ?
                ORDER BY rank
                """,
                (match_query,),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [(int(row["row_position"]), float(row["rank"])) for row in ranked_rows]

    def _fts_match_query(self, terms: list[str]) -> str:
        quoted_terms = []
        for term in terms:
            if not term:
                continue
            escaped = term.replace('"', '""')
            quoted_terms.append(f'"{escaped}"')
        return " OR ".join(quoted_terms)

    def _source_locator(
        self,
        *,
        document: dict[str, Any],
        page_number: int | None,
        chunk_id: int | None = None,
        chunk_index: int | None = None,
        item_id: int | None = None,
        item_ref: str | None = None,
        bbox: dict[str, Any] | None = None,
        page_width: float | None = None,
        page_height: float | None = None,
    ) -> dict[str, Any]:
        return {
            "document_id": int(document["id"]),
            "relative_path": document["relative_path"],
            "file_name": document["file_name"],
            "page_number": page_number,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "item_id": item_id,
            "item_ref": item_ref,
            "bbox": bbox,
            "normalized_bbox": normalize_bbox(
                bbox=bbox,
                page_width=page_width or 0.0,
                page_height=page_height or 0.0,
            ),
        }

    def _source_key(self, locator: dict[str, Any]) -> str:
        page = locator.get("page_number") or 0
        if locator.get("chunk_id") is not None:
            return f"doc:{locator['document_id']}:page:{page}:chunk:{locator['chunk_id']}"
        if locator.get("item_id") is not None:
            return f"doc:{locator['document_id']}:page:{page}:item:{locator['item_id']}"
        return f"doc:{locator['document_id']}:page:{page}"

    def _best_toc_match(self, entries: list[dict[str, Any]], section_query: str) -> dict[str, Any] | None:
        query = section_query.lower().strip()
        if not query:
            return None
        for entry in entries:
            if query in entry["title"].lower():
                return entry
        terms = self._terms(section_query)
        scored = [
            (self._score(terms, entry["title"]), entry)
            for entry in entries
        ]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]["page_number"] or 0))
        return scored[0][1]

    def _citation_items(self, *, document: dict[str, Any], citation: str) -> list[dict[str, Any]]:
        strict_patterns, fallback_patterns = self._citation_patterns(citation)
        matches = []
        for item in self._document_items(document=document):
            text = item.get("display_text", "")
            if any(pattern.search(text) for pattern in strict_patterns):
                matches.append(item)
        if matches:
            return matches
        for item in self._document_items(document=document):
            text = item.get("display_text", "")
            if any(pattern.search(text) for pattern in fallback_patterns):
                matches.append(item)
        return matches

    def _citation_patterns(self, citation: str) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
        stripped = citation.strip()
        strict_patterns = [re.compile(re.escape(stripped), flags=re.IGNORECASE)]
        fallback_patterns: list[re.Pattern[str]] = []
        number_match = re.search(r"\d+", stripped)
        if number_match:
            number = re.escape(number_match.group(0))
            strict_patterns.append(re.compile(rf"\[{number}\]", flags=re.IGNORECASE))
            fallback_patterns.extend(
                [
                    re.compile(rf"^\s*{number}\s*[\].)]", flags=re.IGNORECASE | re.MULTILINE),
                ]
            )
        return strict_patterns, fallback_patterns

    def _matched_documents_from_text(self, text: str, current_document_id: int) -> list[dict[str, Any]]:
        candidates = []
        for document in self._documents_by_id().values():
            document_id = int(document["id"])
            if document_id == current_document_id or document["status"] != "done":
                continue
            score = self._document_reference_score(text, document)
            if score <= 0:
                continue
            candidates.append({"document": self._compact_document(document), "score": score})
        candidates.sort(key=lambda candidate: (-candidate["score"], candidate["document"]["relative_path"]))
        return candidates

    def _document_reference_score(self, text: str, document: dict[str, Any]) -> int:
        normalized_text = self._reference_text(text)
        names = [
            str(document.get("file_name") or ""),
            str(Path(str(document.get("file_name") or "")).stem),
            str(document.get("relative_path") or ""),
        ]
        score = 0
        for name in names:
            normalized_name = self._reference_text(name)
            if normalized_name and normalized_name in normalized_text:
                score += 20
        name_terms = {
            term
            for name in names
            for term in self._terms(name)
            if term not in {"pdf", "rev"} and len(term) > 1
        }
        text_terms = set(self._terms(text))
        score += len(name_terms & text_terms)
        return score

    def _looks_like_reference_section(self, section_path: list[str]) -> bool:
        text = " ".join(section_path).lower()
        return any(marker in text for marker in ["literatur", "reference", "bibliograph"])

    def _reference_text(self, value: str) -> str:
        return " ".join(self._terms(value)).lower()

    def _is_structured_evidence_item(self, item: dict[str, Any]) -> bool:
        label = str(item.get("label", "")).lower()
        item_type = str(item.get("item_type", "")).lower()
        text = str(item.get("display_text", "")).lower()
        if label == "document_index":
            return False
        if label in {"table", "picture", "caption", "figure"}:
            return True
        if any(marker in item_type for marker in ["table", "picture", "image", "figure"]):
            return True
        return bool(re.match(r"\s*(tab\.?|table|abb\.?|fig\.?)\b", text))

    def _is_asset_item(self, item: dict[str, Any]) -> bool:
        label = str(item.get("label", "")).lower()
        item_type = str(item.get("item_type", "")).lower()
        if label == "document_index":
            return False
        return label in {"table", "picture", "figure"} or any(
            marker in item_type for marker in ["table", "picture", "image", "figure"]
        )

    def _item_display_text(self, *, row: Any, payload: dict[str, Any]) -> str:
        parts = [
            payload.get("asset_caption"),
            row["orig_text"],
            row["text"],
            payload.get("table_text"),
        ]
        text_parts: list[str] = []
        for part in parts:
            cleaned = self._clean_text(part or "")
            if cleaned and cleaned not in text_parts:
                text_parts.append(cleaned)
        return "\n".join(text_parts)

    def _document_index_entries(self, document_id: int) -> list[dict[str, Any]]:
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM document_items
                WHERE document_id = ? AND label = 'document_index'
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()

        entries: list[dict[str, Any]] = []
        for row in rows:
            payload = json_loads(row["payload_json"], {})
            entries.extend(self._document_index_entries_from_cells(payload.get("table_cells") or []))
            if not payload.get("table_cells") and payload.get("table_text"):
                entries.extend(self._document_index_entries_from_text(str(payload["table_text"])))
        return entries

    def _document_index_entries_from_cells(self, table_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
        row_map: dict[int, dict[int, str]] = {}
        for cell in table_cells:
            text = self._clean_text(cell.get("text") or "")
            if not text:
                continue
            row_map.setdefault(int(cell.get("row", 0)), {})[int(cell.get("col", 0))] = text
        return [
            entry
            for row_index in sorted(row_map)
            for entry in self._document_index_entries_from_values(
                [value for _, value in sorted(row_map[row_index].items())]
            )
        ]

    def _document_index_entries_from_text(self, table_text: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for line in table_text.splitlines():
            values = [self._clean_text(value) for value in line.split("|")]
            entries.extend(self._document_index_entries_from_values(values))
        return entries

    def _document_index_entries_from_values(self, values: list[str]) -> list[dict[str, Any]]:
        cleaned_values = [value for value in values if value]
        if len(cleaned_values) < 2:
            return []
        page_number = self._parse_index_page_number(cleaned_values[-1])
        if page_number is None:
            return []
        title = self._clean_text(" ".join(cleaned_values[:-1]))
        return [
            {
                "reference_key": reference_key,
                "page_number": page_number,
                "title": title,
            }
            for reference_key in sorted(self._structured_reference_keys(title))
        ]

    def _parse_index_page_number(self, value: str) -> int | None:
        text = self._clean_text(value)
        return int(text) if re.fullmatch(r"\d+", text) else None

    def _item_matches_structured_reference(
        self,
        *,
        item: dict[str, Any],
        all_items: list[dict[str, Any]],
        reference_keys: set[str],
        toc_target_pages: set[int],
    ) -> bool:
        if not reference_keys:
            return False
        if self._structured_reference_keys(self._item_reference_text(item)) & reference_keys:
            return True
        page_number = item["source_locator"].get("page_number")
        if page_number not in toc_target_pages:
            return False
        return bool(self._nearby_caption_reference_keys(item=item, all_items=all_items) & reference_keys)

    def _item_reference_text(self, item: dict[str, Any]) -> str:
        payload = item.get("payload") or {}
        parts = [
            item.get("display_text") or "",
            payload.get("asset_caption") or "",
            payload.get("table_text") or "",
            *[
                str(candidate.get("text") or "")
                for candidate in payload.get("asset_caption_candidates", [])
                if isinstance(candidate, dict)
            ],
        ]
        return "\n".join(part for part in parts if part)

    def _nearby_caption_reference_keys(self, *, item: dict[str, Any], all_items: list[dict[str, Any]]) -> set[str]:
        return {
            reference_key
            for caption in self._nearby_caption_items(item=item, all_items=all_items)
            for reference_key in self._structured_reference_keys(str(caption.get("display_text") or caption.get("text") or ""))
        }

    def _with_nearby_caption_text(
        self,
        *,
        item: dict[str, Any],
        all_items: list[dict[str, Any]],
        reference_keys: set[str],
        require_reference_match: bool = True,
    ) -> dict[str, Any]:
        captions = self._nearby_caption_items(item=item, all_items=all_items)
        matching_captions = [
            caption
            for caption in captions
            if self._structured_reference_keys(str(caption.get("display_text") or caption.get("text") or "")) & reference_keys
        ]
        if matching_captions:
            captions = matching_captions
        elif require_reference_match:
            return item
        for caption in captions:
            text = self._clean_text(caption.get("display_text") or caption.get("text") or "")
            if not text:
                continue
            display_text = item.get("display_text") or item.get("text") or ""
            if text in display_text:
                return item
            enriched = dict(item)
            enriched["payload"] = dict(item.get("payload") or {})
            enriched["payload"]["asset_caption"] = text
            enriched["display_text"] = f"{text}\n{display_text}".strip()
            enriched["text"] = enriched["display_text"]
            enriched["raw_text"] = enriched["display_text"]
            return enriched
        return item

    def _nearby_caption_items(self, *, item: dict[str, Any], all_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        page_number = item["source_locator"].get("page_number")
        position = int(item.get("position", 0))
        captions: list[dict[str, Any]] = []
        for candidate in all_items:
            if candidate["source_key"] == item["source_key"]:
                continue
            if candidate["source_locator"].get("page_number") != page_number:
                continue
            if abs(int(candidate.get("position", 0)) - position) > 4:
                continue
            text = candidate.get("display_text") or candidate.get("text") or ""
            if is_asset_caption_text(str(text)):
                captions.append(candidate)
        captions.sort(key=lambda candidate: abs(int(candidate.get("position", 0)) - position))
        return captions

    def _structured_reference_keys(self, text: str) -> set[str]:
        keys: set[str] = set()
        for match in re.finditer(r"\b(?:tab(?:elle)?|table)\.?\s*(\d+[A-Za-z]?)\b", text, flags=re.IGNORECASE):
            keys.add(f"table:{match.group(1).lower()}")
        for match in re.finditer(r"\b(?:abb(?:ildung)?|fig(?:ure)?)\.?\s*(\d+[A-Za-z]?)\b", text, flags=re.IGNORECASE):
            keys.add(f"figure:{match.group(1).lower()}")
        return keys

    def _linked_document_references(self, text: str, current_document_id: int) -> list[dict[str, Any]]:
        normalized = text.lower()
        matches = []
        for document in self._documents_by_id().values():
            document_id = int(document["id"])
            if document_id == current_document_id or document["status"] != "done":
                continue
            file_name = str(document["file_name"])
            stem = Path(file_name).stem
            candidates = {file_name.lower(), stem.lower(), stem.lower().replace("_", " "), stem.lower().replace("-", " ")}
            if any(candidate and candidate in normalized for candidate in candidates):
                matches.append({"document_id": document_id, "file_name": file_name})
        return matches

    def _score(self, terms: list[str], value: str) -> int:
        if not terms:
            return 1
        normalized = value.lower()
        score = sum(1 for term in terms if self._term_matches_text(term, normalized))
        phrase = " ".join(terms)
        if phrase and phrase in normalized:
            score += len(terms)
        return score

    def _terms(self, value: str) -> list[str]:
        return [match.group(0).lower() for match in WORD_RE.finditer(value)]

    def _important_terms(self, value: str) -> list[str]:
        return [
            term
            for term in self._terms(value)
            if (term.isdigit() or len(term) > 2) and term not in QUERY_STOPWORDS
        ]

    def _toc_entry_relevant(self, query: str, entry: dict[str, Any]) -> bool:
        title = str(entry["title"]).lower()
        terms = self._important_terms(query)
        if any(self._term_matches_text(term, title) for term in terms):
            return True
        if self._has_count_intent(query) and any(marker in title for marker in ["count", "number", "quantity", "anzahl", "menge"]):
            return True
        return False

    def _has_count_intent(self, query: str) -> bool:
        return any(marker in query.lower() for marker in ["how many", "count", "number", "anzahl", "wie viele"])

    def _has_count_signal(self, text: str) -> bool:
        if any(
            marker in text
            for marker in ["anzahl", "bis zu", "umfasst", "max", "summe", "total", "number", "count"]
        ):
            return True
        return bool(re.search(r"\b\d+\s*(?:be|brennelement|brennst)", text))

    def _term_matches_text(self, term: str, text: str) -> bool:
        if term in text:
            return True
        for candidate in self._terms(text):
            if len(candidate) < 5:
                continue
            if SequenceMatcher(None, term, candidate).ratio() >= 0.84:
                return True
        return False

    def _searchable_result_text(self, result: dict[str, Any]) -> str:
        return "\n".join(
            [
                result["document"]["file_name"],
                result["document"]["relative_path"],
                " ".join(result.get("section_path", [])),
                result.get("text", ""),
            ]
        )

    def _result_relevant_to_query(self, query: str, result: dict[str, Any]) -> bool:
        raw_text = str(result.get("raw_text") or "").lower()
        if any(self._term_matches_text(term, raw_text) for term in self._important_terms(query)):
            return True
        if self._has_count_intent(query) and self._has_count_signal(raw_text):
            return True
        return False

    def _first_page(self, pages_json: str) -> int | None:
        pages = json_loads(pages_json, [])
        return int(pages[0]) if pages else None

    def _clean_text(self, value: str) -> str:
        return " ".join(str(value).split())
