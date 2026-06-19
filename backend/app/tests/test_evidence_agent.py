from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.api.schemas import EvidenceRunRequest
from app.core.config import get_settings
from app.services.evidence_agent import (
    ActorDecision,
    AgentRuntimeError,
    CuratorDecision,
    EvidenceAgentRunner,
    EvidenceStanceDecision,
    PlannerDecision,
    parse_json_payload,
)


class AgentServiceStub:
    def __init__(self) -> None:
        self.last_search_document_ids: list[int] | None = None
        self.last_search_limit: int | None = None
        self.last_search_query: str | None = None
        self.last_list_tables_figures_args: dict[str, object] | None = None
        self.resolve_citation_calls = 0

    def _candidate_documents(self, document_scope: set[int]) -> list[dict[str, object]]:
        return [
            {
                "id": 1,
                "file_name": "CASTOR.pdf",
                "relative_path": "docs/CASTOR.pdf",
                "status": "done",
            }
        ]

    def get_readable_toc(self, document_id: int) -> dict[str, object]:
        return {
            "entries": [
                {"page_number": 1, "title": "1 Overview"},
                {"page_number": 2, "title": "2 Brennelemente"},
            ]
        }

    def search_keywords(self, *, query: str, document_ids: list[int] | None = None, limit: int = 8) -> dict[str, object]:
        self.last_search_query = query
        self.last_search_document_ids = document_ids
        self.last_search_limit = limit
        return {"results": []}

    def navigate_to_section(self, *, document_id: int, section_query: str) -> dict[str, object]:
        return {
            "document": {"id": 1, "relative_path": "docs/CASTOR.pdf", "file_name": "CASTOR.pdf", "status": "done", "page_count": 2},
            "matched_section": {"title": "2 Brennelemente"},
            "merged_text": "body",
            "ordered_items": [
                {
                    "result_type": "item",
                    "source_key": "doc:1:page:2:item:9",
                    "source_locator": {"document_id": 1, "page_number": 2, "item_id": 9},
                    "document": {"id": 1, "relative_path": "docs/CASTOR.pdf", "file_name": "CASTOR.pdf", "status": "done", "page_count": 2},
                    "label": "table",
                    "item_type": "TableItem",
                    "section_path": ["2 Brennelemente"],
                    "text": "Anzahl | 17",
                    "raw_text": "Anzahl | 17",
                    "display_text": "Anzahl | 17",
                    "payload": {"table_text": "Anzahl | 17"},
                }
            ],
            "chunks": [
                {
                    "result_type": "chunk",
                    "source_key": "doc:1:page:2:chunk:10",
                    "source_locator": {"document_id": 1, "page_number": 2, "chunk_id": 10},
                    "document": {"id": 1, "relative_path": "docs/CASTOR.pdf", "file_name": "CASTOR.pdf", "status": "done", "page_count": 2},
                    "section_path": ["2 Brennelemente"],
                    "text": "Der Behälter umfasst bis zu 17 Brennelemente.",
                    "raw_text": "Der Behälter umfasst bis zu 17 Brennelemente.",
                }
            ],
        }

    def list_tables_figures(self, **_: object) -> dict[str, object]:
        self.last_list_tables_figures_args = dict(_)
        return {
            "items": [
                {
                    "result_type": "item",
                    "source_key": "doc:1:page:2:item:9",
                    "source_locator": {"document_id": 1, "page_number": 2, "item_id": 9},
                    "document": {"id": 1, "relative_path": "docs/CASTOR.pdf", "file_name": "CASTOR.pdf", "status": "done", "page_count": 2},
                    "label": "table",
                    "item_type": "TableItem",
                    "section_path": ["2 Brennelemente"],
                    "text": "Anzahl | 17",
                    "raw_text": "Anzahl | 17",
                    "display_text": "Anzahl | 17",
                    "payload": {"table_text": "Anzahl | 17"},
                }
            ]
        }

    def inspect_item(self, **_: object) -> dict[str, object]:
        return {"item": self.list_tables_figures()["items"][0]}

    def resolve_citation(self, **_: object) -> dict[str, object]:
        self.resolve_citation_calls += 1
        reference = {
            "result_type": "item",
            "source_key": "doc:1:page:2:item:12",
            "source_locator": {"document_id": 1, "page_number": 2, "item_id": 12},
            "document": {"id": 1, "relative_path": "docs/CASTOR.pdf", "file_name": "CASTOR.pdf", "status": "done", "page_count": 2},
            "label": "text",
            "item_type": "TextItem",
            "section_path": ["References"],
            "text": "[3] Inventory.pdf",
            "raw_text": "[3] Inventory.pdf",
            "display_text": "[3] Inventory.pdf",
            "payload": {},
        }
        return {
            "citation": "[3]",
            "reference": reference,
            "references": [reference],
            "matched_document": {"id": 3, "relative_path": "docs/Inventory.pdf", "file_name": "Inventory.pdf", "status": "done", "page_count": 8},
        }

    def _append_section_results(self, **_: object) -> None:
        return None

    def _append_tool_results(self, **_: object) -> None:
        return None

    def _append_toc_results(self, **_: object) -> None:
        return None


class MultiDocumentServiceStub(AgentServiceStub):
    def _candidate_documents(self, document_scope: set[int]) -> list[dict[str, object]]:
        documents = [
            {
                "id": 1,
                "file_name": "CASTOR.pdf",
                "relative_path": "docs/CASTOR.pdf",
                "status": "done",
            },
            {
                "id": 3,
                "file_name": "Inventory.pdf",
                "relative_path": "docs/Inventory.pdf",
                "status": "done",
            },
        ]
        return [document for document in documents if not document_scope or int(document["id"]) in document_scope]


class EvidenceAgentDecisionParsingTests(unittest.TestCase):
    def test_agent_run_requires_exactly_one_start_document(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        self.assertEqual(runner._start_document_ids([1]), [1])
        with self.assertRaises(AgentRuntimeError):
            runner._start_document_ids(None)
        with self.assertRaises(AgentRuntimeError):
            runner._start_document_ids([1, 3])

    def test_planner_context_only_exposes_start_document(self) -> None:
        runner = EvidenceAgentRunner(
            service=MultiDocumentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        context = runner._planner_context({"document_ids": [1]})

        self.assertIn("docs/CASTOR.pdf", context)
        self.assertNotIn("docs/Inventory.pdf", context)

    def test_planner_context_includes_current_document(self) -> None:
        runner = EvidenceAgentRunner(
            service=MultiDocumentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        context = runner._planner_context({"document_ids": [1], "current_document_id": 3})

        self.assertIn("docs/CASTOR.pdf", context)
        self.assertIn("docs/Inventory.pdf", context)

    def test_invoke_json_records_step_keyed_model_input_and_output_traces(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(content='{"rationale":"start broad","tasks":[{"type":"search_keywords","reason":"find terms","params":{"query":"heat"}}]}')

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        runner.llm = LlmStub()
        state = {"counters": {"trace_step": 0}, "model_input_lines": [], "model_output_lines": []}

        decision = runner._invoke_json(PlannerDecision, system="system prompt", human="human prompt", state=state, node="planner")

        self.assertEqual(decision.rationale, "start broad")
        self.assertIn("[S001] planner PlannerDecision input", "\n".join(state["model_input_lines"]))
        self.assertIn("system prompt", "\n".join(state["model_input_lines"]))
        self.assertIn("[S001] planner PlannerDecision output", "\n".join(state["model_output_lines"]))
        self.assertIn("start broad", "\n".join(state["model_output_lines"]))

    def test_invoke_json_wraps_llm_transport_errors(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                raise RuntimeError("server disconnected")

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        runner.llm = LlmStub()
        state = {"counters": {"trace_step": 0}, "model_input_lines": [], "model_output_lines": []}

        with self.assertRaises(AgentRuntimeError) as exc:
            runner._invoke_json(PlannerDecision, system="system prompt", human="human prompt", state=state, node="planner")

        self.assertIn("LLM invocation failed for PlannerDecision", str(exc.exception))

    def test_planner_recovers_repeated_top_level_task_fields(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(
                    content="""
                    {
                      "type": "navigate_to_section",
                      "params": {
                        "document_id": 3,
                        "section_query": "4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert"
                      },
                      "type": "search_keywords",
                      "params": {
                        "query": "Nachzerfallsleistung",
                        "document_ids": [3],
                        "limit": 12
                      }
                    }
                    """
                )

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        runner.llm = LlmStub()
        state = {"counters": {"trace_step": 0}, "model_input_lines": [], "model_output_lines": []}

        decision = runner._invoke_json(PlannerDecision, system="system prompt", human="human prompt", state=state, node="planner")

        self.assertEqual(
            [task.type for task in decision.tasks],
            ["navigate_to_section", "search_keywords"],
        )
        self.assertEqual(decision.tasks[0].params["section_query"], "4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert")

    def test_planner_recovers_complete_tasks_before_truncated_trailing_task(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(
                    content="""
                    {
                      "tasks": [
                        { "type": "inspect_toc", "params": { "document_id": 3 } },
                        {
                          "type": "search_keywords",
                          "params": {
                            "query": "Stimmt die Anzahl von 17 Brennelementen mit der Inhaltsbeschreibung überein?",
                            "document_ids": [3],
                            "limit": 12
                          }
                        },
                        { "type": "list_tables_figures", "params": { "document_id": 3 } },
                        { "type": "resolve_citation", "params": { "document_id": 1, "citation": "[5]" } },
                        { "type": "navigate_to_section", "params": { "document_id": 3, "section_query": "2.1.1 Brennelemente" } },
                        { "type": "search_keywords", "params": { "query": "17 Brennelemente", "document_ids": [3] } },
                        { "type": "inspect_page", "params": { "document_id": 3, "page_num
                    """
                )

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        runner.llm = LlmStub()
        state = {"counters": {"trace_step": 0}, "model_input_lines": [], "model_output_lines": []}

        decision = runner._invoke_json(PlannerDecision, system="system prompt", human="human prompt", state=state, node="planner")

        self.assertEqual(
            [task.type for task in decision.tasks],
            [
                "inspect_toc",
                "search_keywords",
                "list_tables_figures",
                "resolve_citation",
                "navigate_to_section",
                "search_keywords",
            ],
        )
        self.assertEqual(decision.tasks[1].params["limit"], 12)
        self.assertEqual(decision.tasks[-1].params["query"], "17 Brennelemente")

    def test_planner_accepts_qwen_task_list_without_rationale_or_reasons(self) -> None:
        decision = PlannerDecision.model_validate(
            {
                "tasks": [
                    {"type": "inspect_toc", "params": {"document_id": "3"}},
                    {"type": "inspect_toc", "params": {"document_id": "1"}},
                ]
            }
        )

        self.assertEqual(decision.rationale, "No rationale provided by model.")
        self.assertEqual([task.reason for task in decision.tasks], ["No reason provided by model.", "No reason provided by model."])

    def test_document_resolver_accepts_numeric_ids_and_file_names(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        self.assertEqual(runner._resolve_document_id("1", {}), 1)
        self.assertEqual(runner._resolve_document_id("CASTOR.pdf", {}), 1)
        self.assertEqual(runner._resolve_document_id("docs/CASTOR.pdf", {}), 1)

    def test_curator_accepts_string_followups_as_planner_request(self) -> None:
        decision = CuratorDecision.model_validate(
            {
                "accepted_source_keys": ["doc:1:page:1:item:1"],
                "follow_up_tasks": [
                    "inspect_section:doc:1:page:1:item:1",
                    "search_for:alternate_terms:doc:1:page:1:item:1",
                ],
            }
        )

        self.assertEqual(decision.follow_up_tasks, [])
        self.assertIn("inspect_section:doc:1:page:1:item:1", decision.planner_request or "")

    def test_curator_accepts_structured_planner_request(self) -> None:
        decision = CuratorDecision.model_validate(
            {
                "accepted_source_keys": ["doc:3:page:7:chunk:589"],
                "planner_request": {
                    "type": "get_table",
                    "params": {
                        "document_id": "3",
                        "section_path": ["2 Behälterinventar", "2.1 Behälterinventar", "2.1.1 Brennelemente"],
                        "table_id": "1",
                    },
                },
            }
        )

        self.assertIn("get_table", decision.planner_request or "")
        self.assertIn("2.1.1 Brennelemente", decision.planner_request or "")

    def test_curator_accepts_multiple_evidence_relevance_notes(self) -> None:
        decision = CuratorDecision.model_validate(
            {
                "accepted_evidence": [
                    {
                        "source_key": "doc:1:page:2:chunk:10",
                        "relevance": "The sentence states the container holds up to 17 Brennelemente.",
                    },
                    {
                        "source_key": "doc:1:page:2:item:9",
                        "relevance": "The table row gives the count as 17.",
                    },
                ]
            }
        )

        self.assertEqual(
            [item.source_key for item in decision.accepted_evidence],
            ["doc:1:page:2:chunk:10", "doc:1:page:2:item:9"],
        )
        self.assertEqual(
            [item.relevance for item in decision.accepted_evidence],
            [
                "The sentence states the container holds up to 17 Brennelemente.",
                "The table row gives the count as 17.",
            ],
        )
        self.assertEqual(decision.accepted_source_keys, ["doc:1:page:2:chunk:10", "doc:1:page:2:item:9"])

    def test_curator_trims_copied_relevance_text(self) -> None:
        decision = CuratorDecision.model_validate(
            {
                "accepted_evidence": [
                    {
                        "source_key": "doc:3:page:22:chunk:651",
                        "relevance": " ".join(["copied table row"] * 40),
                    }
                ]
            }
        )

        self.assertLessEqual(len(decision.accepted_evidence[0].relevance), 240)

    def test_curator_decision_payload_omits_redundant_rationale(self) -> None:
        decision = CuratorDecision.model_validate(
            {
                "rationale": "This global explanation is ignored.",
                "accepted_evidence": [
                    {
                        "source_key": "doc:1:page:2:chunk:10",
                        "relevance": "The sentence gives the requested count.",
                    }
                ],
            }
        )

        self.assertNotIn("rationale", decision.model_dump())

    def test_evidence_stance_accepts_aliases_and_trims_reason(self) -> None:
        decision = EvidenceStanceDecision.model_validate(
            {
                "side": "against",
                "reason": " ".join(["The paragraph gives a conflicting count."] * 20),
            }
        )

        self.assertEqual(decision.stance, "against")
        self.assertLessEqual(len(decision.rationale), 180)

    def test_evidence_stance_accepts_neutral_when_unclear(self) -> None:
        decision = EvidenceStanceDecision.model_validate(
            {
                "classification": "unclear",
                "summary": "The chunk is relevant background but does not decide the claim.",
            }
        )

        self.assertEqual(decision.stance, "neutral")
        self.assertEqual(decision.rationale, "The chunk is relevant background but does not decide the claim.")

    def test_add_evidence_classifies_new_source_before_streaming_event(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(content='{"stance":"against","rationale":"The evidence gives a different count."}')

        events = []
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
            emit=events.append,
        )
        runner.llm = LlmStub()
        state = {
            "query": "Does the document say there are 12 fuel elements?",
            "run_id": "run-1",
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {},
            "graph_edges": [],
            "model_input_lines": [],
            "model_output_lines": [],
            "counters": {"trace_step": 0},
            "active_task": {"id": "T1"},
        }
        item = AgentServiceStub().list_tables_figures()["items"][0]

        runner._add_evidence(state, item, "The table row gives the count as 17.")

        evidence = state["evidence_by_key"]["doc:1:page:2:item:9"]
        self.assertEqual(evidence["stance"], "against")
        self.assertEqual(evidence["stance_rationale"], "The evidence gives a different count.")
        self.assertEqual(events[0]["evidence"]["stance"], "against")
        self.assertIn("Evaluator classified evidence doc:1:page:2:item:9 as against.", state["trace_events"][0]["message"])

    def test_add_evidence_can_stream_neutral_evidence(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(content='{"stance":"neutral","rationale":"The chunk is relevant but not decisive."}')

        events = []
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
            emit=events.append,
        )
        runner.llm = LlmStub()
        state = {
            "query": "Does the document say there are 12 fuel elements?",
            "run_id": "run-1",
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {},
            "graph_edges": [],
            "model_input_lines": [],
            "model_output_lines": [],
            "counters": {"trace_step": 0},
            "active_task": {"id": "T1"},
        }
        item = AgentServiceStub().list_tables_figures()["items"][0]

        runner._add_evidence(state, item, "The table row mentions fuel elements.")

        evidence = state["evidence_by_key"]["doc:1:page:2:item:9"]
        self.assertEqual(evidence["stance"], "neutral")
        self.assertEqual(events[0]["evidence"]["stance"], "neutral")

    def test_structured_curator_followups_are_planner_requests(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        decision = CuratorDecision.model_validate(
            {
                "follow_up_tasks": [
                    {
                        "type": "search_keywords",
                        "reason": "Search alternate term.",
                        "params": {"query": "fuel elements", "document_ids": [1]},
                    }
                ]
            }
        )

        requests = runner._planner_requests_from_curator(decision)

        self.assertEqual(len(requests), 1)
        self.assertIn("search_keywords", requests[0])

    def test_json_parser_repairs_missing_final_object_brace(self) -> None:
        payload = parse_json_payload('{"tasks": [{"type": "search_keywords", "params": {"query": "Brennelemente"}}]')

        self.assertEqual(payload["tasks"][0]["type"], "search_keywords")

    def test_planner_prompt_uses_isolated_task_summaries(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "query": "how many brennelemente",
            "document_ids": [1],
            "task_queue": [
                {
                    "id": "T2",
                    "type": "navigate_to_section",
                    "status": "queued",
                    "reason": "check section",
                    "params": {"document_id": 1, "section_query": "2 Brennelemente"},
                    "observation_count": 4,
                }
            ],
            "completed_tasks": [
                {
                    "id": "T1",
                    "type": "inspect_toc",
                    "status": "done",
                    "reason": "huge model-generated explanation that should not go back to planner",
                    "params": {"document_id": 1},
                    "observation_count": 80,
                }
            ],
            "planner_requests": ["Follow section 2 because the curator selected it."],
            "max_tasks": 32,
            "counters": {"task": 2},
        }

        prompt = runner._planner_prompt(state, runner._planner_context(state))

        self.assertIn("Already queued tasks", prompt)
        self.assertIn("Recently completed task summaries", prompt)
        self.assertIn("Remaining task budget: 30", prompt)
        self.assertIn("navigate_to_section", prompt)
        self.assertNotIn("huge model-generated explanation", prompt)
        self.assertNotIn("observation_count", prompt)
        self.assertNotIn("at most 4 tasks", prompt)

    def test_add_task_skips_duplicate_type_and_params(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "max_tasks": 4,
            "task_queue": [],
            "completed_tasks": [],
            "graph_nodes": {},
            "graph_edges": [],
            "counters": {"task": 0},
        }

        runner._add_task(state, "inspect_toc", "first", {"document_id": 1})
        runner._add_task(state, "inspect_toc", "duplicate", {"document_id": 1})

        self.assertEqual(len(state["task_queue"]), 1)

    def test_curator_recovers_accepted_evidence_when_optional_tail_is_truncated(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(
                    content=(
                        '{"accepted_evidence": ['
                        '{"source_key": "doc:3:page:22:chunk:651", "relevance": "Row 6 gives 1,963 kW."}'
                        '], "planner_request": {"type": "resolve_'
                    )
                )

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        runner.llm = LlmStub()

        decision = runner._invoke_json(
            CuratorDecision,
            system="system prompt",
            human="human prompt",
            state={"counters": {"trace_step": 0}, "model_input_lines": [], "model_output_lines": []},
            node="curator",
        )

        self.assertEqual(decision.accepted_source_keys, ["doc:3:page:22:chunk:651"])
        self.assertIsNone(decision.planner_request)

    def test_agentic_retrieval_uses_larger_default_and_safety_budget(self) -> None:
        self.assertEqual(EvidenceRunRequest(query="compare referenced limits", document_ids=[1]).max_tasks, 32)
        settings = get_settings()
        self.assertEqual(settings.agent_max_tasks, 250)
        self.assertEqual(settings.agent_max_tool_calls, 250)
        self.assertEqual(settings.agent_max_evidence, 250)
        self.assertEqual(settings.agent_max_graph_steps, 800)

    def test_add_task_normalizes_model_tool_aliases(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "max_tasks": 4,
            "task_queue": [],
            "completed_tasks": [],
            "graph_nodes": {},
            "graph_edges": [],
            "counters": {"task": 0},
        }

        runner._add_task(state, "resolve_reference", "follow citation", {"document_id": 1, "citation": "[3]"})

        self.assertEqual(state["task_queue"][0]["type"], "resolve_citation")

    def test_structured_planner_requests_accept_model_tool_aliases(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "max_tasks": 4,
            "task_queue": [],
            "completed_tasks": [],
            "active_task": {"id": "T1"},
            "planner_requests": ['{"type": "resolve_reference", "params": {"document_id": 1, "citation": "[3]"}}'],
            "graph_nodes": {},
            "graph_edges": [],
            "counters": {"task": 0},
        }

        added = runner._add_tasks_from_structured_planner_requests(state)

        self.assertEqual(len(added), 1)
        self.assertEqual(state["task_queue"][0]["type"], "resolve_citation")

    def test_structured_planner_requests_accept_top_level_task_lists(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "max_tasks": 4,
            "task_queue": [],
            "completed_tasks": [],
            "active_task": {"id": "T1"},
            "planner_requests": [
                '[{"type": "inspect_toc", "params": {"document_id": 1}}, {"type": "search_keywords", "params": {"query": "heat", "document_ids": [1]}}]'
            ],
            "graph_nodes": {},
            "graph_edges": [],
            "counters": {"task": 0},
        }

        added = runner._add_tasks_from_structured_planner_requests(state)

        self.assertEqual([task["type"] for task in added], ["inspect_toc", "search_keywords"])
        self.assertEqual([task["type"] for task in state["task_queue"]], ["inspect_toc", "search_keywords"])

    def test_structured_planner_requests_skip_tasks_missing_required_params(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "max_tasks": 4,
            "task_queue": [],
            "completed_tasks": [],
            "active_task": {"id": "T1"},
            "planner_requests": ['{"type": "resolve_reference", "reference": "[1]"}'],
            "graph_nodes": {},
            "graph_edges": [],
            "counters": {"task": 0},
        }

        added = runner._add_tasks_from_structured_planner_requests(state)

        self.assertEqual(added, [])
        self.assertEqual(state["task_queue"], [])

    def test_section_query_can_come_from_section_path(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        section_query = runner._section_query_from_args(
            {"section_path": ["2 Behälterinventar", "2.1 Behälterinventar", "2.1.1 Brennelemente"]}
        )

        self.assertEqual(section_query, "2.1.1 Brennelemente")

    def test_actor_prompt_advertises_structured_retrieval_tools(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        prompt = runner._planner_prompt(
            {"query": "check reference", "task_queue": [], "completed_tasks": [], "planner_requests": []},
            "Document 1: docs/CASTOR.pdf",
        )

        self.assertIn("resolve_citation", prompt)
        self.assertIn("list_tables_figures", prompt)
        self.assertIn("inspect_item", prompt)

    def test_section_navigation_returns_chunk_and_structured_item_candidates(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "inspect section"},
            "tool_result_lines": [],
            "document_ids": [1],
        }

        observations = runner._execute_tool("navigate_to_section", {"document_id": 1, "section_query": "2 Brennelemente"}, state)

        self.assertEqual(
            {observation["source_key"] for observation in observations},
            {"doc:1:page:2:chunk:10", "doc:1:page:2:item:9"},
        )

    def test_agent_executes_reference_and_item_tools(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "follow citation"},
            "tool_result_lines": [],
            "document_ids": [1],
        }

        reference_observations = runner._execute_tool("resolve_citation", {"document_id": 1, "citation": "[3]"}, state)
        table_observations = runner._execute_tool(
            "list_tables_figures",
            {"document_id": 1, "section_query": "2 Brennelemente"},
            state,
        )
        item_observations = runner._execute_tool("inspect_item", {"document_id": 1, "item_id": 9}, state)

        self.assertEqual(reference_observations[0]["resolved_document"]["id"], 3)
        self.assertEqual(table_observations[0]["source_key"], "doc:1:page:2:item:9")
        self.assertEqual(item_observations[0]["source_key"], "doc:1:page:2:item:9")

    def test_item_candidates_can_be_added_as_evidence(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "evidence_by_key": {},
            "graph_nodes": {},
            "graph_edges": [],
            "active_task": {"id": "T1"},
        }
        item = AgentServiceStub().list_tables_figures()["items"][0]

        runner._add_evidence(state, item, "The table row gives the count as 17.")

        evidence = state["evidence_by_key"]["doc:1:page:2:item:9"]
        self.assertEqual(evidence["text"], "Anzahl | 17")
        self.assertEqual(evidence["relevance"], "The table row gives the count as 17.")
        self.assertEqual(evidence["source_locator"]["item_id"], 9)
        self.assertNotIn("rationales", evidence)

    def test_add_evidence_emits_evidence_added_event_for_new_source(self) -> None:
        events = []
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_graph_steps=20,
                agent_max_tool_calls=20,
                agent_max_evidence=20,
            ),
            emit=events.append,
        )
        state = {
            "run_id": "run-1",
            "evidence_by_key": {},
            "graph_nodes": {},
            "graph_edges": [],
            "active_task": {"id": "T1"},
        }
        item = AgentServiceStub().list_tables_figures()["items"][0]

        runner._add_evidence(state, item, "The table row gives the count as 17.")

        self.assertEqual(events[0]["type"], "evidence_added")
        self.assertEqual(events[0]["run_id"], "run-1")
        self.assertEqual(events[0]["evidence"]["source_key"], "doc:1:page:2:item:9")
        self.assertEqual(events[0]["evidence_count"], 1)

    def test_add_evidence_deduplicates_repeated_acceptance_from_same_task(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "evidence_by_key": {},
            "graph_nodes": {},
            "graph_edges": [],
            "active_task": {"id": "T1"},
        }
        item = AgentServiceStub().list_tables_figures()["items"][0]

        runner._add_evidence(state, item, "The table row gives the count as 17.")
        runner._add_evidence(state, item, "The table row gives the count as 17.")

        evidence = state["evidence_by_key"]["doc:1:page:2:item:9"]
        self.assertEqual(evidence["found_by_tasks"], ["T1"])
        self.assertEqual(state["graph_edges"], [{"from": "T1", "to": "E1", "type": "found"}])

    def test_actor_node_emits_current_task_status_before_running_tool(self) -> None:
        events = []
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_graph_steps=20,
                agent_max_tool_calls=20,
                agent_max_evidence=20,
            ),
            emit=events.append,
        )
        runner._invoke_json = lambda *_args, **_kwargs: ActorDecision(
            rationale="Use the planned search.",
            tool_name="search_keywords",
            tool_args={"query": "heat"},
        )
        state = {
            "run_id": "run-1",
            "query": "heat",
            "document_ids": [1],
            "current_document_id": 1,
            "max_tasks": 4,
            "task_queue": [
                {
                    "id": "T1",
                    "type": "search_keywords",
                    "status": "queued",
                    "reason": "Find heat references.",
                    "params": {"query": "heat"},
                }
            ],
            "completed_tasks": [],
            "active_task": None,
            "candidate_observations": [],
            "trace_events": [],
            "tool_result_lines": [],
            "graph_nodes": {},
            "graph_edges": [],
            "counters": {"graph_steps": 0, "tool_calls": 0, "trace_step": 0},
        }

        runner._actor_node(state)

        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[0]["node"], "actor")
        self.assertEqual(events[0]["label"], "Actor: search_keywords")
        self.assertEqual(events[0]["task"]["id"], "T1")

    def test_compact_candidates_does_not_hide_tables_behind_empty_pictures(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        empty_pictures = [
            {
                "result_type": "item",
                "source_key": f"doc:3:page:{index}:item:{900 + index}",
                "label": "picture",
                "document": {"relative_path": "docs/Inventory.pdf"},
                "section_path": [],
                "display_text": "",
                "payload": {},
            }
            for index in range(12)
        ]
        candidates = [
            *empty_pictures,
            {
                "result_type": "item",
                "source_key": "doc:3:page:6:item:937",
                "label": "table",
                "document": {"relative_path": "docs/Inventory.pdf"},
                "section_path": ["1 Einleitung und Zusammenfassung"],
                "display_text": "Nachzerfallsleistung: max. 5,597 kW I Behälter",
                "payload": {"table_text": "Nachzerfallsleistung: | max. 5,597 kW I Behälter"},
            },
        ]

        compact = runner._compact_candidates(candidates)

        self.assertIn("doc:3:page:6:item:937", [candidate["source_key"] for candidate in compact])
        self.assertFalse(any(candidate["label"] == "picture" and not candidate["text"] for candidate in compact))

    def test_curator_prompt_is_limited_to_evidence_selection(self) -> None:
        class LlmStub:
            def __init__(self) -> None:
                self.messages: object | None = None

            def invoke(self, messages: object) -> object:
                self.messages = messages
                return SimpleNamespace(content='{"accepted_evidence": []}')

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_tool_calls=10,
                agent_max_evidence=10,
                agent_max_graph_steps=10,
            ),
        )
        llm = LlmStub()
        runner.llm = llm
        state = {
            "query": "Ist die Wärmezerfallsleistung konservativ?",
            "max_tasks": 4,
            "task_queue": [],
            "completed_tasks": [],
            "active_task": {"id": "T1", "type": "inspect_chunk", "params": {"document_id": 3, "chunk_id": 651}},
            "candidate_observations": [
                {
                    "result_type": "chunk",
                    "source_key": "doc:3:page:22:chunk:651",
                    "source_locator": {"document_id": 3, "page_number": 22, "chunk_id": 651},
                    "document": {"id": 3, "relative_path": "docs/Inventory.pdf"},
                    "section_path": ["4.2 Nachzerfallsleistung"],
                    "raw_text": "Der Referenzwert fuer die Nachzerfallsleistung ist konservativ.",
                    "payload": {},
                }
            ],
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {},
            "graph_edges": [],
            "planner_requests": [],
            "counters": {"graph_steps": 0, "tool_calls": 0, "trace_step": 0},
        }

        runner._curator_node(state)

        assert llm.messages is not None
        system_prompt = llm.messages[0].content
        human_prompt = llm.messages[1].content
        self.assertIn("Do not return planner_request", system_prompt)
        self.assertIn("relevance under 160 characters", system_prompt)
        self.assertIn('"chunk_id": 651', human_prompt)
        self.assertIn('{"accepted_evidence": []}', human_prompt)

    def test_compact_candidates_exposes_asset_caption_text(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        compact = runner._compact_candidates(
            [
                {
                    "result_type": "item",
                    "source_key": "doc:1:page:17:item:1202",
                    "label": "table",
                    "document": {"relative_path": "docs/Report.pdf"},
                    "section_path": ["3.4 Brennelemente und Wärmeleistung"],
                    "display_text": "Qmax,BE | 572 W",
                    "payload": {
                        "asset_caption": "Tabelle 3: Vergleich des maximalen Wärmestroms",
                        "table_text": "Qmax,BE | 572 W",
                    },
                }
            ]
        )

        self.assertEqual(compact[0]["asset_caption"], "Tabelle 3: Vergleich des maximalen Wärmestroms")

    def test_search_tool_resolves_document_id_strings_before_calling_service(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "search document"},
            "tool_result_lines": [],
            "document_ids": [1],
        }

        runner._execute_tool("search_keywords", {"query": "17 Brennelemente", "document_ids": ["docs/CASTOR.pdf"]}, state)

        self.assertEqual(service.last_search_document_ids, [1])

    def test_search_without_scope_defaults_to_current_document(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "query": "fallback query",
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "search current document"},
            "tool_result_lines": [],
            "document_ids": [1],
            "current_document_id": 3,
        }

        runner._execute_tool("search_keywords", {"query": "heat"}, state)

        self.assertEqual(service.last_search_document_ids, [3])

    def test_search_scope_all_defaults_to_current_when_corpus_search_is_not_enabled(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "query": "fallback query",
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "search all documents"},
            "tool_result_lines": [],
            "document_ids": [1],
            "current_document_id": 1,
        }

        runner._execute_tool("search_keywords", {"query": "heat", "scope": "all"}, state)

        self.assertEqual(service.last_search_document_ids, [1])

    def test_search_scope_all_searches_across_documents_when_enabled(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "query": "fallback query",
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "search all documents"},
            "tool_result_lines": [],
            "document_ids": [1],
            "current_document_id": 1,
            "search_scope": "all",
        }

        runner._execute_tool("search_keywords", {"query": "heat", "scope": "all"}, state)

        self.assertIsNone(service.last_search_document_ids)

    def test_active_task_search_query_is_authoritative_over_actor_rewrite(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {
                "id": "T1",
                "reason": "search exact term",
                "params": {"query": "Wärmeleistung", "document_ids": [1]},
            },
            "tool_result_lines": [],
            "document_ids": [1, 3],
        }

        runner._execute_tool(
            "search_keywords",
            {"query": "Wärmezerfallsleistung", "document_ids": [1]},
            state,
        )

        self.assertEqual(service.last_search_query, "Wärmeleistung")

    def test_active_task_params_drop_actor_optional_narrowing(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {
                "id": "T1",
                "type": "list_tables_figures",
                "reason": "inspect all structured evidence",
                "params": {"document_id": 3},
            },
            "tool_result_lines": [],
            "document_ids": [1],
            "current_document_id": 3,
        }

        runner._execute_tool(
            "list_tables_figures",
            {"document_id": 3, "section_query": "Wärmeleistung", "page_number": 1},
            state,
        )

        self.assertEqual(
            service.last_list_tables_figures_args,
            {"document_id": 3, "section_query": None, "page_number": None},
        )

    def test_active_task_search_limit_is_authoritative(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {
                "id": "T1",
                "type": "search_keywords",
                "reason": "search exact term",
                "params": {"query": "Wärmeleistung", "document_ids": [1]},
            },
            "tool_result_lines": [],
            "document_ids": [1],
        }

        runner._execute_tool("search_keywords", {"query": "Wärmeleistung", "document_ids": [1], "limit": 1}, state)

        self.assertEqual(service.last_search_limit, 8)

    def test_candidate_citations_become_structured_planner_requests(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        candidates = [
            {
                "result_type": "chunk",
                "source_key": "doc:1:page:2:chunk:10",
                "source_locator": {"document_id": 1},
                "raw_text": "In [3] wurde die Referenzbeladung festgelegt.",
            }
        ]

        requests = runner._planner_requests_from_candidate_leads(candidates, query="check")

        self.assertIn('"type": "resolve_citation"', requests[0])
        self.assertIn('"citation": "[3]"', requests[0])

    def test_candidate_section_hits_become_navigation_requests(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        candidates = [
            {
                "result_type": "chunk",
                "source_key": "doc:3:page:14:chunk:626",
                "source_locator": {"document_id": 3},
                "section_path": ["4 Ergebnisse", "4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert"],
                "raw_text": "Der Referenzwert für die Nachzerfallsleistung ist der abdeckende konservative Wert.",
            }
        ]

        requests = runner._planner_requests_from_candidate_leads(
            candidates,
            query="Entspricht die Wärmeleistung der Wärmezerfallsleistung?",
            state={"task_queue": [], "completed_tasks": []},
        )

        self.assertTrue(any('"type": "navigate_to_section"' in request for request in requests))
        self.assertTrue(any('"section_query": "4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert"' in request for request in requests))
        self.assertTrue(any('"priority": "high"' in request for request in requests))

    def test_candidate_section_hits_ignore_generic_heat_model_sections(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        candidates = [
            {
                "result_type": "chunk",
                "source_key": "doc:1:page:29:chunk:838",
                "source_locator": {"document_id": 1},
                "section_path": ["5.2 Behälter und Inventar", "5.2.2 Radialmodell"],
                "raw_text": (
                    "Im Radialmodell werden axiale Wärmeströme durch Anpassung der Wärmeleistung "
                    "der Brennelemente modelliert."
                ),
            }
        ]

        requests = runner._planner_requests_from_candidate_leads(
            candidates,
            query="Entspricht die Wärmeleistung der Wärmezerfallsleistung?",
            state={"task_queue": [], "completed_tasks": []},
        )

        self.assertFalse(any('"type": "navigate_to_section"' in request for request in requests))

    def test_compact_candidates_prioritizes_relevant_structured_items(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        document = {"id": 3, "relative_path": "docs/Inventory.pdf", "file_name": "Inventory.pdf", "status": "done"}
        candidates = [
            {
                "result_type": "item",
                "source_key": f"doc:3:page:{index}:item:{index}",
                "source_locator": {"document_id": 3, "page_number": index, "item_id": index},
                "document": document,
                "label": "table",
                "item_type": "TableItem",
                "section_path": ["4 Ergebnisse", "4.4 Aktivitätsinventar der Referenzbeladungen"],
                "text": "Aktivitätsinventar und Referenzwert für den Behälter",
                "payload": {"table_text": "Aktivitätsinventar | konservativ"},
            }
            for index in range(14)
        ]
        candidates.append(
            {
                "result_type": "item",
                "source_key": "doc:3:page:22:item:1041",
                "source_locator": {"document_id": 3, "page_number": 22, "item_id": 1041},
                "document": document,
                "label": "table",
                "item_type": "TableItem",
                "section_path": ["4 Ergebnisse", "Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter"],
                "text": "Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter",
                "payload": {"table_text": "Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter | 5,597 kW"},
            }
        )

        compact = runner._compact_candidates(
            candidates,
            query="Entspricht die Wärmeleistung der Wärmezerfallsleistung und ist der Wert konservativ?",
        )

        source_keys = [candidate["source_key"] for candidate in compact]
        self.assertIn("doc:3:page:22:item:1041", source_keys)
        self.assertEqual(source_keys[0], "doc:3:page:22:item:1041")

    def test_decay_power_query_rejects_unrelated_inventory_evidence(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        query = "Ist die Wärmezerfallsleistung konservativ?"
        unrelated = {
            "result_type": "chunk",
            "source_key": "doc:3:page:15:chunk:634",
            "section_path": ["4.5 Spaltgasinventar der Referenzbeladungen und Referenzwert für den Behälter"],
            "raw_text": "Das Spaltgasinventar wird konservativ bestimmt.",
            "payload": {},
        }
        related = {
            "result_type": "chunk",
            "source_key": "doc:3:page:14:chunk:626",
            "section_path": ["4.2 Nachzerfallsleistung der Referenzbeladungen und Referenzwert"],
            "raw_text": "Der Referenzwert für die Nachzerfallsleistung ist konservativ.",
            "payload": {},
        }

        self.assertFalse(runner._candidate_is_query_supporting_evidence(unrelated, query=query))
        self.assertTrue(runner._candidate_is_query_supporting_evidence(related, query=query))

    def test_candidate_citations_use_cached_reference_resolution(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "resolved_references": {
                "docs/CASTOR.pdf#[3]": {
                    "citation": "[3]",
                    "document_id": 1,
                    "matched_document": {"id": 3, "relative_path": "docs/Inventory.pdf"},
                }
            }
        }
        candidates = [
            {
                "result_type": "chunk",
                "source_key": "doc:1:page:2:chunk:10",
                "source_locator": {"document_id": 1},
                "raw_text": "In [3] wurde die Referenzbeladung festgelegt.",
            }
        ]

        requests = runner._planner_requests_from_candidate_leads(candidates, query="check", state=state)

        self.assertFalse(any('"type": "resolve_citation"' in request for request in requests))
        self.assertTrue(any('"type": "inspect_toc"' in request and '"document_id": 3' in request for request in requests))
        self.assertTrue(any('"type": "search_keywords"' in request and '"document_ids": [3]' in request for request in requests))

    def test_resolved_document_candidate_becomes_search_and_toc_requests(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        candidates = [
            {
                "result_type": "item",
                "source_key": "doc:1:page:2:item:12",
                "source_locator": {"document_id": 1},
                "resolved_document": {"id": 3, "relative_path": "docs/Inventory.pdf"},
                "display_text": "[3] Inventory.pdf",
            }
        ]

        requests = runner._planner_requests_from_candidate_leads(candidates, query="how many fuel elements")

        self.assertTrue(any('"type": "inspect_toc"' in request and '"document_id": 3' in request for request in requests))
        self.assertTrue(any('"type": "search_keywords"' in request and '"document_ids": [3]' in request for request in requests))
        self.assertTrue(any('"type": "list_tables_figures"' in request and '"document_id": 3' in request for request in requests))

    def test_resolve_citation_uses_cache_for_repeated_reference(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {"id": "T1", "reason": "resolve reference"},
            "tool_result_lines": [],
            "document_ids": [1],
            "resolved_references": {},
        }

        first = runner._execute_tool("resolve_citation", {"document_id": 1, "citation": "[3]"}, state)
        second = runner._execute_tool("resolve_citation", {"document_id": 1, "citation": "3"}, state)

        self.assertEqual(service.resolve_citation_calls, 1)
        self.assertEqual(first[0]["resolved_document"]["id"], 3)
        self.assertEqual(second[0]["resolved_document"]["id"], 3)
        self.assertIn("docs/CASTOR.pdf#[3]", state["resolved_references"])

    def test_resolve_citation_inserts_resolved_document_tasks_before_existing_queue(self) -> None:
        service = AgentServiceStub()
        runner = EvidenceAgentRunner(
            service=service,
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "query": "how many fuel elements",
            "max_tasks": 8,
            "counters": {"task": 2, "tool_calls": 0},
            "active_task": {
                "id": "T2",
                "type": "resolve_citation",
                "reason": "resolve reference",
                "params": {"document_id": 1, "citation": "[3]"},
            },
            "task_queue": [
                {
                    "id": "T99",
                    "type": "search_keywords",
                    "status": "queued",
                    "reason": "older queued work",
                    "params": {"query": "older", "document_ids": [1]},
                }
            ],
            "completed_tasks": [{"id": "T1", "type": "inspect_toc", "params": {"document_id": 1}}],
            "graph_nodes": {"T2": {"id": "T2", "type": "task", "label": "resolve_citation"}},
            "graph_edges": [],
            "tool_result_lines": [],
            "document_ids": [1],
            "root_document_id": 1,
            "current_document_id": 1,
            "resolved_references": {},
        }

        runner._execute_tool("resolve_citation", {"document_id": 1, "citation": "[3]"}, state)

        inserted = state["task_queue"][:3]
        self.assertEqual([task["type"] for task in inserted], ["inspect_toc", "search_keywords", "list_tables_figures"])
        self.assertEqual(state["task_queue"][3]["id"], "T99")
        for task in inserted:
            self.assertEqual(task["document_id"], 3)
            self.assertEqual(task["origin_document_id"], 1)
            self.assertEqual(task["origin_citation"], "[3]")
            self.assertEqual(task["return_document_id"], 1)
        self.assertEqual(inserted[1]["params"]["document_ids"], [3])
        self.assertEqual(
            [edge for edge in state["graph_edges"] if edge["from"] == "T2"],
            [
                {"from": "T2", "to": inserted[0]["id"], "type": "suggested_followup"},
                {"from": "T2", "to": inserted[1]["id"], "type": "suggested_followup"},
                {"from": "T2", "to": inserted[2]["id"], "type": "suggested_followup"},
            ],
        )

    def test_structured_planner_requests_insert_before_existing_queue(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_tool_calls=10,
                agent_max_evidence=10,
                agent_max_graph_steps=10,
            ),
        )
        state = {
            "query": "compare heat values",
            "document_ids": [1],
            "root_document_id": 1,
            "current_document_id": 1,
            "max_tasks": 4,
            "task_queue": [
                {
                    "id": "T1",
                    "type": "search_keywords",
                    "status": "queued",
                    "reason": "older queued work",
                    "params": {"query": "older", "document_ids": [1]},
                }
            ],
            "completed_tasks": [],
            "active_task": {"id": "T0"},
            "candidate_observations": [],
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {"T1": {"id": "T1", "type": "task", "label": "search_keywords"}},
            "graph_edges": [],
            "tool_result_lines": [],
            "model_input_lines": [],
            "model_output_lines": [],
            "planner_requests": [
                (
                    '{"type": "navigate_to_section", "priority": "high", '
                    '"reason": "fresh section", '
                    '"params": {"document_id": 3, "section_query": "4.2 Nachzerfallsleistung"}}'
                )
            ],
            "resolved_references": {},
            "counters": {"task": 1, "tool_calls": 0, "graph_steps": 0, "trace_step": 0},
            "stop_reason": "",
        }

        next_state = runner._planner_node(state)

        self.assertEqual(next_state["task_queue"][0]["type"], "navigate_to_section")
        self.assertEqual(next_state["task_queue"][1]["reason"], "older queued work")

    def test_planner_followup_tasks_insert_before_existing_queue(self) -> None:
        class LlmStub:
            def invoke(self, _messages: object) -> object:
                return SimpleNamespace(
                    content=(
                        '{"rationale": "fresh lead", "tasks": ['
                        '{"type": "navigate_to_section", "reason": "fresh section", '
                        '"params": {"document_id": 3, "section_query": "4.2 Nachzerfallsleistung"}}'
                        "]}"
                    )
                )

        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_tool_calls=10,
                agent_max_evidence=10,
                agent_max_graph_steps=10,
            ),
        )
        runner.llm = LlmStub()
        state = {
            "query": "compare heat values",
            "document_ids": [1],
            "root_document_id": 1,
            "current_document_id": 1,
            "max_tasks": 4,
            "task_queue": [
                {
                    "id": "T1",
                    "type": "search_keywords",
                    "status": "queued",
                    "reason": "older queued work",
                    "params": {"query": "older", "document_ids": [1]},
                }
            ],
            "completed_tasks": [],
            "active_task": None,
            "candidate_observations": [],
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {"T1": {"id": "T1", "type": "task", "label": "search_keywords"}},
            "graph_edges": [],
            "tool_result_lines": [],
            "model_input_lines": [],
            "model_output_lines": [],
            "planner_requests": ["Inspect the newly discovered Nachzerfallsleistung section."],
            "resolved_references": {},
            "counters": {"task": 1, "tool_calls": 0, "graph_steps": 0, "trace_step": 0},
            "stop_reason": "",
        }

        next_state = runner._planner_node(state)

        self.assertEqual(next_state["task_queue"][0]["type"], "navigate_to_section")
        self.assertEqual(next_state["task_queue"][1]["reason"], "older queued work")

    def test_tool_execution_uses_active_task_params_when_actor_returns_placeholders(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )
        state = {
            "counters": {"tool_calls": 0},
            "active_task": {
                "id": "T1",
                "reason": "inspect section",
                "params": {"document_id": 1, "section_query": "2 Brennelemente"},
            },
            "tool_result_lines": [],
            "document_ids": [1],
        }

        observations = runner._execute_tool(
            "navigate_to_section",
            {"document_id": "document_id", "section_query": "section_query"},
            state,
        )

        self.assertIn("doc:1:page:2:chunk:10", {observation["source_key"] for observation in observations})

    def test_actor_uses_active_task_tool_when_model_selects_wrong_tool(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_tool_calls=10,
                agent_max_evidence=10,
                agent_max_graph_steps=10,
            ),
        )
        runner._invoke_json = lambda *_args, **_kwargs: ActorDecision(
            tool_name="inspect_item",
            tool_args={"document_id": 1, "item_id": 9},
        )
        state = {
            "query": "inspect toc",
            "document_ids": [1],
            "max_tasks": 4,
            "task_queue": [
                {
                    "id": "T1",
                    "type": "inspect_toc",
                    "status": "queued",
                    "reason": "read the table of contents",
                    "params": {"document_id": 1},
                }
            ],
            "completed_tasks": [],
            "active_task": None,
            "candidate_observations": [],
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {},
            "graph_edges": [],
            "tool_result_lines": [],
            "planner_requests": [],
            "counters": {"task": 1, "tool_calls": 0, "graph_steps": 0},
            "stop_reason": "",
        }

        next_state = runner._actor_node(state)

        self.assertEqual(next_state["completed_tasks"][0]["status"], "done")
        self.assertEqual([observation["title"] for observation in next_state["candidate_observations"]], ["1 Overview", "2 Brennelemente"])
        self.assertEqual(next_state["trace_events"][-1]["data"]["tool"]["tool_name"], "inspect_toc")

    def test_actor_records_failed_tool_call_instead_of_aborting_run(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(
                ollama_model="qwen3:latest",
                ollama_base_url="http://127.0.0.1:8880",
                agent_max_tool_calls=10,
                agent_max_evidence=10,
                agent_max_graph_steps=10,
            ),
        )
        runner._invoke_json = lambda *_args, **_kwargs: ActorDecision(
            tool_name="navigate_to_section",
            tool_args={"document_id": "document_id", "section_query": "Wärmeleistung"},
        )
        state = {
            "query": "compare heat values",
            "document_ids": [1, 3],
            "max_tasks": 4,
            "task_queue": [{"id": "T1", "type": "navigate_to_section", "status": "queued", "reason": "bad model args", "params": {}}],
            "completed_tasks": [],
            "active_task": None,
            "candidate_observations": [],
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {},
            "graph_edges": [],
            "tool_result_lines": [],
            "planner_requests": [],
            "counters": {"task": 1, "tool_calls": 0, "graph_steps": 0},
            "stop_reason": "",
        }

        next_state = runner._actor_node(state)

        self.assertEqual(next_state["completed_tasks"][0]["status"], "failed")
        self.assertEqual(next_state["candidate_observations"], [])
        self.assertIn("Could not resolve document_id", next_state["completed_tasks"][0]["error"])
        self.assertTrue(any(event["node"] == "actor_tool_error" for event in next_state["trace_events"]))

    def test_document_placeholder_resolves_to_single_scoped_document(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        self.assertEqual(runner._resolve_document_id("document_id", {"document_ids": [1]}), 1)

    def test_document_placeholder_resolves_to_current_document(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        self.assertEqual(runner._resolve_document_id("document_id", {"document_ids": [1], "current_document_id": 3}), 3)

    def test_source_key_ids_can_be_used_for_chunk_and_item_args(self) -> None:
        runner = EvidenceAgentRunner(
            service=AgentServiceStub(),
            settings=SimpleNamespace(ollama_model="qwen3:latest", ollama_base_url="http://127.0.0.1:8880"),
        )

        self.assertEqual(runner._source_key_int("doc:1:page:2:chunk:10", "chunk"), 10)
        self.assertEqual(runner._source_key_int("doc:1:page:2:item:9", "item"), 9)


if __name__ == "__main__":
    unittest.main()
