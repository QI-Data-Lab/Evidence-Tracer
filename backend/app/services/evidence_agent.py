from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, model_validator


DEFAULT_MODEL_REASON = "No reason provided by model."
DEFAULT_MODEL_RATIONALE = "No rationale provided by model."
MAX_RELEVANCE_CHARS = 240
MAX_ANALYSIS_TEXT_CHARS = 700
MAX_STANCE_RATIONALE_CHARS = 180


class AgentRuntimeError(RuntimeError):
    pass


AgentEventEmitter = Callable[[dict[str, Any]], None]


def parse_json_payload(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*", content, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(_close_json_payload(match.group(0).strip()))
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object.")
    return payload


def _close_json_payload(content: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
    return content + "".join(reversed(stack))


class AgentTaskPlan(BaseModel):
    type: str = Field(
        description=(
            "One of inventory, inspect_toc, search_keywords, navigate_to_section, "
            "inspect_chunk, inspect_page, resolve_citation, list_tables_figures, inspect_item, follow_reference."
        )
    )
    reason: str = DEFAULT_MODEL_REASON
    params: dict[str, Any] = Field(default_factory=dict)


class PlannerDecision(BaseModel):
    rationale: str = DEFAULT_MODEL_RATIONALE
    tasks: list[AgentTaskPlan] = Field(default_factory=list)


class ActorDecision(BaseModel):
    rationale: str = DEFAULT_MODEL_RATIONALE
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class AcceptedEvidence(BaseModel):
    source_key: str
    relevance: str = Field(
        default="",
        description="Concise note identifying the specific sentence, row, phrase, or value that makes this source relevant.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_reasoning_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("relevance"):
            for key in ("reasoning", "reason", "rationale"):
                if data.get(key):
                    data["relevance"] = str(data[key])
                    break
        return data

    @model_validator(mode="after")
    def trim_relevance_text(self) -> AcceptedEvidence:
        self.relevance = " ".join(self.relevance.split())
        if len(self.relevance) > MAX_RELEVANCE_CHARS:
            self.relevance = self.relevance[: MAX_RELEVANCE_CHARS - 3].rstrip() + "..."
        return self


class CuratorDecision(BaseModel):
    accepted_evidence: list[AcceptedEvidence] = Field(default_factory=list)
    accepted_source_keys: list[str] = Field(default_factory=list)
    follow_up_tasks: list[AgentTaskPlan] = Field(default_factory=list)
    planner_request: str | None = Field(
        default=None,
        description="Ask the planner for broader follow-up planning when the next path is unclear.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_string_followups(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("accepted_evidence") and isinstance(data.get("accepted_source_keys"), list):
            relevance_by_source_key = data.get("relevance_by_source_key")
            relevance_by_source_key = relevance_by_source_key if isinstance(relevance_by_source_key, dict) else {}
            data["accepted_evidence"] = [
                {"source_key": source_key, "relevance": str(relevance_by_source_key.get(source_key) or "")}
                for source_key in data["accepted_source_keys"]
            ]
        if isinstance(data.get("accepted_evidence"), list) and not data.get("accepted_source_keys"):
            data["accepted_source_keys"] = [
                item.get("source_key") if isinstance(item, dict) else item
                for item in data["accepted_evidence"]
            ]
        planner_request = data.get("planner_request")
        if planner_request is not None and not isinstance(planner_request, str):
            data["planner_request"] = json.dumps(planner_request, ensure_ascii=False)
        followups = data.get("follow_up_tasks")
        if not isinstance(followups, list) or not any(isinstance(item, str) for item in followups):
            return data
        string_followups = [item for item in followups if isinstance(item, str)]
        data["follow_up_tasks"] = [item for item in followups if isinstance(item, dict)]
        existing_request = str(data.get("planner_request") or "").strip()
        shorthand_request = "Model suggested shorthand follow-ups: " + "; ".join(string_followups)
        data["planner_request"] = f"{existing_request}\n{shorthand_request}".strip() if existing_request else shorthand_request
        return data

    @model_validator(mode="after")
    def sync_accepted_source_keys(self) -> CuratorDecision:
        if self.accepted_evidence:
            self.accepted_source_keys = [item.source_key for item in self.accepted_evidence]
        return self


class EvidenceStanceDecision(BaseModel):
    stance: Literal["for", "against", "neutral"]
    rationale: str = DEFAULT_MODEL_RATIONALE

    @model_validator(mode="before")
    @classmethod
    def coerce_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("stance"):
            for key in ("side", "classification", "label"):
                if data.get(key):
                    data["stance"] = str(data[key]).lower()
                    break
        if not data.get("rationale"):
            for key in ("reason", "summary"):
                if data.get(key):
                    data["rationale"] = str(data[key])
                    break
        stance = str(data.get("stance") or "").lower()
        if stance in {"support", "supports", "supporting", "pro"}:
            data["stance"] = "for"
        elif stance in {"refute", "refutes", "refuting", "con"}:
            data["stance"] = "against"
        elif stance in {"unknown", "unclear", "mixed", "ambiguous", "irrelevant", "background", "neutral"}:
            data["stance"] = "neutral"
        return data

    @model_validator(mode="after")
    def normalize_rationale(self) -> EvidenceStanceDecision:
        self.rationale = " ".join(self.rationale.split())
        if len(self.rationale) > MAX_STANCE_RATIONALE_CHARS:
            self.rationale = self.rationale[: MAX_STANCE_RATIONALE_CHARS - 3].rstrip() + "..."
        return self


class AgentState(TypedDict, total=False):
    run_id: str
    query: str
    document_ids: list[int] | None
    root_document_id: int
    current_document_id: int
    search_scope: Literal["current", "all"]
    max_tasks: int
    task_queue: list[dict[str, Any]]
    completed_tasks: list[dict[str, Any]]
    active_task: dict[str, Any] | None
    candidate_observations: list[dict[str, Any]]
    evidence_by_key: dict[str, dict[str, Any]]
    trace_events: list[dict[str, Any]]
    graph_nodes: dict[str, dict[str, Any]]
    graph_edges: list[dict[str, str]]
    tool_result_lines: list[str]
    model_input_lines: list[str]
    model_output_lines: list[str]
    planner_requests: list[str]
    resolved_references: dict[str, dict[str, Any]]
    counters: dict[str, int]
    stop_reason: str


class EvidenceAgentRunner:
    def __init__(self, *, service: Any, settings: Any, emit: AgentEventEmitter | None = None) -> None:
        self.service = service
        self.settings = settings
        self.emit = emit or (lambda _event: None)
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            format="json",
            num_ctx=8192,
            num_predict=int(getattr(settings, "ollama_num_predict", 512)),
            reasoning=False,
        )

    def run(self, *, run_id: str, query: str, document_ids: list[int] | None, max_tasks: int) -> dict[str, Any]:
        start_document_ids = self._start_document_ids(document_ids)
        self._preflight_ollama()
        graph = self._build_graph()
        initial_state: AgentState = {
            "run_id": run_id,
            "query": query,
            "document_ids": start_document_ids,
            "root_document_id": start_document_ids[0],
            "current_document_id": start_document_ids[0],
            "search_scope": "current",
            "max_tasks": min(max_tasks, int(self.settings.agent_max_tasks)),
            "task_queue": [],
            "completed_tasks": [],
            "active_task": None,
            "candidate_observations": [],
            "evidence_by_key": {},
            "trace_events": [],
            "graph_nodes": {},
            "graph_edges": [],
            "tool_result_lines": [f"Tool Results For Query: {query}", f"Run: {run_id}", ""],
            "model_input_lines": [f"Model Inputs For Query: {query}", f"Run: {run_id}", ""],
            "model_output_lines": [f"Model Outputs For Query: {query}", f"Run: {run_id}", ""],
            "planner_requests": [],
            "resolved_references": {},
            "counters": {"task": 0, "tool_calls": 0, "graph_steps": 0, "trace_step": 0},
            "stop_reason": "",
        }
        self._emit_event(initial_state, type="run_started", query=query)
        recursion_limit = int(self.settings.agent_max_graph_steps) + 4
        return graph.invoke(initial_state, {"recursion_limit": recursion_limit})

    def _start_document_ids(self, document_ids: list[int] | None) -> list[int]:
        unique_ids = list(dict.fromkeys(document_ids or []))
        if len(unique_ids) != 1:
            raise AgentRuntimeError("Agentic retrieval must start from exactly one selected document.")
        return unique_ids

    def _preflight_ollama(self) -> None:
        url = f"{str(self.settings.ollama_base_url).rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentRuntimeError(f"Ollama is not reachable at {self.settings.ollama_base_url}: {exc}") from exc

        model_names = {model.get("name") for model in payload.get("models", [])}
        if self.settings.ollama_model not in model_names:
            raise AgentRuntimeError(
                f"Ollama model {self.settings.ollama_model!r} is not available at {self.settings.ollama_base_url}. "
                f"Run `ollama pull {self.settings.ollama_model}`."
            )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("actor", self._actor_node)
        graph.add_node("curator", self._curator_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "actor")
        graph.add_conditional_edges("actor", self._route_after_actor, {"curator": "curator", "actor": "actor", "finalizer": END})
        graph.add_conditional_edges("curator", self._route_after_curator, {"planner": "planner", "actor": "actor", "finalizer": END})
        return graph.compile()

    def _planner_node(self, state: AgentState) -> AgentState:
        state = dict(state)
        if self._enter_node(state):
            return state
        self._emit_status(state, node="planner", label="Planner working...")
        had_planner_requests = bool(state.get("planner_requests"))
        structured_tasks = self._add_tasks_from_structured_planner_requests(state)
        if structured_tasks:
            self._emit_tasks_planned(state, structured_tasks)
            self._add_trace(
                state,
                node="planner",
                message=f"Converted {len(structured_tasks)} structured planner request(s) into executable tasks.",
                data={"structured_requests": state.get("planner_requests", []), "added_tasks": structured_tasks},
            )
            state["planner_requests"] = []
            return state
        context = self._planner_context(state)
        decision = self._invoke_json(
            PlannerDecision,
            system=(
                "You are the planner for a document evidence retrieval agent. "
                "Create concrete investigation tasks relevant to the user query and available context. "
                "Prefer specific document-scoped TOC, section navigation, and keyword search tasks. "
                "Return all useful non-duplicate tasks needed to advance the investigation. "
                "Do not include task ids or status fields. "
                "Do not answer the user. Return only JSON matching the schema."
            ),
            human=self._planner_prompt(state, context),
            state=state,
            node="planner",
        )
        step_id = state.pop("last_model_step_id", None)
        added_tasks = []
        for planned_task in decision.tasks:
            task = self._add_task(state, planned_task.type, planned_task.reason, planned_task.params, parent_id=None)
            if task is not None:
                added_tasks.append(task)
        if had_planner_requests:
            self._promote_tasks(state, added_tasks)
        state["planner_requests"] = []
        if not state["task_queue"]:
            search_params: dict[str, Any] = {"query": state["query"], "limit": 8}
            if state.get("document_ids"):
                search_params["document_ids"] = state["document_ids"]
            task = self._add_task(state, "search_keywords", "Fallback planner task because no LLM tasks were returned.", search_params)
            if task is not None:
                added_tasks.append(task)
        data = decision.model_dump()
        data["added_tasks"] = added_tasks
        self._emit_tasks_planned(state, added_tasks)
        self._add_trace(state, node="planner", message=decision.rationale, data=data, step_id=step_id)
        return state

    def _actor_node(self, state: AgentState) -> AgentState:
        state = dict(state)
        if self._enter_node(state):
            return state
        if self._limit_hit(state):
            state["stop_reason"] = "limit_hit"
            return state
        if not state["task_queue"]:
            state["stop_reason"] = "task_queue_empty"
            return state

        task = state["task_queue"].pop(0)
        task["status"] = "running"
        state["active_task"] = task
        self._emit_status(state, node="actor", label=f"Actor: {task['type']}", task=task)
        decision = self._invoke_json(
            ActorDecision,
            system=(
                "You are the actor for a document evidence retrieval agent. "
                "Use the active task type as the tool name and provide JSON arguments. "
                "Do not answer the user."
            ),
            human=(
                f"User query: {state['query']}\n"
                f"Active task: {json.dumps(task, ensure_ascii=False)}\n\n"
                "Tools:\n"
                "- inventory()\n"
                "- inspect_toc(document_id)\n"
                "- search_keywords(query, document_ids optional, limit optional)\n"
                "- navigate_to_section(document_id, section_query)\n"
                "- inspect_chunk(document_id, chunk_id)\n"
                "- inspect_page(document_id, page_number)\n"
                "- resolve_citation(document_id, citation)\n"
                "- list_tables_figures(document_id, section_query optional, page_number optional)\n"
                "- inspect_item(document_id, item_id optional, item_ref optional)\n"
                "- follow_reference(target_document_id, query)\n\n"
                "Return tool_name and tool_args."
            ),
            state=state,
            node="actor",
        )
        step_id = state.get("last_model_step_id")
        if step_id:
            state["active_trace_step_id"] = step_id
        effective_tool = decision.model_dump()
        try:
            observations = self._execute_tool(decision.tool_name, decision.tool_args, state)
            effective_tool = state.pop("last_effective_tool", effective_tool)
            task["status"] = "done"
            task["observation_count"] = len(observations)
        except (AgentRuntimeError, KeyError, TypeError, ValueError) as exc:
            observations = []
            effective_tool = state.pop("last_effective_tool", effective_tool)
            task["status"] = "failed"
            task["observation_count"] = 0
            task["error"] = str(exc)
            self._append_tool_error(state, task=task, tool=effective_tool, error=str(exc))
        finally:
            state.pop("active_trace_step_id", None)
        state["completed_tasks"].append(task)
        state["candidate_observations"] = observations
        if not observations and not state["task_queue"]:
            state["stop_reason"] = "task_queue_empty"
        data = {
            "task": task,
            "tool": effective_tool,
            "observation_source_keys": [item.get("source_key") for item in observations if item.get("source_key")],
        }
        self._add_trace(state, node="actor", message=decision.rationale, data=data, step_id=state.pop("last_model_step_id", None))
        return state

    def _curator_node(self, state: AgentState) -> AgentState:
        state = dict(state)
        if self._enter_node(state):
            return state
        candidates = state.get("candidate_observations", [])
        if not candidates:
            state["candidate_observations"] = []
            return state
        self._emit_status(
            state,
            node="curator",
            label=f"Curator reviewing {len(candidates)} candidate(s)...",
            candidate_count=len(candidates),
            task=state.get("active_task"),
        )
        decision = self._invoke_json(
            CuratorDecision,
            system=(
                "You are the curator for a document evidence retrieval agent. "
                "Your only job is selecting direct evidence already present in the candidate observations. "
                "Direct evidence can support or refute the user's query. "
                "Return accepted_evidence with source_key and relevance. "
                "Keep each relevance under 160 characters, in your own words, naming the specific sentence, row, phrase, or value. "
                "Do not copy full chunk, table, or OCR text into relevance. "
                "If the query asks whether one document agrees with an inventory/contents description, "
                "references like [3] are navigation leads, not final agreement evidence. "
                "Do not return planner_request, follow_up_tasks, accepted_source_keys, executable tasks, or a final answer. "
                "The application derives citation and section follow-ups automatically. "
                "Do not answer the user."
            ),
            human=(
                f"User query: {state['query']}\n"
                f"Active task: {json.dumps(state.get('active_task'), ensure_ascii=False)}\n"
                f"Candidate observations: {json.dumps(self._compact_candidates(candidates, query=state['query']), ensure_ascii=False)[:8000]}\n\n"
                'Return exactly {"accepted_evidence": [{"source_key": "...", "relevance": "..."}]}. '
                'If no candidate directly supports or refutes the query, return {"accepted_evidence": []}.'
            ),
            state=state,
            node="curator",
        )
        step_id = state.pop("last_model_step_id", None)
        by_key = {candidate.get("source_key"): candidate for candidate in candidates if candidate.get("source_key")}
        added_tasks = []
        rejected_evidence = []
        for accepted in decision.accepted_evidence:
            source_key = accepted.source_key
            candidate = by_key.get(source_key)
            if candidate is not None:
                if candidate.get("result_type") in {"chunk", "item"}:
                    if not self._candidate_is_query_supporting_evidence(candidate, query=state["query"]):
                        rejected_evidence.append(source_key)
                        continue
                    self._add_evidence(state, candidate, accepted.relevance)
                elif candidate.get("title") and candidate.get("source_locator"):
                    task = self._add_task(
                        state,
                        "navigate_to_section",
                        f"Curator selected TOC entry '{candidate['title']}'; inspect the section body.",
                        {
                            "document_id": candidate["source_locator"]["document_id"],
                            "section_query": candidate["title"],
                        },
                        parent_id=state["active_task"]["id"] if state.get("active_task") else None,
                    )
                    if task is not None:
                        added_tasks.append(task)
        planner_requests = [
            *self._planner_requests_from_curator(decision),
            *self._planner_requests_from_candidate_leads(candidates, query=state["query"], state=state),
        ]
        state.setdefault("planner_requests", []).extend(dict.fromkeys(planner_requests))
        data = decision.model_dump()
        data["candidate_source_keys"] = [candidate.get("source_key") for candidate in candidates if candidate.get("source_key")]
        data["planner_requests"] = planner_requests
        data["added_tasks"] = added_tasks
        data["rejected_accepted_evidence"] = rejected_evidence
        message = (
            f"Accepted {len(decision.accepted_evidence)} evidence source(s)."
            if decision.accepted_evidence
            else "No direct evidence accepted."
        )
        self._add_trace(state, node="curator", message=message, data=data, step_id=step_id)
        state["candidate_observations"] = []
        state["active_task"] = None
        if self._limit_hit(state):
            state["stop_reason"] = "limit_hit"
        elif not state["task_queue"] and not state.get("planner_requests"):
            state["stop_reason"] = "task_queue_empty"
        return state

    def _route_after_actor(self, state: AgentState) -> Literal["curator", "actor", "finalizer"]:
        if state.get("candidate_observations"):
            return "curator"
        if state.get("stop_reason"):
            return "finalizer"
        return "actor" if state.get("task_queue") else "finalizer"

    def _route_after_curator(self, state: AgentState) -> Literal["planner", "actor", "finalizer"]:
        if state.get("stop_reason"):
            return "finalizer"
        if self._limit_hit(state):
            return "finalizer"
        if state.get("planner_requests"):
            return "planner"
        if state.get("task_queue"):
            return "actor"
        return "finalizer"

    def _planner_requests_from_curator(self, decision: CuratorDecision) -> list[str]:
        requests = []
        if decision.planner_request:
            requests.append(decision.planner_request)
        if decision.follow_up_tasks:
            requests.append(
                "Curator suggested follow-up directions; planner must convert only relevant ones into executable tasks: "
                + json.dumps([task.model_dump() for task in decision.follow_up_tasks], ensure_ascii=False)
            )
        return requests

    def _planner_requests_from_candidate_leads(
        self,
        candidates: list[dict[str, Any]],
        *,
        query: str,
        state: AgentState | None = None,
    ) -> list[str]:
        requests: list[str] = []
        for candidate in candidates:
            document_id = (candidate.get("source_locator") or {}).get("document_id")
            text = str(candidate.get("raw_text") or candidate.get("text") or candidate.get("display_text") or "")
            if document_id is not None:
                for citation in sorted(set(re.findall(r"\[\d+\]", text))):
                    cached = self._cached_reference(state, int(document_id), citation)
                    matched_document = cached.get("matched_document") if cached else None
                    if isinstance(matched_document, dict) and matched_document.get("id") is not None:
                        if self._resolved_document_tasks_queued(
                            state,
                            resolved_id=int(matched_document["id"]),
                            origin_document_id=int(document_id),
                            origin_citation=citation,
                        ):
                            continue
                        requests.extend(
                            self._resolved_document_requests(
                                source_key=str(candidate.get("source_key") or "-"),
                                resolved_id=int(matched_document["id"]),
                                query=query,
                            )
                        )
                        continue
                    requests.append(
                        json.dumps(
                            {
                                "type": "resolve_citation",
                                "reason": f"Candidate {candidate.get('source_key')} mentions citation {citation}; resolve it before cross-document curation.",
                                "params": {"document_id": document_id, "citation": citation},
                            },
                            ensure_ascii=False,
                        )
                    )
            resolved_document = candidate.get("resolved_document")
            if isinstance(resolved_document, dict) and resolved_document.get("id") is not None:
                resolved_id = int(resolved_document["id"])
                if document_id is not None:
                    citations = sorted(set(re.findall(r"\[\d+\]", text)))
                    fallback_citation = candidate.get("citation") or candidate.get("origin_citation")
                    if not citations and fallback_citation:
                        citations = [str(fallback_citation)]
                    if any(
                        self._resolved_document_tasks_queued(
                            state,
                            resolved_id=resolved_id,
                            origin_document_id=int(document_id),
                            origin_citation=citation,
                        )
                        for citation in citations
                    ):
                        continue
                requests.extend(
                    self._resolved_document_requests(
                        source_key=str(candidate.get("source_key") or "-"),
                        resolved_id=resolved_id,
                        query=query,
                    )
                )
            section_request = self._section_navigation_request_from_candidate(candidate, query=query, state=state)
            if section_request is not None:
                requests.append(json.dumps(section_request, ensure_ascii=False))
        return list(dict.fromkeys(requests))

    def _section_navigation_request_from_candidate(
        self,
        candidate: dict[str, Any],
        *,
        query: str,
        state: AgentState | None,
    ) -> dict[str, Any] | None:
        if candidate.get("result_type") not in {"chunk", "item"}:
            return None
        locator = candidate.get("source_locator") if isinstance(candidate.get("source_locator"), dict) else {}
        document_id = locator.get("document_id")
        section_path = candidate.get("section_path") if isinstance(candidate.get("section_path"), list) else []
        section_query = next((str(part) for part in reversed(section_path) if str(part).strip()), "")
        if document_id is None or not section_query:
            return None
        if self._section_navigation_task_seen(state, document_id=int(document_id), section_query=section_query):
            return None
        candidate_text = str(candidate.get("raw_text") or candidate.get("text") or candidate.get("display_text") or "")
        searchable_terms = set(self._terms(" ".join([*map(str, section_path), candidate_text])))
        if not (self._section_followup_terms(query) & searchable_terms):
            return None
        return {
            "type": "navigate_to_section",
            "priority": "high",
            "reason": f"Candidate {candidate.get('source_key')} is inside a relevant section; inspect the full section and nearby structured evidence.",
            "params": {"document_id": int(document_id), "section_query": section_query},
        }

    def _section_followup_terms(self, query: str) -> set[str]:
        terms = {term for term in self._terms(query) if len(term) >= 5}
        if self._query_has_decay_power_focus(query):
            return {term for term in terms if "zerfall" in term} | {
                "zerfall",
                "zerfallsleistung",
                "wärmezerfallsleistung",
                "warmezerfallsleistung",
                "nachzerfallsleistung",
            }
        ignored = {
            "angegebene",
            "bestehen",
            "bericht",
            "falls",
            "gleich",
            "konservativ",
            "verwendeten",
            "wert",
        }
        expanded = {term for term in terms if term not in ignored}
        for term in list(expanded):
            if "wärme" in term or "warme" in term:
                expanded.update({"wärmeleistung", "warmeleistung", "gesamtwärmeleistung"})
        return expanded

    def _query_has_decay_power_focus(self, query: str) -> bool:
        return any("zerfall" in term for term in self._terms(query))

    def _terms(self, text: str) -> list[str]:
        return re.findall(r"\w[\w.-]*", text.lower(), flags=re.UNICODE)

    def _section_navigation_task_seen(self, state: AgentState | None, *, document_id: int, section_query: str) -> bool:
        if state is None:
            return False
        for task in [*state.get("task_queue", []), *state.get("completed_tasks", [])]:
            if self._canonical_tool_name(str(task.get("type") or "")) != "navigate_to_section":
                continue
            params = task.get("params") if isinstance(task.get("params"), dict) else {}
            if int(params.get("document_id") or -1) == document_id and str(params.get("section_query") or "") == section_query:
                return True
        return False

    def _resolved_document_requests(self, *, source_key: str, resolved_id: int, query: str) -> list[str]:
        return [
            json.dumps(task, ensure_ascii=False)
            for task in self._resolved_document_task_specs(source_key=source_key, resolved_id=resolved_id, query=query)
        ]

    def _resolved_document_task_specs(self, *, source_key: str, resolved_id: int, query: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "inspect_toc",
                "reason": f"Resolved reference from {source_key} to document {resolved_id}; inspect its TOC.",
                "params": {"document_id": resolved_id},
            },
            {
                "type": "search_keywords",
                "reason": f"Search resolved reference document {resolved_id} for the user query.",
                "params": {"query": query, "document_ids": [resolved_id], "limit": 12},
            },
            {
                "type": "list_tables_figures",
                "reason": f"Inspect structured tables and figures in resolved reference document {resolved_id}.",
                "params": {"document_id": resolved_id},
            },
        ]

    def _insert_resolved_document_tasks(
        self,
        state: AgentState,
        *,
        result: dict[str, Any],
        origin_document_id: int,
        origin_citation: str,
    ) -> list[dict[str, Any]]:
        matched_document = result.get("matched_document")
        if not isinstance(matched_document, dict) or matched_document.get("id") is None:
            return []
        resolved_id = int(matched_document["id"])
        normalized_citation = self._normalized_citation(origin_citation)
        query = str(state.get("query") or "")
        if not query:
            return []
        if self._resolved_document_tasks_queued(
            state,
            resolved_id=resolved_id,
            origin_document_id=origin_document_id,
            origin_citation=normalized_citation,
        ):
            return []

        reference = result.get("reference") if isinstance(result.get("reference"), dict) else {}
        source_key = str(reference.get("source_key") or f"doc:{origin_document_id}#{normalized_citation}")
        metadata = {
            "document_id": resolved_id,
            "origin_document_id": origin_document_id,
            "origin_citation": normalized_citation,
            "return_document_id": origin_document_id,
        }
        parent_id = state["active_task"]["id"] if state.get("active_task") else None
        inserted: list[dict[str, Any]] = []
        for spec in self._resolved_document_task_specs(source_key=source_key, resolved_id=resolved_id, query=query):
            task = self._add_task(
                state,
                str(spec["type"]),
                str(spec["reason"]),
                dict(spec["params"]),
                parent_id=parent_id,
            )
            if task is not None:
                task.update(metadata)
                inserted.append(task)

        if inserted:
            state["task_queue"] = [*inserted, *[task for task in state.get("task_queue", []) if task not in inserted]]
        return inserted

    def _resolved_document_tasks_queued(
        self,
        state: AgentState | None,
        *,
        resolved_id: int,
        origin_document_id: int,
        origin_citation: str,
    ) -> bool:
        if state is None:
            return False
        normalized_citation = self._normalized_citation(origin_citation)
        expected = {"inspect_toc", "search_keywords", "list_tables_figures"}
        queued = {
            str(task.get("type"))
            for task in state.get("task_queue", [])
            if task.get("document_id") == resolved_id
            and task.get("origin_document_id") == origin_document_id
            and task.get("origin_citation") == normalized_citation
        }
        return expected.issubset(queued)

    def _add_tasks_from_structured_planner_requests(self, state: AgentState) -> list[dict[str, Any]]:
        valid_task_types = {
            "inventory",
            "inspect_toc",
            "search_keywords",
            "navigate_to_section",
            "inspect_chunk",
            "inspect_page",
            "resolve_citation",
            "list_tables_figures",
            "inspect_item",
            "follow_reference",
        }
        added: list[dict[str, Any]] = []
        for request in state.get("planner_requests", []):
            try:
                payload = json.loads(request)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, list):
                tasks = payload
            elif isinstance(payload, dict):
                tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else [payload]
            else:
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_type = self._canonical_tool_name(str(task.get("type", "")))
                if task_type not in valid_task_types:
                    continue
                params = dict(task.get("params") or {})
                if not self._has_required_task_params(task_type, params):
                    continue
                before = len(state.get("task_queue", []))
                added_task = self._add_task(
                    state,
                    task_type,
                    str(task.get("reason") or "Structured planner request."),
                    params,
                    parent_id=state["active_task"]["id"] if state.get("active_task") else None,
                )
                if len(state.get("task_queue", [])) > before and added_task is not None:
                    added.append(added_task)
        self._promote_tasks(state, added)
        return added

    def _promote_tasks(self, state: AgentState, tasks: list[dict[str, Any]]) -> None:
        task_ids = [task.get("id") for task in tasks if task.get("id")]
        if not task_ids:
            return
        task_ids_set = set(task_ids)
        by_id = {task.get("id"): task for task in state.get("task_queue", []) if task.get("id") in task_ids_set}
        promoted = [by_id[task_id] for task_id in task_ids if task_id in by_id]
        remaining = [task for task in state.get("task_queue", []) if task.get("id") not in task_ids_set]
        state["task_queue"] = [*promoted, *remaining]

    def _has_required_task_params(self, task_type: str, params: dict[str, Any]) -> bool:
        if task_type == "inventory":
            return True
        if task_type == "search_keywords":
            return bool(params.get("query"))
        if task_type in {"inspect_toc"}:
            return params.get("document_id") is not None
        if task_type == "navigate_to_section":
            return params.get("document_id") is not None and bool(params.get("section_query") or params.get("section_path"))
        if task_type == "inspect_chunk":
            return params.get("document_id") is not None and params.get("chunk_id") is not None
        if task_type == "inspect_page":
            return params.get("document_id") is not None and params.get("page_number") is not None
        if task_type == "resolve_citation":
            return params.get("document_id") is not None and bool(params.get("citation") or params.get("reference"))
        if task_type == "list_tables_figures":
            return params.get("document_id") is not None
        if task_type == "inspect_item":
            return params.get("document_id") is not None and (params.get("item_id") is not None or params.get("item_ref") is not None)
        if task_type == "follow_reference":
            return params.get("target_document_id") is not None
        return True

    def _execute_tool(self, tool_name: str, args: dict[str, Any], state: AgentState) -> list[dict[str, Any]]:
        state["counters"]["tool_calls"] += 1
        tool_name = self._tool_name_for_active_task(tool_name, state)
        args = self._tool_args_with_task_defaults(args, state)
        state["last_effective_tool"] = {"tool_name": tool_name, "tool_args": args}
        self._emit_status(
            state,
            node="actor",
            label=f"Actor using {tool_name}",
            task=state.get("active_task"),
            tool=state["last_effective_tool"],
        )
        if tool_name == "inventory":
            inventory = self.service.list_inventory(document_ids=state.get("document_ids"))
            task = self._task_for_log(state)
            state["tool_result_lines"].append(f"{task.id} inventory")
            self._append_tool_reason(state["tool_result_lines"], task.reason)
            state["tool_result_lines"].append(f"Results: {len(inventory['documents'])}")
            for document in inventory["documents"]:
                state["tool_result_lines"].append(
                    f"- doc:{document['id']} {document['relative_path']} pages:{document.get('page_count') or '?'}"
                )
            state["tool_result_lines"].append("")
            return inventory["documents"]
        if tool_name == "inspect_toc":
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            toc = self.service.get_readable_toc(document_id)
            self.service._append_toc_results(lines=state["tool_result_lines"], task=self._task_for_log(state), toc=toc)
            return toc["entries"]
        if tool_name == "search_keywords":
            document_ids = self._resolve_search_document_ids(args, state)
            if document_ids and len(document_ids) == 1:
                self._set_current_document(state, document_ids[0])
            result = self.service.search_keywords(
                query=str(args.get("query") or state["query"]),
                document_ids=document_ids,
                limit=int(args.get("limit", 8)),
            )
            observations = result["results"]
            self.service._append_tool_results(
                lines=state["tool_result_lines"],
                task=self._task_for_log(state),
                tool_name="search_keywords",
                results=observations,
            )
            return observations
        if tool_name == "navigate_to_section":
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            section = self.service.navigate_to_section(
                document_id=document_id,
                section_query=self._section_query_from_args(args),
            )
            observations = [*section["chunks"], *self._structured_item_candidates(section.get("ordered_items", []))]
            self.service._append_section_results(lines=state["tool_result_lines"], task=self._task_for_log(state), section=section)
            return observations
        if tool_name == "inspect_chunk":
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            observations = self.service.inspect_chunk(
                document_id=document_id,
                chunk_id=self._coerce_source_id(args["chunk_id"], "chunk"),
            )["results"]
            self.service._append_tool_results(
                lines=state["tool_result_lines"],
                task=self._task_for_log(state),
                tool_name="inspect_chunk",
                results=observations,
            )
            return observations
        if tool_name == "inspect_page":
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            observations = self.service.inspect_page(
                document_id=document_id,
                page_number=int(args["page_number"]),
            )["results"]
            self.service._append_tool_results(
                lines=state["tool_result_lines"],
                task=self._task_for_log(state),
                tool_name="inspect_page",
                results=observations,
            )
            return observations
        if tool_name == "resolve_citation":
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            citation = str(args.get("citation") or args.get("reference") or args.get("citation_label") or "")
            cached = self._cached_reference(state, document_id, citation)
            if cached and isinstance(cached.get("result"), dict):
                result = cached["result"]
                observations = self._citation_observations(result)
                self._insert_resolved_document_tasks(
                    state,
                    result=result,
                    origin_document_id=document_id,
                    origin_citation=citation,
                )
                self._append_citation_results(state=state, result=result, observations=observations, cache_hit=True)
                return observations
            result = self.service.resolve_citation(
                document_id=document_id,
                citation=citation,
            )
            self._cache_reference_result(state, document_id=document_id, citation=citation, result=result)
            observations = self._citation_observations(result)
            self._insert_resolved_document_tasks(
                state,
                result=result,
                origin_document_id=document_id,
                origin_citation=citation,
            )
            self._append_citation_results(state=state, result=result, observations=observations)
            return observations
        if tool_name == "list_tables_figures":
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            result = self.service.list_tables_figures(
                document_id=document_id,
                section_query=args.get("section_query") or self._optional_section_query_from_args(args),
                page_number=int(args["page_number"]) if args.get("page_number") is not None else None,
            )
            observations = result["items"]
            self.service._append_tool_results(
                lines=state["tool_result_lines"],
                task=self._task_for_log(state),
                tool_name="list_tables_figures",
                results=observations,
            )
            return observations
        if tool_name == "inspect_item":
            item_id = self._coerce_source_id(args["item_id"], "item") if args.get("item_id") is not None else None
            item_ref = args.get("item_ref") or args.get("source_ref") or args.get("self_ref")
            document_id = self._resolve_document_id(args["document_id"], state)
            self._set_current_document(state, document_id)
            item = self.service.inspect_item(
                document_id=document_id,
                item_id=item_id,
                item_ref=str(item_ref) if item_ref is not None else None,
            )["item"]
            observations = [item]
            self.service._append_tool_results(
                lines=state["tool_result_lines"],
                task=self._task_for_log(state),
                tool_name="inspect_item",
                results=observations,
            )
            return observations
        if tool_name == "follow_reference":
            document_id = self._resolve_document_id(args["target_document_id"], state)
            self._set_current_document(state, document_id)
            result = self.service.search_keywords(
                query=str(args.get("query") or state["query"]),
                document_ids=[document_id],
                limit=int(args.get("limit", 5)),
            )
            observations = result["results"]
            self.service._append_tool_results(
                lines=state["tool_result_lines"],
                task=self._task_for_log(state),
                tool_name="follow_reference/search_keywords",
                results=observations,
            )
            return observations
        raise AgentRuntimeError(f"Actor selected unsupported tool: {tool_name}")

    def _canonical_tool_name(self, tool_name: str) -> str:
        return {
            "get_readable_toc": "inspect_toc",
            "readable_toc": "inspect_toc",
            "keyword_search": "search_keywords",
            "search": "search_keywords",
            "section": "navigate_to_section",
            "inspect_section": "navigate_to_section",
            "get_table": "list_tables_figures",
            "inspect_table": "list_tables_figures",
            "get_figure": "list_tables_figures",
            "inspect_figure": "list_tables_figures",
            "follow_citation": "resolve_citation",
            "resolve_reference": "resolve_citation",
            "citation_resolution": "resolve_citation",
        }.get(tool_name, tool_name)

    def _tool_name_for_active_task(self, tool_name: str, state: AgentState) -> str:
        active = state.get("active_task") or {}
        active_type = active.get("type")
        if active_type:
            return self._canonical_tool_name(str(active_type))
        return self._canonical_tool_name(tool_name)

    def _tool_args_with_task_defaults(self, args: dict[str, Any], state: AgentState) -> dict[str, Any]:
        active = state.get("active_task") or {}
        task_params = dict(active.get("params") or {})
        if task_params:
            authoritative = dict(task_params)
            actor_args = dict(args or {})
            for key, value in list(authoritative.items()):
                if not self._is_usable_task_arg(key, value) and self._is_usable_task_arg(key, actor_args.get(key)):
                    authoritative[key] = actor_args[key]
            return authoritative
        merged = dict(args or {})
        for key, value in task_params.items():
            if self._is_usable_task_arg(key, value):
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        return merged

    def _is_placeholder_arg(self, key: str, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower().strip("<>{}")
        return normalized in {key.lower(), key.lower().replace("_", " "), "same", "current"}

    def _is_usable_task_arg(self, key: str, value: Any) -> bool:
        if value is None or value == "":
            return False
        return not self._is_placeholder_arg(key, value)

    def _section_query_from_args(self, args: dict[str, Any]) -> str:
        if args.get("section_query"):
            return str(args["section_query"])
        section_path = args.get("section_path")
        if isinstance(section_path, list) and section_path:
            return str(section_path[-1])
        if section_path:
            return str(section_path)
        raise AgentRuntimeError("Section navigation requires section_query or section_path.")

    def _optional_section_query_from_args(self, args: dict[str, Any]) -> str | None:
        try:
            return self._section_query_from_args(args)
        except AgentRuntimeError:
            return None

    def _resolve_document_id(self, value: Any, state: AgentState) -> int:
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        normalized = text.lower()
        if normalized.strip("<>{}") in {"document_id", "document id", "same", "current", "current_document_id"}:
            try:
                return self._current_document_id(state)
            except AgentRuntimeError as exc:
                raise AgentRuntimeError(f"Could not resolve document_id from model value: {value!r}") from exc
        scope = set(state.get("document_ids") or [])
        for document in self.service._candidate_documents(scope):
            file_name = str(document.get("file_name", "")).lower()
            relative_path = str(document.get("relative_path", "")).lower()
            if normalized in {file_name, relative_path} or normalized.endswith(f"/{file_name}"):
                return int(document["id"])
        raise AgentRuntimeError(f"Could not resolve document_id from model value: {value!r}")

    def _resolve_document_ids(self, values: Any, state: AgentState) -> list[int] | None:
        if values is None:
            return None
        if isinstance(values, (str, int)):
            values = [values]
        return [self._resolve_document_id(value, state) for value in values]

    def _resolve_search_document_ids(self, args: dict[str, Any], state: AgentState) -> list[int] | None:
        if args.get("document_ids") is not None:
            return self._resolve_document_ids(args.get("document_ids"), state)
        scope = str(args.get("scope") or args.get("search_scope") or state.get("search_scope") or "current").strip().lower()
        if scope in {"all", "all_documents", "corpus"}:
            if str(state.get("search_scope") or "").strip().lower() != "all":
                return [self._current_document_id(state)]
            return None
        if scope in {"current", "current_document", "document"}:
            return [self._current_document_id(state)]
        if scope in {"root", "start", "start_document"}:
            return [int(state.get("root_document_id") or self._current_document_id(state))]
        raise AgentRuntimeError(f"Unsupported search scope: {scope!r}")

    def _current_document_id(self, state: AgentState) -> int:
        if state.get("current_document_id") is not None:
            return int(state["current_document_id"])
        scope = list(state.get("document_ids") or [])
        if len(scope) == 1:
            return int(scope[0])
        raise AgentRuntimeError("Current document is not set.")

    def _set_current_document(self, state: AgentState, document_id: int) -> None:
        state["current_document_id"] = int(document_id)

    def _coerce_source_id(self, value: Any, kind: str) -> int:
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        parsed = self._source_key_int(text, kind)
        if parsed is None:
            raise AgentRuntimeError(f"Could not parse {kind}_id from model value: {value!r}")
        return parsed

    def _source_key_int(self, value: str, kind: str) -> int | None:
        match = re.search(rf"(?:^|:){re.escape(kind)}:(\d+)(?:$|:)", value)
        return int(match.group(1)) if match else None

    def _cached_reference(self, state: AgentState | None, document_id: int, citation: str) -> dict[str, Any] | None:
        if state is None:
            return None
        return state.get("resolved_references", {}).get(self._reference_cache_key(document_id, citation))

    def _cache_reference_result(
        self,
        state: AgentState,
        *,
        document_id: int,
        citation: str,
        result: dict[str, Any],
    ) -> None:
        key = self._reference_cache_key(document_id, citation)
        state.setdefault("resolved_references", {})[key] = {
            "cache_key": key,
            "document_id": document_id,
            "citation": self._normalized_citation(citation),
            "matched_document": result.get("matched_document"),
            "reference_source_key": (result.get("reference") or {}).get("source_key"),
            "result": result,
        }

    def _reference_cache_key(self, document_id: int, citation: str) -> str:
        document_path = f"doc:{document_id}"
        for document in self.service._candidate_documents({document_id}):
            if int(document["id"]) == document_id:
                document_path = str(document.get("relative_path") or document.get("file_name") or document_path)
                break
        return f"{document_path}#{self._normalized_citation(citation)}"

    def _normalized_citation(self, citation: str) -> str:
        text = str(citation).strip()
        match = re.search(r"\d+", text)
        if match:
            return f"[{match.group(0)}]"
        return text.lower()

    def _planner_context(self, state: AgentState) -> str:
        scope = set(state.get("document_ids") or [])
        if state.get("current_document_id") is not None:
            scope.add(int(state["current_document_id"]))
        documents = self.service._candidate_documents(scope)
        lines: list[str] = []
        for document in documents[:8]:
            lines.append(f"Document {document['id']}: {document['relative_path']}")
            toc = self.service.get_readable_toc(int(document["id"]))
            for entry in toc["entries"][:80]:
                lines.append(f"  - p.{entry['page_number'] or '?'} {entry['title']}")
        return "\n".join(lines) or "No processed documents available."

    def _planner_prompt(self, state: AgentState, context: str) -> str:
        queued = [self._task_summary(task) for task in state.get("task_queue", [])]
        completed = [self._task_summary(task) for task in state.get("completed_tasks", [])[-8:]]
        task_count = int(state.get("counters", {}).get("task", len(queued) + len(completed)))
        remaining_task_budget = max(int(state.get("max_tasks", task_count)) - task_count, 0)
        return (
            f"User query: {state['query']}\n\n"
            f"Root document id: {state.get('root_document_id') or '-'}\n"
            f"Current document id: {state.get('current_document_id') or '-'}\n"
            f"Default keyword search scope: {state.get('search_scope') or 'current'}\n\n"
            f"Already queued tasks, do not repeat these: {json.dumps(queued, ensure_ascii=False)}\n\n"
            f"Recently completed task summaries, avoid repeating exact work: {json.dumps(completed, ensure_ascii=False)}\n\n"
            f"Remaining task budget: {remaining_task_budget}\n\n"
            f"Planner requests from curator: {json.dumps(state.get('planner_requests', []), ensure_ascii=False)}\n\n"
            f"Available corpus context:\n{context}\n\n"
            "Return only new tasks that are relevant to the query and the context above. "
            "Every task must be specific enough to execute, including document_id when the tool needs a document. "
            "Use the remaining task budget for useful non-duplicate work, but do not pad with speculative tasks. "
            "Do not include task ids or status fields. "
            "Use task types from: inventory, inspect_toc, search_keywords, navigate_to_section, "
            "inspect_chunk, inspect_page, resolve_citation, list_tables_figures, inspect_item, follow_reference. "
            "For inspect_toc use params document_id. "
            "For navigate_to_section use params document_id and section_query. "
            "For search_keywords use params query, optional document_ids, scope=current|all, limit; omitted document_ids search the current document unless corpus-wide search is explicitly enabled. "
            "For inspect_chunk use params document_id and chunk_id. "
            "For inspect_page use params document_id and page_number. "
            "For resolve_citation use params document_id and citation, for example citation='[3]'. "
            "For list_tables_figures use params document_id and optional section_query or page_number. "
            "For inspect_item use params document_id and item_id or item_ref. "
            "When a section contains references like [3], add a resolve_citation task before claiming cross-document agreement."
        )

    def _stance_prompt(self, state: AgentState, evidence: dict[str, Any]) -> str:
        return (
            f"User query: {state['query']}\n\n"
            f"Evidence: {json.dumps(self._compact_stance_evidence(evidence), ensure_ascii=False)}\n\n"
            'Return exactly {"stance": "for", "rationale": "..."}, {"stance": "against", "rationale": "..."}, '
            'or {"stance": "neutral", "rationale": "..."}. '
            "Use for when the evidence directly supports the user's claim, premise, or expected answer. "
            "Use against when the evidence directly refutes or weakens it. "
            "Use neutral when the evidence is relevant background, ambiguous, mixed, or not enough to decide. "
            "Keep rationale short and do not copy full chunk text."
        )

    def _compact_stance_evidence(self, evidence: dict[str, Any]) -> dict[str, Any]:
        locator = evidence.get("source_locator") if isinstance(evidence.get("source_locator"), dict) else {}
        document = evidence.get("document") if isinstance(evidence.get("document"), dict) else {}
        return {
            "source_key": evidence.get("source_key"),
            "evidence_type": evidence.get("evidence_type", "chunk"),
            "document": document.get("relative_path"),
            "page_number": locator.get("page_number"),
            "section_path": evidence.get("section_path", []),
            "relevance": evidence.get("relevance", ""),
            "text": " ".join(str(evidence.get("text") or "").split())[:MAX_ANALYSIS_TEXT_CHARS],
        }

    def _classify_evidence_stance(self, state: AgentState, evidence: dict[str, Any]) -> EvidenceStanceDecision:
        if not state.get("query"):
            return EvidenceStanceDecision(stance="for", rationale="Accepted by curator.")
        try:
            decision = self._invoke_json(
                EvidenceStanceDecision,
                system=(
                    "You are an evaluator for one evidence item. "
                    "Classify the evidence as for, against, or neutral for the user's query. "
                    "Do not answer the user. Return only JSON matching the schema."
                ),
                human=self._stance_prompt(state, evidence),
                state=state,
                node="evaluator",
            )
            step_id = state.pop("last_model_step_id", None)
        except AgentRuntimeError as exc:
            step_id = state.pop("last_model_step_id", None)
            decision = EvidenceStanceDecision(stance="neutral", rationale=f"Evaluator failed; left evidence neutral: {exc}")
            self._add_trace(
                state,
                node="evaluator",
                message=f"Evaluator failed for evidence {evidence.get('source_key')}; left neutral.",
                data={"error": str(exc), "stance": decision.model_dump(), "source_key": evidence.get("source_key")},
                step_id=step_id,
            )
            return decision

        self._add_trace(
            state,
            node="evaluator",
            message=f"Evaluator classified evidence {evidence.get('source_key')} as {decision.stance}.",
            data={"stance": decision.model_dump(), "source_key": evidence.get("source_key")},
            step_id=step_id,
        )
        return decision

    def _task_summary(self, task: dict[str, Any]) -> dict[str, Any]:
        return {"id": task.get("id"), "type": task.get("type"), "params": task.get("params", {})}

    def _invoke_json(
        self,
        schema: type[BaseModel],
        *,
        system: str,
        human: str,
        state: AgentState | None = None,
        node: str | None = None,
    ) -> BaseModel:
        step_id = self._next_trace_step(state) if state is not None else None
        if state is not None and step_id is not None:
            label = f"[{step_id}] {node or schema.__name__} {schema.__name__} input"
            state.setdefault("model_input_lines", []).extend(
                [label, "System:", system, "", "Human:", human, ""]
            )
        try:
            response = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        except Exception as exc:
            raise AgentRuntimeError(f"LLM invocation failed for {schema.__name__}: {exc}") from exc
        content = str(response.content)
        if state is not None and step_id is not None:
            state.setdefault("model_output_lines", []).extend(
                [f"[{step_id}] {node or schema.__name__} {schema.__name__} output", "Raw JSON:", content, ""]
            )
            state["last_model_step_id"] = step_id
        try:
            payload = parse_json_payload(content)
        except (ValueError, json.JSONDecodeError) as exc:
            if schema is PlannerDecision:
                payload = self._recover_planner_payload(content)
            elif schema is CuratorDecision:
                payload = self._recover_curator_payload(content)
            else:
                payload = None
            if payload is None:
                raise AgentRuntimeError(f"LLM returned malformed JSON for {schema.__name__}: {content[:1200]}") from exc
        if schema is PlannerDecision and not payload.get("tasks"):
            recovered_tasks = self._planner_tasks_from_repeated_top_level_fields(content)
            if recovered_tasks:
                payload = {"rationale": payload.get("rationale", DEFAULT_MODEL_RATIONALE), "tasks": recovered_tasks}
        try:
            decision = schema.model_validate(payload)
        except ValidationError as exc:
            raise AgentRuntimeError(f"LLM JSON did not match {schema.__name__}: {payload}") from exc
        if state is not None and step_id is not None:
            state.setdefault("model_output_lines", []).extend(
                ["Parsed:", json.dumps(decision.model_dump(), ensure_ascii=False, indent=2), ""]
            )
        return decision

    def _recover_planner_payload(self, content: str) -> dict[str, Any] | None:
        key_index = content.find(json.dumps("tasks"))
        if key_index == -1:
            return None
        start = content.find("[", key_index)
        if start == -1:
            return None

        tasks: list[dict[str, Any]] = []
        stack: list[str] = []
        in_string = False
        escaped = False
        object_start: int | None = None
        pairs = {"[": "]", "{": "}"}

        for index in range(start + 1, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in pairs:
                if not stack and char == "{":
                    object_start = index
                stack.append(pairs[char])
            elif stack and char == stack[-1]:
                stack.pop()
                if not stack and object_start is not None and char == "}":
                    try:
                        task = json.loads(content[object_start : index + 1])
                    except json.JSONDecodeError:
                        task = None
                    if isinstance(task, dict):
                        tasks.append(task)
                    object_start = None
            elif not stack and char == "]":
                break

        return {"tasks": tasks} if tasks else None

    def _recover_curator_payload(self, content: str) -> dict[str, Any] | None:
        array_text = self._json_array_after_key(content, "accepted_evidence")
        if array_text is None:
            return None
        try:
            accepted_evidence = json.loads(array_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(accepted_evidence, list):
            return None
        return {"accepted_evidence": accepted_evidence}

    def _json_array_after_key(self, content: str, key: str) -> str | None:
        key_index = content.find(json.dumps(key))
        if key_index == -1:
            return None
        start = content.find("[", key_index)
        if start == -1:
            return None
        stack: list[str] = []
        in_string = False
        escaped = False
        pairs = {"[": "]", "{": "}"}
        for index in range(start, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in pairs:
                stack.append(pairs[char])
            elif stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return content[start : index + 1]
        return None

    def _planner_tasks_from_repeated_top_level_fields(self, content: str) -> list[dict[str, Any]]:
        match = re.search(r"\{.*", content, flags=re.DOTALL)
        if match is None:
            return []
        try:
            pairs = json.loads(_close_json_payload(match.group(0).strip()), object_pairs_hook=list)
        except (TypeError, json.JSONDecodeError):
            return []
        if not self._is_json_object_pairs(pairs) or any(key == "tasks" for key, _value in pairs):
            return []
        tasks: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for key, value in pairs:
            if key == "type":
                if current.get("type"):
                    tasks.append(current)
                current = {"type": str(value), "params": {}}
            elif key == "params" and current.get("type"):
                params = self._plain_json_value(value)
                current["params"] = params if isinstance(params, dict) else {}
            elif key == "reason" and current.get("type"):
                current["reason"] = str(value)
        if current.get("type"):
            tasks.append(current)
        return tasks

    def _plain_json_value(self, value: Any) -> Any:
        if self._is_json_object_pairs(value):
            return {key: self._plain_json_value(item) for key, item in value}
        if isinstance(value, list):
            return [self._plain_json_value(item) for item in value]
        return value

    def _is_json_object_pairs(self, value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        )

    def _add_task(
        self,
        state: AgentState,
        task_type: str,
        reason: str,
        params: dict[str, Any],
        parent_id: str | None = None,
    ) -> dict[str, Any] | None:
        task_type = self._canonical_tool_name(task_type)
        if state["counters"]["task"] >= int(state["max_tasks"]):
            return None
        signature = self._task_signature(task_type, params)
        existing_tasks = [*state.get("task_queue", []), *state.get("completed_tasks", [])]
        if any(self._task_signature(str(task.get("type")), dict(task.get("params", {}))) == signature for task in existing_tasks):
            return None
        state["counters"]["task"] += 1
        task_id = f"T{state['counters']['task']}"
        task = {"id": task_id, "type": task_type, "status": "queued", "reason": reason, "params": params}
        state["task_queue"].append(task)
        state["graph_nodes"][task_id] = {"id": task_id, "type": "task", "label": task_type}
        if parent_id:
            state["graph_edges"].append({"from": parent_id, "to": task_id, "type": "suggested_followup"})
        return task

    def _task_signature(self, task_type: str, params: dict[str, Any]) -> str:
        return json.dumps({"type": task_type, "params": params}, sort_keys=True)

    def _add_evidence(self, state: AgentState, candidate: dict[str, Any], relevance: str) -> None:
        source_key = candidate["source_key"]
        relevance = relevance.strip()
        is_new_source = source_key not in state["evidence_by_key"]
        if is_new_source:
            evidence_id = f"E{len(state['evidence_by_key']) + 1}"
            state["evidence_by_key"][source_key] = {
                "id": evidence_id,
                "source_key": source_key,
                "source_locator": candidate["source_locator"],
                "document": candidate["document"],
                "evidence_type": candidate.get("result_type", "chunk"),
                "section_path": candidate.get("section_path", []),
                "text": candidate.get("text") or candidate.get("display_text") or candidate.get("raw_text", ""),
                "payload": candidate.get("payload", {}),
                "relevance": relevance,
                "found_by_tasks": [],
            }
            state["graph_nodes"][evidence_id] = {"id": evidence_id, "type": "evidence", "label": source_key, "source_key": source_key}
        evidence = state["evidence_by_key"][source_key]
        if relevance:
            evidence["relevance"] = relevance
        if is_new_source:
            stance = self._classify_evidence_stance(state, evidence)
            evidence["stance"] = stance.stance
            evidence["stance_rationale"] = stance.rationale
        if state.get("active_task"):
            task_id = state["active_task"]["id"]
            if task_id not in evidence["found_by_tasks"]:
                evidence["found_by_tasks"].append(task_id)
            edge = {"from": task_id, "to": evidence["id"], "type": "found"}
            if edge not in state["graph_edges"]:
                state["graph_edges"].append(edge)
        if is_new_source:
            self._emit_event(
                state,
                type="evidence_added",
                label=f"Evidence found: {source_key}",
                evidence=evidence,
                evidence_count=len(state["evidence_by_key"]),
            )

    def _next_trace_step(self, state: AgentState | None) -> str | None:
        if state is None:
            return None
        counters = state.setdefault("counters", {})
        counters["trace_step"] = counters.get("trace_step", 0) + 1
        return f"S{counters['trace_step']:03d}"

    def _add_trace(
        self,
        state: AgentState,
        *,
        node: str,
        message: str,
        data: dict[str, Any],
        step_id: str | None = None,
    ) -> None:
        state["trace_events"].append(
            {"step_id": step_id or self._next_trace_step(state), "node": node, "message": message, "data": data}
        )

    def _emit_status(self, state: AgentState, *, node: str, label: str, **data: Any) -> None:
        self._emit_event(state, type="status", node=node, label=label, **data)

    def _emit_tasks_planned(self, state: AgentState, tasks: list[dict[str, Any]]) -> None:
        if not tasks:
            return
        self._emit_event(
            state,
            type="tasks_planned",
            label=f"Planner queued {len(tasks)} task(s).",
            tasks=tasks,
            task_count=len(tasks),
        )

    def _emit_event(self, state: AgentState, **event: Any) -> None:
        run_id = state.get("run_id")
        if run_id:
            event["run_id"] = run_id
        self.emit(dict(event))

    def _append_tool_error(self, state: AgentState, *, task: dict[str, Any], tool: dict[str, Any], error: str) -> None:
        state["tool_result_lines"].extend(
            [
                f"{self._tool_log_prefix(state, task.get('id', 'T?'))} {tool.get('tool_name', '-')}",
            ]
        )
        self._append_tool_reason(state["tool_result_lines"], str(task.get("reason", "")))
        state["tool_result_lines"].extend([f"Tool error: {error}", ""])
        self._add_trace(state, node="actor_tool_error", message=error, data={"task": task, "tool": tool})

    def _task_for_log(self, state: AgentState):
        active = state.get("active_task") or {}
        return type(
            "TaskLog",
            (),
            {
                "id": self._tool_log_prefix(state, active.get("id", "T?")),
                "reason": active.get("reason", ""),
            },
        )()

    def _tool_log_prefix(self, state: AgentState, task_id: str) -> str:
        step_id = state.get("active_trace_step_id")
        return f"[{step_id}] {task_id}" if step_id else task_id

    def _append_tool_reason(self, lines: list[str], reason: str) -> None:
        if reason and reason != DEFAULT_MODEL_REASON:
            lines.append(f"Reason: {reason}")

    def _compact_candidates(self, candidates: list[dict[str, Any]], query: str | None = None) -> list[dict[str, Any]]:
        compact = []
        visible_candidates = [
            candidate
            for candidate in candidates
            if not self._is_empty_visual_candidate(candidate)
        ]
        if query:
            ranked_candidates = [
                candidate
                for _score, _index, candidate in sorted(
                    (
                        (self._candidate_query_score(candidate, query=query), index, candidate)
                        for index, candidate in enumerate(visible_candidates)
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
            ]
        else:
            ranked_candidates = visible_candidates
        for candidate in ranked_candidates[:12]:
            payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
            locator = candidate.get("source_locator") if isinstance(candidate.get("source_locator"), dict) else {}
            caption_candidates = [
                str(candidate.get("text") or "")[:300]
                for candidate in payload.get("asset_caption_candidates", [])
                if isinstance(candidate, dict) and candidate.get("text")
            ][:3]
            compact.append(
                {
                    "source_key": candidate.get("source_key"),
                    "result_type": candidate.get("result_type"),
                    "label": candidate.get("label"),
                    "document_id": locator.get("document_id"),
                    "page_number": locator.get("page_number"),
                    "chunk_id": locator.get("chunk_id"),
                    "item_id": locator.get("item_id"),
                    "document": candidate.get("document", {}).get("relative_path"),
                    "resolved_document": candidate.get("resolved_document"),
                    "section_path": candidate.get("section_path", []),
                    "text": self._candidate_text(candidate, payload)[:1000],
                    "asset_caption": str(payload.get("asset_caption") or "")[:1000],
                    "asset_caption_candidates": caption_candidates,
                    "table_text": str(payload.get("table_text") or "")[:1000],
                }
            )
        return compact

    def _candidate_query_score(self, candidate: dict[str, Any], *, query: str) -> int:
        text = self._candidate_search_text(candidate)
        searchable_terms = set(self._terms(text))
        score = 0
        for term in self._section_followup_terms(query):
            if term in searchable_terms:
                score += 4
            elif term in text.lower():
                score += 2
        for term in {term for term in self._terms(query) if len(term) >= 5}:
            if term in searchable_terms:
                score += 1
        if score and self._is_structured_item_candidate(candidate):
            score += 2
        return score

    def _candidate_is_query_supporting_evidence(self, candidate: dict[str, Any], *, query: str) -> bool:
        if not self._query_has_decay_power_focus(query):
            return True
        terms = set(self._terms(self._candidate_search_text(candidate)))
        if any("zerfall" in term for term in terms):
            return True
        return bool({"gesamtwärmeleistung", "wärmeleistung", "warmeleistung"} & terms)

    def _candidate_search_text(self, candidate: dict[str, Any]) -> str:
        payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
        return " ".join(
            str(part)
            for part in [
                " ".join(map(str, candidate.get("section_path", []))),
                self._candidate_text(candidate, payload),
                payload.get("table_text", ""),
                payload.get("asset_caption", ""),
            ]
            if part
        )

    def _candidate_text(self, candidate: dict[str, Any], payload: dict[str, Any]) -> str:
        return str(
            candidate.get("raw_text")
            or candidate.get("text")
            or candidate.get("display_text")
            or candidate.get("title")
            or payload.get("asset_caption")
            or ""
        )

    def _is_empty_visual_candidate(self, candidate: dict[str, Any]) -> bool:
        label = str(candidate.get("label") or "").lower()
        item_type = str(candidate.get("item_type") or "").lower()
        if label not in {"picture", "figure", "image"} and not any(
            marker in item_type for marker in ["picture", "image", "figure"]
        ):
            return False
        payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
        return not any(
            [
                self._candidate_text(candidate, payload).strip(),
                str(payload.get("table_text") or "").strip(),
                str(payload.get("asset_caption") or "").strip(),
            ]
        )

    def _structured_item_candidates(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in items if self._is_structured_item_candidate(item)]

    def _is_structured_item_candidate(self, item: dict[str, Any]) -> bool:
        label = str(item.get("label") or "").lower()
        item_type = str(item.get("item_type") or "").lower()
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        text = str(item.get("display_text") or item.get("text") or "").lower()
        if label in {"table", "picture", "caption", "figure"}:
            return True
        if any(marker in item_type for marker in ["table", "picture", "image", "figure"]):
            return True
        if payload.get("table_text"):
            return True
        return bool(re.match(r"\s*(tab\.?|table|abb\.?|fig\.?)\b", text))

    def _citation_observations(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        matched_document = result.get("matched_document")
        observations = []
        for reference in result.get("references") or []:
            candidate = dict(reference)
            if matched_document:
                candidate["resolved_document"] = matched_document
            observations.append(candidate)
        return observations

    def _append_citation_results(
        self,
        *,
        state: AgentState,
        result: dict[str, Any],
        observations: list[dict[str, Any]],
        cache_hit: bool = False,
    ) -> None:
        task = self._task_for_log(state)
        matched = result.get("matched_document") or {}
        state["tool_result_lines"].append(f"{task.id} resolve_citation")
        self._append_tool_reason(state["tool_result_lines"], task.reason)
        state["tool_result_lines"].extend(
            [
                f"Citation: {result.get('citation', '-')}",
                f"Matched document: {matched.get('relative_path', '-')}",
                f"Cache: {'hit' if cache_hit else 'miss'}",
                f"Results: {len(observations)}",
            ]
        )
        for index, observation in enumerate(observations, start=1):
            state["tool_result_lines"].extend(
                [
                    f"{index}. {observation['source_key']}",
                    f"   document: {observation['document']['relative_path']}",
                    f"   page: {observation['source_locator'].get('page_number') or '?'}",
                    f"   text: {str(observation.get('display_text') or observation.get('text') or '')[:500]}",
                ]
            )
        state["tool_result_lines"].append("")

    def _limit_hit(self, state: AgentState) -> bool:
        return (
            len(state.get("completed_tasks", [])) >= int(state["max_tasks"])
            or state["counters"].get("tool_calls", 0) >= int(self.settings.agent_max_tool_calls)
            or len(state.get("evidence_by_key", {})) >= int(self.settings.agent_max_evidence)
        )

    def _enter_node(self, state: AgentState) -> bool:
        state["counters"]["graph_steps"] = state["counters"].get("graph_steps", 0) + 1
        if state["counters"]["graph_steps"] >= int(self.settings.agent_max_graph_steps):
            state["stop_reason"] = "limit_hit"
            return True
        return False
