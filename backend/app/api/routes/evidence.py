from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.schemas import EvidenceRunRequest
from app.services.evidence_agent import AgentRuntimeError
from app.services.evidence_service import EvidenceService


router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("/documents/{document_id}/toc")
def get_readable_toc(document_id: int) -> dict[str, object]:
    service = EvidenceService()
    try:
        return service.get_readable_toc(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/documents/{document_id}/sections")
def navigate_to_section(document_id: int, section_query: str = Query(min_length=1)) -> dict[str, object]:
    service = EvidenceService()
    try:
        return service.navigate_to_section(document_id=document_id, section_query=section_query)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/search")
def search_keywords(
    q: str = Query(min_length=1),
    document_id: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    service = EvidenceService()
    document_ids = [document_id] if document_id is not None else None
    return service.search_keywords(query=q, document_ids=document_ids, limit=limit)


@router.post("/run")
def run_agentic_retrieval(payload: EvidenceRunRequest) -> dict[str, object]:
    service = EvidenceService()
    try:
        return service.run_agentic_retrieval(query=payload.query, document_ids=payload.document_ids, max_tasks=payload.max_tasks)
    except AgentRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/run/stream")
def stream_agentic_retrieval(payload: EvidenceRunRequest) -> StreamingResponse:
    service = EvidenceService()
    return StreamingResponse(
        service.run_agentic_retrieval_stream(
            query=payload.query,
            document_ids=payload.document_ids,
            max_tasks=payload.max_tasks,
        ),
        media_type="application/x-ndjson",
    )
