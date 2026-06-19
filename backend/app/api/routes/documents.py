from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.api.schemas import ProcessDocumentsRequest
from app.core.config import get_settings
from app.services.catalog_service import CatalogService
from app.services.pdf_service import render_page_preview
from app.services.processing_service import ProcessingService
from app.utils.common import json_dumps


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def list_documents(status: str | None = Query(default=None)) -> dict[str, object]:
    catalog = CatalogService()
    return {"items": catalog.list_documents(status=status)}


@router.get("/query")
def query_documents(
    q: str = Query(default=""),
    kind: str = Query(default="all"),
    document_id: int | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    catalog = CatalogService()
    try:
        return catalog.query_artifacts(query=q, kind=kind, document_id=document_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{document_id}")
def get_document(document_id: int) -> dict[str, object]:
    catalog = CatalogService()
    document = catalog.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.post("/process")
def process_documents(payload: ProcessDocumentsRequest) -> dict[str, object]:
    service = ProcessingService()
    return service.process_documents(document_ids=payload.document_ids, only_stale=payload.only_stale)


@router.post("/process/stream")
def stream_process_documents(payload: ProcessDocumentsRequest) -> StreamingResponse:
    service = ProcessingService()

    def lines():
        for event in service.process_documents_stream(document_ids=payload.document_ids, only_stale=payload.only_stale):
            yield json_dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@router.post("/{document_id}/process")
def process_document(document_id: int) -> dict[str, object]:
    service = ProcessingService()
    try:
        return service.process_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{document_id}/pages")
def list_document_pages(document_id: int) -> dict[str, object]:
    catalog = CatalogService()
    document = catalog.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "document": document,
        "pages": catalog.list_pages(document_id),
    }


@router.get("/{document_id}/pages/{page_number}")
def get_document_page(document_id: int, page_number: int) -> dict[str, object]:
    catalog = CatalogService()
    bundle = catalog.get_page_bundle(document_id, page_number)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Page not found.")
    return bundle


@router.get("/{document_id}/pages/{page_number}/image")
def get_document_page_image(document_id: int, page_number: int, scale: float = Query(default=1.5, ge=1.0, le=4.0)) -> FileResponse:
    catalog = CatalogService()
    document = catalog.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    source_path = Path(document["source_path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF not found.")

    settings = get_settings()
    image_path = settings.preview_dir / str(document_id) / f"page-{page_number:03d}-s{int(scale * 100)}.png"

    try:
        render_page_preview(pdf_path=source_path, page_number=page_number, output_path=image_path, scale=scale)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(image_path, media_type="image/png")


@router.get("/{document_id}/file")
def get_document_file(document_id: int) -> FileResponse:
    catalog = CatalogService()
    document = catalog.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    source_path = Path(document["source_path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF not found.")

    return FileResponse(source_path, media_type="application/pdf", filename=document["file_name"])
