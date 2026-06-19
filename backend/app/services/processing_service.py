from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.pipelines.docling_pipeline import process_pdf
from app.services.catalog_service import CatalogService
from app.services.pdf_service import read_pdf_page_data


class ProcessingService:
    def __init__(self, catalog: CatalogService | None = None) -> None:
        self.catalog = catalog or CatalogService()

    def _clear_preview_cache(self, document_id: int) -> None:
        preview_dir = get_settings().preview_dir / str(document_id)
        if preview_dir.exists():
            shutil.rmtree(preview_dir)

    def process_document(self, document_id: int) -> dict[str, Any]:
        document = self.catalog.get_document(document_id)
        if document is None:
            raise ValueError(f"Unknown document: {document_id}")

        source_path = Path(document["source_path"])
        if not source_path.exists():
            self.catalog.update_document_status(document_id, status="error", error_message="Source PDF no longer exists.")
            raise FileNotFoundError(f"Missing source PDF: {source_path}")

        self.catalog.update_document_status(document_id, status="processing", error_message=None)

        try:
            processed = process_pdf(source_path)
            self.catalog.replace_document_content(
                document_id=document_id,
                page_records=processed["page_records"],
                item_records=processed["item_records"],
                chunk_records=processed["chunk_records"],
                summary=processed["summary"],
            )
            self._clear_preview_cache(document_id)
        except Exception as exc:
            self.catalog.update_document_status(document_id, status="error", error_message=str(exc))
            raise

        refreshed = self.catalog.get_document(document_id)
        if refreshed is None:
            raise RuntimeError(f"Document disappeared after processing: {document_id}")
        return refreshed

    def process_documents(
        self,
        *,
        document_ids: list[int] | None = None,
        only_stale: bool = False,
    ) -> dict[str, Any]:
        documents = self.catalog.get_documents_for_processing(document_ids, only_stale=only_stale)

        processed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for document in documents:
            try:
                processed.append(self.process_document(int(document["id"])))
            except Exception as exc:
                failed.append(
                    {
                        "document_id": int(document["id"]),
                        "relative_path": document["relative_path"],
                        "error": str(exc),
                    }
                )

        return {
            "requested": len(documents),
            "processed": processed,
            "failed": failed,
        }

    def _document_page_count(self, document: dict[str, Any]) -> int:
        try:
            page_data = read_pdf_page_data(Path(document["source_path"]))
        except Exception:
            return 0
        return int(page_data.get("page_count") or 0)

    def process_documents_stream(
        self,
        *,
        document_ids: list[int] | None = None,
        only_stale: bool = False,
    ):
        documents = self.catalog.get_documents_for_processing(document_ids, only_stale=only_stale)
        page_counts = {int(document["id"]): self._document_page_count(document) for document in documents}
        total_pages = sum(page_counts.values())
        total_documents = len(documents)
        processed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        processed_pages = 0
        processed_documents = 0

        yield {
            "type": "process_started",
            "total_documents": total_documents,
            "total_pages": total_pages,
            "processed_documents": 0,
            "processed_pages": 0,
        }

        for document in documents:
            document_id = int(document["id"])
            page_count = page_counts[document_id]
            yield {
                "type": "document_started",
                "document": document,
                "page_count": page_count,
                "total_documents": total_documents,
                "total_pages": total_pages,
                "processed_documents": processed_documents,
                "processed_pages": processed_pages,
            }

            try:
                refreshed = self.process_document(document_id)
                processed.append(refreshed)
                event: dict[str, Any] = {
                    "type": "document_completed",
                    "document": refreshed,
                }
            except Exception as exc:
                failed_item = {
                    "document_id": document_id,
                    "relative_path": document["relative_path"],
                    "error": str(exc),
                }
                failed.append(failed_item)
                event = {
                    "type": "document_failed",
                    **failed_item,
                }

            processed_documents += 1
            processed_pages += page_count
            event.update(
                {
                    "page_count": page_count,
                    "total_documents": total_documents,
                    "total_pages": total_pages,
                    "processed_documents": processed_documents,
                    "processed_pages": processed_pages,
                }
            )
            yield event

        yield {
            "type": "process_completed",
            "total_documents": total_documents,
            "total_pages": total_pages,
            "processed_documents": processed_documents,
            "processed_pages": processed_pages,
            "result": {
                "requested": total_documents,
                "processed": processed,
                "failed": failed,
            },
        }
