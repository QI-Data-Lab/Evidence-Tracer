from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.pdf_service import read_pdf_page_data
from app.utils.docling_records import (
    build_chunk_records,
    build_docling_converter,
    build_document_summary,
    build_item_records,
    convert_pdf,
)


def process_pdf(pdf_path: Path) -> dict[str, Any]:
    converter = build_docling_converter()
    document = convert_pdf(converter, pdf_path)
    page_data = read_pdf_page_data(pdf_path)

    return {
        "summary": build_document_summary(document),
        "page_records": page_data["pages"],
        "item_records": build_item_records(document),
        "chunk_records": build_chunk_records(document),
    }

