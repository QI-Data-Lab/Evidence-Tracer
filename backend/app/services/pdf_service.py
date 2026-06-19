from __future__ import annotations

from pathlib import Path
from typing import Any


def read_pdf_page_data(pdf_path: Path) -> dict[str, Any]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required for PDF metadata and page previews.") from exc

    pdf = pdfium.PdfDocument(str(pdf_path))
    pages: list[dict[str, Any]] = []

    try:
        for index in range(len(pdf)):
            page = pdf[index]
            width, height = page.get_size()
            pages.append(
                {
                    "page_number": index + 1,
                    "width": float(width),
                    "height": float(height),
                    "rotation": 0,
                }
            )
            close_pdf_object(page)
    finally:
        close_pdf_object(pdf)

    return {
        "page_count": len(pages),
        "pages": pages,
    }


def render_page_preview(*, pdf_path: Path, page_number: int, output_path: Path, scale: float = 1.5) -> Path:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required for PDF page preview rendering.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_number - 1]
    bitmap = None
    image = None

    try:
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(output_path)
    finally:
        if image is not None and hasattr(image, "close"):
            image.close()
        close_pdf_object(bitmap)
        close_pdf_object(page)
        close_pdf_object(pdf)

    return output_path


def close_pdf_object(value: Any) -> None:
    if value is None:
        return
    close = getattr(value, "close", None)
    if callable(close):
        close()

