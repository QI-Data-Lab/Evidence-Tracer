#!/usr/bin/env python3
"""Standalone Docling bbox repro utility using the backend uv environment.

Examples:
  cd backend
  python test/docling_bbox_debug.py /path/to/document.pdf
  python test/docling_bbox_debug.py /path/to/document.pdf --page 2
  uv run python test/docling_bbox_debug.py /path/to/document.pdf 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from docling.chunking import HierarchicalChunker


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pdf_service import close_pdf_object, read_pdf_page_data
from app.utils.bbox import normalize_bbox
from app.utils.docling_records import build_chunk_records, build_docling_converter, convert_pdf


DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "test_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one PDF page and draw Docling HierarchicalChunker chunk bboxes "
            "using the same backend dependencies as the app."
        )
    )
    parser.add_argument("document", type=Path, help="PDF document to inspect.")
    parser.add_argument(
        "page_number",
        type=int,
        nargs="?",
        default=None,
        help="1-based page number to render. Defaults to page 1.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="1-based page number to render. Overrides positional page_number.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for annotated output. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="PDF render scale used for the output image.",
    )
    return parser.parse_args()


def selected_page(args: argparse.Namespace) -> int:
    return args.page if args.page is not None else args.page_number or 1


def render_pdf_page(pdf_path: Path, page_number: int, scale: float) -> Image.Image:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required for bbox debug page rendering.") from exc

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_number - 1]
    bitmap = None
    image = None

    try:
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        return image.copy()
    finally:
        if image is not None and hasattr(image, "close"):
            image.close()
        close_pdf_object(bitmap)
        close_pdf_object(page)
        close_pdf_object(pdf)


def chunk_boxes_for_page(chunks: list[dict[str, Any]], page_number: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for chunk in chunks:
        for page_bbox in chunk.get("page_bboxes", []):
            if int(page_bbox.get("page_number", 0)) != page_number:
                continue
            bbox = page_bbox.get("bbox")
            if bbox is None:
                continue
            boxes.append(
                {
                    "chunk_index": int(chunk["index"]),
                    "chunk_type": chunk.get("chunk_type"),
                    "labels": chunk.get("labels", []),
                    "bbox": bbox,
                    "text_preview": str(chunk.get("text", "")).replace("\n", " ")[:140],
                }
            )
    return boxes


def build_debug_chunks(document: Any) -> list[dict[str, Any]]:
    chunks = build_chunk_records(
        document,
        chunker=HierarchicalChunker(),
        chunker_name="hierarchical",
    )
    if any(chunk.get("chunker") != "hierarchical" for chunk in chunks):
        raise RuntimeError("Debug bbox script requires hierarchical chunking.")
    return chunks


def bbox_to_pixels(
    *,
    bbox: dict[str, Any],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    normalized = normalize_bbox(bbox=bbox, page_width=page_width, page_height=page_height)
    if normalized is None:
        return None

    left = normalized["left_pct"] / 100.0 * image_width
    top = normalized["top_pct"] / 100.0 * image_height
    right = left + normalized["width_pct"] / 100.0 * image_width
    bottom = top + normalized["height_pct"] / 100.0 * image_height

    x0 = max(0.0, min(float(image_width), left))
    y0 = max(0.0, min(float(image_height), top))
    x1 = max(0.0, min(float(image_width), right))
    y1 = max(0.0, min(float(image_height), bottom))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_boxes(
    *,
    image: Image.Image,
    boxes: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (220, 38, 38, 255),
        (22, 163, 74, 255),
        (37, 99, 235, 255),
        (217, 119, 6, 255),
        (8, 145, 178, 255),
        (190, 24, 93, 255),
    ]
    rendered_boxes: list[dict[str, Any]] = []

    for offset, box in enumerate(boxes):
        pixel_rect = bbox_to_pixels(
            bbox=box["bbox"],
            page_width=page_width,
            page_height=page_height,
            image_width=image.width,
            image_height=image.height,
        )
        if pixel_rect is None:
            continue

        color = colors[offset % len(colors)]
        fill = (*color[:3], 36)
        x0, y0, x1, y1 = pixel_rect
        draw.rectangle(pixel_rect, outline=color, width=3, fill=fill)

        label = f"chunk {box['chunk_index']}"
        label_bbox = draw.textbbox((x0 + 4, y0 + 4), label)
        draw.rectangle(label_bbox, fill=(*color[:3], 220))
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255, 255))

        rendered_boxes.append({**box, "pixel_rect": [x0, y0, x1, y1]})

    annotated = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.paste(annotated)
    return rendered_boxes


def output_stem(document_path: Path, page_number: int) -> str:
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in document_path.stem)
    return f"{safe_stem}_page_{page_number:03d}_docling_chunks"


def main() -> int:
    args = parse_args()
    page_number = selected_page(args)
    pdf_path = args.document.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    page_data = read_pdf_page_data(pdf_path)
    page_count = int(page_data["page_count"])
    if page_number < 1 or page_number > page_count:
        raise ValueError(f"page_number must be between 1 and {page_count}; got {page_number}")

    page_info = page_data["pages"][page_number - 1]
    converter = build_docling_converter()
    document = convert_pdf(converter, pdf_path)
    chunks = build_debug_chunks(document)
    boxes = chunk_boxes_for_page(chunks, page_number)

    image = render_pdf_page(pdf_path, page_number, args.scale)
    rendered_boxes = draw_boxes(
        image=image,
        boxes=boxes,
        page_width=float(page_info["width"]),
        page_height=float(page_info["height"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(pdf_path, page_number)
    image_path = args.output_dir / f"{stem}.png"
    json_path = args.output_dir / f"{stem}.json"
    image.save(image_path)
    json_path.write_text(
        json.dumps(
            {
                "document": str(pdf_path),
                "chunker": "hierarchical",
                "page_number": page_number,
                "page_size": {
                    "width": float(page_info["width"]),
                    "height": float(page_info["height"]),
                },
                "image_size": {"width": image.width, "height": image.height},
                "chunk_count": len(chunks),
                "page_box_count": len(boxes),
                "rendered_box_count": len(rendered_boxes),
                "boxes": rendered_boxes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {image_path}")
    print(f"wrote {json_path}")
    print(f"chunks: {len(chunks)}; page boxes: {len(boxes)}; rendered boxes: {len(rendered_boxes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
