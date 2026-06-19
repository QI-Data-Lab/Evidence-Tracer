"""Docling helpers adapted from the sister repository's proven record shape."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from app.utils.bbox import merge_bboxes


DEFAULT_NUM_THREADS = 4
HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+")
ASSET_CAPTION_RE = re.compile(
    r"^\s*(?:tab(?:elle)?|table|abb(?:ildung)?|fig(?:ure)?)\.?\s*\d+",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", None) or str(value)


def build_docling_converter():
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.force_backend_text = False
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=DEFAULT_NUM_THREADS,
        device="auto",
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def convert_pdf(converter: Any, pdf_path: Path):
    return converter.convert(source=pdf_path.resolve()).document


def build_document_summary(document: Any) -> dict[str, int]:
    return {
        "texts": len(getattr(document, "texts", [])),
        "tables": len(getattr(document, "tables", [])),
        "pictures": len(getattr(document, "pictures", [])),
        "groups": len(getattr(document, "groups", [])),
        "key_value_items": len(getattr(document, "key_value_items", [])),
    }


def parse_heading_number(text: str) -> tuple[int, ...] | None:
    match = HEADING_NUMBER_RE.match(text)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def serialize_bbox(bbox: Any) -> dict[str, Any] | None:
    if bbox is None:
        return None
    return {
        "l": float(getattr(bbox, "l")),
        "t": float(getattr(bbox, "t")),
        "r": float(getattr(bbox, "r")),
        "b": float(getattr(bbox, "b")),
        "coord_origin": str(getattr(bbox, "coord_origin", "TOPLEFT")),
    }


def serialize_provenance(prov_item: Any) -> dict[str, Any]:
    charspan = getattr(prov_item, "charspan", None)
    page_no = getattr(prov_item, "page_no", None)
    return {
        "page_no": int(page_no) if page_no is not None else None,
        "bbox": serialize_bbox(getattr(prov_item, "bbox", None)),
        "charspan": list(charspan) if charspan is not None else None,
    }


def build_section_paths(document: Any) -> dict[str, list[str]]:
    section_paths: dict[str, list[str]] = {}
    numbered_stack: list[str] = []
    current_path: list[str] = []

    for item, _level in document.iterate_items():
        item_ref = getattr(item, "self_ref", None)
        label = enum_value(getattr(item, "label", None))

        if label == "section_header":
            heading = normalize_text(getattr(item, "text", None) or getattr(item, "orig", None))
            if not heading:
                continue
            if is_asset_caption_text(heading):
                if item_ref and current_path:
                    section_paths[item_ref] = list(current_path)
                continue

            heading_number = parse_heading_number(heading)
            if heading_number is not None:
                numbered_stack = numbered_stack[: max(0, len(heading_number) - 1)] + [heading]
                current_path = list(numbered_stack)
            elif numbered_stack:
                current_path = numbered_stack + [heading]
            else:
                current_path = [heading]

            if item_ref:
                section_paths[item_ref] = list(current_path)
            continue

        if item_ref and current_path:
            section_paths[item_ref] = list(current_path)

    return section_paths


def serialize_table_cells(item: Any) -> list[dict[str, Any]]:
    table_data = getattr(item, "data", None)
    table_cells = getattr(table_data, "table_cells", None) or []
    records: list[dict[str, Any]] = []

    for cell in table_cells:
        records.append(
            {
                "text": normalize_text(getattr(cell, "text", None)),
                "row": int(getattr(cell, "start_row_offset_idx")),
                "col": int(getattr(cell, "start_col_offset_idx")),
                "row_span": int(getattr(cell, "row_span")),
                "col_span": int(getattr(cell, "col_span")),
                "column_header": bool(getattr(cell, "column_header")),
                "row_header": bool(getattr(cell, "row_header")),
                "row_section": bool(getattr(cell, "row_section")),
                "bbox": serialize_bbox(getattr(cell, "bbox", None)),
            }
        )

    return records


def build_table_text(table_cells: list[dict[str, Any]]) -> str:
    row_map: dict[int, list[tuple[int, str]]] = {}
    for cell in table_cells:
        text = cell.get("text", "")
        if not text:
            continue
        row = int(cell["row"])
        col = int(cell["col"])
        row_map.setdefault(row, []).append((col, text))

    lines = []
    for row in sorted(row_map):
        values = [text for _, text in sorted(row_map[row])]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def normalize_dataframe_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return normalize_text(value)
    return value


def serialize_table_dataframe(*, item: Any, document: Any | None) -> dict[str, Any] | None:
    export = getattr(item, "export_to_dataframe", None)
    if export is None or document is None:
        return None

    try:
        dataframe = export(document)
    except Exception:
        return None

    columns = [normalize_text(str(column)) or f"col_{index}" for index, column in enumerate(dataframe.columns)]
    rows = [
        [normalize_dataframe_value(value) for value in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]
    records = [dict(zip(columns, row, strict=False)) for row in rows]
    return {
        "columns": columns,
        "rows": rows,
        "records": records,
        "text": dataframe.to_string(index=False),
    }


def serialize_doc_item(*, item: Any, level: int | None, document: Any | None = None) -> dict[str, Any]:
    prov = getattr(item, "prov", None) or []
    record = {
        "level": int(level) if level is not None else None,
        "type": type(item).__name__,
        "label": enum_value(getattr(item, "label", None)) or type(item).__name__,
        "self_ref": getattr(item, "self_ref", None),
        "parent_ref": getattr(getattr(item, "parent", None), "cref", None),
        "child_refs": [
            cref
            for cref in (getattr(child, "cref", None) for child in (getattr(item, "children", None) or []))
            if cref
        ],
        "text": getattr(item, "text", None),
        "orig": getattr(item, "orig", None),
        "pages": sorted(
            {
                int(getattr(prov_item, "page_no"))
                for prov_item in prov
                if getattr(prov_item, "page_no", None) is not None
            }
        ),
        "provenance": [serialize_provenance(prov_item) for prov_item in prov],
    }
    table_cells = serialize_table_cells(item)
    if table_cells:
        record["table_cells"] = table_cells
        record["table_text"] = build_table_text(table_cells)
        table_dataframe = serialize_table_dataframe(item=item, document=document)
        if table_dataframe is not None:
            record["table_dataframe"] = table_dataframe
    return record


def build_item_records(document: Any) -> list[dict[str, Any]]:
    section_paths = build_section_paths(document)
    records: list[dict[str, Any]] = []

    for position, (item, level) in enumerate(document.iterate_items()):
        record = serialize_doc_item(item=item, level=level, document=document)
        record["position"] = position
        record["content_layer"] = enum_value(getattr(item, "content_layer", None))
        record["section_path"] = section_paths.get(record["self_ref"], [])
        records.append(record)

    associate_asset_captions(records)
    return records


def associate_asset_captions(records: list[dict[str, Any]]) -> None:
    by_ref = {record["self_ref"]: record for record in records if record.get("self_ref")}
    page_candidates: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if not is_caption_candidate(record):
            continue
        for page in record.get("pages", []):
            page_candidates.setdefault(int(page), []).append(record)

    for record in records:
        if record.get("label") not in {"table", "picture"}:
            continue

        explicit_caption = get_explicit_caption(record=record, by_ref=by_ref)
        if explicit_caption is not None:
            record["asset_caption"] = explicit_caption.get("orig") or explicit_caption.get("text") or ""
            record["asset_caption_ref"] = explicit_caption.get("self_ref")
            record["asset_caption_source"] = "explicit_child"
            continue

        pages = [int(page) for page in record.get("pages", [])]
        if len(pages) != 1:
            continue
        candidates = [
            candidate
            for candidate in page_candidates.get(pages[0], [])
            if candidate.get("self_ref") != record.get("self_ref")
        ]
        if not candidates:
            continue

        record["asset_caption_candidates"] = [
            {
                "ref": candidate.get("self_ref"),
                "text": candidate.get("orig") or candidate.get("text") or "",
                "label": candidate.get("label"),
            }
            for candidate in candidates
        ]
        if len(candidates) == 1:
            chosen = candidates[0]
            source = "single_candidate_on_page"
        else:
            distances = [(caption_distance(record, candidate), candidate) for candidate in candidates]
            distances.sort(key=lambda item: (item[0], item[1].get("position", 0)))
            chosen = distances[0][1]
            source = "nearest_candidate_on_page"

        record["asset_caption"] = chosen.get("orig") or chosen.get("text") or ""
        record["asset_caption_ref"] = chosen.get("self_ref")
        record["asset_caption_source"] = source


def is_caption_candidate(record: dict[str, Any]) -> bool:
    if record.get("label") not in {"caption", "section_header"}:
        return False
    text = normalize_text(record.get("orig") or record.get("text"))
    return bool(text and is_asset_caption_text(text))


def is_asset_caption_text(text: str) -> bool:
    return bool(ASSET_CAPTION_RE.match(normalize_text(text)))


def get_explicit_caption(
    *,
    record: dict[str, Any],
    by_ref: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for child_ref in record.get("child_refs", []):
        child = by_ref.get(child_ref)
        if child is not None and child.get("label") == "caption":
            return child
    return None


def caption_distance(asset: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float]:
    asset_bbox = first_bbox(asset)
    candidate_bbox = first_bbox(candidate)
    if asset_bbox is None or candidate_bbox is None:
        return (float("inf"), float("inf"))

    asset_cx = (asset_bbox["l"] + asset_bbox["r"]) / 2.0
    asset_cy = (asset_bbox["t"] + asset_bbox["b"]) / 2.0
    candidate_cx = (candidate_bbox["l"] + candidate_bbox["r"]) / 2.0
    candidate_cy = (candidate_bbox["t"] + candidate_bbox["b"]) / 2.0
    vertical_gap = max(
        0.0,
        max(candidate_bbox["t"] - asset_bbox["b"], asset_bbox["t"] - candidate_bbox["b"]),
    )
    center_distance = ((asset_cx - candidate_cx) ** 2 + (asset_cy - candidate_cy) ** 2) ** 0.5
    return (vertical_gap, center_distance)


def first_bbox(record: dict[str, Any]) -> dict[str, Any] | None:
    for prov in record.get("provenance", []):
        bbox = prov.get("bbox")
        if bbox is not None:
            return bbox
    return None


def build_contextualized_text(*, section_path: list[str], text: str, captions: list[str] | None = None) -> str:
    parts: list[str] = []
    for value in [*section_path, *(captions or []), text]:
        cleaned = normalize_text(value)
        if cleaned and (not parts or parts[-1] != cleaned):
            parts.append(cleaned)
    return "\n\n".join(parts)


def resolve_chunk_section_path(*, meta: Any, section_paths: dict[str, list[str]]) -> list[str]:
    doc_items = getattr(meta, "doc_items", None) or []
    chunk_paths = [
        section_paths[item_ref]
        for item_ref in [getattr(item, "self_ref", None) for item in doc_items]
        if item_ref in section_paths
    ]
    if not chunk_paths:
        headings = getattr(meta, "headings", None) or []
        return [normalize_text(value) for value in headings if normalize_text(value)]
    return max(chunk_paths, key=len)


def derive_chunk_type(labels: list[str]) -> str:
    if "table" in labels:
        return "table"
    if "picture" in labels:
        return "picture"
    return "text"


def build_chunk_records(
    document: Any,
    *,
    chunker: Any | None = None,
    chunker_name: str = "hierarchical",
) -> list[dict[str, Any]]:
    if chunker is None:
        from docling.chunking import HierarchicalChunker

        chunker = HierarchicalChunker()
    section_paths = build_section_paths(document)
    records: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunker.chunk(document)):
        meta = chunk.meta
        doc_items = list(getattr(meta, "doc_items", []) or [])
        labels = sorted(
            {
                enum_value(getattr(item, "label", None)) or type(item).__name__
                for item in doc_items
            }
        )
        pages = sorted(
            {
                int(getattr(prov_item, "page_no"))
                for item in doc_items
                for prov_item in (getattr(item, "prov", None) or [])
                if getattr(prov_item, "page_no", None) is not None
            }
        )

        page_boxes: dict[int, list[dict[str, Any]]] = {}
        for item in doc_items:
            for prov_item in getattr(item, "prov", None) or []:
                page_no = getattr(prov_item, "page_no", None)
                bbox = serialize_bbox(getattr(prov_item, "bbox", None))
                if page_no is None or bbox is None:
                    continue
                page_boxes.setdefault(int(page_no), []).append(bbox)

        page_bboxes = [
            {
                "page_number": page_number,
                "bbox": merge_bboxes(bboxes),
            }
            for page_number, bboxes in sorted(page_boxes.items())
            if merge_bboxes(bboxes) is not None
        ]

        section_path = resolve_chunk_section_path(meta=meta, section_paths=section_paths)
        captions = [normalize_text(value) for value in (getattr(meta, "captions", None) or []) if normalize_text(value)]
        records.append(
            {
                "index": index,
                "chunker": chunker_name,
                "chunk_type": derive_chunk_type(labels),
                "text": chunk.text,
                "contextualized_text": build_contextualized_text(
                    section_path=section_path,
                    captions=captions,
                    text=chunk.text,
                ),
                "section_path": section_path,
                "captions": captions,
                "labels": labels,
                "pages": pages,
                "doc_item_refs": [getattr(item, "self_ref", None) for item in doc_items if getattr(item, "self_ref", None)],
                "page_bboxes": page_bboxes,
            }
        )

    return records
