from __future__ import annotations

from typing import Any


def is_bottom_origin(value: Any) -> bool:
    return "bottom" in str(value or "TOPLEFT").lower()


def _horizontal_bounds(bbox: dict[str, Any]) -> tuple[float, float]:
    left = float(bbox["l"])
    right = float(bbox["r"])
    return min(left, right), max(left, right)


def _vertical_bounds(bbox: dict[str, Any]) -> tuple[float, float]:
    top = float(bbox["t"])
    bottom = float(bbox["b"])
    if is_bottom_origin(bbox.get("coord_origin")):
        return max(top, bottom), min(top, bottom)
    return min(top, bottom), max(top, bottom)


def merge_bboxes(bboxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not bboxes:
        return None

    first_bbox = bboxes[0]
    bottom_origin = is_bottom_origin(first_bbox.get("coord_origin"))
    left_edges = []
    right_edges = []
    top_edges = []
    bottom_edges = []

    for bbox in bboxes:
        left, right = _horizontal_bounds(bbox)
        top, bottom = _vertical_bounds(bbox)
        left_edges.append(left)
        right_edges.append(right)
        top_edges.append(top)
        bottom_edges.append(bottom)

    return {
        "l": min(left_edges),
        "r": max(right_edges),
        "t": max(top_edges) if bottom_origin else min(top_edges),
        "b": min(bottom_edges) if bottom_origin else max(bottom_edges),
        "coord_origin": first_bbox.get("coord_origin", "TOPLEFT"),
    }


def normalize_bbox(*, bbox: dict[str, Any] | None, page_width: float, page_height: float) -> dict[str, float] | None:
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None

    left, right = _horizontal_bounds(bbox)
    top, bottom = _vertical_bounds(bbox)

    if is_bottom_origin(bbox.get("coord_origin")):
        normalized_top = page_height - top
        normalized_bottom = page_height - bottom
    else:
        normalized_top = top
        normalized_bottom = bottom

    width = max(0.0, right - left)
    height = max(0.0, normalized_bottom - normalized_top)

    return {
        "left_pct": max(0.0, min(100.0, left / page_width * 100.0)),
        "top_pct": max(0.0, min(100.0, normalized_top / page_height * 100.0)),
        "width_pct": max(0.0, min(100.0, width / page_width * 100.0)),
        "height_pct": max(0.0, min(100.0, height / page_height * 100.0)),
    }
