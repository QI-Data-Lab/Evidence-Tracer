from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "test" / "docling_bbox_debug.py"
SPEC = importlib.util.spec_from_file_location("docling_bbox_debug", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
docling_bbox_debug = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docling_bbox_debug)


class DoclingBboxDebugChunkingTests(unittest.TestCase):
    def test_build_debug_chunks_requires_hierarchical_chunker(self) -> None:
        document = object()

        with patch.object(docling_bbox_debug, "build_chunk_records", return_value=[{"chunker": "flat"}]):
            with self.assertRaisesRegex(RuntimeError, "hierarchical"):
                docling_bbox_debug.build_debug_chunks(document)

    def test_build_debug_chunks_returns_hierarchical_chunks(self) -> None:
        document = object()
        expected_chunks = [{"chunker": "hierarchical", "page_bboxes": [], "index": 0}]

        with patch.object(docling_bbox_debug, "HierarchicalChunker", return_value=SimpleNamespace()), patch.object(
            docling_bbox_debug,
            "build_chunk_records",
            return_value=expected_chunks,
        ):
            self.assertEqual(docling_bbox_debug.build_debug_chunks(document), expected_chunks)

    def test_bbox_to_pixels_preserves_fractional_coordinates(self) -> None:
        pixel_rect = docling_bbox_debug.bbox_to_pixels(
            bbox={"l": 1.0, "r": 2.0, "t": 9.0, "b": 7.0, "coord_origin": "TOPLEFT"},
            page_width=3.0,
            page_height=10.0,
            image_width=100,
            image_height=100,
        )

        self.assertEqual(pixel_rect, (33.33333333333333, 70.0, 66.66666666666666, 90.0))


if __name__ == "__main__":
    unittest.main()
