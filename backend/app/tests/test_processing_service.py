from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.processing_service import ProcessingService


class ProcessingServicePreviewInvalidationTests(unittest.TestCase):
    def test_process_document_removes_cached_previews_after_successful_reprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "sample.pdf"
            source_path.write_bytes(b"%PDF-1.4\n")

            preview_dir = temp_path / "page_previews" / "42"
            preview_dir.mkdir(parents=True)
            cached_preview = preview_dir / "page-001-s160.png"
            cached_preview.write_bytes(b"stale image")

            catalog = Mock()
            catalog.get_document.side_effect = [
                {
                    "id": 42,
                    "source_path": str(source_path),
                },
                {
                    "id": 42,
                    "source_path": str(source_path),
                },
            ]

            processed_payload = {
                "page_records": [],
                "item_records": [],
                "chunk_records": [],
                "summary": {},
            }

            service = ProcessingService(catalog=catalog)

            with patch("app.services.processing_service.process_pdf", return_value=processed_payload), patch(
                "app.services.processing_service.get_settings",
                return_value=SimpleNamespace(preview_dir=temp_path / "page_previews"),
                create=True,
            ):
                service.process_document(42)

            self.assertFalse(cached_preview.exists())
            self.assertFalse(preview_dir.exists())


class ProcessingServiceProgressTests(unittest.TestCase):
    def test_process_documents_stream_reports_page_progress_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_pdf = temp_path / "first.pdf"
            second_pdf = temp_path / "second.pdf"
            first_pdf.write_bytes(b"%PDF-1.4\n")
            second_pdf.write_bytes(b"%PDF-1.4\n")

            first_document = {
                "id": 1,
                "relative_path": "first.pdf",
                "source_path": str(first_pdf),
                "file_name": "first.pdf",
            }
            second_document = {
                "id": 2,
                "relative_path": "second.pdf",
                "source_path": str(second_pdf),
                "file_name": "second.pdf",
            }
            refreshed_first = {
                **first_document,
                "status": "done",
                "page_count": 2,
            }

            catalog = Mock()
            catalog.get_documents_for_processing.return_value = [first_document, second_document]
            catalog.get_document.side_effect = [first_document, refreshed_first, second_document]

            processed_payload = {
                "page_records": [{"page_number": 1, "width": 100, "height": 100}, {"page_number": 2, "width": 100, "height": 100}],
                "item_records": [],
                "chunk_records": [],
                "summary": {},
            }

            service = ProcessingService(catalog=catalog)

            with patch(
                "app.services.processing_service.read_pdf_page_data",
                side_effect=[
                    {"page_count": 2, "pages": []},
                    {"page_count": 3, "pages": []},
                ],
                create=True,
            ), patch(
                "app.services.processing_service.process_pdf",
                side_effect=[processed_payload, RuntimeError("docling failed")],
            ), patch(
                "app.services.processing_service.get_settings",
                return_value=SimpleNamespace(preview_dir=temp_path / "page_previews"),
                create=True,
            ):
                events = list(service.process_documents_stream(only_stale=True))

            self.assertEqual(
                [event["type"] for event in events],
                [
                    "process_started",
                    "document_started",
                    "document_completed",
                    "document_started",
                    "document_failed",
                    "process_completed",
                ],
            )
            self.assertEqual(events[0]["total_documents"], 2)
            self.assertEqual(events[0]["total_pages"], 5)
            self.assertEqual(events[2]["processed_documents"], 1)
            self.assertEqual(events[2]["processed_pages"], 2)
            self.assertEqual(events[4]["processed_documents"], 2)
            self.assertEqual(events[4]["processed_pages"], 5)
            self.assertEqual(events[-1]["result"]["requested"], 2)
            self.assertEqual(len(events[-1]["result"]["processed"]), 1)
            self.assertEqual(len(events[-1]["result"]["failed"]), 1)


if __name__ == "__main__":
    unittest.main()
