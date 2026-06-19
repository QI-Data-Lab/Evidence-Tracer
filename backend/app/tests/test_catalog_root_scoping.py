from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.database import connection, init_database
from app.services.catalog_service import CatalogService


class CatalogRootScopingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "catalog.sqlite3"

        settings = SimpleNamespace(db_path=self.db_path)
        self.get_settings_patch = patch("app.core.database.get_settings", return_value=settings)
        self.get_settings_patch.start()
        self.addCleanup(self.get_settings_patch.stop)

        init_database()
        self.catalog = CatalogService()

        self.root_a = str((self.temp_path / "root-a").resolve())
        self.root_b = str((self.temp_path / "root-b").resolve())
        Path(self.root_a).mkdir()
        Path(self.root_b).mkdir()

        self.catalog.set_root_path(self.root_a)

        with connection() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                    root_path, relative_path, source_path, file_name, status, page_count,
                    file_size, modified_time, fingerprint, summary_json, metadata_json,
                    error_message, last_scanned_at, last_processed_at, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, 'done', NULL, NULL, NULL, NULL, '{}', '{}', NULL, NULL, NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
                """,
                (self.root_a, "a/doc-a.pdf", f"{self.root_a}/a/doc-a.pdf", "doc-a.pdf"),
            )
            conn.execute(
                """
                INSERT INTO documents(
                    root_path, relative_path, source_path, file_name, status, page_count,
                    file_size, modified_time, fingerprint, summary_json, metadata_json,
                    error_message, last_scanned_at, last_processed_at, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, 'done', NULL, NULL, NULL, NULL, '{}', '{}', NULL, NULL, NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
                """,
                (self.root_b, "b/doc-b.pdf", f"{self.root_b}/b/doc-b.pdf", "doc-b.pdf"),
            )

    def test_list_documents_only_returns_active_root(self) -> None:
        documents = self.catalog.list_documents()

        self.assertEqual([document["relative_path"] for document in documents], ["a/doc-a.pdf"])

    def test_build_tree_only_contains_active_root_documents(self) -> None:
        tree = self.catalog.build_tree()

        child_names = [child["name"] for child in tree["children"]]
        self.assertEqual(child_names, ["a"])

    def test_get_documents_for_processing_only_returns_active_root(self) -> None:
        documents = self.catalog.get_documents_for_processing()

        self.assertEqual([document["relative_path"] for document in documents], ["a/doc-a.pdf"])


if __name__ == "__main__":
    unittest.main()
