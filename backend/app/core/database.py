from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.core.config import get_settings


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        root_path TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        source_path TEXT NOT NULL UNIQUE,
        file_name TEXT NOT NULL,
        status TEXT NOT NULL,
        page_count INTEGER,
        file_size INTEGER,
        modified_time REAL,
        fingerprint TEXT,
        summary_json TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        error_message TEXT,
        last_scanned_at TEXT,
        last_processed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_root_idx ON documents(root_path)",
    "CREATE INDEX IF NOT EXISTS documents_relative_idx ON documents(relative_path)",
    "CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(status)",
    """
    CREATE TABLE IF NOT EXISTS document_pages (
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        width REAL NOT NULL,
        height REAL NOT NULL,
        rotation INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (document_id, page_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        self_ref TEXT,
        parent_ref TEXT,
        position INTEGER NOT NULL,
        level INTEGER,
        label TEXT NOT NULL,
        item_type TEXT NOT NULL,
        text TEXT,
        orig_text TEXT,
        section_path_json TEXT NOT NULL,
        pages_json TEXT NOT NULL,
        provenance_json TEXT NOT NULL,
        bbox_json TEXT,
        payload_json TEXT NOT NULL,
        UNIQUE(document_id, self_ref)
    )
    """,
    "CREATE INDEX IF NOT EXISTS document_items_doc_idx ON document_items(document_id)",
    "CREATE INDEX IF NOT EXISTS document_items_ref_idx ON document_items(document_id, self_ref)",
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunker TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_type TEXT NOT NULL,
        text TEXT NOT NULL,
        contextualized_text TEXT,
        section_path_json TEXT NOT NULL,
        pages_json TEXT NOT NULL,
        item_refs_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        UNIQUE(document_id, chunker, chunk_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS document_chunks_doc_idx ON document_chunks(document_id)",
    "CREATE INDEX IF NOT EXISTS document_chunks_doc_chunker_idx ON document_chunks(document_id, chunker)",
    """
    CREATE TABLE IF NOT EXISTS chunk_pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        bbox_json TEXT,
        UNIQUE(chunk_id, page_number)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chunk_pages_doc_page_idx ON chunk_pages(document_id, page_number)",
]


def create_connection() -> sqlite3.Connection:
    settings = get_settings()
    connection = sqlite3.connect(settings.db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database() -> None:
    with connection() as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = create_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

