from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from app.core.database import connection
from app.utils.bbox import normalize_bbox
from app.utils.common import compute_fingerprint, json_dumps, json_loads, utc_now_iso


PROCESSABLE_STATUSES = {"new", "stale", "error"}


class CatalogService:
    def _active_root_path(self) -> str | None:
        root_path = self.get_root_path()
        return str(Path(root_path).expanduser().resolve()) if root_path else None

    def get_config(self) -> dict[str, Any]:
        return {
            "root_path": self.get_root_path(),
        }

    def get_root_path(self) -> str | None:
        with connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = 'root_path'").fetchone()
        return None if row is None else str(row["value"])

    def set_root_path(self, root_path: str) -> dict[str, Any]:
        resolved = Path(root_path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"Invalid root path: {resolved}")

        now = utc_now_iso()
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES('root_path', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (str(resolved), now),
            )
        return {"root_path": str(resolved)}

    def scan_documents(self, root_path: str | None = None) -> dict[str, Any]:
        configured_root = root_path or self.get_root_path()
        if not configured_root:
            raise ValueError("A valid root path must be configured before scanning.")

        active_root = Path(configured_root).expanduser().resolve()
        if not active_root.exists() or not active_root.is_dir():
            raise ValueError("A valid root path must be configured before scanning.")

        self.set_root_path(str(active_root))
        now = utc_now_iso()
        seen_paths: set[str] = set()

        with connection() as conn:
            existing_rows = conn.execute(
                "SELECT id, source_path, status, fingerprint FROM documents WHERE root_path = ?",
                (str(active_root),),
            ).fetchall()
            existing_map = {str(row["source_path"]): row for row in existing_rows}

            scanned = 0
            new_count = 0
            stale_count = 0
            unchanged_count = 0

            for pdf_path in sorted(active_root.rglob("*.pdf")):
                if not pdf_path.is_file():
                    continue

                scanned += 1
                resolved = pdf_path.resolve()
                source_path = str(resolved)
                relative_path = str(PurePosixPath(resolved.relative_to(active_root)))
                stat = resolved.stat()
                fingerprint = compute_fingerprint(resolved)
                existing = existing_map.get(source_path)

                if existing is None:
                    status = "new"
                    new_count += 1
                elif existing["fingerprint"] != fingerprint:
                    status = "stale"
                    stale_count += 1
                elif existing["status"] == "missing":
                    status = "stale"
                    stale_count += 1
                elif existing["status"] == "processing":
                    status = "stale"
                    stale_count += 1
                else:
                    status = str(existing["status"])
                    unchanged_count += 1

                seen_paths.add(source_path)
                conn.execute(
                    """
                    INSERT INTO documents(
                        root_path,
                        relative_path,
                        source_path,
                        file_name,
                        status,
                        page_count,
                        file_size,
                        modified_time,
                        fingerprint,
                        summary_json,
                        metadata_json,
                        error_message,
                        last_scanned_at,
                        last_processed_at,
                        created_at,
                        updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, NULL, ?, ?, ?, '{}', '{}', NULL, ?, NULL, ?, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                        root_path = excluded.root_path,
                        relative_path = excluded.relative_path,
                        file_name = excluded.file_name,
                        status = excluded.status,
                        file_size = excluded.file_size,
                        modified_time = excluded.modified_time,
                        fingerprint = excluded.fingerprint,
                        error_message = CASE WHEN excluded.status IN ('new', 'stale') THEN NULL ELSE documents.error_message END,
                        last_scanned_at = excluded.last_scanned_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(active_root),
                        relative_path,
                        source_path,
                        resolved.name,
                        status,
                        int(stat.st_size),
                        float(stat.st_mtime),
                        fingerprint,
                        now,
                        now,
                        now,
                    ),
                )

            missing_rows = conn.execute(
                "SELECT source_path FROM documents WHERE root_path = ?",
                (str(active_root),),
            ).fetchall()
            missing_count = 0
            for row in missing_rows:
                source_path = str(row["source_path"])
                if source_path in seen_paths:
                    continue
                missing_count += 1
                conn.execute(
                    "UPDATE documents SET status = 'missing', updated_at = ? WHERE source_path = ?",
                    (now, source_path),
                )

        return {
            "root_path": str(active_root),
            "scanned": scanned,
            "new": new_count,
            "stale": stale_count,
            "unchanged": unchanged_count,
            "missing": missing_count,
        }

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT
                d.*, 
                (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunk_count,
                (SELECT COUNT(*) FROM document_items i WHERE i.document_id = d.id) AS item_count
            FROM documents d
        """
        params: list[Any] = []
        clauses: list[str] = []
        active_root = self._active_root_path()
        if active_root:
            clauses.append("d.root_path = ?")
            params.append(active_root)
        if status:
            clauses.append("d.status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY d.relative_path"

        with connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with connection() as conn:
            row = conn.execute(
                """
                SELECT
                    d.*, 
                    (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunk_count,
                    (SELECT COUNT(*) FROM document_items i WHERE i.document_id = d.id) AS item_count
                FROM documents d
                WHERE d.id = ?
                """,
                (document_id,),
            ).fetchone()
        return None if row is None else self._row_to_document(row)

    def get_documents_for_processing(
        self,
        document_ids: list[int] | None = None,
        *,
        only_stale: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM documents WHERE status != 'missing'"
        params: list[Any] = []
        active_root = self._active_root_path()
        if active_root:
            query += " AND root_path = ?"
            params.append(active_root)
        if document_ids:
            placeholders = ", ".join("?" for _ in document_ids)
            query += f" AND id IN ({placeholders})"
            params.extend(document_ids)
        elif only_stale:
            query += " AND status IN ('new', 'stale', 'error')"
        query += " ORDER BY relative_path"

        with connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def update_document_status(self, document_id: int, *, status: str, error_message: str | None = None) -> None:
        with connection() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
                (status, error_message, utc_now_iso(), document_id),
            )

    def replace_document_content(
        self,
        *,
        document_id: int,
        page_records: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
        chunk_records: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        now = utc_now_iso()
        metadata = {
            "item_count": len(item_records),
            "chunk_count": len(chunk_records),
        }

        with connection() as conn:
            conn.execute("DELETE FROM document_pages WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_items WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))

            for page in page_records:
                conn.execute(
                    """
                    INSERT INTO document_pages(document_id, page_number, width, height, rotation)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        int(page["page_number"]),
                        float(page["width"]),
                        float(page["height"]),
                        int(page.get("rotation", 0)),
                    ),
                )

            for item in item_records:
                provenance = item.get("provenance", [])
                first_bbox = None
                for prov in provenance:
                    bbox = prov.get("bbox")
                    if bbox is not None:
                        first_bbox = bbox
                        break
                payload = {
                    "child_refs": item.get("child_refs", []),
                    "content_layer": item.get("content_layer"),
                    "asset_caption": item.get("asset_caption"),
                    "asset_caption_ref": item.get("asset_caption_ref"),
                    "asset_caption_source": item.get("asset_caption_source"),
                    "asset_caption_candidates": item.get("asset_caption_candidates", []),
                    "table_cells": item.get("table_cells", []),
                    "table_text": item.get("table_text"),
                    "table_dataframe": item.get("table_dataframe"),
                }
                conn.execute(
                    """
                    INSERT INTO document_items(
                        document_id,
                        self_ref,
                        parent_ref,
                        position,
                        level,
                        label,
                        item_type,
                        text,
                        orig_text,
                        section_path_json,
                        pages_json,
                        provenance_json,
                        bbox_json,
                        payload_json
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        item.get("self_ref"),
                        item.get("parent_ref"),
                        int(item.get("position", 0)),
                        item.get("level"),
                        item.get("label", "unknown"),
                        item.get("type", "UnknownItem"),
                        item.get("text"),
                        item.get("orig"),
                        json_dumps(item.get("section_path", [])),
                        json_dumps(item.get("pages", [])),
                        json_dumps(provenance),
                        json_dumps(first_bbox) if first_bbox is not None else None,
                        json_dumps(payload),
                    ),
                )

            for chunk in chunk_records:
                cursor = conn.execute(
                    """
                    INSERT INTO document_chunks(
                        document_id,
                        chunker,
                        chunk_index,
                        chunk_type,
                        text,
                        contextualized_text,
                        section_path_json,
                        pages_json,
                        item_refs_json,
                        payload_json
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk.get("chunker", "hierarchical"),
                        int(chunk.get("index", 0)),
                        chunk.get("chunk_type", "text"),
                        chunk.get("text", ""),
                        chunk.get("contextualized_text", ""),
                        json_dumps(chunk.get("section_path", [])),
                        json_dumps(chunk.get("pages", [])),
                        json_dumps(chunk.get("doc_item_refs", [])),
                        json_dumps(
                            {
                                "labels": chunk.get("labels", []),
                            }
                        ),
                    ),
                )
                chunk_id = int(cursor.lastrowid)

                for page_bbox in chunk.get("page_bboxes", []):
                    conn.execute(
                        """
                        INSERT INTO chunk_pages(chunk_id, document_id, page_number, bbox_json)
                        VALUES(?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            document_id,
                            int(page_bbox["page_number"]),
                            json_dumps(page_bbox.get("bbox")) if page_bbox.get("bbox") is not None else None,
                        ),
                    )

            conn.execute(
                """
                UPDATE documents
                SET status = 'done',
                    page_count = ?,
                    summary_json = ?,
                    metadata_json = ?,
                    error_message = NULL,
                    last_processed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    len(page_records),
                    json_dumps(summary),
                    json_dumps(metadata),
                    now,
                    now,
                    document_id,
                ),
            )

    def list_pages(self, document_id: int) -> list[dict[str, Any]]:
        with connection() as conn:
            page_rows = conn.execute(
                """
                SELECT
                    dp.page_number,
                    dp.width,
                    dp.height,
                    dp.rotation,
                    COUNT(cp.id) AS chunk_count
                FROM document_pages dp
                LEFT JOIN chunk_pages cp
                    ON cp.document_id = dp.document_id
                    AND cp.page_number = dp.page_number
                WHERE dp.document_id = ?
                GROUP BY dp.document_id, dp.page_number, dp.width, dp.height, dp.rotation
                ORDER BY dp.page_number
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in page_rows]

    def get_page_bundle(self, document_id: int, page_number: int) -> dict[str, Any] | None:
        document = self.get_document(document_id)
        if document is None:
            return None

        with connection() as conn:
            page_row = conn.execute(
                "SELECT * FROM document_pages WHERE document_id = ? AND page_number = ?",
                (document_id, page_number),
            ).fetchone()
            if page_row is None:
                return None

            chunk_rows = conn.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.chunker,
                    c.chunk_index,
                    c.chunk_type,
                    c.text,
                    c.contextualized_text,
                    c.section_path_json,
                    c.pages_json,
                    c.item_refs_json,
                    c.payload_json,
                    cp.bbox_json
                FROM chunk_pages cp
                JOIN document_chunks c ON c.id = cp.chunk_id
                WHERE cp.document_id = ? AND cp.page_number = ?
                ORDER BY c.chunk_index
                """,
                (document_id, page_number),
            ).fetchall()

        width = float(page_row["width"])
        height = float(page_row["height"])
        chunks = []
        for row in chunk_rows:
            bbox = json_loads(row["bbox_json"], None)
            chunks.append(
                {
                    "id": int(row["chunk_id"]),
                    "chunker": row["chunker"],
                    "chunk_index": int(row["chunk_index"]),
                    "chunk_type": row["chunk_type"],
                    "text": row["text"],
                    "contextualized_text": row["contextualized_text"],
                    "section_path": json_loads(row["section_path_json"], []),
                    "pages": json_loads(row["pages_json"], []),
                    "item_refs": json_loads(row["item_refs_json"], []),
                    "labels": json_loads(row["payload_json"], {}).get("labels", []),
                    "bbox": bbox,
                    "normalized_bbox": normalize_bbox(bbox=bbox, page_width=width, page_height=height),
                }
            )

        return {
            "document": document,
            "page": {
                "page_number": int(page_row["page_number"]),
                "width": width,
                "height": height,
                "rotation": int(page_row["rotation"]),
            },
            "chunks": chunks,
        }

    def query_artifacts(
        self,
        *,
        query: str = "",
        kind: str = "all",
        document_id: int | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        normalized_query = " ".join(query.lower().split())
        terms = [term for term in normalized_query.split(" ") if term]
        valid_kinds = {"all", "chunks", "index", "sections", "tables", "figures", "references", "items"}
        if kind not in valid_kinds:
            raise ValueError(f"Unsupported query kind: {kind}")

        documents = self.list_documents()
        document_index = {int(document["id"]): document for document in documents}
        processed_documents = [document for document in documents if document["status"] == "done"]

        with connection() as conn:
            coverage_row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents WHERE status = 'done') AS processed_documents,
                    (SELECT COUNT(*) FROM document_pages) AS pages,
                    (SELECT COUNT(*) FROM document_chunks) AS chunks,
                    (SELECT COUNT(*) FROM document_items) AS items,
                    (SELECT COUNT(*) FROM document_items WHERE label = 'section_header') AS sections,
                    (SELECT COUNT(*) FROM document_items WHERE label = 'table') AS tables,
                    (SELECT COUNT(*) FROM document_items WHERE label = 'picture') AS figures
                """
            ).fetchone()

            chunk_rows = []
            if kind in {"all", "chunks", "references"}:
                chunk_query = """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.chunker,
                        c.chunk_index,
                        c.chunk_type,
                        c.text,
                        c.contextualized_text,
                        c.section_path_json,
                        c.pages_json,
                        c.item_refs_json,
                        c.payload_json,
                        cp.page_number,
                        cp.bbox_json,
                        dp.width,
                        dp.height
                    FROM document_chunks c
                    LEFT JOIN chunk_pages cp ON cp.chunk_id = c.id
                    LEFT JOIN document_pages dp
                        ON dp.document_id = c.document_id
                        AND dp.page_number = cp.page_number
                """
                params: list[Any] = []
                if document_id is not None:
                    chunk_query += " WHERE c.document_id = ?"
                    params.append(document_id)
                chunk_query += " ORDER BY c.document_id, c.chunk_index, cp.page_number"
                chunk_rows = conn.execute(chunk_query, params).fetchall()

            item_rows = []
            if kind in {"all", "index", "sections", "tables", "figures", "references", "items"}:
                item_query = "SELECT * FROM document_items"
                clauses = []
                params = []
                if document_id is not None:
                    clauses.append("document_id = ?")
                    params.append(document_id)
                if kind in {"index", "sections"}:
                    clauses.append("label = 'section_header'")
                elif kind == "tables":
                    clauses.append("label = 'table'")
                elif kind == "figures":
                    clauses.append("label = 'picture'")
                if clauses:
                    item_query += " WHERE " + " AND ".join(clauses)
                item_query += " ORDER BY document_id, position"
                item_rows = conn.execute(item_query, params).fetchall()

        results: list[dict[str, Any]] = []

        for row in chunk_rows:
            document = document_index.get(int(row["document_id"]))
            if document is None:
                continue
            payload = json_loads(row["payload_json"], {})
            text = row["text"] or ""
            contextualized_text = row["contextualized_text"] or ""
            searchable = "\n".join(
                [
                    document["file_name"],
                    document["relative_path"],
                    contextualized_text,
                    " ".join(json_loads(row["section_path_json"], [])),
                    " ".join(payload.get("labels", [])),
                    text,
                ]
            )
            linked_documents = self._find_linked_documents(
                searchable,
                current_document_id=int(row["document_id"]),
                documents=processed_documents,
            )
            score = self._query_score(terms, searchable)
            if kind == "references" and not linked_documents:
                continue
            if terms and score == 0:
                continue

            bbox = json_loads(row["bbox_json"], None)
            page_number = int(row["page_number"]) if row["page_number"] is not None else None
            width = float(row["width"]) if row["width"] is not None else 0.0
            height = float(row["height"]) if row["height"] is not None else 0.0
            results.append(
                {
                    "id": f"chunk:{row['chunk_id']}:{page_number or 0}",
                    "result_type": "reference" if kind == "references" and linked_documents else "chunk",
                    "score": score,
                    "document": self._compact_document(document),
                    "page_number": page_number,
                    "page_numbers": json_loads(row["pages_json"], []),
                    "chunk_id": int(row["chunk_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "chunk_type": row["chunk_type"],
                    "title": self._result_title(
                        fallback=f"Chunk {int(row['chunk_index'])}",
                        section_path=json_loads(row["section_path_json"], []),
                    ),
                    "section_path": json_loads(row["section_path_json"], []),
                    "labels": payload.get("labels", []),
                    "text": contextualized_text or text,
                    "bbox": bbox,
                    "normalized_bbox": normalize_bbox(bbox=bbox, page_width=width, page_height=height),
                    "linked_documents": linked_documents,
                }
            )

        for row in item_rows:
            document = document_index.get(int(row["document_id"]))
            if document is None:
                continue
            payload = json_loads(row["payload_json"], {})
            section_path = json_loads(row["section_path_json"], [])
            pages = json_loads(row["pages_json"], [])
            table_text = payload.get("table_text") or ""
            text = row["text"] or row["orig_text"] or table_text or ""
            searchable = "\n".join(
                [
                    document["file_name"],
                    document["relative_path"],
                    row["label"],
                    row["item_type"],
                    " ".join(section_path),
                    str(payload.get("asset_caption") or ""),
                    text,
                    table_text,
                ]
            )
            linked_documents = self._find_linked_documents(
                searchable,
                current_document_id=int(row["document_id"]),
                documents=processed_documents,
            )
            score = self._query_score(terms, searchable)
            if kind == "references" and not linked_documents:
                continue
            if terms and score == 0:
                continue

            result_type = self._item_result_type(row["label"], kind)
            results.append(
                {
                    "id": f"item:{row['id']}",
                    "result_type": result_type,
                    "score": score,
                    "document": self._compact_document(document),
                    "page_number": int(pages[0]) if pages else None,
                    "page_numbers": pages,
                    "item_id": int(row["id"]),
                    "item_ref": row["self_ref"],
                    "label": row["label"],
                    "item_type": row["item_type"],
                    "title": self._result_title(fallback=text or row["label"], section_path=section_path),
                    "section_path": section_path,
                    "text": text,
                    "bbox": json_loads(row["bbox_json"], None),
                    "linked_documents": linked_documents,
                    "payload": {
                        "table_text": table_text,
                        "table_cells": payload.get("table_cells", []),
                        "asset_caption": payload.get("asset_caption"),
                        "content_layer": payload.get("content_layer"),
                    },
                }
            )

        if not terms and kind in {"index", "sections", "tables", "figures"}:
            results.sort(key=lambda result: (result["document"]["relative_path"], result.get("page_number") or 0))
        else:
            results.sort(
                key=lambda result: (
                    -int(result.get("score", 0)),
                    result["document"]["relative_path"],
                    result.get("page_number") or 0,
                )
            )

        return {
            "query": query,
            "kind": kind,
            "document_id": document_id,
            "limit": limit,
            "coverage": dict(coverage_row) if coverage_row is not None else {},
            "results": results[:limit],
        }

    def build_tree(self) -> dict[str, Any]:
        root_path = self.get_root_path()
        documents = [doc for doc in self.list_documents() if doc["status"] != "missing"]
        root_name = Path(root_path).name if root_path else "Documents"
        root_node: dict[str, Any] = {
            "type": "folder",
            "name": root_name,
            "path": "",
            "children": [],
        }
        folder_index: dict[str, dict[str, Any]] = {"": root_node}

        for document in documents:
            parts = PurePosixPath(document["relative_path"]).parts
            current_path = ""

            for part in parts[:-1]:
                next_path = part if not current_path else f"{current_path}/{part}"
                if next_path not in folder_index:
                    node = {
                        "type": "folder",
                        "name": part,
                        "path": next_path,
                        "children": [],
                    }
                    folder_index[next_path] = node
                    folder_index[current_path]["children"].append(node)
                current_path = next_path

            folder_index[current_path]["children"].append(
                {
                    "type": "document",
                    "name": document["file_name"],
                    "path": document["relative_path"],
                    "document_id": document["id"],
                    "status": document["status"],
                    "page_count": document["page_count"],
                    "chunk_count": document["chunk_count"],
                }
            )

        self._sort_tree(root_node)
        return root_node

    def _sort_tree(self, node: dict[str, Any]) -> None:
        children = node.get("children")
        if not children:
            return
        children.sort(key=lambda child: (child["type"] != "folder", child["name"].lower()))
        for child in children:
            if child["type"] == "folder":
                self._sort_tree(child)

    def _row_to_document(self, row: Any) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "root_path": row["root_path"],
            "relative_path": row["relative_path"],
            "source_path": row["source_path"],
            "file_name": row["file_name"],
            "status": row["status"],
            "page_count": row["page_count"],
            "file_size": row["file_size"],
            "modified_time": row["modified_time"],
            "fingerprint": row["fingerprint"],
            "summary": json_loads(row["summary_json"], {}),
            "metadata": json_loads(row["metadata_json"], {}),
            "error_message": row["error_message"],
            "last_scanned_at": row["last_scanned_at"],
            "last_processed_at": row["last_processed_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "chunk_count": int(row["chunk_count"]) if "chunk_count" in row.keys() else 0,
            "item_count": int(row["item_count"]) if "item_count" in row.keys() else 0,
            "needs_processing": row["status"] in PROCESSABLE_STATUSES,
        }

    def _query_score(self, terms: list[str], value: str) -> int:
        if not terms:
            return 1
        normalized = value.lower()
        score = 0
        for term in terms:
            if term in normalized:
                score += 1
        if " ".join(terms) in normalized:
            score += len(terms)
        return score

    def _compact_document(self, document: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": document["id"],
            "file_name": document["file_name"],
            "relative_path": document["relative_path"],
            "status": document["status"],
            "page_count": document["page_count"],
        }

    def _find_linked_documents(
        self,
        value: str,
        *,
        current_document_id: int,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = value.lower()
        linked = []
        for document in documents:
            if int(document["id"]) == current_document_id:
                continue
            file_name = str(document["file_name"])
            stem = Path(file_name).stem
            candidates = {
                file_name.lower(),
                stem.lower(),
                stem.lower().replace("_", " "),
            }
            if any(candidate and candidate in normalized for candidate in candidates):
                linked.append(self._compact_document(document))
        return linked

    def _item_result_type(self, label: str, kind: str) -> str:
        if kind == "references":
            return "reference"
        if label == "section_header":
            return "section"
        if label == "table":
            return "table"
        if label == "picture":
            return "figure"
        return "item"

    def _result_title(self, fallback: str, section_path: list[str]) -> str:
        if section_path:
            return " / ".join(section_path)
        normalized = " ".join(str(fallback).split())
        return normalized[:140] if normalized else "Untitled"
