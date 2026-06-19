from __future__ import annotations

from pydantic import BaseModel


class RootPathRequest(BaseModel):
    root_path: str


class ScanRequest(BaseModel):
    root_path: str | None = None


class ProcessDocumentsRequest(BaseModel):
    document_ids: list[int] | None = None
    only_stale: bool = False


class EvidenceRunRequest(BaseModel):
    query: str
    document_ids: list[int] | None = None
    max_tasks: int = 32
