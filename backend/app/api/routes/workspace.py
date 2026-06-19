from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from app.api.schemas import RootPathRequest, ScanRequest
from app.services.catalog_service import CatalogService


router = APIRouter(tags=["workspace"])


@router.get("/config")
def get_config() -> dict[str, str | None]:
    catalog = CatalogService()
    return catalog.get_config()


@router.put("/config/root")
def set_root_path(payload: RootPathRequest) -> dict[str, str]:
    catalog = CatalogService()
    try:
        return catalog.set_root_path(payload.root_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/scan")
def scan_documents(payload: ScanRequest | None = Body(default=None)) -> dict[str, int | str]:
    catalog = CatalogService()
    try:
        return catalog.scan_documents(root_path=None if payload is None else payload.root_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/tree")
def get_tree() -> dict[str, object]:
    catalog = CatalogService()
    return catalog.build_tree()

