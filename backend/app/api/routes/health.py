from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.services.catalog_service import CatalogService


router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str | None]:
    settings = get_settings()
    catalog = CatalogService()
    return {
        "status": "ok",
        "version": settings.api_version,
        "root_path": catalog.get_root_path(),
        "db_path": str(settings.db_path),
    }

