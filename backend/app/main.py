from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.evidence import router as evidence_router
from app.api.routes.health import router as health_router
from app.api.routes.workspace import router as workspace_router
from app.core.config import get_settings
from app.core.database import init_database
from app.core.observability import init_observability


settings = get_settings()
init_observability(settings)
app = FastAPI(title=settings.api_title, version=settings.api_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_database()


app.include_router(health_router)
app.include_router(workspace_router)
app.include_router(documents_router)
app.include_router(evidence_router)
