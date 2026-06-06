"""Rotas de sistema (públicas): health e metrics."""
from fastapi import APIRouter
from pydantic import BaseModel
from app.inference_queue import get_metrics
from app import gguf_backend
from config.settings import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    loaded_model: str | None
    version: str


@router.get("/health", response_model=HealthResponse, summary="Saúde da API", tags=["Sistema"])
def health():
    return HealthResponse(
        status="ok",
        loaded_model=gguf_backend.current_name(),
        version=get_settings().api_version,
    )


@router.get("/metrics", summary="Métricas de concorrência", tags=["Sistema"])
def metrics():
    return get_metrics()
