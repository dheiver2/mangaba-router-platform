"""Endpoints de gerenciamento de modelos."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from app import model_registry

router = APIRouter(prefix="/models", tags=["Modelos"])


class ModelEntry(BaseModel):
    name: str
    repo_id: str
    description: str
    params: str
    available_locally: bool
    loaded: bool


class LoadRequest(BaseModel):
    name: str


@router.get("", response_model=list[ModelEntry], summary="Listar todos os modelos Gemma 4 disponíveis")
def list_models():
    return model_registry.list_models()


@router.post("/load", summary="Carregar um modelo específico em memória")
def load_model(req: LoadRequest, background_tasks: BackgroundTasks):
    if req.name not in model_registry.CATALOG:
        raise HTTPException(404, f"Modelo '{req.name}' não encontrado. Use GET /models para ver a lista.")
    if model_registry.current_name() == req.name and model_registry.is_loaded():
        return {"message": f"Modelo '{req.name}' já está carregado."}
    background_tasks.add_task(model_registry.load, req.name)
    return {"message": f"Carregamento de '{req.name}' iniciado em background."}


@router.get("/current", summary="Modelo atualmente carregado")
def current_model():
    return {
        "name": model_registry.current_name(),
        "loaded": model_registry.is_loaded(),
        "device": model_registry.get_device(),
    }
