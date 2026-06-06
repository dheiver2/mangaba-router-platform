import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from app.schemas import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
    ModelInfoResponse, HealthResponse, Message, RoleEnum, MetricsResponse,
)
from app import inference, model_loader
from app.inference_queue import enqueue, get_metrics
from config.settings import get_settings

router = APIRouter()


def _check_loaded():
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Chame POST /model/load primeiro.",
        )


@router.get("/health", response_model=HealthResponse, summary="Verificar saúde da API", tags=["Sistema"])
def health_check():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded(),
        version=settings.api_version,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Métricas de concorrência", tags=["Sistema"])
def metrics():
    return MetricsResponse(**get_metrics())


@router.get("/model/info", response_model=ModelInfoResponse, summary="Informações do modelo", tags=["Modelo"])
def model_info():
    settings = get_settings()
    return ModelInfoResponse(
        model_id=settings.model_id,
        device=model_loader.get_device(),
        loaded=model_loader.is_loaded(),
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        max_concurrent_requests=settings.max_concurrent_requests,
    )


@router.post("/model/load", summary="Carregar modelo em memória", tags=["Modelo"])
def load_model(background_tasks: BackgroundTasks):
    if model_loader.is_loaded():
        return {"message": "Modelo já está carregado."}
    background_tasks.add_task(model_loader.load_model)
    return {"message": "Carregamento do modelo iniciado em background."}


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Gerar texto a partir de um prompt",
    tags=["Inferência"],
)
async def generate(req: GenerateRequest):
    _check_loaded()
    settings = get_settings()
    try:
        text, tokens = await asyncio.wait_for(
            enqueue(
                inference.generate_text,
                req.prompt,
                req.max_new_tokens,
                req.temperature,
                req.top_p,
                req.top_k,
            ),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Tempo limite excedido. Tente novamente.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        generated_text=text,
        model=settings.model_id,
        tokens_generated=tokens,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat com o modelo no formato de mensagens",
    tags=["Inferência"],
)
async def chat(req: ChatRequest):
    _check_loaded()
    settings = get_settings()
    try:
        text, tokens = await asyncio.wait_for(
            enqueue(
                inference.chat,
                req.messages,
                req.max_new_tokens,
                req.temperature,
                req.top_p,
                req.top_k,
            ),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Tempo limite excedido. Tente novamente.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        message=Message(role=RoleEnum.assistant, content=text),
        model=settings.model_id,
        tokens_generated=tokens,
    )
