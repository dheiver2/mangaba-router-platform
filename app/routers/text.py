import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
    Message, RoleEnum,
)
from app import inference, model_registry
from app.inference_queue import enqueue
from config.settings import get_settings

router = APIRouter(prefix="/text", tags=["Texto"])


def _check_and_switch(model_name: str | None):
    name = model_name or model_registry.DEFAULT_MODEL
    if not model_registry.is_loaded() or model_registry.current_name() != name:
        model_registry.load(name)
    if not model_registry.is_loaded():
        raise HTTPException(503, "Modelo não carregado.")


@router.post("/generate", response_model=GenerateResponse, summary="Gerar texto a partir de um prompt")
async def generate(req: GenerateRequest):
    _check_and_switch(req.model_name)
    settings = get_settings()
    try:
        text, tokens = await asyncio.wait_for(
            enqueue(inference.generate_text, req.prompt, req.max_new_tokens, req.temperature, req.top_p, req.top_k),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Tempo limite excedido.")
    except Exception as e:
        raise HTTPException(500, str(e))
    return GenerateResponse(
        generated_text=text,
        model=model_registry.current_name(),
        tokens_generated=tokens,
    )


@router.post("/chat", response_model=ChatResponse, summary="Chat com histórico de mensagens")
async def chat(req: ChatRequest):
    _check_and_switch(req.model_name)
    settings = get_settings()
    try:
        text, tokens = await asyncio.wait_for(
            enqueue(inference.chat, req.messages, req.max_new_tokens, req.temperature, req.top_p, req.top_k),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Tempo limite excedido.")
    except Exception as e:
        raise HTTPException(500, str(e))
    return ChatResponse(
        message=Message(role=RoleEnum.assistant, content=text),
        model=model_registry.current_name(),
        tokens_generated=tokens,
    )
