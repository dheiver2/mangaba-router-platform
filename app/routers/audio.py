"""
Endpoint de áudio — usa Whisper para transcrição e opcionalmente
passa o texto transcrito para o Gemma 4 responder.
"""
import asyncio
import io
import tempfile
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app import inference, model_loader
from app.inference_queue import enqueue
from config.settings import get_settings

router = APIRouter(prefix="/audio", tags=["Áudio"])

_whisper_model = None


def _load_whisper(size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def _transcribe(audio_bytes: bytes, language: Optional[str]) -> str:
    import whisper
    model = _load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        opts = {"language": language} if language else {}
        result = model.transcribe(tmp_path, **opts)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def _transcribe_and_chat(audio_bytes: bytes, language: Optional[str], system_prompt: str,
                          max_new_tokens: int, temperature: float) -> tuple[str, str, int]:
    transcription = _transcribe(audio_bytes, language)
    from app.schemas import Message, RoleEnum
    messages = [
        Message(role=RoleEnum.system, content=system_prompt),
        Message(role=RoleEnum.user, content=transcription),
    ]
    reply, tokens = inference.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return transcription, reply, tokens


class TranscribeResponse(BaseModel):
    transcription: str
    language: Optional[str]


class AudioChatResponse(BaseModel):
    transcription: str
    reply: str
    model: str
    tokens_generated: int


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcrever áudio para texto (Whisper)",
)
async def transcribe(
    file: UploadFile = File(..., description="Arquivo de áudio (MP3, WAV, M4A, OGG)"),
    language: Optional[str] = Form(None, description="Código do idioma, ex: 'pt', 'en'. Deixe vazio para detecção automática."),
):
    content = await file.read()
    try:
        transcription = await asyncio.get_running_loop().run_in_executor(
            None, _transcribe, content, language
        )
    except Exception as e:
        raise HTTPException(500, f"Erro na transcrição: {e}")
    return TranscribeResponse(transcription=transcription, language=language)


@router.post(
    "/chat",
    response_model=AudioChatResponse,
    summary="Enviar áudio e receber resposta do Gemma 4 (transcrição + chat)",
)
async def audio_chat(
    file: UploadFile = File(..., description="Arquivo de áudio"),
    language: Optional[str] = Form(None, description="Idioma do áudio (opcional)"),
    system_prompt: str = Form("Você é um assistente útil em português. Responda de forma clara e objetiva."),
    max_new_tokens: int = Form(256),
    temperature: float = Form(0.7),
):
    if not model_loader.is_loaded():
        raise HTTPException(503, "Modelo não carregado.")

    settings = get_settings()
    content = await file.read()

    try:
        transcription, reply, tokens = await asyncio.wait_for(
            enqueue(_transcribe_and_chat, content, language, system_prompt, max_new_tokens, temperature),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Tempo limite excedido.")
    except Exception as e:
        raise HTTPException(500, str(e))

    return AudioChatResponse(
        transcription=transcription,
        reply=reply,
        model=settings.model_id,
        tokens_generated=tokens,
    )
