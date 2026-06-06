"""
Rotas para os modelos GGUF quantizados (Q4_0) — rodam em 16GB via Metal.
Prefixo: /api/v1/{slug}/...   (e2b, e4b, 12b, 26b)
"""
import asyncio, io, os, tempfile
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app import gguf_backend
from app.inference_queue import enqueue
from app.schemas import GenerateRequest, ChatRequest, Message, RoleEnum
from config.settings import get_settings


class QResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int


class TranscribeResponse(BaseModel):
    transcription: str
    language: Optional[str] = None


class AudioChatResponse(BaseModel):
    transcription: str
    reply: str
    model: str
    tokens_generated: int


# ── Whisper (transcrição) ────────────────────────────────────────────────────
_whisper = None

def _ensure_ffmpeg():
    import shutil
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = os.path.expanduser("~/.cache/mangaba-router/bin")
        os.makedirs(bin_dir, exist_ok=True)
        dest = os.path.join(bin_dir, "ffmpeg")
        if not os.path.exists(dest):
            shutil.copy2(exe, dest); os.chmod(dest, 0o755)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

def _load_whisper():
    global _whisper
    if _whisper is None:
        _ensure_ffmpeg()
        import whisper
        _whisper = whisper.load_model("base")
    return _whisper

def _transcribe(audio_bytes: bytes, language: Optional[str]) -> str:
    m = _load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes); tmp = f.name
    try:
        opts = {"language": language} if language else {}
        return m.transcribe(tmp, **opts)["text"].strip()
    finally:
        os.unlink(tmp)


def create_gguf_router(slug: str, label: str) -> APIRouter:
    router = APIRouter(prefix=f"/{slug}", tags=[f"[{label} Q4_0]"])
    settings = get_settings()

    def _ensure():
        if not gguf_backend.is_available(slug):
            raise HTTPException(503, f"Modelo '{slug}' não baixado. Rode scripts/download_gguf.py {slug}")

    @router.post("/text/chat", response_model=QResponse,
                 summary=f"[{label}] Chat",
                 description=f"Conversa com **Mangaba {label}** (Q4_0, GPU Metal). Rota recomendada.")
    async def chat(req: ChatRequest):
        _ensure()
        msgs = [{"role": m.role.value, "content": m.content} for m in req.messages]
        try:
            text, toks = await asyncio.wait_for(
                enqueue(gguf_backend.chat, slug, msgs,
                        req.max_new_tokens or 256, req.temperature or 0.7, req.top_p or 0.9),
                timeout=settings.request_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return QResponse(text=text, model=f"{slug}-q4_0", tokens_generated=toks)

    @router.post("/text/generate", response_model=QResponse,
                 summary=f"[Q4_0 {label}] Gerar texto (quantizado)")
    async def generate(req: GenerateRequest):
        _ensure()
        try:
            text, toks = await asyncio.wait_for(
                enqueue(gguf_backend.generate, slug, req.prompt,
                        req.max_new_tokens or 256, req.temperature or 0.7, req.top_p or 0.9),
                timeout=settings.request_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return QResponse(text=text, model=f"{slug}-q4_0", tokens_generated=toks)

    @router.post("/image/describe", response_model=QResponse,
                 summary=f"[{label}] Descrever imagem",
                 description=f"Visão multimodal com **Mangaba {label}** (Q4_0 + mmproj).")
    async def describe(file: UploadFile = File(...),
                       prompt: str = Form("Descreva esta imagem."),
                       max_new_tokens: int = Form(256)):
        _ensure()
        content = await file.read()
        try:
            text, toks = await asyncio.wait_for(
                enqueue(gguf_backend.describe_image, slug, content, prompt, max_new_tokens, 0.3),
                timeout=settings.request_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return QResponse(text=text, model=f"{slug}-q4_0", tokens_generated=toks)

    @router.post("/audio/transcribe", response_model=TranscribeResponse,
                 summary=f"[{label}] Transcrever áudio (Whisper)",
                 description="Fala → texto via Whisper. Não usa o LLM.")
    async def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None)):
        content = await file.read()
        try:
            text = await asyncio.get_running_loop().run_in_executor(None, _transcribe, content, language)
        except Exception as e:
            raise HTTPException(500, f"Erro na transcrição: {e}")
        return TranscribeResponse(transcription=text, language=language)

    @router.post("/audio/chat", response_model=AudioChatResponse,
                 summary=f"[{label}] Áudio → resposta",
                 description=f"Transcreve a fala e **Mangaba {label}** responde (assistente por voz).")
    async def audio_chat(file: UploadFile = File(...), language: Optional[str] = Form(None),
                         system_prompt: str = Form("Você é um assistente útil em português."),
                         max_new_tokens: int = Form(256)):
        _ensure()
        content = await file.read()
        def _run():
            transcription = _transcribe(content, language)
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}]
            reply, toks = gguf_backend.chat(slug, msgs, max_new_tokens)
            return transcription, reply, toks
        try:
            transcription, reply, toks = await asyncio.wait_for(
                enqueue(_run), timeout=settings.request_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return AudioChatResponse(transcription=transcription, reply=reply,
                                 model=f"{slug}-q4_0", tokens_generated=toks)

    return router
