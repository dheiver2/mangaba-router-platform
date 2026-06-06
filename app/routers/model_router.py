"""
Factory de routers por modelo.
Cria endpoints dedicados para cada modelo Gemma 4:
  /api/v1/{slug}/text/generate
  /api/v1/{slug}/text/chat
  /api/v1/{slug}/image/describe
  /api/v1/{slug}/audio/transcribe
  /api/v1/{slug}/audio/chat
"""
import asyncio
import io
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

from app import inference, model_registry
from app.inference_queue import enqueue
from app.schemas import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
    Message, RoleEnum,
)
from config.settings import get_settings


# ── Schemas específicos ──────────────────────────────────────────────────────

class ImageResponse(BaseModel):
    description: str
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


# ── Funções de inferência ────────────────────────────────────────────────────

def _run_image_inference(image, prompt: str, max_new_tokens: int, temperature: float):
    import torch
    processor = model_registry.get_processor()
    model = model_registry.get_model()
    if processor is None:
        raise RuntimeError("Modelo atual não suporta imagem (sem AutoProcessor).")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    # alinha o dtype dos pixels ao do modelo (float16 na GPU)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated = output[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip(), len(generated)


_whisper_model = None

def _ensure_ffmpeg():
    """
    Disponibiliza o ffmpeg empacotado (imageio-ffmpeg) com o nome 'ffmpeg' num
    diretório do SSD do notebook (ExFAT não suporta symlink/exec confiável),
    e o adiciona ao PATH para o Whisper encontrá-lo.
    """
    import shutil
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = os.path.expanduser("~/.cache/mangaba-router/bin")
        os.makedirs(bin_dir, exist_ok=True)
        dest = os.path.join(bin_dir, "ffmpeg")
        if not os.path.exists(dest):
            shutil.copy2(exe, dest)
            os.chmod(dest, 0o755)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        _ensure_ffmpeg()
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def _transcribe(audio_bytes: bytes, language: Optional[str]) -> str:
    model = _load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        opts = {"language": language} if language else {}
        return model.transcribe(tmp, **opts)["text"].strip()
    finally:
        os.unlink(tmp)

def _transcribe_and_chat(audio_bytes, language, system_prompt, max_new_tokens, temperature):
    transcription = _transcribe(audio_bytes, language)
    messages = [
        Message(role=RoleEnum.system, content=system_prompt),
        Message(role=RoleEnum.user, content=transcription),
    ]
    reply, tokens = inference.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return transcription, reply, tokens


# ── Factory ──────────────────────────────────────────────────────────────────

def create_model_router(model_name: str, slug: str, label: str) -> APIRouter:
    """Cria um APIRouter completo para um modelo específico."""

    router = APIRouter(prefix=f"/{slug}", tags=[f"[{label}]"])
    settings = get_settings()

    def _ensure():
        if model_registry.current_name() != model_name:
            model_registry.load(model_name)
        if not model_registry.is_loaded():
            raise HTTPException(503, f"Modelo {model_name} não carregado.")

    # ── Texto ────────────────────────────────────────────────────────────────

    @router.post("/text/generate", response_model=GenerateResponse,
                 summary=f"[{label}] Gerar texto")
    async def generate(req: GenerateRequest):
        _ensure()
        try:
            text, tokens = await asyncio.wait_for(
                enqueue(inference.generate_text, req.prompt, req.max_new_tokens,
                        req.temperature, req.top_p, req.top_k),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return GenerateResponse(generated_text=text, model=model_name, tokens_generated=tokens)

    @router.post("/text/chat", response_model=ChatResponse,
                 summary=f"[{label}] Chat com histórico")
    async def chat(req: ChatRequest):
        _ensure()
        try:
            text, tokens = await asyncio.wait_for(
                enqueue(inference.chat, req.messages, req.max_new_tokens,
                        req.temperature, req.top_p, req.top_k),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return ChatResponse(
            message=Message(role=RoleEnum.assistant, content=text),
            model=model_name, tokens_generated=tokens,
        )

    # ── Imagem ───────────────────────────────────────────────────────────────

    @router.post("/image/describe", response_model=ImageResponse,
                 summary=f"[{label}] Descrever imagem")
    async def describe_image(
        file: UploadFile = File(...),
        prompt: str = Form("Descreva detalhadamente o que você vê nesta imagem."),
        max_new_tokens: int = Form(256),
        temperature: float = Form(0.3),
    ):
        _ensure()
        from PIL import Image
        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Arquivo de imagem inválido.")
        try:
            text, tokens = await asyncio.wait_for(
                enqueue(_run_image_inference, image, prompt, max_new_tokens, temperature),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return ImageResponse(description=text, model=model_name, tokens_generated=tokens)

    # ── Áudio ────────────────────────────────────────────────────────────────

    @router.post("/audio/transcribe", response_model=TranscribeResponse,
                 summary=f"[{label}] Transcrever áudio (Whisper)")
    async def transcribe(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
    ):
        content = await file.read()
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, _transcribe, content, language
            )
        except Exception as e:
            raise HTTPException(500, f"Erro na transcrição: {e}")
        return TranscribeResponse(transcription=text, language=language)

    @router.post("/audio/chat", response_model=AudioChatResponse,
                 summary=f"[{label}] Áudio → transcrição + resposta")
    async def audio_chat(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        system_prompt: str = Form("Você é um assistente útil em português."),
        max_new_tokens: int = Form(256),
        temperature: float = Form(0.7),
    ):
        _ensure()
        content = await file.read()
        try:
            transcription, reply, tokens = await asyncio.wait_for(
                enqueue(_transcribe_and_chat, content, language,
                        system_prompt, max_new_tokens, temperature),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Tempo limite excedido.")
        except Exception as e:
            raise HTTPException(500, str(e))
        return AudioChatResponse(
            transcription=transcription, reply=reply,
            model=model_name, tokens_generated=tokens,
        )

    return router
