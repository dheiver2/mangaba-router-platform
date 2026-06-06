"""
Endpoint de imagem — usa o Gemma 4 multimodal para descrever/analisar imagens.
Aceita upload de arquivo ou URL pública.
"""
import asyncio
import io
import base64
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image

from app import model_loader
from app.inference_queue import enqueue
from config.settings import get_settings

router = APIRouter(prefix="/image", tags=["Imagem"])


class ImageResponse(BaseModel):
    description: str
    model: str
    tokens_generated: int


def _check():
    if not model_loader.is_loaded():
        raise HTTPException(503, "Modelo não carregado.")


def _run_image_inference(image: Image.Image, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, int]:
    import torch
    tokenizer = model_loader.get_tokenizer()
    model = model_loader.get_model()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=text_input, images=[image], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip(), len(generated)


@router.post(
    "/describe",
    response_model=ImageResponse,
    summary="Descrever ou analisar uma imagem (upload de arquivo)",
)
async def describe_image(
    file: UploadFile = File(..., description="Arquivo de imagem (JPG, PNG, WEBP)"),
    prompt: str = Form("Descreva detalhadamente o que você vê nesta imagem.", description="Instrução para o modelo"),
    max_new_tokens: int = Form(256),
    temperature: float = Form(0.3),
):
    _check()
    settings = get_settings()

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

    return ImageResponse(description=text, model=settings.model_id, tokens_generated=tokens)


@router.post(
    "/describe-base64",
    response_model=ImageResponse,
    summary="Descrever imagem enviada em base64 (JSON)",
)
async def describe_base64(body: dict):
    _check()
    settings = get_settings()

    b64 = body.get("image_base64", "")
    prompt = body.get("prompt", "Descreva detalhadamente o que você vê nesta imagem.")
    max_new_tokens = int(body.get("max_new_tokens", 256))
    temperature = float(body.get("temperature", 0.3))

    try:
        image_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Base64 inválido ou não é uma imagem.")

    try:
        text, tokens = await asyncio.wait_for(
            enqueue(_run_image_inference, image, prompt, max_new_tokens, temperature),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Tempo limite excedido.")
    except Exception as e:
        raise HTTPException(500, str(e))

    return ImageResponse(description=text, model=settings.model_id, tokens_generated=tokens)
