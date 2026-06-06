"""
Backend GGUF (llama.cpp) — modelos quantizados Q4_0 que rodam em 16GB.
Acelerado por Metal (GPU Apple Silicon) via n_gpu_layers=-1.

Suporta:
  - texto (chat / generate)
  - visão (imagem) quando há mmproj (projetor multimodal)

Carrega um modelo por vez, igual ao registry float16.
"""
import os
import base64
import logging
from threading import Lock

logger = logging.getLogger(__name__)

_GGUF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_gguf")
# Cache no SSD do notebook — evita contenção/lentidão do USB ao carregar.
_SSD_CACHE = os.path.expanduser("~/.cache/mangaba-router/gguf")
_STAGE = os.getenv("STAGE_TO_SSD", "0") == "1"


def _find_files(slug: str):
    """Localiza (principal, mmproj) do GGUF. Prefere o cache no SSD se existir."""
    for base in ([os.path.join(_SSD_CACHE, slug)] if _STAGE else []) + [os.path.join(_GGUF_DIR, slug)]:
        if not os.path.isdir(base):
            continue
        main, proj = None, None
        for root, _, files in os.walk(base):
            for f in files:
                if not f.endswith(".gguf"):
                    continue
                p = os.path.join(root, f)
                if "mmproj" in f.lower():
                    proj = p
                else:
                    main = p
        if main:
            return main, proj
    return None, None


def _stage(slug: str):
    """Copia os .gguf do USB para o SSD (uma vez) se ainda não estiverem lá."""
    if not _STAGE:
        return
    import shutil, glob
    src = os.path.join(_GGUF_DIR, slug)
    dst = os.path.join(_SSD_CACHE, slug)
    if not os.path.isdir(src):
        return
    src_gguf = glob.glob(os.path.join(src, "**", "*.gguf"), recursive=True)
    have = glob.glob(os.path.join(dst, "*.gguf"))
    if have and len(have) >= len(src_gguf):
        return
    os.makedirs(dst, exist_ok=True)
    for f in src_gguf:
        d = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(d):
            logger.info(f"Copiando {os.path.basename(f)} p/ SSD...")
            shutil.copy2(f, d)


CATALOG = {
    "e2b": {"display_name": "Mangaba E2B (Q4_0)", "params": "2B",  "ctx": 8192},
    "e4b": {"display_name": "Mangaba E4B (Q4_0)", "params": "4B",  "ctx": 8192},
    "12b": {"display_name": "Mangaba 12B (Q4_0)", "params": "12B", "ctx": 4096},
}

_lock = Lock()
_current: str | None = None
_vision_mode = False          # se o modelo atual está carregado com handler de visão
_llm = None
_chat_handler = None


def is_available(slug: str) -> bool:
    main, _ = _find_files(slug)
    return main is not None


def has_vision(slug: str) -> bool:
    _, proj = _find_files(slug)
    return proj is not None


def load(slug: str, vision: bool = False):
    """
    Carrega o GGUF. vision=False → texto puro (rápido).
    vision=True → anexa o mmproj (projetor) para análise de imagem.
    O handler de visão atrapalha o texto, então carregamos sob demanda.
    """
    global _current, _vision_mode, _llm, _chat_handler
    with _lock:
        if _current == slug and _llm is not None and _vision_mode == vision:
            return
        _stage(slug)  # copia p/ SSD na 1ª vez (evita lentidão do USB)
        main, proj = _find_files(slug)
        if main is None:
            raise FileNotFoundError(f"GGUF '{slug}' não baixado em {_GGUF_DIR}/{slug}")

        _llm = None
        _chat_handler = None

        from llama_cpp import Llama
        ctx = CATALOG.get(slug, {}).get("ctx", 8192)
        kwargs = dict(model_path=main, n_ctx=ctx, n_gpu_layers=-1, verbose=False)

        if vision and proj:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            _chat_handler = Llava15ChatHandler(clip_model_path=proj, verbose=False)
            kwargs["chat_handler"] = _chat_handler

        mode = "visão" if (vision and proj) else "texto"
        logger.info(f"Carregando GGUF '{slug}' ({mode}, Metal) de {os.path.basename(main)}")
        _llm = Llama(**kwargs)
        _current = slug
        _vision_mode = bool(vision and proj)
        logger.info(f"GGUF '{slug}' carregado ({mode}).")


def _ensure(slug: str, vision: bool = False):
    if _current != slug or _llm is None or _vision_mode != vision:
        load(slug, vision=vision)


def chat(slug: str, messages: list[dict], max_new_tokens: int = 256,
         temperature: float = 0.7, top_p: float = 0.9) -> tuple[str, int]:
    _ensure(slug)
    out = _llm.create_chat_completion(
        messages=messages, max_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    text = out["choices"][0]["message"]["content"].strip()
    toks = out.get("usage", {}).get("completion_tokens", 0)
    return text, toks


def generate(slug: str, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.7, top_p: float = 0.9) -> tuple[str, int]:
    _ensure(slug)
    out = _llm.create_completion(
        prompt=prompt, max_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    text = out["choices"][0]["text"].strip()
    toks = out.get("usage", {}).get("completion_tokens", 0)
    return text, toks


def describe_image(slug: str, image_bytes: bytes, prompt: str,
                   max_new_tokens: int = 256, temperature: float = 0.3) -> tuple[str, int]:
    _ensure(slug, vision=True)
    if _chat_handler is None:
        raise RuntimeError(f"Modelo GGUF '{slug}' sem suporte a visão (sem mmproj).")
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": prompt},
    ]}]
    out = _llm.create_chat_completion(
        messages=messages, max_tokens=max_new_tokens, temperature=temperature,
    )
    text = out["choices"][0]["message"]["content"].strip()
    toks = out.get("usage", {}).get("completion_tokens", 0)
    return text, toks


def current_name() -> str | None:
    return _current


def is_loaded() -> bool:
    return _llm is not None


def list_models() -> list[dict]:
    res = []
    for slug, info in CATALOG.items():
        res.append({
            "slug": slug,
            "display_name": info["display_name"],
            "params": info["params"],
            "quantization": "Q4_0",
            "available_locally": is_available(slug),
            "loaded": slug == _current,
        })
    return res
