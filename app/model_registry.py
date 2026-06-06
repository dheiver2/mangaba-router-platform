"""
Registro de múltiplos modelos Gemma 4.
Carrega um modelo por vez (lazy) e mantém cache do último usado.
"""
import os
import torch
import logging
from threading import Lock
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Catálogo de modelos disponíveis
CATALOG: dict[str, dict] = {
    "gemma-4-E2B-it": {
        "repo_id": "google/gemma-4-E2B-it",
        "local_dir": "models/google--gemma-4-E2B-it",
        "description": "Gemma 4 E2B Instruct — 2B params efetivos (MoE compacto)",
        "params": "2B",
    },
    "gemma-4-E4B-it": {
        "repo_id": "google/gemma-4-E4B-it",
        "local_dir": "models/google--gemma-4-E4B-it",
        "description": "Gemma 4 E4B Instruct — 4B params efetivos",
        "params": "4B",
    },
    "gemma-4-12B-it": {
        "repo_id": "google/gemma-4-12B-it",
        "local_dir": "models/google--gemma-4-12B-it",
        "description": "Gemma 4 12B Instruct — multimodal, maior capacidade",
        "params": "12B",
    },
    "gemma-4-26B-A4B-it": {
        "repo_id": "google/gemma-4-26B-A4B-it",
        "local_dir": "models/google--gemma-4-26B-A4B-it",
        "description": "Gemma 4 26B MoE (4B ativos) Instruct — melhor custo-benefício",
        "params": "26B/4B-active",
    },
}

DEFAULT_MODEL = "gemma-4-E2B-it"

_lock = Lock()
_current_name: str | None = None
_tokenizer = None
_model = None


def _resolve_path(name: str) -> str:
    entry = CATALOG[name]
    local = entry["local_dir"]
    if os.path.exists(local) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(local) for f in files
    ):
        return local
    return entry["repo_id"]


def _is_available(name: str) -> bool:
    if name not in CATALOG:
        return False
    local = CATALOG[name]["local_dir"]
    return os.path.exists(local) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(local) for f in files
    )


def load(name: str = DEFAULT_MODEL):
    global _current_name, _tokenizer, _model

    with _lock:
        if _current_name == name and _model is not None:
            return

        if name not in CATALOG:
            raise ValueError(f"Modelo '{name}' não encontrado no catálogo.")

        if _model is not None:
            logger.info(f"Descarregando modelo anterior: {_current_name}")
            del _model
            del _tokenizer
            _model = None
            _tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        path = _resolve_path(name)
        logger.info(f"Carregando modelo '{name}' de: {path}")

        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        _tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        _model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map=device_map, **kwargs
        )
        _model.eval()
        _current_name = name
        logger.info(f"Modelo '{name}' carregado com sucesso.")


def get_tokenizer():
    if _tokenizer is None:
        load()
    return _tokenizer


def get_model():
    if _model is None:
        load()
    return _model


def current_name() -> str | None:
    return _current_name


def is_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    if _model is None:
        return "não carregado"
    try:
        return str(next(_model.parameters()).device)
    except Exception:
        return "desconhecido"


def list_models() -> list[dict]:
    result = []
    for name, info in CATALOG.items():
        result.append({
            "name": name,
            "repo_id": info["repo_id"],
            "description": info["description"],
            "params": info["params"],
            "available_locally": _is_available(name),
            "loaded": name == _current_name,
        })
    return result
