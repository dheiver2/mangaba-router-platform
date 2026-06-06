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

# Catálogo de modelos disponíveis (família Mangaba, baseada em Google Gemma 4)
CATALOG: dict[str, dict] = {
    "gemma-4-E2B-it": {
        "repo_id": "google/gemma-4-E2B-it",
        "local_dir": "models/google--gemma-4-E2B-it",
        "display_name": "Mangaba E2B 🥭",
        "description": "Mangaba E2B 🥭 — 2B params efetivos (MoE compacto, mais rápido)",
        "params": "2B",
    },
    "gemma-4-E4B-it": {
        "repo_id": "google/gemma-4-E4B-it",
        "local_dir": "models/google--gemma-4-E4B-it",
        "display_name": "Mangaba E4B 🥭",
        "description": "Mangaba E4B 🥭 — 4B params efetivos (equilíbrio)",
        "params": "4B",
    },
    "gemma-4-12B-it": {
        "repo_id": "google/gemma-4-12B-it",
        "local_dir": "models/google--gemma-4-12B-it",
        "display_name": "Mangaba 12B 🥭",
        "description": "Mangaba 12B 🥭 — multimodal, maior capacidade",
        "params": "12B",
    },
    "gemma-4-26B-A4B-it": {
        "repo_id": "google/gemma-4-26B-A4B-it",
        "local_dir": "models/google--gemma-4-26B-A4B-it",
        "display_name": "Mangaba 26B 🥭",
        "description": "Mangaba 26B 🥭 — MoE 26B (4B ativos), melhor custo-benefício",
        "params": "26B/4B-active",
    },
}

DEFAULT_MODEL = "gemma-4-E2B-it"

_lock = Lock()
_current_name: str | None = None
_tokenizer = None
_model = None


def _detect_device() -> str:
    """CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Cache no SSD do notebook — leitura MUITO mais rápida que o USB/ExFAT.
# Ative com a variável de ambiente STAGE_TO_SSD=1
_SSD_CACHE = os.path.expanduser("~/.cache/mangaba-router")


def _stage_to_ssd(local_dir: str, name: str) -> str:
    """
    Copia o modelo do HD externo para o SSD do notebook (uma vez) e retorna
    o caminho no SSD. Elimina o gargalo de leitura do USB nas próximas cargas.
    """
    import shutil
    dest = os.path.join(_SSD_CACHE, name)
    have = os.path.exists(dest) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(dest) for f in files
    )
    if have:
        logger.info(f"Usando cache SSD: {dest}")
        return dest
    os.makedirs(_SSD_CACHE, exist_ok=True)
    logger.info(f"Copiando '{name}' do HD externo para o SSD (uma vez)...")
    tmp = dest + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(local_dir, tmp)
    os.replace(tmp, dest)
    logger.info(f"Modelo em cache no SSD: {dest}")
    return dest


def _resolve_path(name: str) -> str:
    entry = CATALOG[name]
    local = entry["local_dir"]
    has_local = os.path.exists(local) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(local) for f in files
    )
    if has_local:
        if os.getenv("STAGE_TO_SSD") == "1":
            try:
                return _stage_to_ssd(local, name)
            except Exception as e:
                logger.warning(f"Falha ao copiar para SSD ({e}); usando HD externo.")
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
        device = _detect_device()
        logger.info(f"Carregando modelo '{name}' de: {path} (device={device})")

        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        # float16: metade da RAM + compute acelerado na GPU (CUDA ou MPS/Metal).
        # Em Apple Silicon, MPS usa a GPU integrada -> muito mais rápido que CPU.
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        _model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            low_cpu_mem_usage=True,   # carrega shard a shard, sem dobrar a RAM
            **kwargs,
        )
        _model = _model.to(device)
        _model.eval()
        _current_name = name
        logger.info(f"Modelo '{name}' carregado (device={device}, dtype={dtype}).")


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
            "display_name": info.get("display_name", name),
            "repo_id": info["repo_id"],
            "description": info["description"],
            "params": info["params"],
            "available_locally": _is_available(name),
            "loaded": name == _current_name,
        })
    return result
