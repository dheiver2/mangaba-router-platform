"""Baixa todos os modelos Gemma 4 instruct em paralelo."""
import os
import sys
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")

MODELS = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B-it",
]


def download(model_id: str):
    local_dir = f"models/{model_id.replace('/', '--')}"
    if os.path.exists(local_dir) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(local_dir) for f in files
    ):
        print(f"[SKIP] {model_id} — já existe em {local_dir}")
        return local_dir

    print(f"[START] Baixando {model_id} → {local_dir}")
    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    path = snapshot_download(repo_id=model_id, local_dir=local_dir, **kwargs)
    print(f"[DONE]  {model_id} → {path}")
    return path


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(download, m): m for m in MODELS}
        for f in concurrent.futures.as_completed(futures):
            m = futures[f]
            try:
                f.result()
            except Exception as e:
                print(f"[ERROR] {m}: {e}")
    print("\nTodos os downloads concluídos.")
