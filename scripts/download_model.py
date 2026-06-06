"""Script para pré-baixar o modelo do HuggingFace Hub."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from huggingface_hub import snapshot_download
from config.settings import get_settings


def main():
    settings = get_settings()
    print(f"Baixando modelo: {settings.model_id}")

    kwargs = {"token": settings.hf_token} if settings.hf_token else {}

    path = snapshot_download(
        repo_id=settings.model_id,
        local_dir=f"models/{settings.model_id.replace('/', '--')}",
        **kwargs,
    )
    print(f"Modelo salvo em: {path}")


if __name__ == "__main__":
    main()
