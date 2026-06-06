from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_id: str = "google/gemma-4-E2B-it"
    hf_token: str = ""
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    # Concorrência: quantas inferências podem estar "esperando" a GPU ao mesmo tempo
    # Requisições acima desse limite retornam 503 imediatamente (fail-fast)
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 120
    api_title: str = "Gemma 4 API"
    api_version: str = "1.0.0"
    api_description: str = "API REST para inferência com o modelo Google Gemma 4 E2B Instruct"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
