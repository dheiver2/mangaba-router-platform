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
    request_timeout_seconds: int = 300
    # Multiusuário: exige cabeçalho X-API-Key (chaves em config/users.json)
    # False = acesso livre pelo Swagger, sem senha (defina AUTH_ENABLED=true p/ reativar)
    auth_enabled: bool = False
    # Rede: 0.0.0.0 expõe na LAN para múltiplas plataformas (web, mobile, desktop)
    host: str = "0.0.0.0"
    port: int = 8000
    api_title: str = "Mangaba Router API"
    api_version: str = "1.0.0"
    api_description: str = "Plataforma multimodal de IA — modelos Mangaba (Gemma 4)"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
