from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from config.settings import get_settings
from app.routes import router
from app.routers.gguf_router import create_gguf_router
from app.auth import verify_api_key
from app import auth, gguf_backend
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Modelos Mangaba — GGUF quantizados Q4_0 (rodam em 16GB via Metal)
GGUF_ROUTES = [
    ("e2b", "Mangaba E2B"),
    ("e4b", "Mangaba E4B"),
    ("12b", "Mangaba 12B"),
    ("26b", "Mangaba 26B"),
]


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Mangaba Router API",
        version=settings.api_version,
        description="""
# Mangaba Router

**Plataforma multimodal de IA** — modelos **Mangaba** (Gemma 4) quantizados **Q4_0**,
acelerados na GPU (Metal/Apple Silicon). Texto, imagem e áudio.

> Multiusuário · Multiplataforma · 100% no HD externo · roda em 16GB de RAM.

---

## 🧠 Qual modelo escolher?

Todos são **quantizados Q4_0** (cabem em 16GB). Carrega **um por vez** (troca automática).

| Modelo | Prefixo | Params | Quando usar |
|--------|---------|--------|-------------|
| **Mangaba E2B** | `/api/v1/e2b/` | 2B  | **Padrão.** Rápido, chat simples, alto volume. |
| **Mangaba E4B** | `/api/v1/e4b/` | 4B  | Mais qualidade mantendo boa velocidade. |
| **Mangaba 12B** | `/api/v1/12b/` | 12B | Tarefas complexas: raciocínio, imagem detalhada, textos longos. |
| **Mangaba 26B** | `/api/v1/26b/` | 26B MoE | Máxima qualidade. |

---

## 🔌 Qual rota usar?

| Rota | Quando usar |
|------|-------------|
| `POST /{modelo}/text/chat` | **Conversas e instruções** (rota recomendada). |
| `POST /{modelo}/text/generate` | Completar texto cru, controle total do prompt. |
| `POST /{modelo}/image/describe` | **Visão:** descrever/analisar imagem, OCR. |
| `POST /{modelo}/audio/transcribe` | **Só transcrever** fala → texto (Whisper). |
| `POST /{modelo}/audio/chat` | **Áudio → resposta:** assistente por voz. |

---

## 🔑 Autenticação
Envie o cabeçalho **`X-API-Key`** nas chamadas de inferência (botão **Authorize** 🔒).
`health`, `metrics`, `users` e `q/models` são públicos.
        """,
        contact={"name": "Mangaba AI", "url": "https://github.com/dheiver2/Mangaba-Router"},
        license_info={"name": "Apache 2.0"},
        docs_url="/swagger",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    auth_dep = [Depends(verify_api_key)]

    # Sistema (health/metrics) — público
    app.include_router(router, prefix="/api/v1")

    # Rotas por modelo GGUF quantizado — protegidas por API key
    for slug, label in GGUF_ROUTES:
        app.include_router(create_gguf_router(slug, label), prefix="/api/v1", dependencies=auth_dep)

    @app.get("/api/v1/models", tags=["Modelos"], summary="Listar modelos Mangaba (Q4_0)")
    def list_models():
        return gguf_backend.list_models()

    @app.get("/api/v1/users", tags=["Usuários"], summary="Listar usuários cadastrados")
    def list_users():
        return {"users": auth.list_users(), "metrics": auth.user_metrics()}

    @app.post("/api/v1/users/reload", tags=["Usuários"], summary="Recarregar users.json")
    def reload_users():
        n = auth.reload_users()
        return {"message": f"{n} usuários carregados."}

    # Esquema de API key no Swagger (botão "Authorize")
    from fastapi.openapi.utils import get_openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title, version=app.version,
            description=app.description, routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
            "type": "apiKey", "in": "header", "name": "X-API-Key",
        }
        schema["security"] = [{"ApiKeyAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    _settings = get_settings()
    uvicorn.run(
        "main:app",
        host=_settings.host,
        port=_settings.port,
        workers=1,
        loop="uvloop",
        http="httptools",
    )
