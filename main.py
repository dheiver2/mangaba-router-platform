from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
from config.settings import get_settings
from app.routes import router
from app.routers.gguf_router import create_gguf_router
from app.auth import verify_api_key
from app import auth, gguf_backend
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Modelos Mangaba — GGUF quantizados Q4_0 (rodam 100% em 16GB via Metal)
GGUF_ROUTES = [
    ("e2b", "Mangaba E2B"),
    ("e4b", "Mangaba E4B"),
    ("12b", "Mangaba 12B"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup em background: pré-carrega o modelo padrão (e2b) sem bloquear o boot.
    # A 1ª requisição do usuário já encontra o modelo quente → baixa latência.
    threading.Thread(target=gguf_backend.warmup, daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Mangaba Router API",
        version=settings.api_version,
        lifespan=lifespan,
        description="""
# Mangaba Router

**Plataforma multimodal de IA** — modelos **Mangaba** (Gemma 4) quantizados **Q4_0**,
acelerados na GPU (Metal/Apple Silicon). Texto · Imagem · Áudio.

> 🔓 Acesso livre (sem senha) · 100% no HD externo · roda em 16 GB de RAM · GPU Metal.

---

## 🧠 Qual modelo escolher?

Todos **quantizados Q4_0** (cabem em 16 GB). A API carrega **um por vez** — ao chamar
outro prefixo, ela troca de modelo automaticamente.

| Modelo | Prefixo | Params | Quando usar |
|--------|---------|--------|-------------|
| **Mangaba E2B** 🟢 | `/api/v1/e2b/` | 2B  | **Padrão.** Rápido, chat simples, alto volume. Pré-carregado no boot. |
| **Mangaba E4B** | `/api/v1/e4b/` | 4B  | Mais qualidade mantendo boa velocidade. |
| **Mangaba 12B** | `/api/v1/12b/` | 12B | Tarefas complexas: raciocínio, imagem detalhada, textos longos. |

> ⏱️ **Troca de modelo:** a 1ª chamada a um modelo ainda não carregado lê o arquivo
> do HD (USB) — pode levar de ~30 s (E2B) a ~60 s (12B). Depois fica **quente**
> (respostas em frações de segundo). O E2B já vem aquecido no boot.

---

## 🔌 Qual rota usar?

Cada modelo expõe as mesmas 5 rotas. Escolha pela tarefa:

| Rota | Quando usar |
|------|-------------|
| `POST /{modelo}/text/chat` | **Conversas e instruções** — rota recomendada (chat template). |
| `POST /{modelo}/text/generate` | Completar texto cru, controle total do prompt. |
| `POST /{modelo}/image/describe` | **Visão:** descrever/analisar imagem, OCR. |
| `POST /{modelo}/audio/transcribe` | **Só transcrever** fala → texto (Whisper, não usa o LLM). |
| `POST /{modelo}/audio/chat` | **Áudio → resposta:** assistente por voz (transcreve + responde). |

---

## ▶️ Como testar aqui no Swagger
1. Abra qualquer rota (ex: **`/api/v1/e2b/text/chat`**)
2. Clique em **Try it out**
3. Edite o JSON e clique em **Execute**

Sem senha, sem cabeçalhos — acesso direto. 🔓

## 📋 Endpoints de sistema (públicos)
`GET /api/v1/health` · `GET /api/v1/metrics` · `GET /api/v1/models` · `GET /api/v1/users`
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

    # Esquema de API key no Swagger — só mostra o botão "Authorize" se a auth estiver ligada
    from fastapi.openapi.utils import get_openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title, version=app.version,
            description=app.description, routes=app.routes,
        )
        if settings.auth_enabled:
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
