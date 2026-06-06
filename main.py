from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.settings import get_settings
from app.routes import router
from app.routers import text, image, audio, models
from app.routers.model_router import create_model_router
from app.auth import verify_api_key
from app import model_loader, auth
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Definição dos modelos e seus slugs de URL
MODEL_ROUTES = [
    ("gemma-4-E2B-it",    "e2b",  "Mangaba E2B 🥭"),
    ("gemma-4-E4B-it",    "e4b",  "Mangaba E4B 🥭"),
    ("gemma-4-12B-it",    "12b",  "Mangaba 12B 🥭"),
    ("gemma-4-26B-A4B-it","26b",  "Mangaba 26B 🥭"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="🥭 Mangaba Router API",
        version=settings.api_version,
        description="""
# 🥭 Mangaba Router

**Plataforma multimodal de IA** — roteamento inteligente entre múltiplos modelos
**Mangaba** (baseados em Google Gemma 4), com texto, imagem e áudio.

> Multiusuário · Multiplataforma · GPU acelerada · 100% autocontido no HD externo.

---

## 🧠 Modelos Mangaba disponíveis

| Modelo | Params | Prefixo |
|--------|--------|---------|
| **Mangaba E2B** 🥭 | 2B   | `/api/v1/e2b/` |
| **Mangaba E4B** 🥭 | 4B   | `/api/v1/e4b/` |
| **Mangaba 12B** 🥭 | 12B  | `/api/v1/12b/` |
| **Mangaba 26B** 🥭 | 26B MoE (4B ativos) | `/api/v1/26b/` |

## 🔌 Endpoints por modelo
- `POST /{modelo}/text/generate` — geração de texto
- `POST /{modelo}/text/chat` — chat com histórico
- `POST /{modelo}/image/describe` — análise de imagem
- `POST /{modelo}/audio/transcribe` — transcrição de áudio
- `POST /{modelo}/audio/chat` — áudio → resposta do modelo

## 🔑 Autenticação
Envie o cabeçalho **`X-API-Key`** em todas as chamadas de inferência.
Clique em **Authorize** 🔒 acima para testar pelo Swagger.
        """,
        contact={"name": "Mangaba AI", "url": "https://github.com/dheiver2/Mangaba-Router"},
        license_info={"name": "Apache 2.0"},
        lifespan=lifespan,
        docs_url="/swagger",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "none",
            "filter": True,
            "tryItOutEnabled": True,
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dependência de autenticação aplicada a todos os endpoints de inferência.
    # (health, swagger, redoc e openapi.json permanecem públicos)
    auth_dep = [Depends(verify_api_key)]

    # Router de sistema (health/metrics) — público
    app.include_router(router, prefix="/api/v1")

    # Routers de inferência — protegidos por API key
    app.include_router(text.router, prefix="/api/v1", dependencies=auth_dep)
    app.include_router(image.router, prefix="/api/v1", dependencies=auth_dep)
    app.include_router(audio.router, prefix="/api/v1", dependencies=auth_dep)
    app.include_router(models.router, prefix="/api/v1", dependencies=auth_dep)

    # Router dedicado por modelo — protegido por API key
    for model_name, slug, label in MODEL_ROUTES:
        model_router = create_model_router(model_name, slug, label)
        app.include_router(model_router, prefix="/api/v1", dependencies=auth_dep)

    # Endpoints de gerenciamento de usuários
    @app.get("/api/v1/users", tags=["Usuários"], summary="Listar usuários cadastrados")
    def list_users():
        return {"users": auth.list_users(), "metrics": auth.user_metrics()}

    @app.post("/api/v1/users/reload", tags=["Usuários"], summary="Recarregar users.json")
    def reload_users():
        n = auth.reload_users()
        return {"message": f"{n} usuários carregados."}

    # Adiciona o esquema de API key no Swagger (botão "Authorize")
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
