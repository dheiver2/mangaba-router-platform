from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.settings import get_settings
from app.routes import router
from app.routers import text, image, audio, models
from app.routers.model_router import create_model_router
from app import model_loader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Definição dos modelos e seus slugs de URL
MODEL_ROUTES = [
    ("gemma-4-E2B-it",    "e2b",  "Gemma4 E2B"),
    ("gemma-4-E4B-it",    "e4b",  "Gemma4 E4B"),
    ("gemma-4-12B-it",    "12b",  "Gemma4 12B"),
    ("gemma-4-26B-A4B-it","26b",  "Gemma4 26B"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
API REST para inferência com os modelos **Google Gemma 4**.

## Endpoints por modelo

Cada modelo tem seu próprio conjunto de endpoints:

| Modelo | Prefixo |
|--------|---------|
| Gemma 4 E2B (2B params) | `/api/v1/e2b/` |
| Gemma 4 E4B (4B params) | `/api/v1/e4b/` |
| Gemma 4 12B             | `/api/v1/12b/` |
| Gemma 4 26B MoE (4B ativos) | `/api/v1/26b/` |

## Endpoints disponíveis por modelo
- `POST /{modelo}/text/generate` — geração de texto
- `POST /{modelo}/text/chat` — chat com histórico
- `POST /{modelo}/image/describe` — análise de imagem
- `POST /{modelo}/audio/transcribe` — transcrição de áudio
- `POST /{modelo}/audio/chat` — áudio → resposta do modelo
        """,
        lifespan=lifespan,
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

    # Routers gerais
    app.include_router(router, prefix="/api/v1")
    app.include_router(text.router, prefix="/api/v1")
    app.include_router(image.router, prefix="/api/v1")
    app.include_router(audio.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")

    # Router dedicado por modelo
    for model_name, slug, label in MODEL_ROUTES:
        model_router = create_model_router(model_name, slug, label)
        app.include_router(model_router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop",
        http="httptools",
    )
