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
    ("gemma-4-E2B-it",    "e2b",  "Mangaba E2B"),
    ("gemma-4-E4B-it",    "e4b",  "Mangaba E4B"),
    ("gemma-4-12B-it",    "12b",  "Mangaba 12B"),
    ("gemma-4-26B-A4B-it","26b",  "Mangaba 26B"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Mangaba Router API",
        version=settings.api_version,
        description="""
# Mangaba Router

**Plataforma multimodal de IA** — roteamento inteligente entre múltiplos modelos
**Mangaba** (baseados em Google Gemma 4), com texto, imagem e áudio.

> Multiusuário · Multiplataforma · GPU acelerada · 100% autocontido no HD externo.

---

## 🧠 Qual modelo escolher?

A API carrega **um modelo por vez** (troca automática ao chamar outro prefixo).
Escolha pelo equilíbrio entre **qualidade** e **RAM/velocidade**:

| Modelo | Prefixo | Params | RAM mín. | Quando usar |
|--------|---------|--------|----------|-------------|
| **Mangaba E2B** | `/api/v1/e2b/` | 2B | ~8 GB | **Padrão.** Respostas rápidas, chat simples, alto volume, hardware modesto (≤16GB). |
| **Mangaba E4B** | `/api/v1/e4b/` | 4B | ~12 GB | Mais qualidade que o E2B mantendo boa velocidade. Bom meio-termo. |
| **Mangaba 12B** | `/api/v1/12b/` | 12B | ~24 GB | Tarefas complexas: raciocínio, análise de imagem detalhada, textos longos. Exige bastante RAM. |
| **Mangaba 26B** | `/api/v1/26b/` | 26B MoE (4B ativos) | ~40 GB | Máxima qualidade. MoE: custo de 4B ativos mas precisa carregar 26B. Só em servidores. |

> 💡 **Regra prática:** comece no **E2B**. Suba para 12B/26B só se precisar de mais
> qualidade **e** tiver RAM suficiente — senão a máquina entra em swap e fica lenta.

---

## 🔌 Qual rota usar?

Cada modelo expõe as mesmas 5 rotas. Escolha pela **tarefa**:

| Rota | Entrada | Quando usar |
|------|---------|-------------|
| `POST /{modelo}/text/chat` | mensagens (system/user/assistant) | **Conversas e instruções.** É a rota recomendada para a maioria dos casos — usa o template de chat do modelo. |
| `POST /{modelo}/text/generate` | prompt cru (string) | Completar texto livre, sem formato de chat. Para quem quer controle total do prompt. |
| `POST /{modelo}/image/describe` | arquivo de imagem + pergunta | **Visão:** descrever/analisar imagem, OCR, perguntar sobre o conteúdo visual. |
| `POST /{modelo}/audio/transcribe` | arquivo de áudio | **Só transcrever** fala → texto (Whisper). Não usa o LLM. |
| `POST /{modelo}/audio/chat` | arquivo de áudio | **Áudio → resposta:** transcreve a fala e o LLM responde. Use para assistente por voz. |

> `chat` vs `generate`: prefira **chat** (segue instruções melhor). Use `generate`
> apenas para autocompletar texto bruto.

---

## 🔑 Autenticação
Envie o cabeçalho **`X-API-Key`** em todas as chamadas de inferência.
Clique em **Authorize** 🔒 acima para testar pelo Swagger.
Endpoints `health`, `metrics` e `users` são públicos.
        """,
        contact={"name": "Mangaba AI", "url": "https://github.com/dheiver2/Mangaba-Router"},
        license_info={"name": "Apache 2.0"},
        lifespan=lifespan,
        docs_url="/swagger",      # Swagger UI padrão do FastAPI
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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
