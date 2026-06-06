# Gemma 4 API — Sistema 100% no HD Externo MangabaAI

API REST multimodal para os modelos **Google Gemma 4**, totalmente autocontida
no HD externo. Não depende de Python, venv ou bibliotecas instaladas no notebook.

## Como iniciar

Plugue o HD externo **MangabaAI** em qualquer Mac (Apple Silicon) e rode:

```bash
cd /Volumes/MangabaAI/gemma4-api
./start.sh
```

Swagger: http://localhost:8000/swagger

## Estrutura (tudo no HD externo)

```
/Volumes/MangabaAI/gemma4-api/
├── start.sh              # inicia a API (usa o Python portátil)
├── python/              # Python 3.11 portátil + todas as dependências
├── main.py
├── app/                 # código da API
├── config/
├── models/              # 4 modelos Gemma 4 (~89 GB)
│   ├── google--gemma-4-E2B-it
│   ├── google--gemma-4-E4B-it
│   ├── google--gemma-4-12B-it
│   └── google--gemma-4-26B-A4B-it
└── .env
```

## Endpoints

### Por modelo (5 cada)
| Modelo | Prefixo |
|--------|---------|
| Gemma 4 E2B (2B) | `/api/v1/e2b/` |
| Gemma 4 E4B (4B) | `/api/v1/e4b/` |
| Gemma 4 12B | `/api/v1/12b/` |
| Gemma 4 26B MoE | `/api/v1/26b/` |

Cada modelo expõe:
- `POST /{modelo}/text/generate`
- `POST /{modelo}/text/chat`
- `POST /{modelo}/image/describe`
- `POST /{modelo}/audio/transcribe`
- `POST /{modelo}/audio/chat`

### Gerais
- `GET /api/v1/health`
- `GET /api/v1/metrics`
- `GET /api/v1/models`
- `POST /api/v1/models/load`
- `GET /api/v1/models/current`

## Observações
- O HD externo é **ExFAT** — o Python portátil contorna a limitação de symlinks.
- Carregar os pesos via USB é mais lento que SSD interno. Um modelo por vez fica
  em memória; ao trocar de modelo o anterior é descarregado.
# Mangaba-Router
