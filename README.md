<div align="center">
  <img src="assets/mangaba-logo.svg" alt="Mangaba AI" width="140"/>

  [![Mangaba AI](https://img.shields.io/badge/Mangaba-AI-F97518?style=for-the-badge)](https://www.mangaba.ia.br)
  [![Site](https://img.shields.io/badge/mangaba.ia.br-1E0D01?style=for-the-badge)](https://www.mangaba.ia.br)
</div>

# 🥭 Mangaba Router

Plataforma **multimodal de IA** (texto · imagem · áudio) servindo os modelos
**Mangaba** — baseados em **Google Gemma 4**, quantizados **Q4_0 (QAT oficial)** —
acelerados na **GPU Metal** (Apple Silicon). Roda **100% de um HD externo**, em 16 GB de RAM.

> Acesso livre (sem senha) · API REST + Swagger · multiplataforma · baixa latência.

---

## ⚡ Início rápido

Plugue o HD externo **MangabaAI** num Mac (Apple Silicon) e rode:

```bash
cd /Volumes/MangabaAI/gemma4-api
./start.sh
```

- **Swagger (interface):** http://localhost:8000/swagger
- Na rede (outros dispositivos): `http://SEU_IP:8000/swagger`

O modelo padrão (**E2B**) é pré-carregado no boot — a primeira requisição já sai rápida.

---

## 🧠 Modelos (quantizados Q4_0)

A API carrega **um modelo por vez** (troca automática ao chamar outro prefixo).

| Modelo | Prefixo | Params | Quando usar |
|--------|---------|--------|-------------|
| **Mangaba E2B** | `/api/v1/e2b/` | 2B  | Padrão. Rápido, chat simples, alto volume. |
| **Mangaba E4B** | `/api/v1/e4b/` | 4B  | Mais qualidade mantendo boa velocidade. |
| **Mangaba 12B** | `/api/v1/12b/` | 12B | Tarefas complexas: raciocínio, imagem detalhada, textos longos. |

> O **26B** foi removido: 15 GB não cabem nos 16 GB de RAM mesmo quantizado.
> Todos baseados em `google/gemma-4-*-it-qat-q4_0-gguf` (QAT oficial do Google).

---

## 🔌 Rotas (5 por modelo)

| Rota | Quando usar |
|------|-------------|
| `POST /{modelo}/text/chat` | **Conversas e instruções** (recomendada). |
| `POST /{modelo}/text/generate` | Completar texto cru, controle total do prompt. |
| `POST /{modelo}/image/describe` | **Visão:** descrever/analisar imagem, OCR. |
| `POST /{modelo}/audio/transcribe` | **Só transcrever** fala → texto (Whisper). |
| `POST /{modelo}/audio/chat` | **Áudio → resposta:** assistente por voz. |

### Sistema / gerência (público)
- `GET /api/v1/health` — status + modelo carregado
- `GET /api/v1/metrics` — métricas de fila/concorrência
- `GET /api/v1/models` — lista os modelos Mangaba
- `GET /api/v1/users` · `POST /api/v1/users/reload`

---

## 💻 Exemplos

**cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/e2b/text/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Olá!"}]}'
```

**Imagem:**
```bash
curl -X POST http://localhost:8000/api/v1/e2b/image/describe \
  -F "file=@foto.jpg" -F "prompt=O que há nesta imagem?"
```

**Áudio → resposta:**
```bash
curl -X POST http://localhost:8000/api/v1/e2b/audio/chat \
  -F "file=@pergunta.mp3" -F "language=pt"
```

---

## 🏗️ Arquitetura

```
/Volumes/MangabaAI/gemma4-api/        ← tudo no HD externo (zero footprint no notebook)
├── start.sh                          # inicia a API (Python portátil do HD)
├── main.py                           # FastAPI: rotas, Swagger
├── python/                           # Python 3.11 portátil + dependências
├── app/
│   ├── gguf_backend.py               # llama.cpp + Metal (carga, chat, visão)
│   ├── routers/gguf_router.py        # rotas por modelo (texto/imagem/áudio)
│   ├── auth.py                       # API key (desativada por padrão)
│   ├── inference_queue.py            # fila assíncrona p/ concorrência
│   └── routes.py                     # health/metrics
├── config/settings.py
├── scripts/download_gguf.py          # baixa os GGUF Q4_0 (marca Mangaba)
└── mangaba_models/                   # modelos GGUF
    ├── e2b/  mangaba-e2b-q4_0.gguf + mangaba-e2b-mmproj.gguf
    ├── e4b/  mangaba-e4b-q4_0.gguf + mangaba-e4b-mmproj.gguf
    └── 12b/  mangaba-12b-q4_0.gguf + mangaba-12b-mmproj.gguf
```

**Processamento:** GPU Metal + RAM do notebook (runtime). **Armazenamento:** 100% no HD externo.

---

## ⚙️ Otimizações de performance

| Técnica | Efeito |
|---------|--------|
| **Warmup no boot** | Pré-carrega o E2B → 1ª requisição instantânea (~0.2 s) |
| **mlock** | Fixa o modelo na RAM, nunca é despejado (sempre quente) |
| **flash attention + n_batch + threads** | ~26 tokens/s na GPU Metal |
| **mmap + page cache** | Reaproveita páginas já lidas |
| **Preload fora do timeout** | Carga a frio do USB não causa 504 |

> **Gargalo conhecido:** o HD em USB 2.0 (~108 MB/s) afeta só a **carga inicial**
> de cada modelo (~30 s E2B, ~60 s 12B). Uma porta **USB 3 / Thunderbolt** reduz isso 4-5×.

---

## 🔧 Configuração (`config/settings.py` ou `.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `AUTH_ENABLED` | `False` | `true` exige cabeçalho `X-API-Key` (chaves em `config/users.json`) |
| `STAGE_TO_SSD` | `0` | `1` copia o modelo p/ o SSD do notebook (mais rápido, mas ocupa o notebook) |
| `MAX_CONCURRENT_REQUESTS` | `10` | Requisições simultâneas na fila |
| `REQUEST_TIMEOUT_SECONDS` | `300` | Timeout de inferência |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind de rede |

---

## 📦 Setup do zero (nova máquina)

```bash
cd /Volumes/MangabaAI/gemma4-api
# Python portátil já incluso; se faltar, recrie deps:
./python/bin/python3.11 -m pip install -r requirements.txt
# baixar os modelos GGUF (precisa de HF_TOKEN no .env)
./python/bin/python3.11 scripts/download_gguf.py
./start.sh
```

---

## Licença
Apache 2.0 · Modelos: Google Gemma 4 (Apache 2.0) · Branding: Mangaba AI
