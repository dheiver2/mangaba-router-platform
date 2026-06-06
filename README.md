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

## Multiusuário e multiplataforma

### Autenticação por API key
Todos os endpoints de inferência exigem o cabeçalho `X-API-Key`.
As chaves ficam em `config/users.json` (cada usuário tem chave + rate limit próprio).

```bash
# editar usuários e chaves
nano config/users.json
# recarregar sem reiniciar a API
curl -X POST http://SEU_IP:8000/api/v1/users/reload
```

### Acesso em rede (qualquer plataforma)
A API escuta em `0.0.0.0:8000` — acessível por web, mobile, desktop, qualquer SO.
Descubra o IP da máquina e use-o nos clientes:

```bash
# IP local (LAN)
ipconfig getifaddr en0     # macOS
```

### Exemplos de clientes

**cURL (Linux/macOS/Windows):**
```bash
curl -X POST http://192.168.15.33:8000/api/v1/e2b/text/chat \
  -H "X-API-Key: mangaba-web-CHANGE-ME-e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Olá!"}]}'
```

**JavaScript (web/mobile):**
```javascript
fetch("http://192.168.15.33:8000/api/v1/e2b/text/chat", {
  method: "POST",
  headers: {
    "X-API-Key": "mangaba-web-CHANGE-ME-e5f6g7h8",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ messages: [{ role: "user", content: "Olá!" }] }),
}).then(r => r.json()).then(console.log);
```

**Python:**
```python
import requests
r = requests.post(
    "http://192.168.15.33:8000/api/v1/e2b/text/chat",
    headers={"X-API-Key": "mangaba-web-CHANGE-ME-e5f6g7h8"},
    json={"messages": [{"role": "user", "content": "Olá!"}]},
)
print(r.json())
```

### Concorrência (múltiplos usuários simultâneos)
- Fila assíncrona com semáforo (`MAX_CONCURRENT_REQUESTS`, padrão 10)
- Rate limit por usuário (req/min em `users.json`)
- `GET /api/v1/users` mostra contadores de uso por usuário

## Observações
- O HD externo é **ExFAT** — o Python portátil contorna a limitação de symlinks.
- Carregar os pesos via USB é mais lento que SSD interno. Um modelo por vez fica
  em memória; ao trocar de modelo o anterior é descarregado.
- ⚠️ **Troque as chaves padrão** em `config/users.json` antes de expor na rede.
# Mangaba-Router
