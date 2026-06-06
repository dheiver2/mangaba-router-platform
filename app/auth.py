"""
Autenticação multiusuário por API key + rate limiting por usuário.

Cada requisição deve enviar o cabeçalho:  X-API-Key: <chave>
As chaves são definidas em config/users.json.

Funciona para QUALQUER plataforma cliente (web, mobile, desktop, curl, etc.)
pois é apenas um header HTTP padrão.
"""
import json
import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, Depends
from config.settings import get_settings

_USERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "users.json")

_users_by_key: dict[str, dict] = {}
_load_lock = Lock()

# histórico de timestamps de requisições por usuário (para rate limit)
_req_history: dict[str, deque] = defaultdict(deque)
_rate_lock = Lock()

# contadores por usuário (métricas)
_user_counts: dict[str, int] = defaultdict(int)


def _load_users():
    global _users_by_key
    with _load_lock:
        try:
            with open(_USERS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _users_by_key = {
                u["api_key"]: u
                for u in data.get("users", [])
                if u.get("enabled", True)
            }
        except FileNotFoundError:
            _users_by_key = {}


_load_users()


def reload_users():
    _load_users()
    return len(_users_by_key)


def _check_rate_limit(user: dict):
    name = user["name"]
    limit = user.get("rate_limit_per_minute", 60)
    now = time.monotonic()
    window_start = now - 60.0

    with _rate_lock:
        hist = _req_history[name]
        while hist and hist[0] < window_start:
            hist.popleft()
        if len(hist) >= limit:
            retry = int(60 - (now - hist[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Limite de {limit} req/min excedido. Tente em {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        hist.append(now)
        _user_counts[name] += 1


async def verify_api_key(x_api_key: str | None = Header(default=None)):
    """
    Dependency do FastAPI. Valida a API key e aplica rate limit.
    Se auth_enabled=False nas settings, libera acesso anônimo.
    """
    settings = get_settings()

    if not settings.auth_enabled:
        return {"name": "anonymous", "rate_limit_per_minute": 0}

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Cabeçalho 'X-API-Key' ausente.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    user = _users_by_key.get(x_api_key)
    if not user:
        raise HTTPException(status_code=403, detail="API key inválida ou desativada.")

    _check_rate_limit(user)
    return user


def user_metrics() -> dict:
    with _rate_lock:
        return {name: count for name, count in _user_counts.items()}


def list_users() -> list[dict]:
    return [
        {
            "name": u["name"],
            "rate_limit_per_minute": u.get("rate_limit_per_minute", 60),
            "enabled": u.get("enabled", True),
        }
        for u in _users_by_key.values()
    ]
