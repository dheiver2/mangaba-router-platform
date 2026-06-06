"""
Fila de inferência com semáforo para atender múltiplos usuários simultaneamente.
A GPU executa uma inferência por vez, mas as requisições ficam enfileiradas
e são processadas sem bloquear o servidor HTTP.
"""
import asyncio
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Any
from config.settings import get_settings

settings = get_settings()

# Um único executor — inferência é CPU/GPU bound, não I/O bound
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma-inference")

# Semáforo controla quantas requisições chegam ao executor de uma vez
# (evita acúmulo ilimitado na fila do ThreadPoolExecutor)
_semaphore: Optional[asyncio.Semaphore] = None

# Métricas simples compartilhadas
_metrics: dict = {
    "total_requests": 0,
    "completed": 0,
    "failed": 0,
    "queue_wait_total_ms": 0.0,
    "inference_total_ms": 0.0,
}


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    return _semaphore


async def run_in_thread(fn, *args, **kwargs) -> Any:
    """Executa fn(*args, **kwargs) em thread pool sem bloquear o event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


async def enqueue(fn, *args, request_id: str = "", **kwargs) -> Any:
    """
    Enfileira uma chamada de inferência.
    - Aguarda semáforo (fila virtual com limite configurável)
    - Executa em thread pool (não bloqueia outros requests HTTP)
    - Registra métricas de tempo
    """
    sem = _get_semaphore()
    _metrics["total_requests"] += 1
    queued_at = time.monotonic()

    async with sem:
        wait_ms = (time.monotonic() - queued_at) * 1000
        _metrics["queue_wait_total_ms"] += wait_ms

        start = time.monotonic()
        try:
            result = await run_in_thread(fn, *args, **kwargs)
            _metrics["completed"] += 1
            return result
        except Exception:
            _metrics["failed"] += 1
            raise
        finally:
            _metrics["inference_total_ms"] += (time.monotonic() - start) * 1000


def get_metrics() -> dict:
    completed = _metrics["completed"] or 1
    return {
        "total_requests": _metrics["total_requests"],
        "completed": _metrics["completed"],
        "failed": _metrics["failed"],
        "avg_queue_wait_ms": round(_metrics["queue_wait_total_ms"] / completed, 1),
        "avg_inference_ms": round(_metrics["inference_total_ms"] / completed, 1),
    }
