from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: RoleEnum = Field(..., description="Papel da mensagem na conversa")
    content: str = Field(..., description="Conteúdo da mensagem")

    model_config = {
        "json_schema_extra": {
            "example": {"role": "user", "content": "Explique o que é machine learning."}
        }
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Texto de entrada para geração")
    model_name: Optional[str] = Field(None, description="Nome do modelo: gemma-4-E2B-it | gemma-4-E4B-it | gemma-4-12B-it | gemma-4-26B-A4B-it")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Máximo de tokens gerados")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperatura de amostragem")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p (nucleus sampling)")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k tokens considerados")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Escreva um poema sobre inteligência artificial.",
                "max_new_tokens": 256,
                "temperature": 0.7,
            }
        }
    }


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Lista de mensagens do histórico")
    model_name: Optional[str] = Field(None, description="Nome do modelo: gemma-4-E2B-it | gemma-4-E4B-it | gemma-4-12B-it | gemma-4-26B-A4B-it")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "system", "content": "Você é um assistente útil em português."},
                    {"role": "user", "content": "Qual é a capital do Brasil?"},
                ],
                "max_new_tokens": 128,
                "temperature": 0.5,
            }
        }
    }


class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Texto gerado pelo modelo")
    model: str = Field(..., description="ID do modelo utilizado")
    tokens_generated: int = Field(..., description="Quantidade de tokens gerados")


class ChatResponse(BaseModel):
    message: Message = Field(..., description="Resposta do assistente")
    model: str = Field(..., description="ID do modelo utilizado")
    tokens_generated: int = Field(..., description="Quantidade de tokens gerados")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_id: str
    device: str
    loaded: bool
    max_new_tokens: int
    temperature: float
    max_concurrent_requests: int


class MetricsResponse(BaseModel):
    total_requests: int = Field(..., description="Total de requisições recebidas")
    completed: int = Field(..., description="Requisições concluídas com sucesso")
    failed: int = Field(..., description="Requisições com erro")
    avg_queue_wait_ms: float = Field(..., description="Tempo médio de espera na fila (ms)")
    avg_inference_ms: float = Field(..., description="Tempo médio de inferência (ms)")
