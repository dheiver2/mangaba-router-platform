import torch
from typing import Optional
from app import model_registry
from app.schemas import Message
from config.settings import get_settings


def generate_text(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> tuple[str, int]:
    settings = get_settings()
    tokenizer = model_registry.get_tokenizer()
    model = model_registry.get_model()

    max_new_tokens = max_new_tokens or settings.max_new_tokens
    temperature = temperature if temperature is not None else settings.temperature
    top_p = top_p if top_p is not None else settings.top_p
    top_k = top_k if top_k is not None else settings.top_k

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip(), len(generated)


def chat(
    messages: list[Message],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> tuple[str, int]:
    tokenizer = model_registry.get_tokenizer()
    chat_messages = [{"role": m.role.value, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    return generate_text(prompt, max_new_tokens, temperature, top_p, top_k)
