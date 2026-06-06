"""Compatibilidade: delega ao model_registry."""
from app import model_registry


def load_model():
    model_registry.load()


def get_tokenizer():
    return model_registry.get_tokenizer()


def get_model():
    return model_registry.get_model()


def is_loaded() -> bool:
    return model_registry.is_loaded()


def get_device() -> str:
    return model_registry.get_device()
