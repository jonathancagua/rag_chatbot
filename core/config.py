from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv


def _get_env(name: str, default: str | None = None):
    val = os.getenv(name, default)
    return val


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass
class AppConfig:
    groq_api_key: str
    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_index: str
    model_name: str
    embed_model: str
    top_k: int
    temperature: float
    max_tokens: int
    memory_k: int
    chunk_size: int
    overlap: int


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        groq_api_key=_get_env("GROQ_API_KEY", ""),
        pinecone_api_key=_get_env("PINECONE_API_KEY", ""),
        pinecone_cloud=_get_env("PINECONE_CLOUD", "aws"),
        pinecone_region=_get_env("PINECONE_REGION", "us-east-1"),
        pinecone_index=_get_env("PINECONE_INDEX_NAME", "cv-index"),
        model_name=_get_env("MODEL_NAME", "llama3-8b-8192"),
        embed_model=_get_env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=_get_int("TOP_K", 4),
        temperature=_get_float("TEMPERATURE", 0.2),
        max_tokens=_get_int("MAX_TOKENS", 800),
        memory_k=_get_int("MEMORY_K", 6),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        overlap=_get_int("OVERLAP", 100),
    )
