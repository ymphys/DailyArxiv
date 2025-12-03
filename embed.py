import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("dailyarxiv.embed")


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    model: str
    batch_size: int
    device: Optional[str]
    cache_path: Path


DEFAULT_CACHE = Path(".cache") / "embeddings.db"
DEFAULT_FALLBACK_MODEL = "BAAI/bge-m3-small"


def _ensure_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            key TEXT PRIMARY KEY,
            backend TEXT,
            model TEXT,
            text TEXT,
            embedding TEXT
        )
        """
    )
    return conn


def _hash_text(text: str, backend: str, model: str) -> str:
    digest = hashlib.sha256(f"{backend}|{model}|{text}".encode("utf-8")).hexdigest()
    return digest


def _fetch_cached(conn: sqlite3.Connection, key: str) -> Optional[List[float]]:
    row = conn.execute("SELECT embedding FROM embeddings WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    return json.loads(row[0])


def _store_embedding(conn: sqlite3.Connection, key: str, backend: str, model: str, text: str, vector: List[float]) -> None:
    conn.execute(
        "REPLACE INTO embeddings (key, backend, model, text, embedding) VALUES (?, ?, ?, ?, ?)",
        (key, backend, model, text, json.dumps(vector)),
    )


def _resolve_backend_model(backend: str, model: Optional[str]) -> Tuple[str, str]:
    if backend == "openai":
        resolved = model or "text-embedding-3-large"
        return backend, resolved
    return "huggingface", model or DEFAULT_FALLBACK_MODEL


def _embed_openai(texts: Sequence[str], model: str, batch_size: int) -> List[List[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for OpenAI embeddings.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for the OpenAI backend.")

    client = OpenAI(api_key=api_key)
    embeddings: List[List[float]] = []
    for idx in range(0, len(texts), batch_size):
        batch = texts[idx : idx + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def _resolve_hf_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _embed_huggingface(texts: Sequence[str], model: str, device: Optional[str]) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers must be installed to use HuggingFace embeddings.") from exc

    resolved_device = _resolve_hf_device(device)
    model_inst = SentenceTransformer(model, device=resolved_device)
    embeddings = model_inst.encode(list(texts), batch_size=16, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def embed_texts(texts: Sequence[str], config: EmbeddingConfig) -> List[List[float]]:
    backend, model = _resolve_backend_model(config.backend, config.model)
    resolved_config = EmbeddingConfig(
        backend=backend,
        model=model,
        batch_size=config.batch_size,
        device=config.device,
        cache_path=config.cache_path,
    )

    conn = _ensure_cache(resolved_config.cache_path)
    results: List[Optional[List[float]]] = [None] * len(texts)
    pending: List[Tuple[int, str]] = []

    for idx, text in enumerate(texts):
        key = _hash_text(text, resolved_config.backend, resolved_config.model)
        cached = _fetch_cached(conn, key)
        if cached:
            results[idx] = cached
        else:
            pending.append((idx, text))

    if pending:
        logger.info("Generating %d new embeddings...", len(pending))
        order = [text for _, text in pending]
        if resolved_config.backend == "openai":
            batch_embeddings = _embed_openai(order, resolved_config.model, resolved_config.batch_size)
        else:
            batch_embeddings = _embed_huggingface(order, resolved_config.model, resolved_config.device)

        for (idx, text), emb in zip(pending, batch_embeddings):
            results[idx] = emb
            key = _hash_text(text, resolved_config.backend, resolved_config.model)
            _store_embedding(conn, key, resolved_config.backend, resolved_config.model, text, emb)
        conn.commit()

    conn.close()
    return [result or [] for result in results]
