from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

import psycopg
from psycopg.rows import dict_row

from servemind.config.policy_registry import POLICY_VERSION
from servemind.config.settings import PROJECT_ROOT

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DIMENSIONS = 1024
TASK_INSTRUCTION = "Given a Chinese customer service question, retrieve relevant policy passages that answer the question"
_MODEL_LOCK = threading.RLock()
MODEL_CACHE = os.getenv("SERVEMIND_MODEL_CACHE", str(PROJECT_ROOT / "backend" / "runtime" / "hf-cache" / "hub"))


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    return SentenceTransformer(snapshot_download(EMBEDDING_MODEL, cache_dir=MODEL_CACHE, local_files_only=True))


@lru_cache(maxsize=1)
def rerank_model():
    from sentence_transformers import CrossEncoder
    from huggingface_hub import snapshot_download
    return CrossEncoder(snapshot_download(RERANK_MODEL, cache_dir=MODEL_CACHE, local_files_only=True),
                        prompts={"query": TASK_INSTRUCTION}, default_prompt_name="query", max_length=1024)


def encode_document(text: str) -> list[float]:
    with _MODEL_LOCK:
        vector = embedding_model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    values = vector.tolist()
    if len(values) != DIMENSIONS:
        raise ValueError(f"embedding dimension mismatch: {len(values)}")
    return values


def encode_query(text: str) -> list[float]:
    query = f"Instruct: {TASK_INSTRUCTION}\nQuery: {text}"
    with _MODEL_LOCK:
        vector = embedding_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    values = vector.tolist()
    if len(values) != DIMENSIONS:
        raise ValueError(f"embedding dimension mismatch: {len(values)}")
    return values


def encode_memory_query(text: str) -> list[float]:
    return encode_document('Instruct: Retrieve earlier customer concerns and unresolved tasks from the same buyer, merchant and product\nQuery: '+text)


def vector_literal(values: list[float]) -> str:
    if len(values) != DIMENSIONS:
        raise ValueError("embedding dimension mismatch")
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class VectorKnowledgeBase:
    def __init__(self, dsn: str | None = None, *, use_reranker: bool | None = None) -> None:
        self.dsn = dsn or os.getenv("SERVEMIND_DATABASE_URL", "postgresql:///servemind")
        self.use_reranker = (os.getenv("SERVEMIND_RERANK_ENABLED", "false").lower() == "true"
                             if use_reranker is None else use_reranker)

    def search(self, query: str, top_k: int = 3, *, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        vector = vector_literal(encode_query(query))
        with psycopg.connect(self.dsn, row_factory=dict_row, options="-c statement_timeout=5000") as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT chunk_id, title, display_title, source, content, document_version,
                              valid_from::text, valid_until::text,
                              1 - (embedding <=> %s::vector) AS similarity
                       FROM knowledge.chunks
                       WHERE embedding_model = %s AND policy_version = %s
                         AND (%s::text[] IS NULL OR document_id = ANY(%s::text[]))
                         AND valid_from <= CURRENT_DATE
                         AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
                       ORDER BY embedding <=> %s::vector LIMIT 12""",
                    (vector, EMBEDDING_MODEL, POLICY_VERSION, document_ids, document_ids, vector),
                )
                vector_rows = [dict(row) for row in cursor.fetchall()]
                cursor.execute(
                    """SELECT chunk_id, title, display_title, source, content, document_version,
                              valid_from::text, valid_until::text,
                              similarity(content, %s) +
                              CASE WHEN strpos(content, %s) > 0 THEN 1.0 ELSE 0.0 END
                              AS lexical_score
                       FROM knowledge.chunks
                       WHERE embedding_model = %s AND policy_version = %s
                         AND (%s::text[] IS NULL OR document_id = ANY(%s::text[]))
                         AND valid_from <= CURRENT_DATE
                         AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
                       ORDER BY lexical_score DESC LIMIT 12""",
                    (query, query, EMBEDDING_MODEL, POLICY_VERSION, document_ids, document_ids),
                )
                lexical_rows = [dict(row) for row in cursor.fetchall()]
        fused: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(vector_rows, start=1):
            item = fused.setdefault(row["chunk_id"], row)
            item["fusion_score"] = item.get("fusion_score", 0.0) + 0.7 / (60 + rank)
            item["vector_rank"] = rank
        for rank, row in enumerate(lexical_rows, start=1):
            item = fused.setdefault(row["chunk_id"], row)
            item["fusion_score"] = item.get("fusion_score", 0.0) + 0.3 / (60 + rank)
            item["lexical_rank"] = rank
        rows = sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)
        # Rerank fused candidates, never documents outside the active rule window.
        if self.use_reranker and len(rows) >= 5:
            with _MODEL_LOCK:
                scores = rerank_model().predict([(query, row["content"]) for row in rows], show_progress_bar=False)
            for row, score in zip(rows, scores):
                row["rerank_score"] = float(score)
            rows.sort(key=lambda row: row["rerank_score"], reverse=True)
        return [{"title": row["title"], "display_title": row["display_title"],
                 "content": row["content"], "source": row["source"],
                 "score": round(float(row.get("rerank_score", row["fusion_score"])), 4),
                 "chunk_id": row["chunk_id"], "document_version": row["document_version"],
                 "valid_from": row["valid_from"], "valid_until": row["valid_until"],
                 "retrieval": {"vector_rank": row.get("vector_rank"),
                               "lexical_rank": row.get("lexical_rank"),
                               "reranked": "rerank_score" in row}}
                for row in rows[:top_k]]

    def summary(self) -> dict[str, Any]:
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT count(*), count(DISTINCT document_id)
                                  FROM knowledge.chunks WHERE policy_version = %s
                                    AND valid_from <= CURRENT_DATE
                                    AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)""",
                               (POLICY_VERSION,))
                chunks, documents = cursor.fetchone()
        return {"backend": "pgvector_trigram_hybrid", "embedding_model": EMBEDDING_MODEL,
                "reranker": RERANK_MODEL if self.use_reranker else None,
                "chunks": chunks, "documents": documents}
