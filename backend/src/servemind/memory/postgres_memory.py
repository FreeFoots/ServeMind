from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from servemind.core.vector_knowledge import EMBEDDING_MODEL, vector_literal
from servemind.memory.semantic_memory import bounded_embedding


SCHEMA_PATH = Path(__file__).resolve().parents[3] / "sql" / "004_memory.sql"


class PostgresConversationMemory:
    """Durable, scoped episode and profile memory without raw buyer messages."""

    def __init__(self, dsn: str | None = None, *, initialize: bool = True, schema: str = 'memory') -> None:
        self.dsn = dsn or os.getenv("SERVEMIND_DATABASE_URL", "postgresql:///servemind")
        if not re.fullmatch(r'memory|test_memory_[a-f0-9]{32}', schema):
            raise ValueError('invalid_memory_schema')
        self.schema = schema
        if initialize:
            with psycopg.connect(self.dsn) as connection:
                for path in (SCHEMA_PATH, SCHEMA_PATH.with_name('006_semantic_memory.sql')):
                    connection.execute(path.read_text(encoding='utf-8').replace('memory.',schema+'.')
                                       .replace('CREATE SCHEMA IF NOT EXISTS memory;',f'CREATE SCHEMA IF NOT EXISTS {schema};'))

    def _execute(self, connection, query, params):
        return connection.execute(query.replace('memory.',self.schema+'.'),params)

    @contextmanager
    def _connection(self, connection=None):
        if connection is not None:
            # The commercial transaction owns commit/rollback. Memory errors
            # must propagate; no independent commit or hidden partial success.
            yield connection
        else:
            with psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=3,
                                 options='-c statement_timeout=5000') as own:
                yield own

    def context(self, conversation_id: str, buyer_id: str, merchant_id: str,
                product_id: str | None = None, *, query: str = '', semantic: bool = False,
                connection=None) -> dict[str, Any]:
        embedding = bounded_embedding(query,query=True) if semantic and query and product_id else None
        episodes = []
        with self._connection(connection) as connection:
            session = self._execute(connection,
                """SELECT summary, turn_count, resolved_topics, pending_topics FROM memory.sessions
                   WHERE conversation_id = %s AND buyer_id = %s AND merchant_id = %s
                     AND (%s::text IS NULL OR product_id=%s)
                     AND updated_at >= now() - interval '30 days'""",
                (conversation_id, buyer_id, merchant_id,product_id,product_id),
            ).fetchone()
            profile = self._execute(connection,
                """SELECT last_intent, last_product_id, interaction_count, response_style FROM memory.profiles
                   WHERE buyer_id = %s AND merchant_id = %s
                     AND updated_at >= now() - interval '90 days'""",
                (buyer_id, merchant_id),
            ).fetchone()
            if embedding is not None:
                vector = vector_literal(embedding)
                episodes = self._execute(connection,
                    """WITH scoped AS MATERIALIZED (
                         SELECT e.message_id,e.conversation_id,e.summary,e.topics,e.pending_topics,
                                e.evidence_ids,e.summary_source,e.embedding,e.created_at
                         FROM memory.episodes e JOIN memory.sessions s USING(conversation_id)
                         WHERE s.buyer_id=%s AND s.merchant_id=%s AND s.product_id=%s
                           AND e.created_at >= now()-interval '30 days'
                           AND e.embedding_model=%s AND e.embedding IS NOT NULL)
                       SELECT message_id,conversation_id,summary,topics,pending_topics,evidence_ids,
                              summary_source,1-(embedding <=> %s::vector) AS similarity
                       FROM scoped WHERE 1-(embedding <=> %s::vector)>=0.55
                       ORDER BY embedding <=> %s::vector,created_at DESC LIMIT 3""",
                    (buyer_id,merchant_id,product_id,EMBEDDING_MODEL,vector,vector,vector)).fetchall()
        return {"summary": session["summary"] if session else "",
                "resolved_topics": session["resolved_topics"] if session else [],
                "pending_topics": session["pending_topics"] if session else [],
                "turn_count": session["turn_count"] if session else 0,
                "profile": dict(profile) if profile else {},
                "episodes": [dict(e) for e in episodes], 'semantic_query_succeeded':embedding is not None,
                'authority':'historical_context_not_current_business_evidence'}

    def record_exchange(self, *, conversation_id: str, buyer_id: str, merchant_id: str,
                        product_id: str, message_id: str, intent: str,
                        topics: list[str], evidence_ids: list[str],
                        needs_merchant: bool, summary: str = "",
                        resolved_topics: list[str] | None = None,
                        pending_topics: list[str] | None = None, response_style: str | None = None,
                        connection=None, prepared: dict | None = None) -> bool:
        if response_style not in {None, "standard", "concise", "detailed"}:
            raise ValueError("invalid_response_style")
        resolved_topics, pending_topics = resolved_topics or [], pending_topics or []
        prepared = prepared or {}
        summary = prepared.get('summary',summary)[:1200]
        summary_source = prepared.get('source','structured_topics')
        if summary_source not in {'structured_topics','deepseek_validated_topics'}:
            raise ValueError('invalid_memory_summary_source')
        embedding = prepared.get('embedding')
        vector = vector_literal(embedding) if embedding is not None else None
        with self._connection(connection) as connection:
            self._execute(connection,
                """INSERT INTO memory.sessions
                   (conversation_id, buyer_id, merchant_id, product_id)
                   VALUES (%s, %s, %s, %s) ON CONFLICT (conversation_id) DO NOTHING""",
                (conversation_id, buyer_id, merchant_id, product_id),
            )
            scope = self._execute(connection,
                """SELECT buyer_id, merchant_id, product_id FROM memory.sessions
                   WHERE conversation_id = %s FOR UPDATE""", (conversation_id,),
            ).fetchone()
            if scope != {"buyer_id": buyer_id, "merchant_id": merchant_id,
                         "product_id": product_id}:
                raise PermissionError("memory_scope_mismatch")
            inserted = self._execute(connection,
                """INSERT INTO memory.episodes
                   (message_id, conversation_id, intent, topics, evidence_ids, needs_merchant, summary,
                    resolved_topics, pending_topics, embedding, embedding_model, summary_source)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                   ON CONFLICT (message_id) DO NOTHING RETURNING message_id""",
                (message_id, conversation_id, intent, topics, evidence_ids, needs_merchant,
                 summary, resolved_topics, pending_topics, vector, EMBEDDING_MODEL if vector else None,summary_source),
            ).fetchone()
            if not inserted:
                return False
            episodes = self._execute(connection,
                """SELECT intent, topics, summary FROM memory.episodes
                   WHERE conversation_id = %s ORDER BY created_at DESC, message_id DESC LIMIT 4""",
                (conversation_id,),
            ).fetchall()
            labels = [row["summary"] or "/".join(row["topics"][:3]) or row["intent"] for row in reversed(episodes)]
            session_summary = (summary if summary_source=='deepseek_validated_topics' else '最近关注：'+' → '.join(labels))
            self._execute(connection,
                """UPDATE memory.sessions SET summary = %s, turn_count = turn_count + 1,
                   resolved_topics = %s, pending_topics = %s,
                   summary_source=%s, updated_at = now() WHERE conversation_id = %s""",
                (session_summary[:2400], resolved_topics, pending_topics, summary_source,conversation_id),
            )
            # A merchant reply is an event, not proof of resolution or a buyer preference.
            if intent == "merchant_reply":
                return True
            self._execute(connection,
                """INSERT INTO memory.profiles
                   (buyer_id, merchant_id, last_product_id, last_intent, interaction_count, response_style)
                   VALUES (%s, %s, %s, %s, 1, COALESCE(%s, 'standard'))
                   ON CONFLICT (buyer_id, merchant_id) DO UPDATE SET
                     last_product_id = EXCLUDED.last_product_id,
                     last_intent = EXCLUDED.last_intent,
                     interaction_count = memory.profiles.interaction_count + 1,
                     response_style = COALESCE(%s, memory.profiles.response_style),
                     updated_at = now()""",
                (buyer_id, merchant_id, product_id, intent, response_style, response_style),
            )
        return True
