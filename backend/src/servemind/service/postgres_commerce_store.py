"""PostgreSQL persistence reusing the tested commercial authorization/workflow.

The transport adapts only bound qmark parameters and row access. DDL, identity,
locking and transactions are native PostgreSQL, not arbitrary SQLite emulation.
"""
from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from servemind.memory.conversation_lock import ConversationLock
from servemind.service.commerce_store import CommerceStore

SCHEMA_PATH = Path(__file__).resolve().parents[3] / 'sql/005_commerce.sql'


def checked_schema(name: str) -> str:
    if not re.fullmatch(r'commerce|test_commerce_[a-f0-9]{32}', name):
        raise ValueError('invalid_commerce_schema')
    return name


def initialize_schema(connection, schema: str = 'commerce') -> None:
    checked_schema(schema)
    connection.execute(SCHEMA_PATH.read_text(encoding='utf-8').replace('commerce.', schema+'.')
                       .replace('CREATE SCHEMA IF NOT EXISTS commerce;', f'CREATE SCHEMA IF NOT EXISTS {schema};'))


def bound_sql(query: str) -> str:
    # Quoted '?' literals are not placeholders; percentages must be escaped for
    # psycopg's bind protocol, never by interpolating parameter values ourselves.
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\?",
                  lambda m: '%s' if m.group() == '?' else m.group(), query.replace('%','%%'))


class PublicRow(dict):
    def __getitem__(self, key):
        value = list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)
        return value


def contract_row(row):
    if row is None:
        return None
    return PublicRow({k: v.isoformat(timespec='microseconds') if isinstance(v, datetime) else v
                      for k, v in row.items()})


class Cursor:
    def __init__(self, cursor):
        self.cursor = cursor
    def fetchone(self):
        return contract_row(self.cursor.fetchone())
    def fetchall(self):
        return [contract_row(r) for r in self.cursor.fetchall()]
    def __iter__(self):
        return (contract_row(r) for r in self.cursor)


class Connection:
    def __init__(self, raw):
        self.raw = raw
    def execute(self, query, params=None):
        return Cursor(self.raw.execute(bound_sql(query) if params is not None else query, params))
    def __enter__(self):
        return self
    def __exit__(self, kind, value, traceback):
        self.raw.rollback() if kind else self.raw.commit()


class PostgresConversationLock:
    """Session advisory lock spans model latency WITHOUT a long SQL transaction.

    Redis is still the fast coordinating layer. The database lock also prevents
    duplicate generation across processes when Redis is down or its lease expires.
    """
    def __init__(self, dsn: str):
        self.dsn, self.fast = dsn, ConversationLock()

    @contextmanager
    def hold(self, conversation_id: str, *, wait_seconds: float = 5):
        with self.fast.hold(conversation_id, wait_seconds=wait_seconds):
            with psycopg.connect(self.dsn, autocommit=True, connect_timeout=3) as connection:
                deadline = time.monotonic() + wait_seconds
                while True:
                    acquired = connection.execute('SELECT pg_try_advisory_lock(hashtextextended(%s, 31))',
                                                  (conversation_id,)).fetchone()[0]
                    if acquired:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('conversation_busy')
                    time.sleep(.05)
                try:
                    yield 'postgres_advisory_and_redis'
                finally:
                    connection.execute('SELECT pg_advisory_unlock(hashtextextended(%s, 31))', (conversation_id,))


class PostgresCommerceStore(CommerceStore):
    backend = 'postgres'

    def __init__(self, dsn: str | None = None, *, schema: str = 'commerce', require_migration: bool = True):
        self.dsn = dsn or os.getenv('SERVEMIND_DATABASE_URL','postgresql:///servemind')
        self.schema = checked_schema(schema)
        self.conversation_lock = PostgresConversationLock(self.dsn)
        # No silent fallback to stale SQLite and no automatic demo regeneration.
        with psycopg.connect(self.dsn) as connection:
            if require_migration:
                exists = connection.execute('SELECT to_regclass(%s)', (self.schema+'.migrations',)).fetchone()[0]
                if not exists:
                    raise RuntimeError('commerce_verified_migration_required')
                ready = connection.execute(sql.SQL("SELECT completed_at FROM {}.migrations WHERE migration_id=%s")
                    .format(sql.Identifier(self.schema)), ('sqlite-commerce-v1',)).fetchone()
                if not ready:
                    raise RuntimeError('commerce_verified_migration_required')
            else:
                if schema == 'commerce':
                    raise ValueError('unverified_store_only_allowed_in_isolated_test_schema')
                initialize_schema(connection, schema)

    @contextmanager
    def _db(self):
        with psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=3,
                            options='-c timezone=UTC -c statement_timeout=10000 -c lock_timeout=3000') as raw:
            raw.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(self.schema)))
            yield Connection(raw)

    def _begin_write(self, connection):
        # psycopg begins a native transaction at the first statement. The outer
        # advisory session lock serializes this conversation, not the whole DB.
        pass

    def _persist_in_transaction(self, connection, support, **turn) -> bool:
        if support.durable_memory is None:
            return False
        support.persist_turn(connection=connection.raw,**turn)
        return True
