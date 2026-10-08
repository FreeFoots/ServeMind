"""Recoverable, atomic SQLite snapshot import with complete logical row hashes."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from servemind.config.settings import PROJECT_ROOT
from servemind.service.postgres_commerce_store import initialize_schema, checked_schema

TABLES = ('accounts','tokens','products','catalog_assignments','demo_purchases',
          'conversations','messages','handoff_events','message_feedback')


def _hash_rows(rows, columns):
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        values = []
        for column, value in zip(columns, row):
            if column in ('created_at','expires_at') and value is not None:
                stamp = value if isinstance(value, datetime) else datetime.fromisoformat(value)
                value = stamp.astimezone(timezone.utc).isoformat(timespec='microseconds')
            values.append(value)
        digest.update(json.dumps(values, ensure_ascii=False, separators=(',',':')).encode('utf-8') + b'\n')
        count += 1
    return {'rows': count, 'sha256': digest.hexdigest()}


def create_snapshot(source: Path, backup_root: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError('commerce_source_missing')
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_root.chmod(0o700)
    name = 'commerce-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex + '.sqlite3'
    backup = backup_root / name
    with backup.open('xb'):
        pass
    backup.chmod(0o600)
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as read:
        with sqlite3.connect(backup) as target:
            read.backup(target)
    return backup


def verify_snapshot(snapshot: Path, connection, schema: str = 'commerce') -> dict:
    checked_schema(schema)
    checks = {}
    with sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro', uri=True) as source:
        for table in TABLES:
            info = source.execute(f'PRAGMA table_info({table})').fetchall()
            columns = [r[1] for r in info]
            keys = [r[1] for r in sorted(info,key=lambda r:r[5]) if r[5]]
            query = sql.SQL('SELECT {} FROM {}.{} ORDER BY {}').format(
                sql.SQL(',').join(map(sql.Identifier,columns)), sql.Identifier(schema), sql.Identifier(table),
                sql.SQL(',').join(map(sql.Identifier,keys)))
            # Table/column names come from a checked local source and psycopg
            # identifiers. No values, account hashes or messages are printed.
            original = _hash_rows(source.execute(f'SELECT {",".join(columns)} FROM {table} ORDER BY {",".join(keys)}'),columns)
            imported = _hash_rows(connection.execute(query), columns)
            checks[table] = {'source':original,'target':imported,'matched':original==imported}
    return {'passed':all(c['matched'] for c in checks.values()),'tables':checks}


def migrate(source: Path, *, dsn: str | None = None, schema: str = 'commerce',
            backup_root: Path | None = None) -> dict:
    checked_schema(schema)
    dsn = dsn or os.getenv('SERVEMIND_DATABASE_URL','postgresql:///servemind')
    backup_root = backup_root or PROJECT_ROOT/'backend/runtime/backups'
    with psycopg.connect(dsn, options='-c timezone=UTC') as connection:
        connection.execute('SELECT pg_advisory_xact_lock(197310071, 1)')
        initialize_schema(connection,schema)
        existing = connection.execute(sql.SQL('SELECT migration_id FROM {}.migrations WHERE migration_id=%s')
            .format(sql.Identifier(schema)),('sqlite-commerce-v1',)).fetchone()
        if existing:
            return {'already_applied':True,'schema':schema,'copied':False}
        for table in TABLES:
            if connection.execute(sql.SQL('SELECT EXISTS(SELECT 1 FROM {}.{})')
                .format(sql.Identifier(schema),sql.Identifier(table))).fetchone()[0]:
                raise RuntimeError('nonempty_target_requires_manual_resolution')
        snapshot = create_snapshot(source,backup_root)
        with sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro',uri=True) as origin:
            if origin.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or origin.execute('PRAGMA foreign_key_check').fetchone():
                raise RuntimeError('source_integrity_failed')
            for table in TABLES:
                info = origin.execute(f'PRAGMA table_info({table})').fetchall()
                if not info:
                    raise RuntimeError('source_table_missing')
                columns = [r[1] for r in info]
                # Known schema columns are verified by COPY; unexpected/missing
                # business columns fail the transaction, rather than dropping data.
                copy = sql.SQL('COPY {}.{} ({}) FROM STDIN').format(sql.Identifier(schema),sql.Identifier(table),
                        sql.SQL(',').join(map(sql.Identifier,columns)))
                with connection.cursor().copy(copy) as sink:
                    for row in origin.execute(f'SELECT {",".join(columns)} FROM {table}'):
                        sink.write_row(row)
        verified = verify_snapshot(snapshot,connection,schema)
        if not verified['passed']:
            raise RuntimeError('commerce_logical_verification_failed')
        connection.execute(sql.SQL("SELECT setval(pg_get_serial_sequence(%s,'sequence'),"
            "GREATEST(COALESCE(max(sequence),0),1), max(sequence) IS NOT NULL) FROM {}.messages")
            .format(sql.Identifier(schema)),(schema+'.messages',))
        fingerprint = hashlib.sha256(json.dumps(verified['tables'],sort_keys=True).encode()).hexdigest()
        connection.execute(sql.SQL('INSERT INTO {}.migrations '
            '(migration_id,source_fingerprint,backup_path,verified_counts,verified_hashes) VALUES (%s,%s,%s,%s,%s)')
            .format(sql.Identifier(schema)),('sqlite-commerce-v1',fingerprint,str(snapshot),
                Jsonb({t:c['source']['rows'] for t,c in verified['tables'].items()}),Jsonb(verified['tables'])))
    return {'already_applied':False,'schema':schema,'backup':str(snapshot),'verification':verified}
