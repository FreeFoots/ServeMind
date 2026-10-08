from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
import csv
import os
from decimal import Decimal
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException
import psycopg
from psycopg.rows import dict_row

from servemind.config.settings import COMMERCE_DB_PATH, DATA_ROOT, DATA_VERSION, DELIVERY_DATA_PATH, ORDER_DATA_PATH
from servemind.memory.conversation_lock import ConversationLock


DEFAULT_COMMERCE_DB = COMMERCE_DB_PATH
PASSWORD_ITERATIONS = 310_000
DEMO_CATEGORIES = ("手机数码", "电脑办公", "家用电器", "家居日用", "母婴童装", "食品酒饮", "美妆护肤", "服饰鞋靴", "汽车用品", "图书文具")
DEFAULT_DEMO_PRICES = (6990, 12990, 29990, 5990, 8990, 3990, 7990, 10990, 19990, 4990)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _password_matches(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_hex, expected_hex = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations_text)
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected_hex))
    except (ValueError, TypeError):
        return False


class CommerceStore:
    """Local transactional state for declared products and three-party conversations.

    No data here is inferred from MSOM CSV. In particular, a product SKU is a
    merchant-declared reference, not a verified link to a historical order.
    """

    backend = 'sqlite'

    def __init__(self, path: Path | str = DEFAULT_COMMERCE_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conversation_lock = ConversationLock()
        self._initialize()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._db() as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_key TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('buyer', 'merchant')),
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL REFERENCES accounts(id),
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    sku_id TEXT,
                    provenance TEXT NOT NULL CHECK (provenance = 'merchant_declared_unverified'),
                    catalog_status TEXT NOT NULL DEFAULT 'available',
                    catalog_label TEXT NOT NULL DEFAULT '演示商品',
                    display_price_cents INTEGER,
                    price_basis TEXT NOT NULL DEFAULT 'not_provided',
                    price_as_of TEXT,
                    category TEXT NOT NULL DEFAULT '其他商品',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS catalog_assignments (
                    sku_id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL REFERENCES accounts(id),
                    display_title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '其他商品',
                    catalog_status TEXT NOT NULL,
                    assignment_type TEXT NOT NULL CHECK (assignment_type = 'simulated'),
                    source_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS demo_purchases (
                    id TEXT PRIMARY KEY,
                    buyer_id TEXT NOT NULL REFERENCES accounts(id),
                    sku_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    merchant_id TEXT NOT NULL REFERENCES accounts(id),
                    order_alias TEXT NOT NULL,
                    fulfillment_status TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (buyer_id, order_id, sku_id)
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL REFERENCES products(id),
                    buyer_id TEXT NOT NULL REFERENCES accounts(id),
                    merchant_id TEXT NOT NULL REFERENCES accounts(id),
                    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
                    access_mode TEXT NOT NULL DEFAULT 'ai_private' CHECK (access_mode IN ('ai_private', 'merchant_shared')),
                    merchant_visible INTEGER NOT NULL DEFAULT 0,
                    handoff_summary_json TEXT NOT NULL DEFAULT '{}',
                    purchase_id TEXT REFERENCES demo_purchases(id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    sender_type TEXT NOT NULL CHECK (sender_type IN ('buyer', 'merchant', 'ai')),
                    sender_id TEXT REFERENCES accounts(id),
                    content TEXT NOT NULL,
                    client_message_id TEXT,
                    reply_to_id TEXT UNIQUE REFERENCES messages(id),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE (conversation_id, sender_id, client_message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_products_merchant ON products(merchant_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_buyer ON conversations(buyer_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_merchant ON conversations(merchant_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_catalog_merchant ON catalog_assignments(merchant_id);
                CREATE INDEX IF NOT EXISTS idx_demo_purchases_buyer ON demo_purchases(buyer_id);
                CREATE TABLE IF NOT EXISTS handoff_events (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    request_id TEXT UNIQUE,
                    state TEXT NOT NULL,
                    actor_id TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_feedback (
                    message_id TEXT NOT NULL REFERENCES messages(id),
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    rating TEXT NOT NULL CHECK(rating IN ('helpful', 'not_helpful', 'incorrect')),
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(message_id, account_id)
                );
                """
            )
            self._add_column(connection, "products", "catalog_status", "TEXT NOT NULL DEFAULT 'available'")
            self._add_column(connection, "products", "catalog_label", "TEXT NOT NULL DEFAULT '演示商品'")
            self._add_column(connection, "products", "display_price_cents", "INTEGER")
            self._add_column(connection, "products", "price_basis", "TEXT NOT NULL DEFAULT 'not_provided'")
            self._add_column(connection, "products", "price_as_of", "TEXT")
            self._add_column(connection, "products", "category", "TEXT NOT NULL DEFAULT '其他商品'")
            self._add_column(connection, "catalog_assignments", "category", "TEXT NOT NULL DEFAULT '其他商品'")
            self._add_column(connection, "conversations", "access_mode", "TEXT NOT NULL DEFAULT 'ai_private'")
            self._add_column(connection, "conversations", "merchant_visible", "INTEGER NOT NULL DEFAULT 0")
            self._add_column(connection, "conversations", "handoff_summary_json", "TEXT NOT NULL DEFAULT '{}'")
            self._add_column(connection, "conversations", "purchase_id", "TEXT")
            self._add_column(connection, "conversations", "handoff_state", "TEXT NOT NULL DEFAULT 'ai_support'")
            connection.execute("UPDATE conversations SET handoff_state='awaiting_merchant' WHERE merchant_visible=1 AND handoff_state='ai_support'")
            if self.path.resolve() == DEFAULT_COMMERCE_DB.resolve():
                self._seed_demo_catalog(connection)
                self._complete_demo_catalog(connection)

    def _begin_write(self, connection) -> None:
        connection.execute('BEGIN IMMEDIATE')

    @staticmethod
    def _backfill_catalog_prices(connection: sqlite3.Connection) -> None:
        """Use historical prices only as visibly simulated reference prices."""
        if os.getenv("SERVEMIND_DATA_BACKEND") != "postgres":
            return
        pending = connection.execute(
            """SELECT id, sku_id FROM products WHERE catalog_label = '数据目录模拟归属'
               AND display_price_cents IS NULL AND sku_id IS NOT NULL"""
        ).fetchall()
        if not pending:
            return
        with psycopg.connect(os.getenv("SERVEMIND_DATABASE_URL", "postgresql:///servemind")) as postgres:
            for product in pending:
                row = postgres.execute(
                    """SELECT final_unit_price, order_date FROM msom.orders
                       WHERE sku_id = %s AND final_unit_price ~ '^[0-9]+(\\.[0-9]{1,2})?$'
                         AND final_unit_price::numeric > 0
                       ORDER BY order_date DESC, order_time DESC LIMIT 1""",
                    (product["sku_id"],),
                ).fetchone()
                if row:
                    connection.execute(
                        """UPDATE products SET display_price_cents = ?,
                           price_basis = 'historical_reference_simulated', price_as_of = ?
                           WHERE id = ?""",
                        (int(Decimal(row[0]) * 100), row[1], product["id"]),
                    )

    @staticmethod
    def _add_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _seed_demo_catalog(self, connection: sqlite3.Connection) -> None:
        """Create local-only personas and stable simulated SKU ownership.

        The CSV files remain read-only. This table is an explicit demo overlay:
        it never claims that a historical SKU has a real seller.
        """
        personas = [
            ("acct_demo_merchant_digital", "demo_merchant_digital", "数码生活馆", "merchant"),
            ("acct_demo_merchant_home", "demo_merchant_home", "安心家居店", "merchant"),
            ("acct_demo_merchant_daily", "demo_merchant_daily", "日用优选铺", "merchant"),
            ("acct_demo_buyer_lin", "demo_buyer_lin", "林女士（演示买家）", "buyer"),
        ]
        for account_id, username, display_name, role in personas:
            exists = connection.execute("SELECT 1 FROM accounts WHERE username_key = ?", (username,)).fetchone()
            if not exists:
                connection.execute(
                    "INSERT INTO accounts (id, username, username_key, display_name, role, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (account_id, username, username, display_name, role, _password_hash(secrets.token_urlsafe(32)), _now()),
                )
        merchants = [row[0] for row in personas if row[3] == "merchant"]
        sku_rows: list[dict[str, str]] = []
        if os.getenv("SERVEMIND_DATA_BACKEND") == "postgres":
            with psycopg.connect(os.getenv("SERVEMIND_DATABASE_URL", "postgresql:///servemind"),
                                 row_factory=dict_row) as postgres:
                sku_rows = [dict(row) for row in postgres.execute(
                    'SELECT sku_id AS "sku_ID", brand_id AS "brand_ID" FROM msom.skus ORDER BY sku_id'
                ).fetchall()]
        else:
            sku_path = DATA_ROOT / "JD_sku_data.csv"
            if sku_path.exists():
                with sku_path.open(newline="", encoding="utf-8") as handle:
                    sku_rows = list(csv.DictReader(handle))
        statuses = ("在售", "库存紧张", "预售", "暂时缺货", "已下架")
        existing_skus = {row[0] for row in connection.execute("SELECT sku_id FROM catalog_assignments").fetchall()}
        for index, row in enumerate(sku_rows):
            sku_id = row.get("sku_ID")
            if not sku_id or sku_id in existing_skus:
                continue
            merchant_id = merchants[index % len(merchants)]
            status = statuses[index % len(statuses)]
            brand = row.get("brand_ID") or "综合品牌"
            category = DEMO_CATEGORIES[index % len(DEMO_CATEGORIES)]
            connection.execute(
                "INSERT INTO catalog_assignments (sku_id, merchant_id, display_title, category, catalog_status, assignment_type, source_version, created_at) VALUES (?, ?, ?, ?, ?, 'simulated', ?, ?)",
                (sku_id, merchant_id, f"{category}{index // len(DEMO_CATEGORIES) + 1}", category, status, DATA_VERSION, _now()),
            )
            existing_skus.add(sku_id)
        # Bind the current buyer account to one historical order line. The
        # alias is intentionally human-readable; raw IDs stay internal.
        freefoot = connection.execute("SELECT id FROM accounts WHERE username_key = 'freefoot'").fetchone()
        if freefoot:
            row = connection.execute("SELECT 1 FROM demo_purchases WHERE buyer_id = ?", (freefoot[0],)).fetchone()
            if not row:
                order_id, sku_id = "81a6fa818d", "ac61f4e10e"
                assignment = connection.execute("SELECT merchant_id FROM catalog_assignments WHERE sku_id = ?", (sku_id,)).fetchone()
                if assignment:
                    connection.execute(
                        "INSERT OR IGNORE INTO demo_purchases (id, buyer_id, sku_id, order_id, merchant_id, order_alias, fulfillment_status, source_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (_new_id("purchase"), freefoot[0], sku_id, order_id, assignment[0], "我的订单 · 001", "已签收", DATA_VERSION, _now()),
                    )
                    self._ensure_catalog_product(connection, sku_id, assignment[0])
        buyer = connection.execute("SELECT id FROM accounts WHERE username_key = 'demo_buyer_lin'").fetchone()
        if buyer:
            row = connection.execute("SELECT 1 FROM demo_purchases WHERE buyer_id = ?", (buyer[0],)).fetchone()
            if not row:
                assignment = connection.execute("SELECT sku_id, merchant_id FROM catalog_assignments ORDER BY sku_id LIMIT 1").fetchone()
                if assignment:
                    connection.execute(
                        "INSERT OR IGNORE INTO demo_purchases (id, buyer_id, sku_id, order_id, merchant_id, order_alias, fulfillment_status, source_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (_new_id("purchase"), buyer[0], assignment[0], "demo-order-001", assignment[1], "演示订单 · 001", "运输中", DATA_VERSION, _now()),
                    )
                    self._ensure_catalog_product(connection, assignment[0], assignment[1])

    @staticmethod
    def _ensure_catalog_product(connection: sqlite3.Connection, sku_id: str, merchant_id: str) -> None:
        existing = connection.execute("SELECT 1 FROM products WHERE sku_id = ?", (sku_id,)).fetchone()
        if existing:
            return
        assignment = connection.execute("SELECT display_title, catalog_status, category FROM catalog_assignments WHERE sku_id = ?", (sku_id,)).fetchone()
        if not assignment:
            return
        connection.execute(
            "INSERT INTO products (id, merchant_id, title, description, sku_id, provenance, catalog_status, catalog_label, category, created_at) VALUES (?, ?, ?, ?, ?, 'merchant_declared_unverified', ?, '数据目录模拟归属', ?, ?)",
            (_new_id("prod"), merchant_id, assignment[0], "商品信息以店铺页面为准。", sku_id, assignment[1], assignment[2], _now()),
        )

    @staticmethod
    def _complete_demo_catalog(connection: sqlite3.Connection) -> None:
        """Materialize every anonymous SKU as a clearly simulated, priced listing.

        Source SKU attributes do not encode a reliable retail category.  Category,
        title and fallback price are demo metadata, never inferred source facts.
        """
        assignments = connection.execute(
            "SELECT sku_id, merchant_id, catalog_status FROM catalog_assignments ORDER BY sku_id"
        ).fetchall()
        if not assignments:
            return
        existing = {row[0] for row in connection.execute("SELECT sku_id FROM products WHERE sku_id IS NOT NULL")}
        now = _now()
        for index, assignment in enumerate(assignments):
            category_index = index % len(DEMO_CATEGORIES)
            category = DEMO_CATEGORIES[category_index]
            title = f"{category}{index // len(DEMO_CATEGORIES) + 1}"
            connection.execute(
                "UPDATE catalog_assignments SET display_title = ?, category = ? WHERE sku_id = ?",
                (title, category, assignment["sku_id"]),
            )
            if assignment["sku_id"] not in existing:
                connection.execute(
                    """INSERT INTO products
                       (id, merchant_id, title, description, sku_id, provenance,
                        catalog_status, catalog_label, category, display_price_cents,
                        price_basis, created_at)
                       VALUES (?, ?, ?, ?, ?, 'merchant_declared_unverified', ?,
                               '数据目录模拟归属', ?, ?, 'simulated_catalog_price', ?)""",
                    (_new_id("prod"), assignment["merchant_id"], title,
                     "商品详情请向商家咨询。", assignment["sku_id"],
                     assignment["catalog_status"], category,
                     DEFAULT_DEMO_PRICES[category_index], now),
                )
        connection.execute(
            """UPDATE products SET title = (SELECT display_title FROM catalog_assignments
                 WHERE sku_id = products.sku_id),
                 category = (SELECT category FROM catalog_assignments WHERE sku_id = products.sku_id)
               WHERE catalog_label = '数据目录模拟归属' AND sku_id IS NOT NULL"""
        )
        connection.execute(
            """UPDATE products SET category = '其他商品'
               WHERE category IS NULL OR trim(category) = ''"""
        )
        connection.execute(
            """UPDATE products SET display_price_cents = 5990,
                      price_basis = 'simulated_catalog_price', price_as_of = NULL
               WHERE display_price_cents IS NULL OR display_price_cents <= 0"""
        )
        connection.execute(
            """UPDATE products SET price_basis = 'simulated_catalog_price', price_as_of = NULL
               WHERE catalog_label = '数据目录模拟归属'
                 AND price_basis = 'historical_reference_simulated'"""
        )

    @staticmethod
    def _account_public(row: sqlite3.Row) -> dict[str, str]:
        return {key: row[key] for key in ("id", "username", "display_name", "role", "created_at")}

    @staticmethod
    def _participant_public(row: sqlite3.Row) -> dict[str, str]:
        # Conversation peers need a stable ID and display name, not a login identifier.
        return {key: row[key] for key in ("id", "display_name", "role")}

    @staticmethod
    def _message_public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "sender_type": row["sender_type"],
            "sender_id": row["sender_id"],
            "content": row["content"],
            "client_message_id": row["client_message_id"],
            "reply_to_id": row["reply_to_id"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def _issue_token(self, connection: sqlite3.Connection, account_id: str) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(timespec="microseconds")
        connection.execute(
            "INSERT INTO tokens (token_hash, account_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token_hash, account_id, _now(), expires_at),
        )
        return token

    def register(self, *, username: str, password: str, display_name: str, role: str) -> dict[str, Any]:
        username = username.strip()
        display_name = display_name.strip()
        account_id = _new_id("acct")
        with self._db() as connection, connection:
            try:
                connection.execute(
                    """INSERT INTO accounts
                    (id, username, username_key, display_name, role, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (account_id, username, username.casefold(), display_name, role, _password_hash(password), _now()),
                )
            except (sqlite3.IntegrityError, psycopg.errors.UniqueViolation) as exc:
                raise HTTPException(status_code=409, detail="username_already_exists") from exc
            token = self._issue_token(connection, account_id)
            row = connection.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return {"token": token, "account": self._account_public(row)}

    def login(self, *, username: str, password: str) -> dict[str, Any]:
        with self._db() as connection, connection:
            row = connection.execute("SELECT * FROM accounts WHERE username_key = ?", (username.strip().casefold(),)).fetchone()
            if row is None or not _password_matches(password, row["password_hash"]):
                raise HTTPException(status_code=401, detail="invalid_credentials")
            token = self._issue_token(connection, row["id"])
        return {"token": token, "account": self._account_public(row)}

    def list_demo_accounts(self) -> list[dict[str, str]]:
        """Public labels only; access is gated by the local-only API route."""
        with self._db() as connection:
            rows = connection.execute(
                "SELECT id, display_name, role FROM accounts WHERE role IN ('buyer', 'merchant') "
                "ORDER BY CASE role WHEN 'buyer' THEN 0 ELSE 1 END, created_at, id"
            ).fetchall()
        return [{"id": row["id"], "display_name": row["display_name"],
                 "role": row["role"]} for row in rows]

    def select_demo_account(self, account_id: str) -> dict[str, Any]:
        with self._db() as connection, connection:
            row = connection.execute(
                "SELECT * FROM accounts WHERE id = ? AND role IN ('buyer', 'merchant')",
                (account_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="demo_account_not_found")
            token = self._issue_token(connection, row["id"])
        return {"token": token, "account": self._account_public(row)}

    def authenticate(self, token: str) -> dict[str, str] | None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._db() as connection:
            row = connection.execute(
                """SELECT a.* FROM tokens t JOIN accounts a ON a.id = t.account_id
                WHERE t.token_hash = ? AND t.expires_at > ?""",
                (token_hash, _now()),
            ).fetchone()
        return self._account_public(row) if row else None

    def _product(self, connection: sqlite3.Connection, product_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            """SELECT p.*, a.display_name AS merchant_display_name
            FROM products p JOIN accounts a ON a.id = p.merchant_id WHERE p.id = ?""",
            (product_id,),
        ).fetchone()
        return self._product_public(row) if row else None

    @staticmethod
    def _product_public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "merchant_id": row["merchant_id"],
            "merchant_display_name": row["merchant_display_name"],
            "merchant": {
                "id": row["merchant_id"],
                "display_name": row["merchant_display_name"],
                "role": "merchant",
            },
            "title": row["title"],
            "category": row["category"],
            "description": row["description"],
            "sku_id": row["sku_id"],
            "provenance": row["provenance"],
            "verification_status": "unverified",
            "catalog_status": row["catalog_status"] if "catalog_status" in row.keys() else "available",
            "catalog_label": row["catalog_label"] if "catalog_label" in row.keys() else "演示商品",
            "display_price": (f"{row['display_price_cents'] // 100}.{row['display_price_cents'] % 100:02d}"
                              if row["display_price_cents"] is not None else None),
            "price_basis": row["price_basis"],
            "price_as_of": row["price_as_of"],
            "created_at": row["created_at"],
        }

    def list_products(self) -> list[dict[str, Any]]:
        with self._db() as connection:
            rows = connection.execute(
                """SELECT p.*, a.display_name AS merchant_display_name
                FROM products p JOIN accounts a ON a.id = p.merchant_id
                ORDER BY CASE WHEN p.sku_id = 'ac61f4e10e' THEN 0 WHEN p.catalog_label = '数据目录模拟归属' THEN 1 ELSE 0 END,
                p.created_at DESC LIMIT 121"""
            ).fetchall()
        return [self._product_public(row) for row in rows]

    def list_purchases(self, *, account: dict[str, str]) -> list[dict[str, Any]]:
        with self._db() as connection:
            rows = connection.execute(
                """SELECT p.*, a.display_name AS merchant_display_name, cp.title, cp.description,
                cp.catalog_status, cp.catalog_label, cp.display_price_cents,
                cp.price_basis, cp.price_as_of
                FROM demo_purchases p JOIN accounts a ON a.id = p.merchant_id
                LEFT JOIN products cp ON cp.sku_id = p.sku_id AND cp.merchant_id = p.merchant_id
                WHERE p.buyer_id = ? ORDER BY p.created_at DESC""", (account["id"],)
            ).fetchall()
        return [{
            "id": row["id"], "order_alias": row["order_alias"], "sku_id": row["sku_id"],
            "order_id": row["order_id"], "fulfillment_status": row["fulfillment_status"],
            "merchant": {"id": row["merchant_id"], "display_name": row["merchant_display_name"], "role": "merchant"},
            "product": {"title": row["title"] or f"商品 · {row['sku_id'][:6]}", "description": row["description"] or "商品详情请向商家咨询。", "catalog_status": row["catalog_status"] or "在售", "catalog_label": row["catalog_label"] or "数据目录模拟归属", "sku_id": row["sku_id"],
                        "display_price": (f"{row['display_price_cents'] // 100}.{row['display_price_cents'] % 100:02d}" if row["display_price_cents"] is not None else None),
                        "price_basis": row["price_basis"]},
            "simulated": True, "source_version": row["source_version"], "created_at": row["created_at"],
        } for row in rows]

    def create_product(
        self, *, account: dict[str, str], title: str, description: str, sku_id: str | None,
        price: Decimal | None = None, category: str = "其他商品",
    ) -> dict[str, Any]:
        if account["role"] != "merchant":
            raise HTTPException(status_code=403, detail="merchant_role_required")
        product_id = _new_id("prod")
        with self._db() as connection, connection:
            connection.execute(
                """INSERT INTO products
                   (id, merchant_id, title, description, sku_id, provenance,
                    catalog_status, catalog_label, category, display_price_cents, price_basis, created_at)
                   VALUES (?, ?, ?, ?, ?, 'merchant_declared_unverified', 'available',
                           '商家自声明', ?, ?, ?, ?)""",
                (product_id, account["id"], title.strip(), description.strip(),
                 sku_id.strip() if sku_id else None,
                 category, int(price * 100) if price is not None else 5990,
                 "merchant_declared" if price is not None else "simulated_catalog_price", _now()),
            )
            product = self._product(connection, product_id)
        return product

    def _conversation(
        self, connection: sqlite3.Connection, conversation_id: str, account: dict[str, str], *, include_messages: bool
    ) -> dict[str, Any]:
        row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None or account["id"] not in (row["buyer_id"], row["merchant_id"]):
            raise HTTPException(status_code=404, detail="conversation_not_found")
        if account["id"] == row["merchant_id"] and not row["merchant_visible"]:
            raise HTTPException(status_code=404, detail="conversation_not_found")
        buyer = connection.execute("SELECT * FROM accounts WHERE id = ?", (row["buyer_id"],)).fetchone()
        merchant = connection.execute("SELECT * FROM accounts WHERE id = ?", (row["merchant_id"],)).fetchone()
        result: dict[str, Any] = {
            "id": row["id"],
            "product": self._product(connection, row["product_id"]),
            "buyer": self._participant_public(buyer),
            "merchant": self._participant_public(merchant),
            "status": row["status"],
            "access_mode": row["access_mode"],
            "merchant_visible": bool(row["merchant_visible"]),
            "handoff_summary": json.loads(row["handoff_summary_json"] or "{}"),
            "handoff_state": row["handoff_state"],
            "purchase_id": row["purchase_id"],
            "created_at": row["created_at"],
        }
        latest = connection.execute(
            "SELECT created_at FROM messages WHERE conversation_id = ? ORDER BY sequence DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        result["updated_at"] = latest["created_at"] if latest else row["created_at"]
        if include_messages:
            messages = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence", (conversation_id,)
            ).fetchall()
            result["messages"] = [self._message_public(message) for message in messages]
        return result

    def create_conversation(self, *, account: dict[str, str], product_id: str) -> dict[str, Any]:
        if account["role"] != "buyer":
            raise HTTPException(status_code=403, detail="buyer_role_required")
        conversation_id = _new_id("conv")
        with self._db() as connection, connection:
            product = self._product(connection, product_id)
            if product is None:
                raise HTTPException(status_code=404, detail="product_not_found")
            purchase = connection.execute(
                """SELECT * FROM demo_purchases WHERE buyer_id = ? AND sku_id = ? AND merchant_id = ?
                ORDER BY created_at DESC LIMIT 1""", (account["id"], product.get("sku_id"), product["merchant_id"])
            ).fetchone() if product.get("sku_id") else None
            connection.execute(
                """INSERT INTO conversations (id, product_id, buyer_id, merchant_id, status, access_mode, merchant_visible, handoff_summary_json, purchase_id, created_at)
                VALUES (?, ?, ?, ?, 'open', 'ai_private', 0, '{}', ?, ?)""",
                (conversation_id, product_id, account["id"], product["merchant_id"], purchase["id"] if purchase else None, _now()),
            )
            return self._conversation(connection, conversation_id, account, include_messages=True)

    def get_conversation(self, *, account: dict[str, str], conversation_id: str) -> dict[str, Any]:
        with self._db() as connection:
            return self._conversation(connection, conversation_id, account, include_messages=True)

    def list_conversations(self, *, account: dict[str, str]) -> list[dict[str, Any]]:
        with self._db() as connection:
            rows = connection.execute(
                """SELECT c.id FROM conversations c WHERE c.buyer_id = ? OR (c.merchant_id = ? AND c.merchant_visible = 1)
                ORDER BY COALESCE(
                    (SELECT MAX(m.sequence) FROM messages m WHERE m.conversation_id = c.id), 0
                ) DESC, c.created_at DESC""",
                (account["id"], account["id"]),
            ).fetchall()
            return [self._conversation(connection, row["id"], account, include_messages=False) for row in rows]

    def send_message(
        self,
        *,
        account: dict[str, str],
        conversation_id: str,
        content: str,
        client_message_id: str | None,
        support: Any,
    ) -> dict[str, Any]:
        try:
            with self.conversation_lock.hold(conversation_id):
                return self._send_message_locked(
                    account=account, conversation_id=conversation_id, content=content,
                    client_message_id=client_message_id, support=support,
                )
        except TimeoutError as exc:
            if str(exc) == 'agent_request_deadline':
                raise HTTPException(status_code=504, detail='agent_request_timeout') from exc
            raise HTTPException(status_code=409, detail="conversation_busy") from exc

    def _persist_in_transaction(self, connection, support, **turn) -> bool:
        return False  # SQLite fixtures cannot share a PostgreSQL transaction.

    def _send_message_locked(
        self, *, account: dict[str, str], conversation_id: str, content: str,
        client_message_id: str | None, support: Any,
    ) -> dict[str, Any]:
        # Commit the buyer message before invoking any remote model. Holding a
        # SQLite write transaction across provider latency would block unrelated
        # buyers and merchants, and make timeout recovery unsafe.
        with self._db() as connection, connection:
            self._begin_write(connection)
            conversation = self._conversation(connection, conversation_id, account, include_messages=False)
            if conversation["status"] != "open":
                raise HTTPException(status_code=409, detail="conversation_closed")
            prior = None
            if client_message_id:
                prior = connection.execute(
                    """SELECT * FROM messages WHERE conversation_id = ? AND sender_id = ?
                    AND client_message_id = ?""",
                    (conversation_id, account["id"], client_message_id),
                ).fetchone()
                if prior:
                    if prior["content"] != content:
                        raise HTTPException(status_code=409, detail="client_message_id_conflict")
                    ai = connection.execute("SELECT * FROM messages WHERE reply_to_id = ?", (prior["id"],)).fetchone()
                    if ai or account["role"] == "merchant":
                        return {"message": self._message_public(prior),
                                "ai_message": self._message_public(ai) if ai else None}
            sender_type = account["role"]
            if sender_type == "merchant" and not conversation["merchant_visible"]:
                raise HTTPException(status_code=404, detail="conversation_not_found")
            message_id = prior["id"] if prior else _new_id("msg")
            if prior is None:
                connection.execute(
                    """INSERT INTO messages
                    (id, conversation_id, sender_type, sender_id, content, client_message_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (message_id, conversation_id, sender_type, account["id"], content, client_message_id, _now()),
                )
            row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
            if sender_type == "merchant":
                self._set_handoff_state(connection, conversation_id, "merchant_replied", account["id"])
                connection.execute("UPDATE messages SET metadata_json=? WHERE id=?",
                                   (json.dumps(support.review_merchant(content), ensure_ascii=False), message_id))
                row = connection.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
                public_row = self._message_public(row)
                memory_committed = self._persist_in_transaction(connection,support,conversation=conversation,
                    message_id=message_id,intent='merchant_reply',topics=['merchant_reply'],
                    evidence_ids=[],needs_merchant=False)
            else:
                purchase = None
                if conversation.get("purchase_id"):
                    purchase_row = connection.execute(
                        "SELECT * FROM demo_purchases WHERE id = ?", (conversation["purchase_id"],)
                    ).fetchone()
                    purchase = dict(purchase_row) if purchase_row else None
                public_row = self._message_public(row)
                recent_rows = connection.execute("SELECT sender_type, content, metadata_json FROM messages WHERE conversation_id=? AND id!=? ORDER BY sequence DESC LIMIT 8",
                                                 (conversation_id, message_id)).fetchall()
                recent_messages = [{"sender_type": r["sender_type"], "content": r['content'], "metadata": json.loads(r["metadata_json"] or "{}")} for r in reversed(recent_rows)]
        if sender_type == "merchant":
            if not memory_committed:
                support.persist_turn(
                    conversation=conversation, message_id=message_id,
                    intent="merchant_reply", topics=["merchant_reply"],
                    evidence_ids=[], needs_merchant=False,
                )
            return {"message": public_row, "ai_message": None}
        ai_answer, metadata = support.reply(content, conversation["product"], purchase=purchase,
                                            conversation_id=conversation_id,
                                            buyer_id=conversation["buyer"]["id"],
                                            merchant_id=conversation["merchant"]["id"], recent_messages=recent_messages,
                                            defer_working_memory=True, handoff_state=conversation.get('handoff_state'))
        prepared = metadata.pop('_memory_prepared',None)
        with self._db() as connection, connection:
            self._begin_write(connection)
            connection.execute(
                """INSERT INTO messages
                   (id, conversation_id, sender_type, sender_id, content, reply_to_id, metadata_json, created_at)
                   VALUES (?, ?, 'ai', NULL, ?, ?, ?, ?)
                   ON CONFLICT(reply_to_id) DO NOTHING""",
                (_new_id("msg"), conversation_id, ai_answer, message_id,
                 json.dumps(metadata, ensure_ascii=False), _now()),
            )
            ai_row = connection.execute("SELECT * FROM messages WHERE reply_to_id = ?", (message_id,)).fetchone()
            effective_metadata = json.loads(ai_row["metadata_json"] or "{}")
            if effective_metadata.get("needs_merchant") and conversation.get('handoff_state') not in {'merchant_processing','merchant_replied'}:
                self._set_handoff_state(connection, conversation_id, "awaiting_merchant", account["id"],
                                        request_id=effective_metadata.get("request_id"),
                                        summary=effective_metadata.get("handoff_summary") or {})
            result = {"message": public_row, "ai_message": self._message_public(ai_row)}
            memory_turn = dict(conversation=conversation, message_id=message_id,
            intent=effective_metadata.get("intent", "other"),
            topics=effective_metadata.get("topics") or [],
            evidence_ids=effective_metadata.get("evidence_ids") or [],
            needs_merchant=bool(effective_metadata.get("needs_merchant")),
            summary=effective_metadata.get("memory_summary", ""),
            resolved_topics=effective_metadata.get("resolved_topics"),
            pending_topics=effective_metadata.get("pending_topics"),
            response_style=effective_metadata.get("requested_response_style"),
            prepared=prepared,
            )
            memory_committed = self._persist_in_transaction(connection,support,**memory_turn)
        if not memory_committed:
            support.persist_turn(**memory_turn)
        support.commit_working_memory(conversation,effective_metadata)
        return result

    def _set_handoff_state(self, connection: sqlite3.Connection, conversation_id: str,
                           state: str, actor_id: str, *, request_id: str | None = None,
                           summary: dict | None = None) -> None:
        summary_json = json.dumps(summary or {}, ensure_ascii=False)
        connection.execute("INSERT INTO handoff_events (id,conversation_id,request_id,state,actor_id,summary_json,created_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                           (_new_id("handoff"), conversation_id, request_id, state, actor_id, summary_json, _now()))
        connection.execute("UPDATE conversations SET handoff_state = ? WHERE id = ?", (state, conversation_id))
        if state == "awaiting_merchant":
            connection.execute("UPDATE conversations SET access_mode='merchant_shared', merchant_visible=1, handoff_summary_json=? WHERE id=?",
                               (summary_json, conversation_id))

    def update_handoff(self, *, account: dict, conversation_id: str, state: str) -> dict:
        with self.conversation_lock.hold(conversation_id):
            with self._db() as connection, connection:
                conversation = self._conversation(connection, conversation_id, account, include_messages=False)
                if account["role"] != "merchant" or not conversation["merchant_visible"]:
                    raise HTTPException(403, "merchant_role_required")
                allowed = {"awaiting_merchant": {"merchant_processing", "resolved"},
                           "merchant_processing": {"resolved"}, "merchant_replied": {"merchant_processing", "resolved"}}
                if state != conversation["handoff_state"] and state not in allowed.get(conversation["handoff_state"], set()):
                    raise HTTPException(409, "invalid_handoff_transition")
                if state != conversation["handoff_state"]:
                    self._set_handoff_state(connection, conversation_id, state, account["id"])
                return self._conversation(connection, conversation_id, account, include_messages=True)

    def record_feedback(self, *, account: dict, conversation_id: str, message_id: str,
                        rating: str, note: str = "") -> dict:
        with self._db() as connection, connection:
            self._conversation(connection, conversation_id, account, include_messages=False)
            row = connection.execute("SELECT id FROM messages WHERE id=? AND conversation_id=? AND sender_type='ai'",
                                     (message_id, conversation_id)).fetchone()
            if not row:
                raise HTTPException(404, "ai_message_not_found")
            connection.execute("INSERT INTO message_feedback VALUES (?, ?, ?, ?, ?) ON CONFLICT(message_id,account_id) DO UPDATE SET rating=excluded.rating,note=excluded.note,created_at=excluded.created_at",
                               (message_id, account["id"], rating, note[:500], _now()))
        return {"saved": True, "rating": rating}

    def diagnostics(self) -> dict:
        from servemind.monitor.performance_monitor import CommerceMonitor
        monitor = CommerceMonitor()
        with self._db() as connection:
            rows = connection.execute("SELECT metadata_json FROM messages WHERE sender_type='ai' ORDER BY sequence DESC LIMIT 2000").fetchall()
            feedback = [dict(row) for row in connection.execute("SELECT rating, count(*) AS count FROM message_feedback GROUP BY rating")]
            handoffs = [dict(row) for row in connection.execute("SELECT handoff_state AS state, count(*) AS count FROM conversations GROUP BY handoff_state")]
        for row in reversed(rows):
            metadata = json.loads(row["metadata_json"] or "{}")
            if all(key in metadata for key in ("grounding", "tools_used", "latency_ms", "request_id")):
                monitor.record_commerce(metadata=metadata, latency_ms=metadata["latency_ms"])
        result = monitor.summary()
        result["window"] = "last_2000_persisted_ai_messages"
        return {**result, "feedback": feedback, "handoff_states": handoffs}
