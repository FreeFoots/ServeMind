"""Index project-local policy Markdown with Qwen3 1024-dimensional embeddings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import psycopg

from servemind.config.policy_registry import POLICY_VERSION
from servemind.config.settings import KNOWLEDGE_ROOT
from servemind.core.vector_knowledge import EMBEDDING_MODEL, encode_document, vector_literal

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "002_knowledge.sql"
DSN = os.getenv("SERVEMIND_DATABASE_URL", "postgresql:///servemind")


def chunks(text: str, limit: int = 800, overlap: int = 80) -> list[str]:
    if overlap >= limit:
        raise ValueError("overlap must be smaller than chunk limit")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            if current:
                result.append(current)
                current = ""
            step = limit - overlap
            for start in range(0, len(paragraph), step):
                part = paragraph[start:start + limit]
                if part:
                    result.append(part)
                if start + limit >= len(paragraph):
                    break
            continue
        if current and len(current) + len(paragraph) + 2 > limit:
            result.append(current)
            current = current[-overlap:] if len(paragraph) + overlap + 2 <= limit else ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        result.append(current)
    return result


def main() -> None:
    manifest = json.loads((KNOWLEDGE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    paths = [path for path in sorted(KNOWLEDGE_ROOT.glob("*.md")) if path.name.lower() != "readme.md"]
    if set(manifest) != {path.stem for path in paths}:
        raise ValueError("knowledge manifest and Markdown files differ")
    with psycopg.connect(DSN) as connection:
        connection.execute(SQL_PATH.read_text(encoding="utf-8"))
        for path in paths:
            content = path.read_text(encoding="utf-8")
            display_title = next((line.removeprefix("# ").strip() for line in content.splitlines()
                                  if line.startswith("# ")), path.stem)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            metadata = manifest[path.stem]
            if not metadata.get("version") or not metadata.get("valid_from"):
                raise ValueError(f"missing version or validity: {path.name}")
            if metadata.get("valid_until") and metadata["valid_until"] < metadata["valid_from"]:
                raise ValueError(f"invalid rule validity: {path.name}")
            existing = connection.execute(
                """SELECT DISTINCT source_sha256, document_version, valid_from::text,
                          valid_until::text, policy_version, embedding_model, display_title
                   FROM knowledge.chunks WHERE document_id = %s""", (path.stem,),
            ).fetchone()
            expected = (digest, metadata["version"], metadata["valid_from"],
                        metadata.get("valid_until"), POLICY_VERSION, EMBEDDING_MODEL,
                        display_title)
            if existing and tuple(existing) == expected:
                print(f"{path.name}: unchanged", flush=True)
                continue
            if existing:
                # Delete one document's stale index records only after new vectors
                # have been computed; the transaction is rolled back on failure.
                connection.execute("DELETE FROM knowledge.chunks WHERE document_id = %s", (path.stem,))
            parts = chunks(content)
            for index, part in enumerate(parts):
                vector = vector_literal(encode_document(part))
                connection.execute(
                    """INSERT INTO knowledge.chunks
                       (chunk_id, document_id, title, display_title, source, content, source_sha256,
                        policy_version, document_version, valid_from, valid_until,
                        embedding_model, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::date,%s::date,%s,%s::vector)""",
                    (f"{path.stem}:{index}", path.stem, path.stem, display_title,
                     f"knowledge/{path.name}",
                     part, digest, POLICY_VERSION, metadata["version"],
                     metadata["valid_from"], metadata.get("valid_until"), EMBEDDING_MODEL, vector),
                )
            connection.commit()
            print(f"{path.name}: indexed {len(parts)} chunks", flush=True)


if __name__ == "__main__":
    main()
