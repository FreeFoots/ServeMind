CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE SCHEMA IF NOT EXISTS knowledge;

CREATE TABLE IF NOT EXISTS knowledge.chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL,
    title text NOT NULL,
    display_title text NOT NULL DEFAULT '',
    source text NOT NULL,
    content text NOT NULL,
    source_sha256 text NOT NULL,
    policy_version text NOT NULL,
    document_version text NOT NULL DEFAULT '1.0.0',
    valid_from date NOT NULL DEFAULT DATE '2026-09-23',
    valid_until date,
    embedding_model text NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE knowledge.chunks ADD COLUMN IF NOT EXISTS document_version text NOT NULL DEFAULT '1.0.0';
ALTER TABLE knowledge.chunks ADD COLUMN IF NOT EXISTS display_title text NOT NULL DEFAULT '';
ALTER TABLE knowledge.chunks ADD COLUMN IF NOT EXISTS valid_from date NOT NULL DEFAULT DATE '2026-09-23';
ALTER TABLE knowledge.chunks ADD COLUMN IF NOT EXISTS valid_until date;
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document ON knowledge.chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_validity
    ON knowledge.chunks (policy_version, valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_text_trgm
    ON knowledge.chunks USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_vector
    ON knowledge.chunks USING hnsw (embedding vector_cosine_ops);
