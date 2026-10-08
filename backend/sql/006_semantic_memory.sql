CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS embedding vector(1024);
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS embedding_model text;
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS summary_source text NOT NULL DEFAULT 'structured_topics';
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS summary_version integer NOT NULL DEFAULT 1;
ALTER TABLE memory.sessions ADD COLUMN IF NOT EXISTS summary_source text NOT NULL DEFAULT 'structured_topics';
CREATE INDEX IF NOT EXISTS idx_memory_sessions_scope
    ON memory.sessions(buyer_id,merchant_id,product_id,updated_at DESC);
-- Exact distance over the scoped, retained candidate set. No approximate index
-- that can lose tenant-filtered neighbors; add one only after scoped benchmarks.
