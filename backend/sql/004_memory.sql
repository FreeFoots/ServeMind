CREATE SCHEMA IF NOT EXISTS memory;

CREATE TABLE IF NOT EXISTS memory.sessions (
    conversation_id text PRIMARY KEY,
    buyer_id text NOT NULL,
    merchant_id text NOT NULL,
    product_id text NOT NULL,
    summary text NOT NULL DEFAULT '',
    turn_count integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory.episodes (
    message_id text PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES memory.sessions(conversation_id),
    intent text NOT NULL,
    topics text[] NOT NULL DEFAULT '{}',
    evidence_ids text[] NOT NULL DEFAULT '{}',
    needs_merchant boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memory_episodes_conversation
    ON memory.episodes (conversation_id, created_at DESC);
ALTER TABLE memory.sessions ADD COLUMN IF NOT EXISTS resolved_topics text[] NOT NULL DEFAULT '{}';
ALTER TABLE memory.sessions ADD COLUMN IF NOT EXISTS pending_topics text[] NOT NULL DEFAULT '{}';
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS summary text NOT NULL DEFAULT '';
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS resolved_topics text[] NOT NULL DEFAULT '{}';
ALTER TABLE memory.episodes ADD COLUMN IF NOT EXISTS pending_topics text[] NOT NULL DEFAULT '{}';

-- A profile is scoped to one buyer and one merchant. There is deliberately no
-- cross-merchant buyer profile or copied historical anonymous-user profile.
CREATE TABLE IF NOT EXISTS memory.profiles (
    buyer_id text NOT NULL,
    merchant_id text NOT NULL,
    last_product_id text NOT NULL,
    last_intent text NOT NULL,
    interaction_count integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (buyer_id, merchant_id)
);
ALTER TABLE memory.profiles ADD COLUMN IF NOT EXISTS response_style text NOT NULL DEFAULT 'standard';
