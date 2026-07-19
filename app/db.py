import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL, EMBEDDING_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repos (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    chunk_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    repo_id BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    start_line INT NOT NULL,
    end_line INT NOT NULL,
    symbol TEXT,
    language TEXT,
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);

-- HNSW gives good approximate-nearest-neighbor recall without needing to
-- pick a training-set size up front (unlike ivfflat, which wants to be
-- built AFTER the table has representative data). For portfolio-scale
-- corpora the difference is small, but HNSW is the safer default.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_conn() -> psycopg.Connection:
    # prepare_threshold=None disables psycopg3's automatic prepared
    # statements. Required when the DATABASE_URL points at a transaction-mode
    # connection pooler (e.g. Neon's -pooler endpoint / PgBouncer): a pooled
    # connection can be handed to a different backend between statements, so
    # a prepared statement created on one won't exist on the next, throwing
    # "prepared statement does not exist". Harmless against a direct
    # connection too, so it's safe to leave on unconditionally.
    conn = psycopg.connect(DATABASE_URL, prepare_threshold=None)
    register_vector(conn)
    return conn


def init_schema() -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(SCHEMA)
