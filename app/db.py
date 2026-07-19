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
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_schema() -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(SCHEMA)
