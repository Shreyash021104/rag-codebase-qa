"""Schema test — verifies the pgvector extension loads and the tables
create cleanly against a real Postgres. Runs in CI against a
pgvector/pgvector service container. Skipped locally if no DATABASE_URL is
set, so `pytest` works offline without a database.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set (needs a Postgres+pgvector instance)",
)


def test_schema_init_and_pgvector_available():
    from app.db import get_conn, init_schema

    init_schema()
    conn = get_conn()
    try:
        ext = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert ext is not None, "pgvector extension not installed"

        # Both tables should exist after init.
        for table in ("repos", "chunks"):
            exists = conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL", (table,)
            ).fetchone()[0]
            assert exists, f"table {table} missing"
    finally:
        conn.close()
