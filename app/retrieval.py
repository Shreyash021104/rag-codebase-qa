"""Hybrid retrieval: vector similarity + BM25 keyword search, fused.

Embeddings are good at "what code does X" (semantic paraphrase) and bad at
exact identifiers — a query for `markPresent` embeds similarly to lots of
presence-related code, not specifically the function with that name. BM25 is
the mirror image: exact-token matching nails identifiers and misses
paraphrase. Fusing both (reciprocal rank fusion) gets the union of their
strengths without having to tune score-normalization between two totally
different scoring scales — RRF only looks at each result's RANK in each
list, never its raw score.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import BM25_TOP_K, FUSED_TOP_K, VECTOR_TOP_K
from .db import get_conn
from .embeddings import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: int
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None
    content: str
    score: float


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Code-aware tokenization for BM25.

    `markPresent` becomes ["markpresent", "mark", "present"] — the whole
    identifier AND its camelCase/snake_case parts. A query mentioning just
    "present" can then still match the identifier, while an exact-identifier
    query gets the strongest possible match on the full token.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lower = raw.lower()
        tokens.append(lower)
        parts = [p.lower() for p in _CAMEL_RE.split(raw) if p]
        subparts: list[str] = []
        for part in parts:
            subparts.extend(s for s in part.split("_") if s)
        if len(subparts) > 1:
            tokens.extend(subparts)
    return tokens


# BM25 index cache, keyed by repo. Invalidated by indexed_at so re-indexing
# a repo naturally rebuilds it. Fine for a single-process deployment; a
# multi-instance deployment would move this into the DB (Postgres full-text
# search) — noted in the README's scaling section.
_bm25_cache: dict[int, tuple[str, BM25Okapi, list[int]]] = {}


def _bm25_for_repo(conn, repo_id: int) -> tuple[BM25Okapi, list[int]] | None:
    row = conn.execute(
        "SELECT indexed_at::text FROM repos WHERE id = %s", (repo_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    version = row[0]

    cached = _bm25_cache.get(repo_id)
    if cached and cached[0] == version:
        return cached[1], cached[2]

    rows = conn.execute(
        "SELECT id, file_path, symbol, content FROM chunks WHERE repo_id = %s ORDER BY id",
        (repo_id,),
    ).fetchall()
    if not rows:
        return None
    corpus = [tokenize(f"{r[1]} {r[2] or ''} {r[3]}") for r in rows]
    bm25 = BM25Okapi(corpus)
    chunk_ids = [r[0] for r in rows]
    _bm25_cache[repo_id] = (version, bm25, chunk_ids)
    return bm25, chunk_ids


def search(repo_id: int, question: str) -> list[RetrievedChunk]:
    conn = get_conn()
    try:
        # --- Vector leg ---
        qvec = embed_query(question)
        vector_rows = conn.execute(
            """SELECT id FROM chunks
               WHERE repo_id = %s
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (repo_id, qvec, VECTOR_TOP_K),
        ).fetchall()
        vector_ranked = [r[0] for r in vector_rows]

        # --- BM25 leg ---
        bm25_ranked: list[int] = []
        bm25_result = _bm25_for_repo(conn, repo_id)
        if bm25_result is not None:
            bm25, chunk_ids = bm25_result
            scores = bm25.get_scores(tokenize(question))
            scored = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
            bm25_ranked = [cid for cid, s in scored[:BM25_TOP_K] if s > 0]

        # --- Reciprocal rank fusion ---
        K = 60  # standard RRF constant; dampens the gap between top ranks
        fused: dict[int, float] = {}
        for ranking in (vector_ranked, bm25_ranked):
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (K + rank + 1)

        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:FUSED_TOP_K]
        if not top_ids:
            return []

        rows = conn.execute(
            """SELECT id, file_path, start_line, end_line, symbol, content
               FROM chunks WHERE id = ANY(%s)""",
            (top_ids,),
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        return [
            RetrievedChunk(
                chunk_id=r[0],
                file_path=r[1],
                start_line=r[2],
                end_line=r[3],
                symbol=r[4],
                content=r[5],
                score=fused[r[0]],
            )
            for cid in top_ids
            if (r := by_id.get(cid)) is not None
        ]
    finally:
        conn.close()
