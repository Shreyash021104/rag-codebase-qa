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

import os
import re
from collections import OrderedDict
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


# BM25 index cache, keyed by repo. Invalidated by indexed_at so re-indexing a
# repo naturally rebuilds it. Fine for a single-process deployment; a
# multi-instance deployment would move this into the DB (Postgres full-text
# search) — noted in the README's scaling section.
#
# Bounded, because each entry holds a whole repo's tokenized corpus in memory.
# An unbounded dict keeps every repo ever queried resident for the lifetime of
# the process, so a server used across many repos grows until it is killed.
# Least-recently-used eviction keeps the hot repos and drops the rest; a
# dropped repo just pays one rebuild on its next query.
_BM25_CACHE_SIZE = int(os.environ.get("BM25_CACHE_SIZE", "8"))
_bm25_cache: "OrderedDict[int, tuple[str, BM25Okapi, list[int]]]" = OrderedDict()


def _bm25_for_repo(conn, repo_id: int) -> tuple[BM25Okapi, list[int]] | None:
    row = conn.execute(
        "SELECT indexed_at::text FROM repos WHERE id = %s", (repo_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    version = row[0]

    cached = _bm25_cache.get(repo_id)
    if cached and cached[0] == version:
        _bm25_cache.move_to_end(repo_id)
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
    _bm25_cache.move_to_end(repo_id)
    while len(_bm25_cache) > _BM25_CACHE_SIZE:
        _bm25_cache.popitem(last=False)
    return bm25, chunk_ids


# Standard RRF constant. It dampens the gap between adjacent top ranks, so a
# document ranked 1st by one leg does not automatically outrank a document
# ranked 2nd by both.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> dict[int, float]:
    """Fuse ranked id lists by rank alone, never by score.

    The two legs score on incompatible scales — cosine distance and BM25
    relevance — so combining the raw numbers would need normalisation tuned
    per corpus. RRF sidesteps that: each list contributes 1/(k + rank) for
    whatever it ranked, and appearing in both lists is what wins.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def rank_fused(fused: dict[int, float], limit: int) -> list[int]:
    """Highest fused score first, ties broken by chunk id.

    The tie-break is explicit rather than relying on dict insertion order,
    which would make results depend on which leg happened to return first.
    """
    return sorted(fused, key=lambda cid: (-fused[cid], cid))[:limit]


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
        fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])
        top_ids = rank_fused(fused, FUSED_TOP_K)
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
