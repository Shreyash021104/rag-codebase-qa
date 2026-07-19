"""Local embeddings via sentence-transformers.

Embeddings run locally (BAAI/bge-small-en-v1.5, 384-dim) instead of through a
paid API. For a portfolio-scale corpus the quality difference vs. hosted
embedding APIs is modest, the cost is zero, and — the practical win — the
entire ingestion and retrieval pipeline is testable end-to-end with no API
key at all. The model downloads (~130MB) on first use, then loads from cache.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import BGE_QUERY_PREFIX, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Lazy + cached: the model costs seconds to load and hundreds of MB of
    # RAM, so it loads once on first use, not at import time (which would
    # slow down every process that imports this module for any reason).
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed chunks for indexing. Normalized so cosine distance in pgvector
    behaves as expected."""
    return _model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )


def embed_query(text: str) -> np.ndarray:
    # bge models are trained with an instruction prefix on the QUERY side
    # only — the indexed passages stay un-prefixed. Skipping this looks
    # harmless and quietly costs retrieval accuracy.
    return _model().encode(
        [BGE_QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
    )[0]
