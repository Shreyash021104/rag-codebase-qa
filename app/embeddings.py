"""Pluggable embeddings: local sentence-transformers, or hosted Gemini.

- "local" (default): BAAI/bge-small-en-v1.5 via sentence-transformers. Runs
  on this machine, no API key, no cost — used for dev and for the eval.
  Downside: torch is a ~1GB runtime-RAM dependency, too big for the smallest
  free hosting tiers.
- "gemini": Google's hosted embedding API. No torch, so the deployed image
  is tiny and fits Render's free 512MB tier. Needs a (free) GEMINI_API_KEY.

The two produce different-dimension vectors and are NOT interchangeable
within one indexed database — pick one per deployment, set EMBEDDING_DIM to
match, and index with it.
"""

from functools import lru_cache

import numpy as np

from .config import (
    BGE_QUERY_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def _local_model():
    # Lazy + cached: importing torch and loading the model costs seconds and
    # hundreds of MB, so it happens on first use, not at import time.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed chunks for indexing."""
    if EMBEDDING_PROVIDER == "gemini":
        from google.genai import types

        resp = _gemini_client().models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return np.array([e.values for e in resp.embeddings], dtype=np.float32)

    # local: normalize so cosine distance in pgvector behaves as expected
    return _local_model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )


def embed_query(text: str) -> np.ndarray:
    if EMBEDDING_PROVIDER == "gemini":
        from google.genai import types

        # Gemini distinguishes query vs document embeddings via task_type
        # (the hosted equivalent of bge's query-prefix trick below).
        resp = _gemini_client().models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return np.array(resp.embeddings[0].values, dtype=np.float32)

    # bge is trained with an instruction prefix on the QUERY side only.
    return _local_model().encode(
        [BGE_QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
    )[0]
