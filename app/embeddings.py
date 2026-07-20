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
    EMBEDDING_DIM,
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


# Gemini free tier is rate-limited per minute. A large single batch bursts
# past that limit, and the SDK's built-in retries then hammer it further, so
# we send small sub-batches with a short gap and back off on 429 instead.
_GEMINI_SUB_BATCH = 16
_GEMINI_INTER_BATCH_SLEEP = 1.5


def _gemini_embed(texts: list[str], task_type: str) -> np.ndarray:
    import time

    from google.genai import types
    from google.genai.errors import ClientError

    client = _gemini_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _GEMINI_SUB_BATCH):
        sub = texts[i : i + _GEMINI_SUB_BATCH]
        delay = _GEMINI_INTER_BATCH_SLEEP
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=sub,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=EMBEDDING_DIM
                    ),
                )
                out.extend(e.values for e in resp.embeddings)
                break
            except ClientError as exc:
                if exc.code == 429 and attempt < 5:
                    time.sleep(delay)
                    delay *= 2  # exponential backoff on rate-limit
                    continue
                raise
        if i + _GEMINI_SUB_BATCH < len(texts):
            time.sleep(_GEMINI_INTER_BATCH_SLEEP)
    return np.array(out, dtype=np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed chunks for indexing."""
    if EMBEDDING_PROVIDER == "gemini":
        return _gemini_embed(texts, "RETRIEVAL_DOCUMENT")

    # local: normalize so cosine distance in pgvector behaves as expected
    return _local_model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )


def embed_query(text: str) -> np.ndarray:
    if EMBEDDING_PROVIDER == "gemini":
        # Gemini distinguishes query vs document embeddings via task_type
        # (the hosted equivalent of bge's query-prefix trick below); reuses
        # the same rate-limit backoff as indexing.
        return _gemini_embed([text], "RETRIEVAL_QUERY")[0]

    # bge is trained with an instruction prefix on the QUERY side only.
    return _local_model().encode(
        [BGE_QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
    )[0]
