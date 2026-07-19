"""Retrieval evaluation against a hand-labeled question set.

Each question names the file(s) that genuinely contain its answer — labeled
by reading the target repo, not generated. The metric is top-k retrieval
accuracy: did at least one expected file appear in the top k retrieved
chunks? That's the number that matters for RAG quality — if retrieval
doesn't surface the right file, no amount of LLM prompting can produce a
grounded answer from it.

Usage:
    .venv/bin/python -m scripts.eval <repo_id>

The repo must already be indexed, and should be
https://github.com/Shreyash021104/distributed-rate-limiter-gateway — the
labels below are specific to it.
"""

import sys

from app.retrieval import search

# fmt: off
LABELED_QUESTIONS: list[tuple[str, set[str]]] = [
    # (question, files that contain the answer — any match counts)
    ("where is the rate limit middleware defined?",
     {"src/rateLimitMiddleware.ts"}),
    ("how does the token bucket algorithm refill tokens?",
     {"src/scripts/tokenBucket.lua"}),
    ("how does the sliding window log evict old requests?",
     {"src/scripts/slidingWindowLog.lua"}),
    ("what happens when redis is unreachable during a rate limit check?",
     {"src/rateLimitMiddleware.ts"}),
    ("how do multiple gateway instances share the same rate limit?",
     {"README.md", "scripts/multi-instance-test.mjs", "src/scripts/tokenBucket.lua"}),
    ("what is the boundary burst problem with fixed window rate limiting?",
     {"src/scripts/fixedWindow.lua", "scripts/boundary-burst-test.mjs", "README.md"}),
    ("which prometheus metrics does the gateway expose?",
     {"src/metrics.ts"}),
    ("how are requests forwarded to the upstream backend?",
     {"src/index.ts"}),
    ("how is the api key extracted from a request?",
     {"src/rateLimitMiddleware.ts"}),
    ("what environment variables configure the rate limit?",
     {"src/env.ts", ".env.example", "README.md"}),
    ("how does the load test force contention on the same api keys?",
     {"loadtest/compare-algorithms.js"}),
    ("how does the fixed window counter reset when a new window starts?",
     {"src/scripts/fixedWindow.lua", "src/limiters/fixedWindow.ts"}),
    ("what does the mock upstream service return?",
     {"src/mockUpstream.ts"}),
    ("how are lua scripts registered as redis commands?",
     {"src/limiters/tokenBucket.ts", "src/limiters/slidingWindow.ts",
      "src/limiters/fixedWindow.ts"}),
    ("what retry-after header is sent when a request is rejected?",
     {"src/rateLimitMiddleware.ts", "src/scripts/tokenBucket.lua"}),
]
# fmt: on


def evaluate(repo_id: int, ks: tuple[int, ...] = (1, 3, 5)) -> None:
    hits = {k: 0 for k in ks}
    misses: list[tuple[str, list[str]]] = []

    for question, expected_files in LABELED_QUESTIONS:
        results = search(repo_id, question)
        retrieved_files = [r.file_path for r in results]
        for k in ks:
            if any(f in expected_files for f in retrieved_files[:k]):
                hits[k] += 1
        if not any(f in expected_files for f in retrieved_files[: max(ks)]):
            misses.append((question, retrieved_files[: max(ks)]))

    n = len(LABELED_QUESTIONS)
    print(f"Evaluated {n} hand-labeled questions\n")
    for k in ks:
        pct = 100.0 * hits[k] / n
        print(f"  top-{k} retrieval accuracy: {hits[k]}/{n}  ({pct:.0f}%)")

    if misses:
        print("\nMisses (expected file absent from top results):")
        for question, got in misses:
            print(f"  Q: {question}")
            for f in got:
                print(f"     got: {f}")
    else:
        print("\nNo complete misses — every question surfaced an expected file.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    evaluate(int(sys.argv[1]))
