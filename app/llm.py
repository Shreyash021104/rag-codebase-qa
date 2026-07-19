"""Answer synthesis with Claude, grounded strictly in retrieved chunks.

Degrades gracefully when no ANTHROPIC_API_KEY is configured: the API still
returns the retrieved chunks with citations, just without a synthesized
prose answer. That keeps the whole pipeline (ingestion, retrieval, eval,
frontend) usable and testable with zero API dependency — the LLM is the
last, optional step, not a load-bearing one.
"""

import anthropic

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are a codebase Q&A assistant. Answer questions about a \
repository using ONLY the code excerpts provided. Rules:

- Ground every claim in the excerpts. If they don't contain the answer, say \
so plainly — do not guess or fill gaps from general knowledge.
- Cite as you go, using the format (file_path:start_line) after each claim, \
e.g. "Authentication is handled by a JWT middleware (src/auth/middleware.ts:8)."
- Be concise and concrete. Prefer naming the actual functions/files involved \
over describing patterns abstractly."""


def llm_available() -> bool:
    return bool(ANTHROPIC_API_KEY)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        symbol = f" ({c.symbol})" if c.symbol else ""
        parts.append(
            f"[Excerpt {i}] {c.file_path}:{c.start_line}-{c.end_line}{symbol}\n"
            f"```\n{c.content}\n```"
        )
    return "\n\n".join(parts)


def synthesize_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Code excerpts from the repository:\n\n{_format_context(chunks)}\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")
