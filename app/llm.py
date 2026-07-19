"""Answer synthesis, grounded strictly in retrieved chunks.

Two interchangeable providers behind one function:
  - Anthropic (Claude) via the anthropic SDK
  - Groq (Llama etc.) via the openai SDK pointed at Groq's OpenAI-compatible
    endpoint — generous free tier, which keeps the whole stack at $0

Provider is auto-detected from which API key is configured (Anthropic wins
if both are set; override with LLM_PROVIDER=groq). With neither configured,
the API degrades gracefully: it returns the retrieved chunks with citations,
just without a synthesized prose answer — the whole pipeline stays usable
and testable with zero API dependency.
"""

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
)
from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are a codebase Q&A assistant. Answer questions about a \
repository using ONLY the code excerpts provided. Rules:

- Ground every claim in the excerpts. If they don't contain the answer, say \
so plainly — do not guess or fill gaps from general knowledge.
- Cite as you go, using the format (file_path:start_line) after each claim, \
e.g. "Authentication is handled by a JWT middleware (src/auth/middleware.ts:8)."
- Be concise and concrete. Prefer naming the actual functions/files involved \
over describing patterns abstractly."""


def active_provider() -> str | None:
    if LLM_PROVIDER in ("anthropic", "groq"):
        # Explicit override — but only honor it if that provider has a key.
        key = ANTHROPIC_API_KEY if LLM_PROVIDER == "anthropic" else GROQ_API_KEY
        return LLM_PROVIDER if key else None
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GROQ_API_KEY:
        return "groq"
    return None


def llm_available() -> bool:
    return active_provider() is not None


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        symbol = f" ({c.symbol})" if c.symbol else ""
        parts.append(
            f"[Excerpt {i}] {c.file_path}:{c.start_line}-{c.end_line}{symbol}\n"
            f"```\n{c.content}\n```"
        )
    return "\n\n".join(parts)


def _user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Code excerpts from the repository:\n\n{_format_context(chunks)}\n\n"
        f"Question: {question}"
    )


def synthesize_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    provider = active_provider()
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_message(question, chunks)}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    if provider == "groq":
        from openai import OpenAI

        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(question, chunks)},
            ],
        )
        return response.choices[0].message.content or ""

    raise RuntimeError("No LLM provider configured")
