import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/rag_codebase"
)

# Embedding provider: "local" (sentence-transformers, runs on this machine,
# no key, no cost — the default for dev and for the eval) or "gemini"
# (Google's hosted embedding API — no torch, so it fits tiny free hosting
# tiers like Render's 512MB). The two produce different-dimension vectors,
# so EMBEDDING_DIM must match the chosen provider and a DB indexed with one
# can't be queried with the other.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")

# Local embedding model (sentence-transformers). 384-dim, ~130MB download on
# first use, then cached.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Gemini embedding model (768-dim). Free via Google AI Studio.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

_DEFAULT_DIM = "768" if EMBEDDING_PROVIDER == "gemini" else "384"
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", _DEFAULT_DIM))

# bge models are trained to expect this prefix on QUERIES only (not on the
# indexed passages) — retrieval quality measurably drops without it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# Groq (OpenAI-compatible API, generous free tier). If both keys are set,
# Anthropic wins — override with LLM_PROVIDER=groq.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "")  # "", "anthropic", or "groq"

# Chunking bounds (in lines). Functions longer than CHUNK_MAX_LINES get
# split; consecutive tiny top-level statements get grouped up to
# CHUNK_TARGET_LINES.
CHUNK_MAX_LINES = int(os.environ.get("CHUNK_MAX_LINES", "120"))
CHUNK_TARGET_LINES = int(os.environ.get("CHUNK_TARGET_LINES", "40"))

# Retrieval fan-out: how many candidates each retriever contributes before
# fusion, and how many fused results feed the LLM.
VECTOR_TOP_K = int(os.environ.get("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.environ.get("BM25_TOP_K", "20"))
FUSED_TOP_K = int(os.environ.get("FUSED_TOP_K", "8"))

CLONE_DIR = os.environ.get("CLONE_DIR", "/tmp/rag-codebase-clones")

MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_BYTES", str(512 * 1024)))
