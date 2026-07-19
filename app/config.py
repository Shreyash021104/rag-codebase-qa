import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/rag_codebase"
)

# Local embedding model (sentence-transformers). 384-dim, ~130MB download on
# first use, then cached. Swappable via env without touching code.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

# bge models are trained to expect this prefix on QUERIES only (not on the
# indexed passages) — retrieval quality measurably drops without it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

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
