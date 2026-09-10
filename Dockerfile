# Self-hostable image for the RAG codebase Q&A app. Uses local embeddings
# (no API key needed to run), so the model is baked in at build time and the
# container works fully offline for indexing + retrieval. A Groq/Anthropic
# key is optional (for synthesized prose answers) and passed at runtime.
FROM python:3.12-slim

# git is needed at runtime to clone the repos users ask about.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the first request doesn't have
# to download ~130MB before it can serve anything.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

ENV CLONE_DIR=/tmp/rag-codebase-clones \
    EMBEDDING_PROVIDER=local
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
