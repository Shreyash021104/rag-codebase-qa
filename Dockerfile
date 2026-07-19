# Dockerfile for Hugging Face Spaces (Docker SDK). Spaces run the container
# as uid 1000 and expect the app on the port declared in the Space README's
# `app_port` — 7860 by convention.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image at build time, so cold starts
# don't re-download ~130MB before the first request can be served.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=user . .

ENV CLONE_DIR=/tmp/rag-codebase-clones
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
