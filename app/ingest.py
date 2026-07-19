"""Ingestion pipeline: clone a repo, chunk its files, embed, store.

Runs as a background job (kicked off by the API) with progress tracked in the
repos table, so the frontend can poll status while indexing runs.
"""

import os
import shutil
import subprocess

from .chunker import Chunk, chunk_file, is_indexable, language_for_path
from .config import CLONE_DIR, MAX_FILE_BYTES
from .db import get_conn
from .embeddings import embed_passages

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "vendor",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    "target",
    "coverage",
}


def clone_repo(url: str, repo_id: int) -> str:
    dest = os.path.join(CLONE_DIR, str(repo_id))
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(CLONE_DIR, exist_ok=True)
    # --depth 1: we only need the current state of the code, not history —
    # this is the difference between cloning a few MB and a few hundred.
    subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return dest


def walk_files(root: str) -> list[str]:
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Mutating dirnames in place is how os.walk prunes traversal — the
        # skipped directories are never even entered.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root)
            if not is_indexable(rel):
                continue
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            paths.append(rel)
    return sorted(paths)


def chunk_repo(root: str, rel_paths: list[str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for rel in rel_paths:
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError:
            continue
        all_chunks.extend(chunk_file(rel, source))
    return all_chunks


def _embedding_text(chunk: Chunk) -> str:
    # What actually gets embedded is the chunk PLUS a one-line header naming
    # the file and symbol. Queries mention names ("where is markPresent
    # defined", "how does room.ts persist state") far more often than they
    # quote code verbatim — without the header, that signal never reaches
    # the embedding.
    header = chunk.file_path
    if chunk.symbol:
        header += f" — {chunk.symbol}"
    return f"{header}\n{chunk.content}"


def index_repo(repo_id: int, url: str) -> None:
    conn = get_conn()
    try:
        conn.execute("UPDATE repos SET status = 'cloning' WHERE id = %s", (repo_id,))
        conn.commit()
        root = clone_repo(url, repo_id)

        conn.execute("UPDATE repos SET status = 'chunking' WHERE id = %s", (repo_id,))
        conn.commit()
        rel_paths = walk_files(root)
        chunks = chunk_repo(root, rel_paths)
        if not chunks:
            raise ValueError("No indexable files found in repository")

        conn.execute("UPDATE repos SET status = 'embedding' WHERE id = %s", (repo_id,))
        conn.commit()

        # Re-indexing the same repo replaces its chunks wholesale — simplest
        # correct behavior, and idempotent if a job is retried.
        conn.execute("DELETE FROM chunks WHERE repo_id = %s", (repo_id,))

        BATCH = 64
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vectors = embed_passages([_embedding_text(c) for c in batch])
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks
                       (repo_id, file_path, start_line, end_line, symbol, language, content, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (
                            repo_id,
                            c.file_path,
                            c.start_line,
                            c.end_line,
                            c.symbol,
                            c.language,
                            c.content,
                            vec,
                        )
                        for c, vec in zip(batch, vectors)
                    ],
                )
            conn.commit()

        conn.execute(
            """UPDATE repos
               SET status = 'ready', chunk_count = %s, indexed_at = now(), error = NULL
               WHERE id = %s""",
            (len(chunks), repo_id),
        )
        conn.commit()
        shutil.rmtree(root, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 — job boundary: record any failure
        conn.rollback()
        conn.execute(
            "UPDATE repos SET status = 'failed', error = %s WHERE id = %s",
            (str(exc)[:2000], repo_id),
        )
        conn.commit()
    finally:
        conn.close()
