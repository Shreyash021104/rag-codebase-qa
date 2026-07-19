from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from . import ingest, llm, retrieval
from .db import get_conn, init_schema


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_schema()
    yield


app = FastAPI(title="RAG Codebase Q&A", lifespan=lifespan)


class CreateRepoRequest(BaseModel):
    url: HttpUrl


class AskRequest(BaseModel):
    question: str


def _repo_row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "url": row[1],
        "name": row[2],
        "status": row[3],
        "error": row[4],
        "chunk_count": row[5],
        "indexed_at": row[6].isoformat() if row[6] else None,
    }


REPO_COLUMNS = "id, url, name, status, error, chunk_count, indexed_at"


@app.get("/health")
def health():
    return {"ok": True, "llm_configured": llm.llm_available()}


@app.post("/api/repos", status_code=201)
def create_repo(body: CreateRepoRequest, background: BackgroundTasks):
    url = str(body.url)
    if not url.startswith("https://"):
        raise HTTPException(400, "Only https git URLs are supported")
    name = url.rstrip("/").removesuffix(".git").split("/")[-1]
    conn = get_conn()
    try:
        row = conn.execute(
            f"INSERT INTO repos (url, name) VALUES (%s, %s) RETURNING {REPO_COLUMNS}",
            (url, name),
        ).fetchone()
        conn.commit()
    finally:
        conn.close()
    background.add_task(ingest.index_repo, row[0], url)
    return _repo_row_to_dict(row)


@app.get("/api/repos")
def list_repos():
    conn = get_conn()
    try:
        rows = conn.execute(
            f"SELECT {REPO_COLUMNS} FROM repos ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return {"repos": [_repo_row_to_dict(r) for r in rows]}


@app.get("/api/repos/{repo_id}")
def get_repo(repo_id: int):
    conn = get_conn()
    try:
        row = conn.execute(
            f"SELECT {REPO_COLUMNS} FROM repos WHERE id = %s", (repo_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "Repo not found")
    return _repo_row_to_dict(row)


@app.post("/api/repos/{repo_id}/ask")
def ask(repo_id: int, body: AskRequest):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT status FROM repos WHERE id = %s", (repo_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "Repo not found")
    if row[0] != "ready":
        raise HTTPException(409, f"Repo is not ready yet (status: {row[0]})")

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty")

    chunks = retrieval.search(repo_id, question)
    sources = [
        {
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "symbol": c.symbol,
            "content": c.content,
        }
        for c in chunks
    ]

    answer = None
    if chunks and llm.llm_available():
        answer = llm.synthesize_answer(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "llm_configured": llm.llm_available(),
        "sources": sources,
    }


@app.get("/")
def index_page():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
