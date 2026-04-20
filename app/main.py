"""FastAPI application — RAG service with provider-agnostic LLM adapter."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.db import delete_document, index_document, init_db, retrieve
from app.providers import ChatRequest, router


# ---------------------------------------------------------------------------
# Lifespan: initialise DB on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = settings.retrieval_top_k


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class IngestRequest(BaseModel):
    doc_id: str
    content: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    """Direct LLM completion (provider-agnostic)."""
    resp = await router.complete(req)
    return resp


@app.post("/v1/ingest")
async def ingest(req: IngestRequest):
    """Chunk, embed, and store a document."""
    index_document(req.doc_id, req.content)
    return {"status": "indexed", "doc_id": req.doc_id}


@app.delete("/v1/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete all chunks for a document."""
    delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """RAG endpoint: retrieve context, then generate an answer."""
    # 1. Retrieve relevant chunks
    sources = retrieve(req.question, k=req.top_k)
    if not sources:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # 2. Build prompt with retrieved context
    context = "\n\n".join(
        f"[Source: {s['doc_id']}]\n{s['content']}" for s in sources
    )
    prompt = (
        f"Use the following context to answer the question. "
        f"Cite your sources using [Source: doc_id] format.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}\n"
        f"Answer:"
    )

    # 3. Generate answer via LLM
    chat_req = ChatRequest(
        messages=[{"role": "user", "content": prompt}],
        model=settings.default_llm_model,
        max_tokens=settings.max_tokens,
    )
    resp = await router.complete(chat_req)

    return AskResponse(
        answer=resp.text,
        sources=sources,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
