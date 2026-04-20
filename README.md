# RAG Project

A local Retrieval Augmented Generation (RAG) service built with FastAPI, pgvector, and sentence-transformers. Features a provider-agnostic LLM adapter supporting Anthropic, OpenAI, and Ollama (local).

## Architecture

```
User Query → FastAPI → Embed Query → pgvector Cosine Search → Top-K Chunks → Build Prompt → LLM → Answer
```

### Components

| File | Role |
|------|------|
| `app/main.py` | FastAPI app with RAG, chat, ingest, and delete endpoints |
| `app/providers.py` | LLM adapter (Anthropic, OpenAI, Ollama) with retries and fallback |
| `app/db.py` | pgvector operations: init, index, retrieve, delete |
| `app/chunker.py` | Text chunking with configurable size and overlap |
| `app/agent.py` | Tool-calling ReAct agent loop |
| `app/config.py` | Settings via environment variables |
| `ingest.py` | CLI script to bulk-index markdown docs |

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ with pgvector extension
- Ollama (for free local inference) or Anthropic/OpenAI API key

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database

```bash
# Create the database
createdb rag_demo

# Or with Docker (includes pgvector):
docker run -d --name rag-postgres \
  -e POSTGRES_DB=rag_demo -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 pgvector/pgvector:pg16
```

### Ingest Documents

```bash
python ingest.py
```

This indexes all `.md` files from the `docs/` directory into pgvector.

### Run the API

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### `POST /v1/ask` — RAG query

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and why use it?"}'
```

Response:
```json
{
  "answer": "RAG stands for...",
  "sources": [{"doc_id": "rag_overview", "content": "..."}],
  "model": "llama3.2",
  "input_tokens": 450,
  "output_tokens": 120
}
```

### `POST /v1/chat` — Direct LLM completion

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "model": "llama3.2"}'
```

### `POST /v1/ingest` — Index a document

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "my_doc", "content": "Document text here..."}'
```

### `DELETE /v1/documents/{doc_id}` — Remove a document

```bash
curl -X DELETE http://localhost:8000/v1/documents/my_doc
```

## Configuration

Settings are configured via environment variables with `RAG_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/rag_demo` | Postgres connection string |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RAG_CHUNK_SIZE` | `500` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RAG_RETRIEVAL_TOP_K` | `4` | Number of chunks to retrieve |
| `RAG_DEFAULT_LLM_MODEL` | `llama3.2` | Default model for generation |

## Running Tests

```bash
pytest tests/ -v
```

## Design Decisions

- **HNSW over IVFFlat**: HNSW works with any dataset size; IVFFlat requires tuning `lists` relative to row count.
- **Provider adapter pattern**: Swap LLMs by changing the model name — no changes to RAG logic.
- **Retry with exponential backoff**: Handles transient provider failures gracefully.
- **Numpy vectors for pgvector**: Required for proper type registration with psycopg's pgvector adapter.
