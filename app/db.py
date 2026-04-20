import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from sentence_transformers import SentenceTransformer

from app.chunker import chunk_text
from app.config import settings

embedder = SentenceTransformer(settings.embedding_model)


def get_conn(register_vec: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    if register_vec:
        register_vector(conn)
    return conn


def init_db():
    with get_conn(register_vec=False) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dim})
                )
            """).format(dim=sql.Literal(settings.embedding_dim))
        )
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)


def index_document(doc_id: str, text: str):
    chunks = chunk_text(text)
    vectors = embedder.encode(chunks).tolist()
    with get_conn() as conn:
        with conn.cursor() as cur:
            for content, vec in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks (doc_id, content, embedding) VALUES (%s, %s, %s)",
                    (doc_id, content, vec),
                )


def retrieve(query: str, k: int = settings.retrieval_top_k) -> list[dict]:
    query_vec = np.array(embedder.encode([query])[0])
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT doc_id, content
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vec, k),
        ).fetchall()
    return [{"doc_id": row[0], "content": row[1]} for row in rows]


def delete_document(doc_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
