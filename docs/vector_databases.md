# Vector Databases

Vector databases are specialized storage systems optimized for storing and querying high-dimensional vectors using approximate nearest neighbor (ANN) search.

## pgvector

pgvector is a PostgreSQL extension that adds vector similarity search. It's an excellent default choice because:

- Runs inside your existing Postgres instance (no new infrastructure).
- Supports exact and approximate nearest neighbor search.
- Provides HNSW and IVFFlat index types.
- Integrates with standard SQL queries for hybrid filtering.

### Index Types

- **IVFFlat**: Partitions vectors into lists and searches a subset. Faster to build, slightly less accurate. Good for datasets under 1M vectors.
- **HNSW**: Hierarchical navigable small world graph. Slower to build, more accurate, better for large datasets. Preferred for production.

## Other Options

- **Pinecone**: Fully managed, serverless. Easy to start, harder to control costs at scale.
- **Weaviate**: Open-source, supports hybrid search natively.
- **Qdrant**: High-performance, written in Rust. Good filtering support.
- **Milvus**: Distributed, designed for billion-scale datasets.
- **Chroma**: Lightweight, good for prototyping and local development.

## Hybrid Search

Combining lexical search (BM25) with dense vector search often outperforms either approach alone. The lexical component catches exact keyword matches that embeddings might miss, while embeddings capture semantic similarity.
