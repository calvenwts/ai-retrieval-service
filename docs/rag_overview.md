# Retrieval Augmented Generation (RAG)

RAG is a technique that enhances LLM responses by retrieving relevant information from an external knowledge base before generating an answer. Instead of relying solely on the model's training data, RAG fetches up-to-date, domain-specific context.

## Pipeline

1. **Load**: Ingest documents from various sources (files, APIs, databases).
2. **Chunk**: Split documents into smaller, retrievable units with optional overlap.
3. **Embed**: Convert chunks into dense vector representations using an embedding model.
4. **Store**: Insert vectors into a vector database (e.g., pgvector, Pinecone, Weaviate).
5. **Retrieve**: Given a user query, embed it and find the top-k most similar chunks.
6. **Rerank**: Optionally reorder candidates using a cross-encoder for higher precision.
7. **Prompt**: Inject retrieved context into the LLM prompt.
8. **Generate**: The LLM produces an answer grounded in the retrieved context.

## Why RAG?

- Context windows are finite and expensive.
- Larger contexts degrade quality (lost-in-the-middle effect).
- RAG scales knowledge independently of the model.
- Enables citation of sources.
- Knowledge can be updated without retraining.

## Common Failure Modes

- **Stale data**: Documents not re-indexed after updates.
- **Wrong chunk size**: Too small loses context, too large adds noise.
- **Lost-in-the-middle**: Models pay less attention to middle sections of long contexts.
- **Hallucinated citations**: Model invents source references.
