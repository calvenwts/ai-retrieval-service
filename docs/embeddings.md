# Embeddings

Embeddings are dense vector representations of text that capture semantic meaning. Similar texts produce vectors that are close together in the embedding space.

## Key Concepts

- **Dimensionality**: The number of values in each vector. Common models produce 384 (MiniLM), 768 (BERT), or 1536 (OpenAI ada-002) dimensions.
- **Cosine similarity**: The primary metric for comparing embeddings. Measures the angle between two vectors, ranging from -1 (opposite) to 1 (identical).
- **Dot product**: An alternative similarity metric, faster to compute but sensitive to vector magnitude.

## Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| BGE-large | 1024 | Medium | Very good |
| OpenAI text-embedding-3-small | 1536 | API call | Very good |
| Cohere embed-v3 | 1024 | API call | Excellent |

## Best Practices

- Use the same embedding model for indexing and querying.
- Normalize vectors if using dot product similarity.
- Batch encoding for efficiency when indexing many documents.
- Consider the trade-off between quality and latency/cost.
