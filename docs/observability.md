# Observability for LLM Applications

Every LLM call in production needs comprehensive observability. Without it, debugging failures, optimizing costs, and improving quality is guesswork.

## What to Track Per Call

- **Model**: Which model and version was used.
- **Prompt template version**: Which prompt was active.
- **Input/output tokens**: For cost calculation and context window monitoring.
- **Latency**: End-to-end and time-to-first-token for streaming.
- **Cost**: Calculated from token counts and model pricing.
- **Finish reason**: Did it complete, hit token limit, or use a tool?
- **Retrieved doc IDs**: Which RAG sources were used (for debugging relevance).
- **Tool calls**: Which tools were invoked and their results.

## Tools

- **OpenTelemetry**: Open standard for distributed tracing. Add span attributes for LLM-specific fields.
- **LangSmith**: LangChain's tracing and evaluation platform. Good integration with LangChain/LangGraph.
- **Langfuse**: Open-source alternative to LangSmith. Self-hostable.
- **Arize Phoenix**: Open-source observability with embedding drift detection.

## Dashboards

Key metrics to display:

- p50/p95 latency by model and endpoint.
- Cost per request and total daily/weekly cost.
- Error rate by provider (detect provider outages quickly).
- Eval scores over time (track quality regressions).
- Token usage trends (spot prompt bloat).

## Evaluation

- **Golden datasets**: Curated inputs with expected outputs.
- **Exact match**: For factual, deterministic answers.
- **LLM-as-judge**: Use a strong model to grade outputs against a rubric.
- **Regression tests**: Ensure prompt changes don't break existing passing cases.
