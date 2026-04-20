from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_demo"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4
    default_llm_model: str = "llama3.2"
    max_tokens: int = 1024

    model_config = {"env_prefix": "RAG_"}


settings = Settings()
