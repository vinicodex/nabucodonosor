from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    groq_api_key: str = ""
    groq_model: str = "llama-3.2-3b-preview"

    # Embeddings
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text"
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 768

    # Vector store
    chroma_path: str = "./data/chroma_db"
    collection_name: str = "cv_knowledge"

    # Cache
    cache_db_path: str = "./data/cache.db"
    exact_cache_ttl_hours: int = 24
    semantic_cache_threshold: float = 0.95

    # Retrieval
    retrieval_k: int = 5
    retrieval_initial_k: int = 20
    reranker_threshold: float = 0.10
    enable_query_rewriting: bool = True

    # Guardrails
    max_question_length: int = 500
    topic_guard_threshold: float = 0.5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
