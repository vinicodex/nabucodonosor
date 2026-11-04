import pytest

from nabucodonosor.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Test settings pointing to isolated temp directories."""
    return Settings(
        llm_provider="ollama",
        embed_provider="ollama",
        chroma_path=str(tmp_path / "chroma_db"),
        cache_db_path=str(tmp_path / "cache.db"),
        collection_name="test_cv",
        retrieval_k=3,
    )


@pytest.fixture
def sample_cv_text() -> str:
    return """
# John Doe
Senior Backend Engineer

## Summary
Backend engineer with 5 years of experience building distributed systems in Python.

## Experience

### Software Engineer · Acme Corp
Jan 2022 – Present

- Built REST APIs with FastAPI serving 1M+ requests/day.
- Designed event-driven pipelines using AWS SQS and Lambda.

## Skills
- Languages: Python, TypeScript
- Cloud: AWS (Lambda, SQS, DynamoDB)
- Databases: PostgreSQL, Redis
""".strip()
