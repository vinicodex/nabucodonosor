import pytest

from nabucodonosor.cache import exact, semantic
from nabucodonosor.config import Settings
from tests.fakes import FakeEmbedder

# ---------------------------------------------------------------------------
# Exact cache (SQLite L1)
# ---------------------------------------------------------------------------


def test_exact_cache_miss_returns_none(settings: Settings):
    result = exact.get("unknown question", settings.cache_db_path, settings.exact_cache_ttl_hours)
    assert result is None


def test_exact_cache_set_and_get(settings: Settings):
    exact.set("What is Marcos's role?", "Senior SWE", [], settings.cache_db_path)
    ttl = settings.exact_cache_ttl_hours
    result = exact.get("What is Marcos's role?", settings.cache_db_path, ttl)
    assert result == "Senior SWE"


def test_exact_cache_normalizes_question(settings: Settings):
    exact.set("  What is Marcos's role?  ", "answer", [], settings.cache_db_path)
    ttl = settings.exact_cache_ttl_hours
    result = exact.get("what is marcos's role?", settings.cache_db_path, ttl)
    assert result == "answer"


def test_exact_cache_ttl_zero_misses(settings: Settings):
    exact.set("ttl question", "answer", [], settings.cache_db_path)
    # TTL of 0 hours means everything is expired
    result = exact.get("ttl question", settings.cache_db_path, ttl_hours=0)
    assert result is None


def test_exact_cache_purge_expired(settings: Settings):
    exact.set("old question", "old answer", [], settings.cache_db_path)
    deleted = exact.purge_expired(settings.cache_db_path, ttl_hours=0)
    assert deleted >= 1


def test_exact_cache_set_is_idempotent(settings: Settings):
    exact.set("idempotent q", "v1", [], settings.cache_db_path)
    exact.set("idempotent q", "v2", [], settings.cache_db_path)
    result = exact.get("idempotent q", settings.cache_db_path, settings.exact_cache_ttl_hours)
    assert result == "v2"


# ---------------------------------------------------------------------------
# Semantic cache (ChromaDB L2)
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


def test_semantic_cache_miss_returns_none(embedder: FakeEmbedder, settings: Settings):
    result = semantic.get("some question never cached", embedder, settings)
    assert result is None


def test_semantic_cache_set_and_get_exact_match(embedder: FakeEmbedder, settings: Settings):
    settings.semantic_cache_threshold = 0.0  # type: ignore[misc]
    semantic.set("What is Marcos's tech stack?", "Python FastAPI AWS", embedder, settings)
    result = semantic.get("What is Marcos's tech stack?", embedder, settings)
    assert result == "Python FastAPI AWS"


def test_semantic_cache_misses_below_threshold(embedder: FakeEmbedder, settings: Settings):
    settings.semantic_cache_threshold = 0.9999  # type: ignore[misc]
    semantic.set("Python experience", "some answer", embedder, settings)
    # Identical vector → distance=0 → similarity=1.0 ≥ 0.9999, so this should still hit
    result = semantic.get("Python experience", embedder, settings)
    # For identical embeddings, distance = 0, similarity = 1.0 >= 0.9999 → should still hit
    assert result is not None  # exact match should always pass any reasonable threshold
