import pytest

from nabucodonosor.config import Settings
from nabucodonosor.embedding.embedder import (
    OllamaEmbedder,
    cosine_similarity,
    get_embedder,
)
from tests.fakes import FakeEmbedder


def test_cosine_similarity_identical_vectors() -> None:
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_get_embedder_returns_ollama_by_default(settings: Settings) -> None:
    embedder = get_embedder(settings)
    assert isinstance(embedder, OllamaEmbedder)


def test_fake_embedder_returns_correct_dim() -> None:
    embedder = FakeEmbedder(dim=16)
    result = embedder.embed("hello world")
    assert len(result) == 16


def test_fake_embedder_output_is_unit_vector() -> None:
    import math

    embedder = FakeEmbedder(dim=8)
    vec = embedder.embed("test")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_fake_embedder_different_texts_differ() -> None:
    embedder = FakeEmbedder(dim=8)
    assert embedder.embed("FastAPI") != embedder.embed("Django")


def test_fake_embedder_same_text_is_deterministic() -> None:
    embedder = FakeEmbedder(dim=8)
    assert embedder.embed("hello") == embedder.embed("hello")


def test_fake_embedder_batch_consistent_with_single() -> None:
    embedder = FakeEmbedder(dim=8)
    texts = ["python", "aws", "fastapi"]
    batch = embedder.embed_batch(texts)
    assert batch == [embedder.embed(t) for t in texts]
