import pytest

from nabucodonosor.config import Settings
from nabucodonosor.ingestion.chunker import Chunk
from nabucodonosor.vectorstore.chroma import (
    build_collection,
    get_embeddings_by_ids,
    similarity_search,
    upsert_chunks,
)
from tests.fakes import FakeEmbedder


@pytest.fixture
def collection(settings: Settings):
    return build_collection(settings)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            content="FastAPI and Python backend development",
            metadata={"source": "cv.md", "section": "Experience", "chunk_index": "0"},
        ),
        Chunk(
            content="AWS infrastructure and cloud services",
            metadata={"source": "cv.md", "section": "Skills", "chunk_index": "1"},
        ),
        Chunk(
            content="Machine learning with PyTorch and scikit-learn",
            metadata={"source": "cv.md", "section": "Skills", "chunk_index": "2"},
        ),
    ]


def test_upsert_returns_count(collection, embedder, sample_chunks):
    count = upsert_chunks(collection, sample_chunks, embedder)
    assert count == len(sample_chunks)


def test_upsert_empty_chunks_returns_zero(collection, embedder):
    assert upsert_chunks(collection, [], embedder) == 0


def test_upsert_persists_to_collection(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    assert collection.count() == len(sample_chunks)


def test_upsert_is_idempotent(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    upsert_chunks(collection, sample_chunks, embedder)
    assert collection.count() == len(sample_chunks)


def test_similarity_search_returns_results(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    query_emb = embedder.embed("Python backend")
    results = similarity_search(collection, query_emb, k=2)
    assert len(results) == 2


def test_similarity_search_scores_between_zero_and_one(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    query_emb = embedder.embed("AWS cloud")
    results = similarity_search(collection, query_emb, k=3)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_similarity_search_results_have_content(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    query_emb = embedder.embed("machine learning")
    results = similarity_search(collection, query_emb, k=1)
    assert results[0].content != ""


def test_similarity_search_includes_embeddings(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    query_emb = embedder.embed("cloud")
    results = similarity_search(collection, query_emb, k=1, include_embeddings=True)
    assert results[0].embedding is not None
    assert len(results[0].embedding) == 8


def test_similarity_search_no_embeddings_by_default(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    query_emb = embedder.embed("cloud")
    results = similarity_search(collection, query_emb, k=1, include_embeddings=False)
    assert results[0].embedding is None


def test_similarity_search_empty_collection_returns_empty(collection, embedder):
    query_emb = embedder.embed("anything")
    results = similarity_search(collection, query_emb, k=5)
    assert results == []


def test_get_embeddings_by_ids(collection, embedder, sample_chunks):
    upsert_chunks(collection, sample_chunks, embedder)
    ids = ["cv.md_0", "cv.md_1"]
    embs = get_embeddings_by_ids(collection, ids)
    assert set(embs.keys()) == set(ids)
    for emb in embs.values():
        assert len(emb) == 8


def test_get_embeddings_by_ids_empty_input(collection, embedder):
    result = get_embeddings_by_ids(collection, [])
    assert result == {}
