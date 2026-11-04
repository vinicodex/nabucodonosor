from unittest.mock import MagicMock

import pytest

from nabucodonosor.config import Settings
from nabucodonosor.ingestion.chunker import Chunk
from nabucodonosor.retrieval.bm25_index import BM25Index
from nabucodonosor.retrieval.query_rewriter import QueryRewriter
from nabucodonosor.retrieval.reranker import ScoredChunk, quality_gate, rerank
from nabucodonosor.retrieval.retriever import BM25IndexBuilder, HybridRetriever, _rrf_fuse
from nabucodonosor.vectorstore.chroma import SearchResult, build_collection, upsert_chunks
from tests.fakes import FakeEmbedder

# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

DOCS = ["FastAPI Python backend", "AWS cloud infrastructure", "Machine learning PyTorch"]
IDS = ["id0", "id1", "id2"]


def test_bm25_search_returns_ranked_results():
    idx = BM25Index()
    idx.build(IDS, DOCS)
    results = idx.search("FastAPI Python", k=2)
    assert len(results) == 2
    assert results[0][0] == "id0"


def test_bm25_scores_normalized_between_zero_and_one():
    idx = BM25Index()
    idx.build(IDS, DOCS)
    results = idx.search("AWS cloud", k=3)
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_bm25_empty_index_returns_empty():
    idx = BM25Index()
    assert idx.search("anything") == []


def test_bm25_no_match_query_returns_empty():
    idx = BM25Index()
    idx.build(IDS, DOCS)
    results = idx.search("zzzzzzzzzzz")
    assert results == []


def test_bm25_size_matches_corpus():
    idx = BM25Index()
    idx.build(IDS, DOCS)
    assert idx.size == len(IDS)


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------


def test_query_rewriter_uses_llm():
    llm = MagicMock()
    llm.complete.return_value = "What backend technologies does Marcos know?"
    rewriter = QueryRewriter(llm)
    result = rewriter.rewrite("What tech does Marcos use?")
    assert result == "What backend technologies does Marcos know?"


def test_query_rewriter_falls_back_on_empty_llm_response():
    llm = MagicMock()
    llm.complete.return_value = "   "
    rewriter = QueryRewriter(llm)
    original = "What tech does Marcos use?"
    assert rewriter.rewrite(original) == original


def test_query_rewriter_falls_back_on_llm_exception():
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("LLM unavailable")
    rewriter = QueryRewriter(llm)
    original = "What tech does Marcos use?"
    assert rewriter.rewrite(original) == original


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, score: float = 0.5) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content="", metadata={}, score=score)


def test_rrf_fuse_merges_both_lists():
    dense = [_make_result("a"), _make_result("b")]
    bm25 = [("b", 0.9), ("c", 0.5)]
    fused = _rrf_fuse(dense, bm25)
    assert set(fused) == {"a", "b", "c"}


def test_rrf_fuse_boosts_items_in_both_lists():
    dense = [_make_result("a"), _make_result("b")]
    bm25 = [("b", 0.9), ("c", 0.5)]
    fused = _rrf_fuse(dense, bm25)
    # "b" appears in both lists — should rank higher than "a" or "c" alone
    assert fused[0] == "b"


def test_rrf_fuse_empty_bm25():
    dense = [_make_result("x"), _make_result("y")]
    fused = _rrf_fuse(dense, [])
    assert fused == ["x", "y"]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def _make_search_result(chunk_id: str, content: str, emb: list[float]) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content=content, metadata={}, score=0.5, embedding=emb)


def test_rerank_orders_by_cosine_similarity():
    embedder = FakeEmbedder(dim=8)
    q_emb = embedder.embed("AWS cloud")
    # Inject embeddings that match the query to different degrees
    high_emb = embedder.embed("AWS cloud infrastructure")
    low_emb = embedder.embed("machine learning keras")
    candidates = [
        _make_search_result("low", "low relevance", low_emb),
        _make_search_result("high", "AWS cloud services", high_emb),
    ]
    collection = MagicMock()
    scored = rerank(q_emb, candidates, collection)
    assert scored[0].chunk_id == "high"


def test_rerank_empty_candidates_returns_empty():
    collection = MagicMock()
    result = rerank([1.0, 0.0], [], collection)
    assert result == []


def test_quality_gate_passes_above_threshold():
    chunks = [ScoredChunk(chunk_id="a", content="x", metadata={}, score=0.5)]
    assert quality_gate(chunks, threshold=0.10) is True


def test_quality_gate_fails_below_threshold():
    chunks = [ScoredChunk(chunk_id="a", content="x", metadata={}, score=0.05)]
    assert quality_gate(chunks, threshold=0.10) is False


def test_quality_gate_empty_list_fails():
    assert quality_gate([], threshold=0.10) is False


# ---------------------------------------------------------------------------
# HybridRetriever (integration with FakeEmbedder + real ChromaDB in tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_collection(settings: Settings):
    collection = build_collection(settings)
    embedder = FakeEmbedder(dim=8)
    chunks = [
        Chunk(
            content="FastAPI Python backend senior engineer",
            metadata={"source": "cv.md", "section": "Experience", "chunk_index": "0"},
        ),
        Chunk(
            content="AWS cloud infrastructure deployment",
            metadata={"source": "cv.md", "section": "Skills", "chunk_index": "1"},
        ),
        Chunk(
            content="CrewAI multi-agent systems production",
            metadata={"source": "cv.md", "section": "Experience", "chunk_index": "2"},
        ),
    ]
    upsert_chunks(collection, chunks, embedder)
    return collection


@pytest.fixture
def bm25(populated_collection):
    builder = BM25IndexBuilder()
    return builder.build_from_collection(populated_collection)


def test_hybrid_retriever_returns_result(settings: Settings, populated_collection, bm25):
    embedder = FakeEmbedder(dim=8)
    retriever = HybridRetriever(embedder, populated_collection, bm25, settings)
    result = retriever.retrieve("What Python frameworks does Marcos use?")
    assert len(result.chunks) > 0
    assert result.rewritten_query != ""


def test_hybrid_retriever_skips_rewriting_when_disabled(
    settings: Settings, populated_collection, bm25
):
    settings.enable_query_rewriting = False  # type: ignore[misc]
    embedder = FakeEmbedder(dim=8)
    retriever = HybridRetriever(embedder, populated_collection, bm25, settings)
    question = "cloud infrastructure?"
    result = retriever.retrieve(question)
    assert result.rewritten_query == question


def test_hybrid_retriever_quality_gate_set(settings: Settings, populated_collection, bm25):
    embedder = FakeEmbedder(dim=8)
    retriever = HybridRetriever(embedder, populated_collection, bm25, settings)
    result = retriever.retrieve("cloud AWS")
    assert isinstance(result.passed_gate, bool)


def test_bm25_index_builder_builds_from_collection(settings: Settings, populated_collection):
    builder = BM25IndexBuilder()
    idx = builder.build_from_collection(populated_collection)
    assert idx.size == 3
