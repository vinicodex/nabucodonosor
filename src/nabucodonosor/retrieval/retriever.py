from dataclasses import dataclass, field

import chromadb

from nabucodonosor.config import Settings
from nabucodonosor.embedding.embedder import BaseEmbedder
from nabucodonosor.retrieval.bm25_index import BM25Index
from nabucodonosor.retrieval.query_rewriter import QueryRewriter
from nabucodonosor.retrieval.reranker import ScoredChunk, quality_gate, rerank
from nabucodonosor.vectorstore.chroma import SearchResult, similarity_search

_RRF_K = 60  # RRF constant; 60 is the standard default


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    rewritten_query: str
    retrieval_score: float
    passed_gate: bool


def _rrf_fuse(
    dense_results: list[SearchResult],
    bm25_results: list[tuple[str, float]],
) -> list[str]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (60 + rank(d))."""
    scores: dict[str, float] = {}

    for rank, result in enumerate(dense_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        collection: chromadb.Collection,
        bm25_index: BM25Index,
        settings: Settings,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self._embedder = embedder
        self._collection = collection
        self._bm25 = bm25_index
        self._settings = settings
        self._rewriter = query_rewriter

    def retrieve(self, question: str) -> RetrievalResult:
        rewritten = question
        if self._settings.enable_query_rewriting and self._rewriter is not None:
            rewritten = self._rewriter.rewrite(question)

        query_embedding = self._embedder.embed(rewritten)
        k = self._settings.retrieval_initial_k

        dense_results = similarity_search(
            self._collection, query_embedding, k=k, include_embeddings=True
        )
        bm25_results = self._bm25.search(rewritten, k=k)

        fused_ids = _rrf_fuse(dense_results, bm25_results)

        id_to_dense: dict[str, SearchResult] = {r.chunk_id: r for r in dense_results}
        candidates: list[SearchResult] = [
            id_to_dense[cid] for cid in fused_ids if cid in id_to_dense
        ]

        scored = rerank(query_embedding, candidates, self._collection)
        top = scored[: self._settings.retrieval_k]

        passed = quality_gate(top, self._settings.reranker_threshold)
        score = top[0].score if top else 0.0

        return RetrievalResult(
            chunks=top,
            rewritten_query=rewritten,
            retrieval_score=score,
            passed_gate=passed,
        )


@dataclass
class BM25IndexBuilder:
    """Builds a BM25Index from a ChromaDB collection."""

    extra: dict[str, object] = field(default_factory=dict)

    def build_from_collection(self, collection: chromadb.Collection) -> BM25Index:
        results = collection.get(include=["documents"])  # type: ignore[arg-type,unused-ignore]
        ids: list[str] = results["ids"]
        documents: list[str] = results["documents"] or []  # type: ignore[assignment,unused-ignore]

        index = BM25Index()
        if ids:
            index.build(ids, documents)
        return index
