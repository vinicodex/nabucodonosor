from dataclasses import dataclass, field

import chromadb

from nabucodonosor.cache import exact as exact_cache
from nabucodonosor.cache import semantic as semantic_cache
from nabucodonosor.config import Settings
from nabucodonosor.embedding.embedder import BaseEmbedder, get_embedder
from nabucodonosor.generation.llm import BaseLLM, get_llm
from nabucodonosor.generation.prompts import NO_CONTEXT_RESPONSE
from nabucodonosor.guardrails.input_guard import check_input
from nabucodonosor.guardrails.output_guard import check_output
from nabucodonosor.retrieval.bm25_index import BM25Index
from nabucodonosor.retrieval.query_rewriter import QueryRewriter
from nabucodonosor.retrieval.retriever import BM25IndexBuilder, HybridRetriever
from nabucodonosor.vectorstore.chroma import build_collection


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    rewritten_query: str
    retrieval_score: float
    cache_hit: bool
    passed_gate: bool
    blocked: bool = False
    block_reason: str = ""


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
        collection: chromadb.Collection | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.llm = llm or get_llm(self.settings)

        self.collection = collection or build_collection(self.settings)

        if bm25_index is not None:
            self._bm25 = bm25_index
        else:
            self._bm25 = BM25IndexBuilder().build_from_collection(self.collection)

        rewriter_llm = _LLMAdapter(self.llm)
        self._retriever = HybridRetriever(
            embedder=self.embedder,
            collection=self.collection,
            bm25_index=self._bm25,
            settings=self.settings,
            query_rewriter=QueryRewriter(rewriter_llm),
        )

    def query(self, question: str) -> RAGResponse:
        # Input guardrail
        guard = check_input(question, self.embedder, self.settings)
        if not guard.allowed:
            return RAGResponse(
                answer=guard.reason,
                sources=[],
                rewritten_query=question,
                retrieval_score=0.0,
                cache_hit=False,
                passed_gate=False,
                blocked=True,
                block_reason=guard.reason,
            )

        # L1 exact cache
        cached = exact_cache.get(
            question, self.settings.cache_db_path, self.settings.exact_cache_ttl_hours
        )
        if cached:
            return RAGResponse(
                answer=cached,
                sources=[],
                rewritten_query=question,
                retrieval_score=1.0,
                cache_hit=True,
                passed_gate=True,
            )

        # L2 semantic cache
        cached_semantic = semantic_cache.get(question, self.embedder, self.settings)
        if cached_semantic:
            return RAGResponse(
                answer=cached_semantic,
                sources=[],
                rewritten_query=question,
                retrieval_score=1.0,
                cache_hit=True,
                passed_gate=True,
            )

        # Retrieval
        retrieval = self._retriever.retrieve(question)

        if not retrieval.passed_gate:
            return RAGResponse(
                answer=NO_CONTEXT_RESPONSE,
                sources=[],
                rewritten_query=retrieval.rewritten_query,
                retrieval_score=retrieval.retrieval_score,
                cache_hit=False,
                passed_gate=False,
            )

        # Generation
        answer = self.llm.generate(question, retrieval.chunks)
        sources = [c.metadata.get("source", "") for c in retrieval.chunks]

        # Output guardrail
        output_check = check_output(answer, retrieval.chunks)
        if not output_check.passed:
            answer = NO_CONTEXT_RESPONSE

        # Store in both caches
        exact_cache.set(question, answer, sources, self.settings.cache_db_path)
        semantic_cache.set(question, answer, self.embedder, self.settings)

        return RAGResponse(
            answer=answer,
            sources=list(set(sources)),
            rewritten_query=retrieval.rewritten_query,
            retrieval_score=retrieval.retrieval_score,
            cache_hit=False,
            passed_gate=retrieval.passed_gate,
        )


@dataclass
class _LLMAdapter:
    """Adapts BaseLLM.complete for use as a QueryRewriter LLMClient."""

    _llm: BaseLLM = field(repr=False)

    def complete(self, prompt: str) -> str:
        return self._llm.complete(prompt)
