"""FastAPI REST layer for the Nabucodonosor RAG pipeline."""

from fastapi import FastAPI
from pydantic import BaseModel

from nabucodonosor.config import Settings
from nabucodonosor.pipeline import RAGPipeline, RAGResponse

app = FastAPI(
    title="Nabucodonosor",
    description="RAG API — Ask questions about Marcos Vinicius Nascimento's CV.",
    version="0.1.0",
)

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(settings=Settings())
    return _pipeline


class QueryRequest(BaseModel):
    question: str


class QueryResponseModel(BaseModel):
    answer: str
    sources: list[str]
    rewritten_query: str
    retrieval_score: float
    cache_hit: bool
    passed_gate: bool
    blocked: bool
    block_reason: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponseModel)
def query(request: QueryRequest) -> QueryResponseModel:
    pipeline = _get_pipeline()
    response: RAGResponse = pipeline.query(request.question)
    return QueryResponseModel(
        answer=response.answer,
        sources=response.sources,
        rewritten_query=response.rewritten_query,
        retrieval_score=response.retrieval_score,
        cache_hit=response.cache_hit,
        passed_gate=response.passed_gate,
        blocked=response.blocked,
        block_reason=response.block_reason,
    )
