from dataclasses import dataclass

import chromadb

from nabucodonosor.embedding.embedder import cosine_similarity
from nabucodonosor.vectorstore.chroma import SearchResult, get_embeddings_by_ids


@dataclass
class ScoredChunk:
    chunk_id: str
    content: str
    metadata: dict[str, str]
    score: float


def rerank(
    question_embedding: list[float],
    candidates: list[SearchResult],
    collection: chromadb.Collection,
) -> list[ScoredChunk]:
    if not candidates:
        return []

    ids = [c.chunk_id for c in candidates]

    # Reuse embeddings that came with candidates, fall back to fetching from ChromaDB
    id_to_emb: dict[str, list[float]] = {
        c.chunk_id: c.embedding for c in candidates if c.embedding is not None
    }
    missing = [cid for cid in ids if cid not in id_to_emb]
    if missing:
        fetched = get_embeddings_by_ids(collection, missing)
        id_to_emb.update(fetched)

    scored: list[ScoredChunk] = []
    for candidate in candidates:
        emb = id_to_emb.get(candidate.chunk_id)
        if emb is None:
            continue
        score = cosine_similarity(question_embedding, emb)
        scored.append(
            ScoredChunk(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                metadata=candidate.metadata,
                score=score,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def quality_gate(scored_chunks: list[ScoredChunk], threshold: float) -> bool:
    if not scored_chunks:
        return False
    return bool(scored_chunks[0].score >= threshold)
