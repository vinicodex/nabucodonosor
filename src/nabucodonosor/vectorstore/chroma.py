from dataclasses import dataclass

import chromadb

from nabucodonosor.config import Settings
from nabucodonosor.embedding.embedder import BaseEmbedder
from nabucodonosor.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: dict[str, str]
    score: float
    embedding: list[float] | None = None


def build_collection(settings: Settings) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    embedder: BaseEmbedder,
) -> int:
    if not chunks:
        return 0

    ids = [
        f"{c.metadata.get('source', 'doc')}_{c.metadata.get('chunk_index', i)}"
        for i, c in enumerate(chunks)
    ]
    documents = [c.content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    embeddings = embedder.embed_batch(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


def similarity_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    k: int = 5,
    include_embeddings: bool = False,
) -> list[SearchResult]:
    include: list[str] = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, count),
        include=include,  # type: ignore[arg-type]
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    output: list[SearchResult] = []
    ids = results["ids"][0]
    documents = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]
    distances = (results["distances"] or [[]])[0]
    raw_embeddings = results.get("embeddings")
    emb_list = raw_embeddings[0] if raw_embeddings else None

    for i, chunk_id in enumerate(ids):
        # ChromaDB cosine distance = 1 - similarity; clamp for float precision
        score = max(0.0, min(1.0, 1.0 - distances[i]))
        emb = [float(x) for x in emb_list[i]] if emb_list is not None else None
        output.append(
            SearchResult(
                chunk_id=chunk_id,
                content=documents[i],
                metadata=dict(metadatas[i]),
                score=score,
                embedding=emb,
            )
        )

    return output


def get_embeddings_by_ids(
    collection: chromadb.Collection,
    ids: list[str],
) -> dict[str, list[float]]:
    if not ids:
        return {}

    results = collection.get(ids=ids, include=["embeddings"])  # type: ignore[arg-type]
    if not results["ids"]:
        return {}

    raw = results.get("embeddings")
    if raw is None:
        return {}
    return {chunk_id: [float(x) for x in emb] for chunk_id, emb in zip(results["ids"], raw)}
