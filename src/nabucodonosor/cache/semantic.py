import chromadb

from nabucodonosor.config import Settings
from nabucodonosor.embedding.embedder import BaseEmbedder, cosine_similarity

_CACHE_COLLECTION = "semantic_cache"


def _get_collection(settings: Settings) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=_CACHE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get(
    question: str,
    embedder: BaseEmbedder,
    settings: Settings,
) -> str | None:
    try:
        collection = _get_collection(settings)
        if collection.count() == 0:
            return None

        q_emb = embedder.embed(question)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=1,
            include=["documents", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = (results["distances"] or [[1.0]])[0][0]
        similarity = 1.0 - distance

        if similarity >= settings.semantic_cache_threshold:
            docs = results["documents"]
            return str((docs or [[]])[0][0])

        return None
    except Exception:
        return None


def set(
    question: str,
    answer: str,
    embedder: BaseEmbedder,
    settings: Settings,
) -> None:
    try:
        collection = _get_collection(settings)
        q_emb = embedder.embed(question)
        # Use embedding hash as a stable ID
        import hashlib

        cache_id = hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]
        collection.upsert(
            ids=[cache_id],
            documents=[answer],
            embeddings=[q_emb],
            metadatas=[{"question": question[:200]}],
        )
    except Exception:
        pass


def similarity_to_cache(
    question_embedding: list[float],
    settings: Settings,
) -> float:
    """Return the highest cosine similarity between the query and any cached question."""
    try:
        collection = _get_collection(settings)
        if collection.count() == 0:
            return 0.0

        results = collection.get(include=["embeddings"])  # type: ignore[arg-type]
        raw = results.get("embeddings")
        if raw is None:
            return 0.0

        return max(cosine_similarity(question_embedding, [float(x) for x in emb]) for emb in raw)
    except Exception:
        return 0.0
