import math
from abc import ABC, abstractmethod

from nabucodonosor.config import Settings


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, base_url: str, model: str) -> None:
        import ollama

        self._client = ollama.Client(host=base_url)
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embed(model=self._model, input=text)
        return list(response.embeddings[0])  # type: ignore[index,unused-ignore]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self._model, input=texts)
        return [list(e) for e in response.embeddings]  # type: ignore[union-attr,unused-ignore]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped,unused-ignore]

            self._model = SentenceTransformer(model)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for EMBED_PROVIDER=sentence-transformers. "
                "Install with: uv sync --extra prod"
            ) from exc

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()  # type: ignore[no-any-return]


def get_embedder(settings: Settings) -> BaseEmbedder:
    if settings.embed_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.hf_embed_model)
    return OllamaEmbedder(settings.ollama_base_url, settings.embed_model)
