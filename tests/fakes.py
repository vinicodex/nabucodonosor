"""Lightweight test doubles shared across the test suite."""

import hashlib
import math

from nabucodonosor.embedding.embedder import BaseEmbedder


class FakeEmbedder(BaseEmbedder):
    """Deterministic embedder for tests — no network, no torch."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        vec = [float((seed + i * 7919) % 100 + 1) for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]
