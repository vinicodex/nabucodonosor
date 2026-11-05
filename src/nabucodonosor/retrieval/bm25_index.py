from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped,unused-ignore]


@dataclass
class BM25Index:
    _corpus: list[str] = field(default_factory=list)
    _ids: list[str] = field(default_factory=list)
    _index: BM25Okapi | None = field(default=None, init=False, repr=False)

    def build(self, ids: list[str], documents: list[str]) -> None:
        self._ids = ids
        self._corpus = documents
        tokenized = [doc.lower().split() for doc in documents]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if self._index is None or not self._ids:
            return []

        tokens = query.lower().split()
        raw_scores: list[float] = self._index.get_scores(tokens).tolist()

        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score == 0.0:
            return []

        scored = [(self._ids[i], raw_scores[i] / max_score) for i in range(len(self._ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    @property
    def size(self) -> int:
        return len(self._ids)
