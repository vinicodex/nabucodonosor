from dataclasses import dataclass

from nabucodonosor.retrieval.reranker import ScoredChunk


@dataclass
class OutputCheckResult:
    passed: bool
    reason: str = ""


def check_output(
    answer: str, chunks: list[ScoredChunk], min_overlap_words: int = 2
) -> OutputCheckResult:
    """Verify that the answer is grounded in the retrieved chunks."""
    if not answer.strip():
        return OutputCheckResult(passed=False, reason="Empty answer.")

    if not chunks:
        return OutputCheckResult(passed=False, reason="No context chunks provided.")

    answer_words = set(answer.lower().split())
    context_words: set[str] = set()
    for chunk in chunks:
        context_words.update(chunk.content.lower().split())

    overlap = answer_words & context_words
    if len(overlap) < min_overlap_words:
        return OutputCheckResult(
            passed=False,
            reason=f"Insufficient context overlap ({len(overlap)} shared words).",
        )

    return OutputCheckResult(passed=True)
