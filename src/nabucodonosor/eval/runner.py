"""RAGAS evaluation runner for the Nabucodonosor RAG pipeline."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from nabucodonosor.pipeline import RAGPipeline


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    passed_gate: bool = False


@dataclass
class EvalResult:
    samples: list[EvalSample]
    ragas_scores: dict[str, float] = field(default_factory=dict)

    @property
    def answer_coverage(self) -> float:
        """Fraction of questions that produced a non-empty, non-fallback answer."""
        if not self.samples:
            return 0.0
        answered = sum(
            1
            for s in self.samples
            if s.passed_gate and s.answer and "don't have enough" not in s.answer
        )
        return answered / len(self.samples)

    @property
    def gate_pass_rate(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.passed_gate) / len(self.samples)


def run_eval(
    pipeline: RAGPipeline,
    golden_path: Path,
    max_samples: int | None = None,
) -> EvalResult:
    with golden_path.open() as f:
        dataset: list[dict[str, str]] = json.load(f)

    if max_samples is not None:
        dataset = dataset[:max_samples]

    samples: list[EvalSample] = []
    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]

        response = pipeline.query(question)
        sample = EvalSample(
            question=question,
            ground_truth=ground_truth,
            answer=response.answer,
            contexts=[c.content for c in response.chunks] if hasattr(response, "chunks") else [],
            retrieval_score=response.retrieval_score,
            passed_gate=response.passed_gate,
        )
        samples.append(sample)

    result = EvalResult(samples=samples)

    try:
        result.ragas_scores = _run_ragas(samples, pipeline)
    except Exception as e:
        print(f"[RAGAS] Skipped: {e}")

    return result


def _run_ragas(samples: list[EvalSample], pipeline: RAGPipeline) -> dict[str, float]:
    from datasets import Dataset  # type: ignore[import-untyped,unused-ignore]
    from ragas import evaluate  # type: ignore[import-untyped,unused-ignore]
    from ragas.metrics import (  # type: ignore[import-untyped,unused-ignore]
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    ragas_data = {
        "question": [s.question for s in samples if s.contexts],
        "answer": [s.answer for s in samples if s.contexts],
        "contexts": [s.contexts for s in samples if s.contexts],
        "ground_truth": [s.ground_truth for s in samples if s.contexts],
    }

    if not ragas_data["question"]:
        return {}

    ds = Dataset.from_dict(ragas_data)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    return {str(k): float(v) for k, v in result.items()}  # type: ignore[union-attr]
