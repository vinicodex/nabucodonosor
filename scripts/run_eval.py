#!/usr/bin/env python
"""Run RAGAS evaluation against the golden dataset.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --max-samples 10
    uv run python scripts/run_eval.py --output eval/results.json
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabucodonosor.config import settings
from nabucodonosor.eval.runner import run_eval
from nabucodonosor.pipeline import RAGPipeline


def main(golden_path: Path, max_samples: int | None, output: Path | None) -> None:
    if not golden_path.exists():
        print(f"Golden dataset not found: {golden_path}")
        sys.exit(1)

    print("Loading pipeline...")
    pipeline = RAGPipeline(settings=settings)

    print(f"Running eval on {golden_path} ...")
    result = run_eval(pipeline, golden_path, max_samples=max_samples)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results ({len(result.samples)} samples)")
    print(f"{'=' * 50}")
    print(f"Answer coverage : {result.answer_coverage:.1%}")
    print(f"Gate pass rate  : {result.gate_pass_rate:.1%}")

    if result.ragas_scores:
        print("\nRAGAS metrics:")
        for metric, score in result.ragas_scores.items():
            print(f"  {metric:25s}: {score:.3f}")

    if output:
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "num_samples": len(result.samples),
            "answer_coverage": result.answer_coverage,
            "gate_pass_rate": result.gate_pass_rate,
            "ragas_scores": result.ragas_scores,
            "samples": [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "retrieval_score": s.retrieval_score,
                    "passed_gate": s.passed_gate,
                }
                for s in result.samples
            ],
        }
        output.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).parent.parent / "eval" / "golden_dataset.json",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    main(args.golden, args.max_samples, args.output)
