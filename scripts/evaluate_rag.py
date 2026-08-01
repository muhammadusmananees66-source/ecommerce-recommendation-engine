#!/usr/bin/env python3
"""
RAG evaluation runner, used as a CI quality gate before deploying a new
build. Exits non-zero (failing the CI job) if metrics fall below threshold.

Note on thresholds: the defaults below are calibrated for the default
TF-IDF/hashing embedder used in CI (no network access, no model download).
If you switch RAG_EMBEDDER_BACKEND to sentence-transformers for production,
re-calibrate these thresholds against a held-out eval set on that backend --
semantic embedding similarity scores are not directly comparable across
embedding backends.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.dependencies.container import Container  # noqa: E402
from src.evaluation.rag_evaluator import RAGEvaluator  # noqa: E402


async def run_evaluation(
    test_data_path: str,
    seed_items_path: str,
    faithfulness_threshold: float,
    relevancy_threshold: float,
    output_file: str,
) -> dict:
    print(f"Loading eval set from {test_data_path}...")
    test_data = pd.read_parquet(test_data_path)

    config = {
        "rag": {
            "seed_items_path": seed_items_path,
            "llm": {"available_models": ["local-echo"], "fallback_chain": ["local-echo"]},
            "generator": {"api_keys": {}},
        },
        "model": {},
    }
    container = Container(config)
    await container.init()

    evaluator = RAGEvaluator({})
    await evaluator.initialize()

    results = []
    for row in test_data.itertuples():
        rag_result = await container.rag_pipeline.query(row.query)
        eval_result = await evaluator.evaluate(
            query=row.query,
            answer=rag_result.generated_response,
            contexts=[d.get("text", "") for d in rag_result.retrieved_docs],
            ground_truth=getattr(row, "ground_truth", None),
        )
        results.append(
            {
                "query": row.query,
                "faithfulness": eval_result.faithfulness,
                "answer_relevancy": eval_result.answer_relevancy,
                "context_precision": eval_result.context_precision,
                "groundedness": rag_result.groundedness_score,
            }
        )

    await container.shutdown()

    df = pd.DataFrame(results)
    avg_faithfulness = float(df["faithfulness"].mean()) if len(df) else 0.0
    avg_relevancy = float(df["answer_relevancy"].mean()) if len(df) else 0.0

    passed = avg_faithfulness >= faithfulness_threshold and avg_relevancy >= relevancy_threshold
    output = {
        "status": "passed" if passed else "failed",
        "metrics": {"avg_faithfulness": avg_faithfulness, "avg_answer_relevancy": avg_relevancy},
        "thresholds": {"faithfulness": faithfulness_threshold, "relevancy": relevancy_threshold},
        "count": len(results),
        "detailed_results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="data/rag_test.parquet")
    parser.add_argument("--seed-items", default="data/items.parquet")
    parser.add_argument("--threshold-faithfulness", type=float, default=0.3)
    parser.add_argument("--threshold-relevancy", type=float, default=0.2)
    parser.add_argument("--output-file", default="rag-evaluation.json")
    args = parser.parse_args()

    result = asyncio.run(
        run_evaluation(
            args.test_data, args.seed_items, args.threshold_faithfulness,
            args.threshold_relevancy, args.output_file,
        )
    )

    print(json.dumps(result["metrics"], indent=2))
    if result["status"] != "passed":
        print("RAG evaluation FAILED thresholds", file=sys.stderr)
        sys.exit(1)
    print("RAG evaluation passed")


if __name__ == "__main__":
    main()