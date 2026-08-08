#!/usr/bin/env python3
"""
Generates a small, clearly-synthetic dataset so the project is runnable
out of the box for local dev, tests, and CI evaluation gates, without
requiring anyone's real production data on day one.

This data is for bootstrapping only -- replace it with real interaction
and catalog data before training a model you intend to actually serve.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    n_users, n_items = 200, 100
    categories = ["electronics", "books", "home", "sports", "toys"]

    items = pd.DataFrame(
        {
            "item_id": [f"item_{i}" for i in range(n_items)],
            "title": [f"{categories[i % len(categories)].title()} Product {i}" for i in range(n_items)],
            "description": [
                f"A {categories[i % len(categories)]} item, SKU {i}, good quality." for i in range(n_items)
            ],
            "category": [categories[i % len(categories)] for i in range(n_items)],
            "price": rng.uniform(5, 500, size=n_items).round(2),
        }
    )

    rows = []
    for u in range(n_users):
        preferred_cat_idx = u % len(categories)
        n_interactions = rng.integers(5, 30)
        for _ in range(n_interactions):
            item_idx = rng.integers(0, n_items)
            same_category = item_idx % len(categories) == preferred_cat_idx
            rating = float(rng.integers(4, 6)) if same_category else float(rng.integers(1, 4))
            rows.append({"user_id": f"user_{u}", "item_id": f"item_{item_idx}", "rating": rating})
    interactions = pd.DataFrame(rows)

    # Small RAG evaluation set: query + a ground-truth answer grounded in one
    # of the item descriptions, so faithfulness/relevancy metrics mean something.
    rag_eval_rows = [
        {
            "query": f"Tell me about {row.title}",
            "ground_truth": row.description,
        }
        for row in items.sample(20, random_state=42).itertuples()
    ]
    rag_eval = pd.DataFrame(rag_eval_rows)

    items.to_parquet(out_dir / "items.parquet", index=False)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    rag_eval.to_parquet(out_dir / "rag_test.parquet", index=False)

    print(
        f"Wrote {len(items)} items, {len(interactions)} interactions, {len(rag_eval)} eval rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
