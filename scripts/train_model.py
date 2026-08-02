#!/usr/bin/env python3
"""
Train the hybrid recommender on interactions/items CSV or Parquet files and
save it to disk in the layout Container expects (model.path).

Usage:
    python scripts/train_model.py \\
        --interactions data/interactions.parquet \\
        --items data/items.parquet \\
        --text-cols title description \\
        --output models/hybrid_v1 \\
        --epochs 20
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models.hybrid import HybridRecommender  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, help="CSV/Parquet with user_id,item_id,rating")
    parser.add_argument("--items", required=True, help="CSV/Parquet with item_id,<text columns>")
    parser.add_argument("--text-cols", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=32)
    args = parser.parse_args()

    interactions = _read_table(args.interactions)
    items = _read_table(args.items)

    required_interaction_cols = {"user_id", "item_id", "rating"}
    missing = required_interaction_cols - set(interactions.columns)
    if missing:
        logger.error("interactions file is missing required columns: %s", missing)
        sys.exit(1)

    model = HybridRecommender({"epochs": args.epochs, "embedding_dim": args.embedding_dim})
    history = model.fit(interactions, items, text_cols=args.text_cols)

    model.save(args.output)
    logger.info(
        "Model trained and saved to %s (final train_loss=%.4f)",
        args.output,
        history["train_loss"][-1],
    )


if __name__ == "__main__":
    main()
