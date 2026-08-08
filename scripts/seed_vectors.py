#!/usr/bin/env python3
"""
Seed the vector store from the items catalog so the RAG pipeline has real
documents to retrieve against. In production, point --backend at pinecone
and run this whenever the catalog changes; for local dev the default
in-memory backend only lives for the process lifetime, so this is meant to
be re-run (or called from Container.init) rather than run once forever.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings import get_embedder
from src.core.retrieval.vector_store import VectorStore


async def seed(items_path: str, vector_store: VectorStore) -> int:
    items = pd.read_parquet(items_path) if items_path.endswith(".parquet") else pd.read_csv(items_path)
    embedder = get_embedder({})
    texts = (items["title"].astype(str) + ". " + items["description"].astype(str)).tolist()
    vecs = embedder.encode(texts)
    metas = [
        {
            "id": row.item_id,
            "text": text,
            "category": getattr(row, "category", None),
            "price": getattr(row, "price", None),
        }
        for row, text in zip(items.itertuples(), texts)
    ]
    vector_store.add_documents(vecs, metas)
    return len(metas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items.parquet")
    args = parser.parse_args()

    import asyncio

    async def run():
        vs = VectorStore({})
        await vs.initialize()
        n = await seed(args.items, vs)
        print(f"Seeded {n} documents into the vector store (len={len(vs)})")

    asyncio.run(run())


if __name__ == "__main__":
    main()
