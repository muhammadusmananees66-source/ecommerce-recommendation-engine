# """
# Vector store with a real (bounded) in-memory fallback.

# Design notes, addressing bugs found in earlier iterations of this project:
# - The fallback store starts genuinely empty. It is populated only through
#   add_documents(), called explicitly by whoever owns the corpus (a seeding
#   script, or the retriever after a successful primary-store hit) -- never
#   auto-seeded with synthetic placeholder text.
# - The store is bounded (max_size, default 50_000 vectors) with FIFO eviction,
#   so long-running processes cannot OOM by growing this structure forever.
# - If nothing matches, or the store is empty, query() returns an empty list
#   and logs a warning. It never fabricates results.
# - Pinecone is an optional backend, imported lazily so `pinecone-client`
#   doesn't need to be installed to run everything else.
# """

# import logging
# from collections import deque
# from typing import Any, Dict, List

# import numpy as np

# logger = logging.getLogger(__name__)


# class VectorStore:
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.backend = config.get("backend", "memory")
#         self.max_size = config.get("max_size", 50_000)
#         self.index = None
#         self._initialized = False

#         # Fallback / primary in-memory store. Bounded FIFO via deque.
#         self._ids: deque = deque(maxlen=self.max_size)
#         self._embeddings: deque = deque(maxlen=self.max_size)
#         self._metadata: deque = deque(maxlen=self.max_size)

#     async def initialize(self) -> None:
#         if self.backend == "pinecone":
#             try:
#                 from pinecone import Pinecone, ServerlessSpec

#                 pc = Pinecone(api_key=self.config.get("api_key"))
#                 index_name = self.config.get("index_name", "recommendations")
#                 if index_name not in pc.list_indexes().names():
#                     pc.create_index(
#                         name=index_name,
#                         dimension=self.config.get("dimension", 256),
#                         metric="cosine",
#                         spec=ServerlessSpec(
#                             cloud=self.config.get("cloud", "aws"),
#                             region=self.config.get("region", "us-west-2"),
#                         ),
#                     )
#                 self.index = pc.Index(index_name)
#                 self._initialized = True
#                 logger.info("Pinecone index '%s' initialized", index_name)
#                 return
#             except Exception as e:
#                 logger.error(
#                     "Pinecone initialization failed (%s); falling back to "
#                     "bounded in-memory vector store", e,
#                 )

#         self._initialized = True
#         logger.info(
#             "VectorStore running in-memory (max_size=%d). This is suitable "
#             "for dev/CI and as a resilience fallback, not as the sole "
#             "production index for a large catalog.", self.max_size,
#         )

#     def add_documents(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
#         """Add documents to the in-memory store. Real data only -- caller's responsibility."""
#         for emb, meta in zip(embeddings, metadatas):
#             self._ids.append(meta.get("id", str(len(self._ids))))
#             self._embeddings.append(np.asarray(emb, dtype=np.float32))
#             self._metadata.append(meta)

#     async def query(self, vector: np.ndarray, top_k: int = 10, **kwargs) -> List[Dict]:
#         if not self._initialized:
#             logger.error("VectorStore.query called before initialize()")
#             return []

#         if self.index is not None:
#             try:
#                 results = self.index.query(
#                     vector=vector.tolist(), top_k=top_k, include_metadata=True
#                 )
#                 return [
#                     {
#                         "id": m["id"],
#                         "score": self._clamp(m["score"]),
#                         "metadata": m.get("metadata", {}),
#                     }
#                     for m in results["matches"]
#                 ]
#             except Exception as e:
#                 logger.error("Pinecone query failed (%s); falling back to in-memory store", e)

#         if not self._embeddings:
#             logger.warning("In-memory vector store is empty; returning no results")
#             return []

#         from sklearn.metrics.pairwise import cosine_similarity

#         matrix = np.stack(list(self._embeddings))
#         sims = cosine_similarity(vector.reshape(1, -1), matrix).flatten()
#         top_indices = np.argsort(sims)[::-1][:top_k]

#         results = []
#         for i in top_indices:
#             if sims[i] <= 0.0:
#                 continue
#             meta = self._metadata[i]
#             results.append(
#                 {
#                     "id": self._ids[i],
#                     "score": self._clamp(float(sims[i])),
#                     # "text" is surfaced at the top level (not just inside
#                     # metadata) because downstream consumers (ContextBuilder,
#                     # RelevanceRanker, GroundednessChecker) all read doc["text"]
#                     # directly -- nesting it only under metadata silently
#                     # produced empty context in an earlier version of this file.
#                     "text": meta.get("text", ""),
#                     "metadata": meta,
#                 }
#             )
#         return results

#     @staticmethod
#     def _clamp(score: float) -> float:
#         return max(0.0, min(1.0, float(score)))

#     def __len__(self) -> int:
#         return len(self._embeddings)

#     async def close(self) -> None:
#         self._initialized = False


"""
Vector store with a real (bounded) in-memory fallback.

Design notes, addressing bugs found in earlier iterations of this project:
- The fallback store starts genuinely empty. It is populated only through
  add_documents(), called explicitly by whoever owns the corpus (a seeding
  script, or the retriever after a successful primary-store hit) -- never
  auto-seeded with synthetic placeholder text.
- The store is bounded (max_size, default 50_000 vectors) with FIFO eviction,
  so long-running processes cannot OOM by growing this structure forever.
- If nothing matches, or the store is empty, query() returns an empty list
  and logs a warning. It never fabricates results.
- Pinecone is an optional backend, imported lazily so `pinecone-client`
  doesn't need to be installed to run everything else.
"""

import logging
from collections import deque
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get("backend", "memory")
        self.max_size = config.get("max_size", 50_000)
        self.index = None
        self._initialized = False

        # Fallback / primary in-memory store. Bounded FIFO via deque.
        self._ids: deque = deque(maxlen=self.max_size)
        self._embeddings: deque = deque(maxlen=self.max_size)
        self._metadata: deque = deque(maxlen=self.max_size)

    async def initialize(self) -> None:
        if self.backend == "pinecone":
            try:
                from pinecone import Pinecone, ServerlessSpec

                pc = Pinecone(api_key=self.config.get("api_key"))
                index_name = self.config.get("index_name", "recommendations")
                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        dimension=self.config.get("dimension", 256),
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=self.config.get("cloud", "aws"),
                            region=self.config.get("region", "us-west-2"),
                        ),
                    )
                self.index = pc.Index(index_name)
                self._initialized = True
                logger.info("Pinecone index '%s' initialized", index_name)
                return
            except Exception as e:
                logger.error(
                    "Pinecone initialization failed (%s); falling back to bounded in-memory vector store",
                    e,
                )

        self._initialized = True
        logger.info(
            "VectorStore running in-memory (max_size=%d). This is suitable "
            "for dev/CI and as a resilience fallback, not as the sole "
            "production index for a large catalog.",
            self.max_size,
        )

    def add_documents(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        """Add documents to the in-memory store. Real data only -- caller's responsibility."""
        for emb, meta in zip(embeddings, metadatas):
            self._ids.append(meta.get("id", str(len(self._ids))))
            self._embeddings.append(np.asarray(emb, dtype=np.float32))
            self._metadata.append(meta)

    async def query(self, vector: np.ndarray, top_k: int = 10, **kwargs) -> List[Dict]:
        if not self._initialized:
            logger.error("VectorStore.query called before initialize()")
            return []

        if self.index is not None:
            try:
                results = self.index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
                return [
                    {
                        "id": m["id"],
                        "score": self._clamp(m["score"]),
                        "metadata": m.get("metadata", {}),
                    }
                    for m in results["matches"]
                ]
            except Exception as e:
                logger.error("Pinecone query failed (%s); falling back to in-memory store", e)

        if not self._embeddings:
            logger.warning("In-memory vector store is empty; returning no results")
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        matrix = np.stack(list(self._embeddings))
        sims = cosine_similarity(vector.reshape(1, -1), matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for i in top_indices:
            if sims[i] <= 0.0:
                continue
            meta = self._metadata[i]
            results.append(
                {
                    "id": self._ids[i],
                    "score": self._clamp(float(sims[i])),
                    # "text" is surfaced at the top level (not just inside
                    # metadata) because downstream consumers (ContextBuilder,
                    # RelevanceRanker, GroundednessChecker) all read doc["text"]
                    # directly -- nesting it only under metadata silently
                    # produced empty context in an earlier version of this file.
                    "text": meta.get("text", ""),
                    "metadata": meta,
                }
            )
        return results

    @staticmethod
    def _clamp(score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    def __len__(self) -> int:
        return len(self._embeddings)

    async def close(self) -> None:
        self._initialized = False
