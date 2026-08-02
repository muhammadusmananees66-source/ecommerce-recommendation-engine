# """
# Dependency-injection container: owns the lifecycle of every stateful service
# (model, RAG pipeline, feature store, cache) and wires them together once at
# startup, per FastAPI's lifespan hook.
# """

# import logging
# from typing import Any, Dict, Optional

# from src.core.models.hybrid import HybridRecommender
# from src.core.rag.pipeline import RAGPipeline
# from src.data.storage.feature_store import FeatureStore
# from src.data.user_history import UserHistoryService
# from src.monitoring.rag_monitoring import RAGMonitor
# from src.serving.caching import CacheManager
# from src.serving.predictor import Predictor

# logger = logging.getLogger(__name__)


# class Container:
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.predictor: Optional[Predictor] = None
#         self.rag_pipeline: Optional[RAGPipeline] = None
#         self.feature_store: Optional[FeatureStore] = None
#         self.user_history: Optional[UserHistoryService] = None
#         self.cache_manager: Optional[CacheManager] = None
#         self.monitor: Optional[RAGMonitor] = None

#     async def init(self) -> None:
#         logger.info("Initializing DI container...")

#         self.feature_store = FeatureStore(self.config.get("feature_store", {}))
#         self.user_history = UserHistoryService(self.config.get("user_history", {}))
#         self.monitor = RAGMonitor()

#         model_config = self.config.get("model", {})
#         model = HybridRecommender(model_config)
#         model_path = model_config.get("path")
#         if model_path:
#             try:
#                 model.load(model_path)
#                 logger.info("Loaded recommender model from %s", model_path)
#             except FileNotFoundError:
#                 logger.warning(
#                     "No trained model found at %s -- recommendation endpoint will "
#                     "return empty results until a model is trained and saved there.",
#                     model_path,
#                 )
#         else:
#             logger.warning(
#                 "No model.path configured -- recommendation endpoint will return "
#                 "empty results until a model is trained and configured."
#             )

#         self.predictor = Predictor(
#             model=model,
#             feature_store=self.feature_store,
#             user_history=self.user_history,
#             config=model_config,
#         )

#         self.rag_pipeline = RAGPipeline(self.config.get("rag", {}))
#         await self.rag_pipeline.initialize()

#         seed_path = self.config.get("rag", {}).get("seed_items_path")
#         if seed_path:
#             await self._seed_vector_store(seed_path)
#         else:
#             logger.warning(
#                 "rag.seed_items_path not configured -- RAG queries will retrieve "
#                 "no documents until the vector store is seeded (see scripts/seed_vectors.py)."
#             )

#         self.cache_manager = CacheManager(
#             redis_client=self.feature_store.redis_client,
#             config=self.config.get("cache", {}),
#         )

#         logger.info("DI container initialized")

#     async def _seed_vector_store(self, items_path: str) -> None:
#         import os

#         import pandas as pd

#         if not os.path.exists(items_path):
#             logger.warning("rag.seed_items_path=%s does not exist; skipping seed", items_path)
#             return

#         items = pd.read_parquet(items_path) if items_path.endswith(".parquet") else pd.read_csv(items_path)
#         embedder = self.rag_pipeline.embedder
#         texts = (items["title"].astype(str) + ". " + items["description"].astype(str)).tolist()
#         vecs = embedder.encode(texts)
#         metas = [
#             {"id": row.item_id, "text": text}
#             for row, text in zip(items.itertuples(), texts)
#         ]
#         self.rag_pipeline.retriever.vector_store.add_documents(vecs, metas)
#         logger.info("Seeded vector store with %d documents from %s", len(metas), items_path)

#     async def shutdown(self) -> None:
#         if self.rag_pipeline:
#             await self.rag_pipeline.close()
#         if self.predictor:
#             self.predictor.shutdown()
#         logger.info("DI container shut down")


"""
Dependency-injection container: owns the lifecycle of every stateful service
(model, RAG pipeline, feature store, cache) and wires them together once at
startup, per FastAPI's lifespan hook.
"""

import logging
from typing import Any, Dict, Optional

from src.core.models.hybrid import HybridRecommender
from src.core.rag.pipeline import RAGPipeline
from src.data.storage.feature_store import FeatureStore
from src.data.user_history import UserHistoryService
from src.monitoring.rag_monitoring import RAGMonitor
from src.serving.caching import CacheManager
from src.serving.predictor import Predictor

logger = logging.getLogger(__name__)


class Container:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.predictor: Optional[Predictor] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.feature_store: Optional[FeatureStore] = None
        self.user_history: Optional[UserHistoryService] = None
        self.cache_manager: Optional[CacheManager] = None
        self.monitor: Optional[RAGMonitor] = None

    async def init(self) -> None:
        logger.info("Initializing DI container...")

        self.feature_store = FeatureStore(self.config.get("feature_store", {}))
        self.user_history = UserHistoryService(self.config.get("user_history", {}))
        self.monitor = RAGMonitor()

        model_config = self.config.get("model", {})
        model = HybridRecommender(model_config)
        model_path = model_config.get("path")
        if model_path:
            try:
                model.load(model_path)
                logger.info("Loaded recommender model from %s", model_path)
            except FileNotFoundError:
                logger.warning(
                    "No trained model found at %s -- recommendation endpoint will "
                    "return empty results until a model is trained and saved there.",
                    model_path,
                )
        else:
            logger.warning(
                "No model.path configured -- recommendation endpoint will return "
                "empty results until a model is trained and configured."
            )

        self.predictor = Predictor(
            model=model,
            feature_store=self.feature_store,
            user_history=self.user_history,
            config=model_config,
        )

        self.rag_pipeline = RAGPipeline(self.config.get("rag", {}))
        await self.rag_pipeline.initialize()

        seed_path = self.config.get("rag", {}).get("seed_items_path")
        if seed_path:
            await self._seed_vector_store(seed_path)
        else:
            logger.warning(
                "rag.seed_items_path not configured -- RAG queries will retrieve "
                "no documents until the vector store is seeded (see scripts/seed_vectors.py)."
            )

        self.cache_manager = CacheManager(
            redis_client=self.feature_store.redis_client,
            config=self.config.get("cache", {}),
        )

        logger.info("DI container initialized")

    async def _seed_vector_store(self, items_path: str) -> None:
        import os

        import pandas as pd

        if not os.path.exists(items_path):
            logger.warning("rag.seed_items_path=%s does not exist; skipping seed", items_path)
            return

        items = pd.read_parquet(items_path) if items_path.endswith(".parquet") else pd.read_csv(items_path)
        embedder = self.rag_pipeline.embedder
        texts = (items["title"].astype(str) + ". " + items["description"].astype(str)).tolist()
        vecs = embedder.encode(texts)
        metas = [{"id": row.item_id, "text": text} for row, text in zip(items.itertuples(), texts)]
        self.rag_pipeline.retriever.vector_store.add_documents(vecs, metas)
        logger.info("Seeded vector store with %d documents from %s", len(metas), items_path)

    async def shutdown(self) -> None:
        if self.rag_pipeline:
            await self.rag_pipeline.close()
        if self.predictor:
            self.predictor.shutdown()
        logger.info("DI container shut down")
