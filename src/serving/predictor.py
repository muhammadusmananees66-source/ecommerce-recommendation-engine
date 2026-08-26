# """
# Serving-time wrapper around HybridRecommender. Model inference (numpy matrix
# ops) is synchronous/CPU-bound, so it's offloaded to a thread pool from the
# async path -- calling it directly inside `async def` would block the event
# loop for every request, which is what happened in an earlier version of
# this pipeline.
# """

# import asyncio
# import logging
# from concurrent.futures import ThreadPoolExecutor
# from typing import Any, Dict, List, Optional

# from src.core.models.hybrid import HybridRecommender
# from src.data.storage.feature_store import FeatureStore
# from src.data.user_history import UserHistoryService

# logger = logging.getLogger(__name__)


# class Predictor:
#     def __init__(
#         self,
#         model: HybridRecommender,
#         feature_store: FeatureStore,
#         user_history: Optional[UserHistoryService],
#         config: Dict[str, Any],
#     ):
#         self.model = model
#         self.feature_store = feature_store
#         self.user_history = user_history
#         self.model_version = config.get("version", "1.0.0")
#         self._executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))

#     async def get_recommendations(self, user_id: str, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
#         if not self.model.is_fitted:
#             logger.warning(
#                 "Recommendation requested for user '%s' but no model is trained/loaded yet; "
#                 "returning empty list rather than a 502 -- this is expected before the first "
#                 "training run, not a service failure.", user_id,
#             )
#             return []
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(self._executor, self._get_recommendations_sync, user_id, n)

#     def _get_recommendations_sync(self, user_id: str, n: int) -> List[Dict[str, Any]]:
#         user_items = self.user_history.get_user_items(user_id) if self.user_history else []
#         return self.model.recommend(user_id=user_id, n=n, user_items=user_items)

#     def shutdown(self) -> None:
#         self._executor.shutdown(wait=True)

















"""
Serving-time wrapper around HybridRecommender. Model inference (numpy matrix
ops) is synchronous/CPU-bound, so it's offloaded to a thread pool from the
async path -- calling it directly inside `async def` would block the event
loop for every request, which is what happened in an earlier version of
this pipeline.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.models.hybrid import HybridRecommender
from src.data.storage.feature_store import FeatureStore
from src.data.user_history import UserHistoryService

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model: HybridRecommender,
        feature_store: FeatureStore,
        user_history: UserHistoryService | None,
        config: dict[str, Any],
    ):
        self.model = model
        self.feature_store = feature_store
        self.user_history = user_history
        self.model_version = config.get("version", "1.0.0")
        self._executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))

    async def get_recommendations(self, user_id: str, n: int = 10, **kwargs) -> list[dict[str, Any]]:
        if not self.model.is_fitted:
            logger.warning(
                "Recommendation requested for user '%s' but no model is trained/loaded yet; "
                "returning empty list rather than a 502 -- this is expected before the first "
                "training run, not a service failure.", user_id,
            )
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_recommendations_sync, user_id, n)

    def _get_recommendations_sync(self, user_id: str, n: int) -> list[dict[str, Any]]:
        user_items = self.user_history.get_user_items(user_id) if self.user_history else []
        return self.model.recommend(user_id=user_id, n=n, user_items=user_items)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)