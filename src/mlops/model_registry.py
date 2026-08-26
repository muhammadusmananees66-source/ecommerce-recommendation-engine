"""Model registry via MLflow, with an explicit RuntimeError (not a silent no-op) if unavailable.

Unlike experiment tracking (where a no-op fallback is harmless -- you just lose
some logging), registering/promoting a model is a decision that should never
silently do nothing. If MLflow isn't reachable, callers need to know loudly.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistryUnavailableError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, config: dict[str, Any]):
        self.tracking_uri = config.get("tracking_uri", "http://localhost:5000")
        self.model_name = config.get("model_name", "rag-recommendation")
        self._client = None
        self._mlflow = None

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow = mlflow
            self._client = MlflowClient()
        except Exception as e:
            logger.warning("MLflow unavailable for model registry (%s)", e)

    def _require_client(self):
        if self._client is None:
            raise ModelRegistryUnavailableError(
                "MLflow is not reachable; cannot register or query model versions"
            )

    def register_model(self, run_id: str, promote_to_production: bool = False) -> str:
        self._require_client()
        model_uri = f"runs:/{run_id}/model"
        result = self._mlflow.register_model(model_uri, self.model_name)
        if promote_to_production:
            self._client.transition_model_version_stage(
                name=self.model_name, version=result.version, stage="Production"
            )
        return result.version

    def get_latest_version(self, stage: str = "Production") -> dict[str, Any] | None:
        self._require_client()
        versions = self._client.get_latest_versions(self.model_name, stages=[stage])
        if not versions:
            return None
        v = versions[0]
        return {"version": v.version, "run_id": v.run_id, "stage": v.current_stage}