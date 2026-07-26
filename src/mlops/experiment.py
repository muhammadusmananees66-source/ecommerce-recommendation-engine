"""Experiment tracking via MLflow, with a safe no-op fallback if MLflow isn't reachable."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ExperimentTracker:
    def __init__(self, config: Dict[str, Any]):
        self.tracking_uri = config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = config.get("experiment_name", "rag-recommendation")
        self._mlflow = None
        self._initialized = False

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._initialized = True
        except Exception as e:
            logger.warning("MLflow unavailable (%s); experiment tracking will no-op", e)

    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        if not self._initialized:
            yield _NullRun()
            return
        with self._mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._initialized:
            return
        for k, v in params.items():
            self._mlflow.log_param(k, v)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._initialized:
            return
        for k, v in metrics.items():
            self._mlflow.log_metric(k, v, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if not self._initialized:
            return
        self._mlflow.log_artifact(local_path, artifact_path)