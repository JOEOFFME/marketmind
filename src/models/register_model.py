"""Register the best logged model (MLflow 3.x) and promote it to champion."""
import os

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

MODEL_NAME = "marketmind-success-model"
CHAMPION_ALIAS = "champion"


def r2_of(client, lm):
    """R2 from the logged model's metrics, falling back to its source run."""
    for m in (lm.metrics or []):
        if m.key == "r2":
            return m.value
    if lm.source_run_id:
        try:
            run = client.get_run(lm.source_run_id)
            if "r2" in run.data.metrics:
                return run.data.metrics["r2"]
        except Exception:
            pass
    return None


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()

    exp_ids = [e.experiment_id for e in client.search_experiments()]
    models = mlflow.search_logged_models(experiment_ids=exp_ids, output_format="list")

    if not models:
        logger.error("No logged models found. Did `make train` finish logging?")
        return

    best = None
    for lm in models:
        score = r2_of(client, lm)
        if score is None:
            continue
        if best is None or score > best[0]:
            best = (score, lm)

    if best is None:
        logger.error("No logged model with an r2 metric found.")
        return

    score, lm = best
    logger.info(
        f"Best model: {lm.name} | model_id={lm.model_id} | "
        f"run={lm.source_run_id} | R2={score:.4f}"
    )

    model_uri = f"models:/{lm.model_id}"
    version = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    logger.success(
        f"Alias '@{CHAMPION_ALIAS}' -> version {version.version}. "
        f"Worker loads: models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )


if __name__ == "__main__":
    main()
