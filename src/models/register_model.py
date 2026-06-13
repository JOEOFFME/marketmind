"""Register the latest logged model (MLflow 3.x) and promote it to champion."""

import os

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

MODEL_NAME = "marketmind-success-model"
CHAMPION_ALIAS = "champion"


def main():
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    client = MlflowClient()

    exp_ids = [e.experiment_id for e in client.search_experiments()]

    # ✅ Prendre uniquement le plus récent
    models = mlflow.search_logged_models(
        experiment_ids=exp_ids,
        output_format="list",
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
    )

    if not models:
        logger.error("No logged models found. Did `make train` finish logging?")
        return

    # ✅ Premier = le plus récent, pas besoin de comparer les R²
    lm = models[0]
    logger.info(
        f"Latest model: {lm.name} | model_id={lm.model_id} | "
        f"run={lm.source_run_id}"
    )

    # ✅ Toujours enregistrer, sans vérification de doublon
    model_uri = f"models:/{lm.model_id}"
    version = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    logger.success(
        f"Registered version {version.version}. "
        f"Alias '@{CHAMPION_ALIAS}' -> version {version.version}. "
        f"Worker loads: models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )


if __name__ == "__main__":
    main()