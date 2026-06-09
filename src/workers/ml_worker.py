"""MarketMind ML Worker — consumes prediction jobs from Redis."""
import json
import os
import time

import redis
from loguru import logger

QUEUE_KEY = "marketmind:predict_jobs"
RESULT_PREFIX = "marketmind:result:"
POLL = 2

MODEL_URI = os.environ.get("MODEL_URI", "models:/marketmind-success-model@champion")


def load_model():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    logger.info(f"Loading model from {MODEL_URI}...")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Model loaded.")
    return model


def run():
    logger.info("ML Worker starting...")
    r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    model = None
    while model is None:
        try:
            model = load_model()
        except Exception as e:
            logger.warning(f"Model load failed, retry in 10s: {e}")
            time.sleep(10)
    logger.info(f"Worker ready — polling '{QUEUE_KEY}'")
    while True:
        try:
            raw = r.blpop(QUEUE_KEY, timeout=POLL)
            if raw is None:
                continue
            job = json.loads(raw[1])
            result = {"job_id": job.get("job_id"), "status": "pending_phase5"}
            r.setex(f"{RESULT_PREFIX}{job.get('job_id')}", 300, json.dumps(result))
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(POLL)


if __name__ == "__main__":
    run()
