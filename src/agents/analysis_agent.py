"""Agent 3 — ML Analysis (predicts success with the champion model)."""
import os

import mlflow
from loguru import logger

from .data import filter_subset, load_features_df
from .state import AgentState

MODEL_URI = os.environ.get("MODEL_URI", "models:/marketmind-success-model@champion")
_model = None


def _get_model():
    global _model
    if _model is None:
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"[analysis] loading model {MODEL_URI}")
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model


def analysis_agent(state: AgentState) -> AgentState:
    parsed = state.get("parsed", {})
    district, btype = parsed.get("district"), parsed.get("type")
    df = load_features_df()
    sub, _ = filter_subset(df, district, btype)
    if len(sub) == 0:
        return {"analysis": {"note": "no matching places to analyze"}}

    model = _get_model()
    feat_names = list(getattr(model, "feature_names_in_", []))
    if feat_names:
        X = sub.reindex(columns=feat_names, fill_value=0)
    else:
        X = sub.select_dtypes("number").drop(
            columns=["success_score", "success_score_pct"], errors="ignore"
        )
    preds = model.predict(X)

    analysis = {
        "predicted_mean": round(float(preds.mean()), 1),
        "predicted_max": round(float(preds.max()), 1),
    }
    if hasattr(model, "feature_importances_") and feat_names:
        ranked = sorted(
            zip(feat_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        analysis["top_drivers"] = [f for f, _ in ranked[:5]]
    logger.info(f"[analysis] predicted_mean={analysis['predicted_mean']}")
    return {"analysis": analysis}


if __name__ == "__main__":
    import json

    from dotenv import load_dotenv

    from .query_agent import query_agent
    from .retrieval_agent import retrieval_agent

    load_dotenv()
    s = {"question": "Où ouvrir un café à Agdal ?"}
    s.update(query_agent(s))
    s.update(retrieval_agent(s))
    s.update(analysis_agent(s))
    print(json.dumps(s["analysis"], indent=2, ensure_ascii=False))
