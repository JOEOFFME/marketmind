"""Agent 3 — ML Analysis.

Predicts the success score for the queried district / business type and builds a
rich, *client-friendly* factor breakdown. Feature importances come from the SHAP
table pre-computed during training (train.py → shap_<model>.csv), falling back to
the model's built-in importances. Every factor is given a plain-language label so
the recommendation never leaks raw column names or jargon.
"""
import glob
import os

import mlflow
import mlflow.models
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from loguru import logger

from .data import filter_subset, load_features_df
from .state import AgentState

MODEL_URI = os.environ.get("MODEL_URI", "models:/marketmind-success-model@champion")
SHAP_DIR = os.environ.get("SHAP_DIR", "data/processed")
_model = None

# Plain-language labels so the insight agent never exposes raw feature names.
FEATURE_LABELS = {
    "income_index": "district income level",
    "foot_traffic_index": "pedestrian foot traffic",
    "population_density": "neighborhood population density",
    "type_count_in_district": "direct (same-type) competitors in the district",
    "district_place_count": "total businesses in the district",
    "cafes_500m": "cafés within 500 m",
    "restaurants_500m": "restaurants within 500 m",
    "pharmacies_500m": "pharmacies within 500 m",
    "banks_500m": "banks within 500 m",
    "total_pois_500m": "points of interest within 500 m",
    "log_total_pois_500m": "commercial activity density nearby",
    "dist_nearest_pharmacy": "distance to nearest pharmacy (m)",
    "dist_nearest_bank": "distance to nearest bank (m)",
    "dist_nearest_supermarket": "distance to nearest supermarket (m)",
    "dist_nearest_fuel": "distance to nearest fuel station (m)",
    "price_level_encoded": "typical price level in the area",
    "opening_days": "weekly opening days",
    "avg_annual_temp": "average annual temperature (°C)",
    "rainy_days_per_year": "rainy days per year",
    "peak_season_months": "peak tourist-season months",
}


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def _get_model():
    global _model
    if _model is None:
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"[analysis] loading model {MODEL_URI}")
        # The champion can be any flavor (sklearn / xgboost / lightgbm); pick the
        # matching native loader instead of assuming sklearn.
        flavors = mlflow.models.get_model_info(MODEL_URI).flavors
        if "sklearn" in flavors:
            _model = mlflow.sklearn.load_model(MODEL_URI)
        elif "xgboost" in flavors:
            _model = mlflow.xgboost.load_model(MODEL_URI)
        elif "lightgbm" in flavors:
            import mlflow.lightgbm as mlflow_lightgbm

            _model = mlflow_lightgbm.load_model(MODEL_URI)
        else:
            _model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info(f"[analysis] loaded flavor(s): {list(flavors)}")
    return _model


def _load_shap_importances() -> dict[str, float]:
    """Load the most recent shap_<model>.csv as {feature: mean_abs_shap}."""
    shap_files = glob.glob(f"{SHAP_DIR}/shap_*.csv")
    if not shap_files:
        logger.warning("[analysis] no SHAP CSV found — falling back to model importances")
        return {}
    latest = max(shap_files, key=os.path.getmtime)
    try:
        df = pd.read_csv(latest)
        if {"feature", "importance"}.issubset(df.columns):
            return dict(zip(df["feature"], df["importance"]))
        logger.warning(f"[analysis] unexpected SHAP columns: {df.columns.tolist()}")
    except Exception as exc:
        logger.warning(f"[analysis] could not read SHAP file {latest}: {exc}")
    return {}


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
        feat_names = list(X.columns)

    preds = model.predict(X)
    analysis: dict = {
        "predicted_mean": round(float(preds.mean()), 1),
        "predicted_max": round(float(preds.max()), 1),
    }

    # Rank features by importance (SHAP preferred, model importances as fallback).
    shap_importances = _load_shap_importances()
    if shap_importances and feat_names:
        source = "shap"
        ranked = sorted(
            ((f, shap_importances.get(f, 0.0)) for f in feat_names),
            key=lambda x: x[1],
            reverse=True,
        )
    elif hasattr(model, "feature_importances_") and feat_names:
        source = "model"
        ranked = sorted(
            zip(feat_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        source = "none"
        ranked = []

    analysis["importance_source"] = source
    if ranked:
        analysis["top_drivers"] = [_label(f) for f, _ in ranked[:5]]

        # Top factors with a representative value (median over the subset) and a
        # human label — the insight agent turns these into plain-language reasons.
        factors: dict = {}
        for fname, imp in ranked[:12]:
            value = None
            if fname in sub.columns:
                try:
                    value = round(float(sub[fname].median()), 2)
                except Exception:
                    pass
            factors[_label(fname)] = {
                "value": value,
                "importance": round(float(imp), 4),
            }
        analysis["factors_detail"] = factors

    logger.info(
        f"[analysis] predicted_mean={analysis['predicted_mean']} "
        f"source={source} factors={len(analysis.get('factors_detail', {}))}"
    )
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