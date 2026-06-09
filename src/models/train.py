"""
Phase 3 — ML Training
Trains and compares multiple models, then builds a stacking ensemble.
All experiments tracked in MLflow.

Models compared:
    - XGBoost
    - LightGBM
    - Random Forest
    - Ridge Regression (baseline)
    - Stacking Ensemble (XGB + RF + GBM → Ridge meta-learner)

Usage:
    make train
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.config import settings

# ── Config ──────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/features.parquet")
TARGET_COL = "success_score"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

DROP_COLS = [
    "place_id",
    "name",
    "rating",
    "review_count",
    "price_level",
    "log_reviews",
    "rating_norm",
    "reviews_norm",
    "log_review_count",
]


# ── Data loading ─────────────────────────────────────────────────────────────
def load_features() -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop + [TARGET_COL])
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    y = df[TARGET_COL]

    logger.info(f"X shape : {X.shape}")
    logger.info(f"y range : {y.min():.1f} → {y.max():.1f}, mean={y.mean():.1f}")
    return X, y


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(model, X_test, y_test, X, y) -> dict:
    y_pred = model.predict(X_test)
    cv = cross_val_score(model, X, y, cv=N_FOLDS, scoring="r2")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
    }


# ── Individual models ─────────────────────────────────────────────────────────
def train_ridge(X_train, X_test, y_train, y_test, X, y) -> tuple:
    logger.info("Training Ridge (baseline)...")
    with mlflow.start_run(run_name="ridge_baseline"):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test, X, y)
        mlflow.log_params({"alpha": 1.0, "model_type": "ridge"})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_saved", "ridge")
        logger.info(f"  Ridge     → R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}")
    return "ridge", model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, X, y) -> tuple:
    logger.info("Training Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 5,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test, X, y)
        mlflow.log_params({**params, "model_type": "random_forest"})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_saved", "rf")
        logger.info(f"  RF        → R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}")
    return "random_forest", model, metrics


def train_xgboost(X_train, X_test, y_train, y_test, X, y) -> tuple:
    logger.info("Training XGBoost with Optuna tuning...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
            "random_state": RANDOM_STATE,
        }
        m = XGBRegressor(**params, verbosity=0)
        cv = cross_val_score(m, X_train, y_train, cv=3, scoring="r2")
        return cv.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params

    with mlflow.start_run(run_name="xgboost_tuned"):
        model = XGBRegressor(**best_params, random_state=RANDOM_STATE, verbosity=0)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test, X, y)
        mlflow.log_params({**best_params, "model_type": "xgboost_tuned"})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_saved", "xgboost")
        logger.info(f"  XGBoost   → R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}")
    return "xgboost", model, metrics


def train_lightgbm(X_train, X_test, y_train, y_test, X, y) -> tuple:
    logger.info("Training LightGBM with Optuna tuning...")
    try:
        import lightgbm as lgb
        import mlflow.lightgbm

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
                "random_state": RANDOM_STATE,
                "verbose": -1,
            }
            m = lgb.LGBMRegressor(**params)
            cv = cross_val_score(m, X_train, y_train, cv=3, scoring="r2")
            return cv.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best_params = {**study.best_params, "verbose": -1, "random_state": RANDOM_STATE}

        with mlflow.start_run(run_name="lightgbm_tuned"):
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
            metrics = compute_metrics(model, X_test, y_test, X, y)
            mlflow.log_params({**best_params, "model_type": "lightgbm_tuned"})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_saved", "lightgbm")
            logger.info(
                f"  LightGBM  → R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}"
            )
        return "lightgbm", model, metrics

    except ImportError:
        logger.warning("LightGBM not installed — skipping")
        return None, None, None


# ── Stacking Ensemble ─────────────────────────────────────────────────────────
def train_stacking(
    xgb_model, rf_model, X_train, X_test, y_train, y_test, X, y
) -> tuple:
    logger.info("Training Stacking Ensemble...")

    estimators = [
        (
            "xgb",
            XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "gbm",
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="stacking_ensemble"):
        stacking.fit(X_train, y_train)
        metrics = compute_metrics(stacking, X_test, y_test, X, y)
        mlflow.log_params(
            {
                "model_type": "stacking",
                "base_learners": "xgboost+random_forest+gbm",
                "meta_learner": "ridge",
                "cv_folds": 5,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_saved", "stacking")
        logger.info(f"  Stacking  → R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}")
    return "stacking", stacking, metrics


# ── SHAP feature importance ───────────────────────────────────────────────────
def compute_shap(model, X_test: pd.DataFrame, model_name: str):
    logger.info(f"Computing SHAP values for {model_name}...")
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_shap = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("importance", ascending=False)

        logger.info(f"\nTop 10 most important features ({model_name}):")
        logger.info(f"\n{mean_shap.head(10).to_string(index=False)}")

        mean_shap.to_csv(f"data/processed/shap_{model_name}.csv", index=False)
        return mean_shap
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("marketmind_marketplace_prediction")

    logger.info("=== Phase 3 — Model Training starting ===")

    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    results = []

    name, ridge_model, metrics = train_ridge(X_train, X_test, y_train, y_test, X, y)
    results.append((name, ridge_model, metrics))

    name, rf_model, metrics = train_random_forest(
        X_train, X_test, y_train, y_test, X, y
    )
    results.append((name, rf_model, metrics))

    name, xgb_model, metrics = train_xgboost(X_train, X_test, y_train, y_test, X, y)
    results.append((name, xgb_model, metrics))

    name, lgb_model, metrics = train_lightgbm(X_train, X_test, y_train, y_test, X, y)
    if lgb_model:
        results.append((name, lgb_model, metrics))

    name, stack_model, metrics = train_stacking(
        xgb_model, rf_model, X_train, X_test, y_train, y_test, X, y
    )
    results.append((name, stack_model, metrics))

    # ── Leaderboard ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MODEL LEADERBOARD")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'CV R²':>8}")
    logger.info("-" * 60)

    results_sorted = sorted(
        [(n, m, r) for n, m, r in results if r],
        key=lambda x: x[2]["r2"],
        reverse=True,
    )

    for name, model, metrics in results_sorted:
        logger.info(
            f"{name:<20} "
            f"{metrics['r2']:>8.3f} "
            f"{metrics['rmse']:>8.2f} "
            f"{metrics['mae']:>8.2f} "
            f"{metrics['cv_r2_mean']:>8.3f}"
        )

    best_name, best_model, best_metrics = results_sorted[0]
    logger.info(f"\nBest model: {best_name} (R²={best_metrics['r2']:.3f})")

    if best_name in ["xgboost", "random_forest"]:
        compute_shap(best_model, X_test, best_name)
    elif best_name == "stacking":
        compute_shap(xgb_model, X_test, "xgboost_from_stack")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": best_name, **best_metrics}]).to_csv(
        "data/processed/best_model.csv", index=False
    )

    logger.info("=== Training complete ===")
    logger.info(f"View experiments at: {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
