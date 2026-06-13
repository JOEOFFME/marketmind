"""Entraînement honnête, sans fuite de la cible. Pour comparer au baseline (0.974)."""
import warnings

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TARGET = "success_score"
LEAKY = [
    "success_score_pct",   # la cible elle-même
    "type_avg_score",      # moyenne du score par type
    "district_avg_score",  # moyenne du score par district
    "above_type_avg",      # utilise le score du lieu
    "rating", "rating_norm",
    "review_count", "reviews_norm",
    "log_reviews", "log_review_count",
]

df = pd.read_parquet("data/processed/features.parquet")
y = df[TARGET]
removed = [c for c in LEAKY if c in df.columns]
X = (
    df.select_dtypes("number")
    .drop(columns=[TARGET] + removed, errors="ignore")
    .fillna(0)
)

print(f"Features conservées : {X.shape[1]}")
print(f"Colonnes retirées (fuite) : {removed}\n")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=42, verbose=-1),
}

print(f"{'Modèle':<14}{'R² (test)':<12}{'MAE':<8}")
print("-" * 34)
results = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    r2 = r2_score(yte, pred)
    results[name] = r2
    print(f"{name:<14}{r2:<12.3f}{mean_absolute_error(yte, pred):<8.2f}")

best = max(results, key=results.get)
print(f"\nMeilleur modèle honnête : {best} (R²={results[best]:.3f})")

# Top features légitimes
xgb = models["XGBoost"]
imp = sorted(zip(X.columns, xgb.feature_importances_), key=lambda t: t[1], reverse=True)
print("\nTop 8 facteurs (légitimes) :")
for f, v in imp[:8]:
    print(f"  {f:<28}{v:.3f}")
