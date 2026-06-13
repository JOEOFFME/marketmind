"""Detect features that leak the target (success_score)."""
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
TARGET = "success_score"

num = df.select_dtypes("number")
if TARGET not in num.columns:
    raise SystemExit(f"{TARGET} introuvable")

corr = num.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)

print("=== Top 20 |corrélation| avec la cible ===")
print(corr.head(20).to_string())

print("\n=== Fuite suspectée (|corr| > 0.85) ===")
leaky = corr[corr > 0.85]
print(leaky.to_string() if len(leaky) else "(aucune)")

print("\n=== Features mentionnant score/avg/above/pct (à examiner) ===")
suspect = [
    c for c in num.columns
    if c != TARGET and any(k in c.lower() for k in ["score", "avg", "above", "pct", "rank"])
]
print("\n".join(suspect) if suspect else "(aucune)")
