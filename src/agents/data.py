"""Shared data access for agents."""
import functools
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("data/processed/features.parquet")

TYPE_COLS = {
    "cafe": "type_cafe",
    "restaurant": "type_restaurant",
    "pharmacy": "type_pharmacy",
    "bank": "type_bank",
    "supermarket": "type_supermarket",
    "shopping_mall": "type_shopping_mall",
}


@functools.lru_cache(maxsize=1)
def load_features_df() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def get_districts() -> list[str]:
    df = load_features_df()
    return sorted(c[len("district_"):] for c in df.columns if c.startswith("district_"))


def get_types() -> list[str]:
    df = load_features_df()
    return sorted(k for k, v in TYPE_COLS.items() if v in df.columns)


def filter_subset(df, district=None, btype=None):
    """Filter rows by district and/or business type. Returns (subset, warnings)."""
    sub = df
    warnings = []
    if district:
        dcol = f"district_{district}"
        if dcol in df.columns:
            sub = sub[sub[dcol] == 1]
        else:
            warnings.append(f"District '{district}' not found in data")
    if btype:
        tcol = TYPE_COLS.get(btype)
        if tcol and tcol in sub.columns:
            sub = sub[sub[tcol] == 1]
        else:
            warnings.append(f"Type '{btype}' not found in data")
    return sub, warnings
