"""
Augment the REAL feature set with synthetic rows.

Loads the real data/processed/features.parquet (produced by `make features` from
the database), generates synthetic places with generate_synthetic_data, aligns
the two schemas (column union), concatenates them, tags each row with a `source`
column ('real' / 'synthetic'), and writes the combined table back to
data/processed/features.parquet (consumed by train.py and the agents).

Run order: `make features` (rebuild real parquet) then this script.
Or simply: `make augment`.

⚠️  The synthetic rows are SIMULATED. The combined R2 is a blend of the strong
synthetic signal and the weak real signal — it does not describe real-world
predictive power. Use `source` to evaluate each subset separately if needed.

Usage: poetry run python src/collection/augment_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from loguru import logger

from src.collection.generate_synthetic_data import (
    add_target_and_derived,
    generate_places,
)

PARQUET = Path("data/processed/features.parquet")
SEED = 42


def align_and_concat(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    real = real.copy()
    synth = synth.copy()
    real["source"] = "real"
    synth["source"] = "synthetic"

    combined = pd.concat([real, synth], ignore_index=True, sort=False)

    # One-hot columns that only exist on one side mean "not that category" for
    # the other side -> fill with 0, not median.
    dummy_cols = [c for c in combined.columns if c.startswith(("type_", "district_"))]
    combined[dummy_cols] = combined[dummy_cols].fillna(0).astype("int8")

    # Remaining numeric gaps (a feature present on only one side) -> column median.
    num_cols = combined.select_dtypes(include=[np.number]).columns
    combined[num_cols] = combined[num_cols].fillna(combined[num_cols].median())

    return combined


def main() -> None:
    if not PARQUET.exists():
        logger.error(f"{PARQUET} not found — run `make features` first to build "
                     "the real feature table from the database.")
        sys.exit(1)

    real = pd.read_parquet(PARQUET)
    if "source" in real.columns and (real["source"] == "synthetic").any():
        logger.error("features.parquet already contains synthetic rows. Rebuild "
                     "the real one with `make features` before augmenting again.")
        sys.exit(1)
    logger.info(f"Loaded {len(real):,} REAL rows from {PARQUET}")

    rng = np.random.default_rng(SEED)
    synth = add_target_and_derived(generate_places(rng))
    logger.info(f"Generated {len(synth):,} SYNTHETIC rows")

    combined = align_and_concat(real, synth)
    combined.to_parquet(PARQUET, index=False)
    logger.success(
        f"Wrote {len(combined):,} rows ({len(real):,} real + {len(synth):,} synthetic) "
        f"× {combined.shape[1]} cols -> {PARQUET}"
    )

    by_source = combined.groupby("source")["success_score"].agg(["count", "mean", "std"])
    logger.info(f"\nsuccess_score by source:\n{by_source}")


if __name__ == "__main__":
    main()