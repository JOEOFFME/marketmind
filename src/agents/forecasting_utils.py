"""Lightweight 5-year financial forecast for a prospective business.

Indicative only — turns a predicted success score into a simple P&L projection
(revenue, EBITDA, cumulative profit, break-even year, 5-year ROI). All amounts in
Moroccan Dirham (MAD). Overrides let the frontend expose editable assumptions.
"""
from __future__ import annotations

from typing import Optional

# Indicative economics per business type (Moroccan market, order-of-magnitude).
#   capex        : upfront investment (MAD)
#   ref_revenue  : expected year-1 revenue (MAD) at a "strong" success score of 70
#   margin       : typical EBITDA margin
TYPE_ECONOMICS = {
    "cafe":          {"capex":   300_000, "ref_revenue":   700_000, "margin": 0.18},
    "restaurant":    {"capex":   600_000, "ref_revenue": 1_300_000, "margin": 0.15},
    "pharmacy":      {"capex":   900_000, "ref_revenue": 2_500_000, "margin": 0.11},
    "bank":          {"capex": 1_500_000, "ref_revenue": 3_000_000, "margin": 0.25},
    "supermarket":   {"capex": 1_500_000, "ref_revenue": 5_000_000, "margin": 0.07},
    "shopping_mall": {"capex": 8_000_000, "ref_revenue": 15_000_000, "margin": 0.20},
}
DEFAULT_ECONOMICS = {"capex": 500_000, "ref_revenue": 1_000_000, "margin": 0.15}

REF_SCORE = 70.0       # score considered a "strong" location
DEFAULT_GROWTH = 0.08  # 8% annual revenue growth
HORIZON = 5            # years


def run_forecast(
    success_score: Optional[float],
    place_type: Optional[str],
    capex_override: Optional[float] = None,
    margin_override: Optional[float] = None,
    growth_override: Optional[float] = None,
) -> Optional[dict]:
    """Return a 5-year financial projection, or None if inputs are missing."""
    if success_score is None or place_type is None:
        return None

    econ = TYPE_ECONOMICS.get(place_type, DEFAULT_ECONOMICS)
    capex = float(capex_override) if capex_override is not None else econ["capex"]
    margin = float(margin_override) if margin_override is not None else econ["margin"]
    growth = float(growth_override) if growth_override is not None else DEFAULT_GROWTH

    # Year-1 revenue scales with the success score relative to a strong location.
    score_factor = max(0.2, float(success_score) / REF_SCORE)
    year1_revenue = econ["ref_revenue"] * score_factor

    projection = []
    cumulative = -capex
    break_even_year = None
    for y in range(1, HORIZON + 1):
        revenue = year1_revenue * ((1 + growth) ** (y - 1))
        ebitda = revenue * margin
        cumulative += ebitda
        if break_even_year is None and cumulative >= 0:
            break_even_year = y
        projection.append(
            {
                "year": y,
                "revenue": round(revenue),
                "ebitda": round(ebitda),
                "cumulative_profit": round(cumulative),
            }
        )

    total_ebitda = sum(p["ebitda"] for p in projection)
    roi_5y = (total_ebitda - capex) / capex if capex else None

    return {
        "currency": "MAD",
        "assumptions": {
            "capex": round(capex),
            "ebitda_margin": round(margin, 3),
            "annual_growth": round(growth, 3),
            "horizon_years": HORIZON,
        },
        "year1_revenue": round(year1_revenue),
        "year1_ebitda": round(year1_revenue * margin),
        "projection": projection,
        "break_even_year": break_even_year,  # None = not reached within horizon
        "roi_5y": round(roi_5y, 2) if roi_5y is not None else None,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_forecast(75, "cafe"), indent=2, ensure_ascii=False))