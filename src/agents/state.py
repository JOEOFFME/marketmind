"""Shared state passed between LangGraph nodes."""
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str                       # raw user question
    parsed: dict[str, Any]              # {type, district, task, language}
    retrieved: dict[str, Any]          # data pulled from the feature table
    analysis: dict[str, Any]           # model predictions + factor breakdown
    forecast: Optional[dict[str, Any]]  # 5-year financial projection (or None)
    forecast_overrides: dict[str, Any]  # optional caller overrides {capex, margin, growth}
    answer: str                        # final natural-language recommendation