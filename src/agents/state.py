"""Shared state passed between LangGraph nodes."""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str            # raw user question
    parsed: dict[str, Any]   # {type, district, task, language}
    retrieved: dict[str, Any]  # data pulled from the DB
    analysis: dict[str, Any]   # model predictions + drivers
    answer: str              # final natural-language insight