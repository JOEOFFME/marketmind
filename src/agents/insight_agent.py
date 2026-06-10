"""Agent 4 — Insight Generation (writes the recommendation in the user's language)."""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .llm import get_llm
from .state import AgentState

LANG_NAME = {"fr": "French", "en": "English", "ar": "Moroccan Arabic (Darija)"}

SYSTEM = """You are MarketMind, advising entrepreneurs on where to open a business in \
Rabat, Morocco. Using ONLY the data provided, write a concise, concrete recommendation \
(3-5 sentences): state the success score, compare it to the district average, mention \
notable existing players, and give a clear verdict. Write in {lang}."""


def insight_agent(state: AgentState) -> AgentState:
    parsed = state.get("parsed", {})
    lang = LANG_NAME.get(parsed.get("language", "fr"), "French")
    # keep the LLM payload lean: drop the heavy map_points list
    retrieved = {k: v for k, v in state.get("retrieved", {}).items() if k != "map_points"}
    payload = {
        "question": state.get("question"),
        "parsed": parsed,
        "retrieved": retrieved,
        "analysis": state.get("analysis", {}),
    }
    llm = get_llm(temperature=0.4)
    resp = llm.invoke(
        [
            SystemMessage(content=SYSTEM.format(lang=lang)),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    logger.info("[insight] generated")
    return {"answer": resp.content.strip()}
