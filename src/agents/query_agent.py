"""Agent 1 — Query Understanding."""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .data import get_districts
from .llm import get_llm
from .state import AgentState


def _system() -> str:
    districts = get_districts()
    return f"""You extract structured info from a question about where to open a \
business in Rabat, Morocco.

Return ONLY a JSON object (no prose, no markdown) with keys:
- "type": one of [cafe, restaurant, pharmacy, bank, supermarket, shopping_mall], or null
- "district": MUST be one of these exact Latin names (transliterate Arabic/Darija to \
the closest match): {districts}. Use null if none matches.
- "task": one of [recommend, evaluate, compare] (default "recommend")
- "language": one of [fr, en, ar] (ar = Arabic or Darija)

Example: "Où ouvrir un café à Agdal ?" -> \
{{"type": "cafe", "district": "Agdal", "task": "recommend", "language": "fr"}}"""


def query_agent(state: AgentState) -> AgentState:
    logger.info(f"[query] {state['question']}")
    llm = get_llm(temperature=0)
    resp = llm.invoke(
        [SystemMessage(content=_system()), HumanMessage(content=state["question"])]
    )
    text = resp.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
    except Exception as e:
        logger.warning(f"[query] JSON parse failed ({e}); fallback. Raw: {text!r}")
        parsed = {"type": None, "district": None, "task": "recommend", "language": "fr"}
    logger.info(f"[query] parsed: {parsed}")
    return {"parsed": parsed}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    for q in ["Où ouvrir un café à Agdal ?", "café à Hassan?", "صيدلية فأكدال"]:
        print(q, "->", query_agent({"question": q})["parsed"])
