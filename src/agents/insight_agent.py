"""Agent 4 — Insight Generation.

Turns the analytical payload into a clear, client-facing recommendation in the
user's language (French / English / Moroccan Darija). The tone is that of a
professional advisor talking to an entrepreneur — never a data scientist.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .llm import get_llm
from .state import AgentState

LANG_NAME = {"fr": "French", "en": "English", "ar": "Moroccan Arabic (Darija)"}

SYSTEM = """You are MarketMind, a professional location-intelligence advisor speaking
directly to an entrepreneur (your client) who is considering opening a business in
Rabat, Morocco. Turn the data into a clear, trustworthy, decision-ready recommendation.

Write in {lang}. Address the client directly and professionally. Use ONLY the data
provided — never invent scores, competitor names, or figures.

SPEAK LIKE A HUMAN ADVISOR, NOT A DATA SCIENTIST:
- NEVER use technical terms: no "SHAP", "model", "feature", "importance", "weight",
  "variable", "dataset", or raw field names. Translate everything into plain business
  language (e.g. "strong nearby competition", "good foot traffic", "close to a bank").
- Use the listed factors, ordered by how much they matter, but explain each in one
  natural sentence about what it means commercially for THIS location.
- All money amounts are in Moroccan Dirham (MAD).

Structure (use the client's language for the headings):

1. Verdict & score — open with the success score (0–100) and what it means
   (strong >= 70 / average 50–69 / weak < 50), and how it compares to the district
   average. One short paragraph.

2. Why — the 3–5 factors that most influence this result, each explained simply
   (what helps, what is a risk). Name 1–2 real existing competitors with their scores
   if available, and say whether they look beatable.

3. Financial outlook — ONLY if a "forecast" block is provided, summarise in plain
   terms: expected first-year revenue, when the business breaks even, and the 5-year
   return on investment. If no forecast is provided, omit this section entirely.

4. Recommendation — finish with ONE clear verdict in bold: **OPEN**,
   **OPEN WITH CONDITIONS**, or **DO NOT OPEN**. If conditional, give the single most
   important condition. Be decisive; never say "it depends".

If {lang} is "Moroccan Arabic (Darija)": write in Arabic script (الدارجة), natural and
professional; keep competitor and district names in Latin script.

Keep it concise and skimmable."""


def insight_agent(state: AgentState) -> AgentState:
    parsed = state.get("parsed", {})
    lang = LANG_NAME.get(parsed.get("language", "fr"), "French")

    # Keep the payload lean: drop the heavy map_points list.
    retrieved = {
        k: v for k, v in state.get("retrieved", {}).items() if k != "map_points"
    }
    payload = {
        "question": state.get("question"),
        "parsed": parsed,
        "retrieved": retrieved,
        "analysis": state.get("analysis", {}),
    }
    if state.get("forecast"):
        payload["forecast"] = state["forecast"]

    llm = get_llm(temperature=0.35)
    resp = llm.invoke(
        [
            SystemMessage(content=SYSTEM.format(lang=lang)),
            HumanMessage(
                content="Here is the data:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        ]
    )
    answer = resp.content.strip()
    logger.info(f"[insight] generated ({len(answer)} chars, lang={lang})")
    return {"answer": answer}


if __name__ == "__main__":
    from dotenv import load_dotenv

    from .analysis_agent import analysis_agent
    from .query_agent import query_agent
    from .retrieval_agent import retrieval_agent

    load_dotenv()
    s = {"question": "Où ouvrir un café à Agdal ?"}
    s.update(query_agent(s))
    s.update(retrieval_agent(s))
    s.update(analysis_agent(s))
    s.update(insight_agent(s))
    print("\n" + "=" * 70)
    print(s["answer"])