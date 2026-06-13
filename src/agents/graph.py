"""LangGraph state machine wiring the 4 MarketMind agents + a forecasting step."""
from langgraph.graph import END, StateGraph

from .analysis_agent import analysis_agent
from .insight_agent import insight_agent
from .query_agent import query_agent
from .retrieval_agent import retrieval_agent
from .state import AgentState


def _forecast_node(state: AgentState) -> AgentState:
    """Run the financial forecast between analysis and insight.

    Reads the predicted success score + business type, projects a simple 5-year
    P&L, and stores it under state["forecast"] for the insight agent. Optional
    caller overrides come from state["forecast_overrides"] = {capex, margin, growth}.
    """
    from .forecasting_utils import run_forecast  # local import — avoids cycles

    analysis = state.get("analysis", {})
    parsed = state.get("parsed", {})
    overrides = state.get("forecast_overrides", {})

    score = analysis.get("predicted_mean")
    btype = parsed.get("type")
    if score is None or btype is None:
        return {"forecast": None}

    forecast = run_forecast(
        success_score=score,
        place_type=btype,
        capex_override=overrides.get("capex"),
        margin_override=overrides.get("margin"),
        growth_override=overrides.get("growth"),
    )
    return {"forecast": forecast}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("query", query_agent)
    g.add_node("retrieval", retrieval_agent)
    g.add_node("analysis", analysis_agent)
    g.add_node("forecast", _forecast_node)
    g.add_node("insight", insight_agent)

    g.set_entry_point("query")
    g.add_edge("query", "retrieval")
    g.add_edge("retrieval", "analysis")
    g.add_edge("analysis", "forecast")
    g.add_edge("forecast", "insight")
    g.add_edge("insight", END)

    return g.compile()


def run(question: str, forecast_overrides: dict | None = None) -> dict:
    initial: AgentState = {"question": question}
    if forecast_overrides:
        initial["forecast_overrides"] = forecast_overrides
    return build_graph().invoke(initial)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    q = sys.argv[1] if len(sys.argv) > 1 else "Où ouvrir un café à Agdal ?"
    result = run(q)
    print("\n" + "=" * 60)
    print("Q:", q)
    print("=" * 60)
    print(result["answer"])
    if result.get("forecast"):
        import json

        print("\n── Forecast ──")
        print(json.dumps(result["forecast"], indent=2, ensure_ascii=False))