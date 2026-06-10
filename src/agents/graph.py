"""LangGraph state machine wiring the 4 MarketMind agents."""
from langgraph.graph import END, StateGraph

from .analysis_agent import analysis_agent
from .insight_agent import insight_agent
from .query_agent import query_agent
from .retrieval_agent import retrieval_agent
from .state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("query", query_agent)
    g.add_node("retrieval", retrieval_agent)
    g.add_node("analysis", analysis_agent)
    g.add_node("insight", insight_agent)
    g.set_entry_point("query")
    g.add_edge("query", "retrieval")
    g.add_edge("retrieval", "analysis")
    g.add_edge("analysis", "insight")
    g.add_edge("insight", END)
    return g.compile()


def run(question: str) -> dict:
    return build_graph().invoke({"question": question})


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
