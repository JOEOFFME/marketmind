"""MarketMind API — FastAPI app exposing the LangGraph agent pipeline + maps."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="MarketMind API",
    description="Intelligent business-location analytics for Rabat, Morocco",
    version="0.3.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator(
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
).instrument(app).expose(app)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from src.agents.graph import build_graph

        _graph = build_graph()
    return _graph


class AskRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return {
        "message": "MarketMind API is running 🚀",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.3.0",
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="question is required",
        )

    try:
        result = get_graph().invoke({"question": req.question})

        return {
            "question": req.question,
            "answer": result.get("answer"),
            "parsed": result.get("parsed"),
            "retrieved": result.get("retrieved"),
            "analysis": result.get("analysis"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"agent pipeline error: {str(e)}",
        )


@app.get("/map", response_class=HTMLResponse)
def map_view(
    district: str | None = None,
    type: str | None = None,
):
    from src.serving.maps import build_map

    return build_map(district, type).get_root().render()
