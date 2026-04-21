"""
MarketMind API — FastAPI skeleton (Phase 5 will wire in AI agents)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="MarketMind API",
    description="Intelligent marketplace analytics for Rabat, Morocco",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/marketplaces")
async def list_marketplaces():
    # Phase 5: query PostGIS + run agent pipeline
    return {"message": "Marketplace listing — coming in Phase 2"}
