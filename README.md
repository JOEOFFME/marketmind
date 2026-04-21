# MarketMind

**Intelligent Marketplace Analytics Platform — Rabat, Morocco**

A production-grade multi-agent AI system for urban economic intelligence.

## Architecture

| Layer | Tools |
|---|---|
| Data collection | Airflow/Prefect, OpenStreetMap, Open-Meteo, HCP |
| Data engineering | PostgreSQL + PostGIS, DVC, Great Expectations |
| ML | XGBoost, LightGBM, Prophet, MLflow |
| CI/CD | GitHub Actions, Docker, Terraform |
| AI agents | LangGraph (4 agents), LangChain, Ollama |
| Serving | FastAPI, Next.js, Folium maps |

## Quick start

```bash
# 1. Clone and enter
git clone <repo> && cd marketmind

# 2. Start the database stack
make db-up

# 3. Install Python dependencies
make install

# 4. Collect first data
make collect-osm
make collect-weather

# 5. Run tests
make test
```

## Services

| Service | URL |
|---|---|
| Adminer (DB GUI) | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |

## Project phases

- [x] Phase 0 — Bootstrap (this commit)
- [ ] Phase 1 — Infrastructure & data collection
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — ML models + MLOps
- [ ] Phase 4 — CI/CD pipeline
- [ ] Phase 5 — LangGraph AI agents
- [ ] Phase 6 — Testing & documentation
