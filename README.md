cat > README.md << 'MARKDOWN'
# MarketMind

**Intelligent business-location analytics for Rabat, Morocco**

MarketMind predicts how successful a business — café, restaurant, pharmacy, bank, supermarket, or mall — is likely to be at a given location in Rabat, and explains *why*, through a multi-agent AI pipeline that answers natural-language questions in **French, English, or Moroccan Darija**.

> *"Où ouvrir un café à Agdal ?"* → MarketMind parses the question, retrieves local market data, runs an ML model, and returns a written recommendation with a success score, competitor analysis, and an interactive map.

---

## Highlights

- **End-to-end ML system** — data collection → feature engineering → training → model registry → serving.
- **Multi-agent AI (LangGraph)** — 4 agents: query understanding, data retrieval, ML analysis, insight generation.
- **Trilingual** — French / English / Moroccan Darija (الدارجة).
- **Champion model R² = 0.974** (XGBoost) on the success-score regression.
- **Full MLOps** — MLflow tracking + model registry, Prometheus + Grafana monitoring, GitHub Actions CI, Dockerized stack, Terraform IaC.
- **Web app** — Next.js frontend + FastAPI backend + Folium maps.

---

## Architecture

| Layer | Tools |
|---|---|
| Data collection | Apache Airflow, OpenStreetMap, Open-Meteo, Google Places API |
| Data engineering | PostgreSQL + PostGIS, Parquet |
| Machine learning | XGBoost, LightGBM, Random Forest, scikit-learn, MLflow |
| AI agents | LangGraph, LangChain, OpenRouter (gpt-oss-120b) |
| Serving | FastAPI, Next.js, Folium |
| DevOps / MLOps | Docker Compose, GitHub Actions, Prometheus, Grafana, Terraform |

### AI agent pipeline
"Où ouvrir un café à Agdal ?"
│
▼

Query Agent       natural language → {type, district, task, language}
│
▼
Retrieval Agent   district + type stats, competitors, coordinates
│
▼
Analysis Agent    champion model prediction + key drivers
│
▼
Insight Agent     written recommendation in the user's language
│
▼
Answer + success score + interactive map


---

## Model performance

Regression target: `success_score` (0–100), derived from Google ratings and review volume.

| Model | R² |
|---|---|
| **XGBoost** (champion) | **0.974** |
| LightGBM | 0.972 |
| Random Forest | 0.958 |
| Ridge (baseline) | 0.917 |

**Dataset:** 2,145 rated places · 21,173 OpenStreetMap POIs · 68 engineered features (28 used for training).

**Key insight:** the strongest predictor of success is *competitive positioning within a business type* relative to district peers — more than raw proximity, density, or weather.

---

## Tech stack

- **Languages:** Python 3.12 (Poetry), TypeScript
- **Data:** PostgreSQL 15 + PostGIS, Pandas, Parquet
- **ML:** scikit-learn, XGBoost, LightGBM, MLflow
- **AI:** LangGraph, LangChain, OpenRouter
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Next.js 15, React, Tailwind CSS
- **Maps:** Folium (Leaflet)
- **Orchestration:** Apache Airflow
- **Infra:** Docker Compose, GitHub Actions, Prometheus, Grafana, Terraform

---

## Project structure
marketmind/
├── src/
│   ├── collection/        # data collectors (OSM, weather, Google Places)
│   ├── features/          # feature engineering pipeline
│   ├── models/            # training + model registry
│   ├── agents/            # LangGraph agents (query, retrieval, analysis, insight, graph)
│   ├── serving/           # Folium map generation
│   ├── api/               # FastAPI app (/ask, /map, /health, /metrics)
│   └── workers/           # ML worker (Redis queue + champion model)
├── frontend/              # Next.js web app
├── airflow/dags/          # scheduled pipelines
├── infrastructure/docker/ # docker-compose + Prometheus/Grafana config
├── terraform/             # Oracle Cloud IaC (deploy-ready)
├── tests/                 # pytest suite
└── data/processed/        # features.parquet

---

## Getting started

### Prerequisites
- Docker (with WSL2 integration on Windows)
- Python 3.12 + Poetry
- Node.js 20+ (for the frontend)
- A free OpenRouter API key (for the AI agents)

### 1. Clone & configure
```bash
git clone https://github.com/JOEOFFME/marketmind.git
cd marketmind
# create a .env file (see "Environment variables" below)
```

### 2. Start the stack
```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d
poetry install
```

### 3. Train & register the model (if not already done)
```bash
make features
poetry run python src/models/train.py
MLFLOW_TRACKING_URI=http://localhost:5000 poetry run python src/models/register_model.py
```

### 4. Run the API
```bash
poetry run uvicorn src.api.main:app --port 8001
```

### 5. Run the frontend
```bash
cd frontend
npm install
npm run dev -- -p 3001
```

Open **http://localhost:3001**.

---

## Usage

### REST API

**`POST /ask`**
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Où ouvrir un café à Agdal ?"}'
```
Returns the written recommendation plus structured data (parsed intent, district stats, model analysis).

**`GET /map?district=Agdal&type=cafe`** — interactive Folium map of the matching places, colored by success score.

**`GET /metrics`** — Prometheus metrics · **`GET /health`** — health check.

### Example questions
- *Où ouvrir un café à Agdal ?* (Français)
- *Is it a good idea to open a pharmacy in Hassan?* (English)
- *واش نحل صيدلية فأكدال؟* (الدارجة)

---

## Services

| Service | URL | Credentials |
|---|---|---|
| Frontend (Next.js) | http://localhost:3001 | — |
| API (FastAPI docs) | http://localhost:8001/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8081 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Adminer (DB) | http://localhost:8080 | server `db`, user/pass `marketmind` |

---

## MLOps & monitoring

- **MLflow** tracks every training run and hosts the **model registry**. The best model is promoted via the `@champion` alias; the API and worker load it through `models:/marketmind-success-model@champion`.
- **Prometheus + Grafana** — the API is instrumented (request rate, latency p50/p95/p99, errors per endpoint) with an auto-provisioned Grafana dashboard.
- **GitHub Actions CI** — linting (black, isort, flake8), tests (pytest + coverage), and Docker image builds on every push.
- **Terraform** — deploy-ready Infrastructure-as-Code for an Oracle Cloud Always-Free Arm instance (VCN, firewall, compute, Docker via cloud-init).

---

## Environment variables (`.env`)

```bash
# Database
DATABASE_URL=postgresql://marketmind:marketmind_dev@localhost:5432/marketmind
POSTGRES_DB=marketmind
POSTGRES_USER=marketmind
POSTGRES_PASSWORD=marketmind_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# AI agents (OpenRouter — free, no card required)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-oss-120b:free

# Data collection (optional)
GOOGLE_PLACES_API_KEY=
```

---

## Roadmap

- [x] **Phase 0** — Project scaffold, CI, tooling
- [x] **Phase 1** — Data collection (OSM, weather, Google Places) + Airflow
- [x] **Phase 2** — Feature engineering (68 features)
- [x] **Phase 3** — ML training + MLflow (XGBoost R² = 0.974)
- [x] **Phase 4** — DevOps/MLOps: Docker, CI/CD, model registry, Prometheus + Grafana, Terraform
- [x] **Phase 5** — LangGraph AI agents + `/ask` API
- [x] **Serving** — FastAPI + Folium maps + Next.js frontend
- [ ] **Phase 6** — Test coverage > 80%, architecture diagrams, demo video

---

*A full-stack ML + AI engineering project — data engineering, MLOps, multi-agent AI, and web serving end to end.*
