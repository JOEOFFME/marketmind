# MarketMind

**Intelligent business-location analytics for Rabat, Morocco**

MarketMind predicts how successful a business — café, restaurant, pharmacy, bank, supermarket, or mall — is likely to be at a given location in Rabat, and explains *why*, through a multi-agent AI pipeline that answers natural-language questions in **French, English, or Moroccan Darija**.

> *"Où ouvrir un café à Agdal ?"* → MarketMind parses the question, retrieves local market data, runs an ML model, and returns a written recommendation with a success score, competitor analysis, and an interactive map.

---

## Highlights

- **End-to-end ML system** — data collection → feature engineering → training → model registry → serving.
- **Multi-agent AI (LangGraph)** — 4 agents: query understanding, data retrieval, ML analysis, insight generation.
- **Trilingual** — French / English / Moroccan Darija (الدارجة).
- **Honest, leakage-free model** — XGBoost champion at **R² = 0.913 on real places** (verified free of target leakage and geographic memorization).
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

Regression target: `success_score` (0–100) — the **Wilson score lower bound (95% CI)** of the Google rating, treating the rating as a satisfaction proportion and the review count as sample size. This is conservative: unproven places (few reviews) and confidently-bad places (many negative reviews) both score low, so review volume can't inflate a poor rating.

The model is trained on **real market features only** — every feature derived from the target (e.g. `success_score_pct`, `district_avg_score`, `type_avg_score`, `above_type_avg`, district means of rating/reviews) was removed to eliminate **data leakage**, along with raw GPS to prevent geographic memorization.

| Metric | R² |
|---|---|
| **XGBoost** champion — combined test set | **0.904** |
| ↳ on **real** places (honest) | **0.913** |
| ↳ on **synthetic** places | 0.892 |

> Earlier iterations showed R² ≈ 0.97, but that was inflated by target leakage. After removing the leaking features the score is lower **and honest** — verified both by audit and empirically (dropping the suspect features did not reduce R²).

**Dataset:** ~2,145 real Google-rated places + ~1,900 synthetic (simulated) places = ~4,045 rows, tagged with a `source` column so real and synthetic performance are reported separately. 24 market features used for training.

**Key insight:** the strongest predictors of success are genuine market signals — **district income, foot traffic, competition density (`type_count_in_district`), and accessibility (distance to key amenities)** — not any target-derived or geographic shortcut.

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
│   ├── collection/        # data collectors (OSM, weather, Google Places) + synthetic generator + augment
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
├── docs/                  # data & method notes, presentation guide
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
make features        # build the real feature table from the DB
# (optional) make augment   # add synthetic rows for a richer training set
make train           # train, show leaderboard + R² by source
make register        # promote the best model to @champion
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
- [x] **Phase 2** — Feature engineering
- [x] **Phase 3** — ML training + MLflow (leakage-free, real-place R² = 0.913)
- [x] **Phase 4** — DevOps/MLOps: Docker, CI/CD, model registry, Prometheus + Grafana, Terraform
- [x] **Phase 5** — LangGraph AI agents + `/ask` API
- [x] **Serving** — FastAPI + Folium maps + Next.js frontend
- [ ] **Phase 6** — Test coverage > 80%, architecture diagrams, demo video

---

*A full-stack ML + AI engineering project — data engineering, MLOps, multi-agent AI, and web serving end to end.*