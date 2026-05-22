# MarketMind — Project Status & Resume Guide

> **Intelligent Marketplace Analytics Platform — Rabat, Morocco**  
> Author: JOEOFFME | Stack: Python 3.12 · Docker · PostgreSQL · Airflow · MLflow  
> Repo: `github.com/JOEOFFME/marketmind`

---

## Quick Resume (run these every time you come back)

```bash
# 1. Open Ubuntu (WSL)
# 2. Navigate to project
cd ~/marketmind

# 3. Start all Docker services
make db-up

# 4. Verify everything is running
docker ps

# 5. Check UIs are accessible (use WSL IP, not localhost)
# Get your WSL IP first:
hostname -I
# Then open in browser:
# Database GUI  → http://<WSL_IP>:8080  (server=db, user=marketmind, pass=marketmind_dev)
# MLflow        → http://<WSL_IP>:5000
# Airflow       → http://<WSL_IP>:8081  (user=admin, pass=admin)
```

---

## What We Built So Far

### ✅ Phase 0 — Project Skeleton (Complete)

Everything was scaffolded using a bootstrap script:

- Full directory structure (`src/`, `tests/`, `airflow/`, `infrastructure/`, `data/`)
- `pyproject.toml` — all Python dependencies locked via Poetry
- `Makefile` — developer shortcuts (`make train`, `make test`, etc.)
- `.env` — secrets file (never committed)
- `.gitignore` — excludes cache, data, models, secrets
- `.pre-commit-config.yaml` — black + isort + flake8 on every commit
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline
- `conftest.py` — fixes Python path for pytest
- `infrastructure/docker/docker-compose.yml` — 7 services
- `infrastructure/docker/init/01_init.sql` — auto-creates DB schema on startup

**Key files:**
```
src/config.py              ← typed settings from .env via pydantic-settings
src/api/main.py            ← FastAPI skeleton (health endpoint)
tests/unit/test_config.py  ← 3 passing tests
```

---

### ✅ Phase 1 — Data Collection (Complete)

Three datasets collected and stored in PostgreSQL + PostGIS:

#### Dataset 1 — OpenStreetMap POIs
- **Script:** `src/collection/collect_osm.py`
- **Method:** Downloaded `morocco-latest.osm.pbf` from Geofabrik, parsed with `osmium`
- **Result:** **2,677 Rabat POIs** in `raw_pois` table (cafés, restaurants, pharmacies, souks, banks...)
- **Run:** `make collect-osm`

#### Dataset 2 — Weather Data
- **Script:** `src/collection/collect_weather.py`
- **Method:** Open-Meteo Archive API (free, no key needed)
- **Result:** **731 daily records** (2023–2024) in `weather_observations` table
- **Run:** `make collect-weather`

#### Dataset 3 — Google Places Ratings (Target Variable)
- **Script:** `src/collection/collect_places.py`
- **Method:** Google Places API (New) — requires `GOOGLE_PLACES_API_KEY` in `.env`
- **Result:** **589 rated places** with `success_score` (0–100) in `places_ratings` table
- **Run:** `make collect-places`

**Success score formula:**
```
success_score = (0.4 × rating_normalized + 0.6 × log_reviews_normalized) × 100
```

#### Airflow DAGs (Automation)
- **File:** `airflow/dags/marketmind_pipeline.py`
- **DAGs running:**
  - `daily_weather_collection` — every day at 6am
  - `weekly_places_collection` — every Monday at 7am (triggers feature rebuild)
  - `monthly_full_pipeline` — 1st of month at 5am (full pipeline + validation)

---

### ✅ Phase 2 — Feature Engineering (Complete)

- **Script:** `src/features/feature_pipeline.py`
- **Run:** `make features`
- **Output:** `data/processed/features.parquet` + `features` table in DB

**47 features computed for 589 places:**

| Category | Features |
|---|---|
| Proximity | dist to nearest pharmacy, bank, supermarket, fuel (meters) |
| Density | cafes/restaurants/banks/pharmacies/total POIs within 500m |
| District-level | avg rating, avg reviews, place count, avg score per district |
| Competition | type count in district, above/below type average |
| Weather | avg annual temp, rainy days, peak season months |
| Encoded | place type dummies (6), district dummies (8), price level |
| Log-transformed | log(review_count), log(total_pois_500m) |

**Target variable stats:**
```
min=7.13  |  mean=50.99  |  max=96.05  |  std=16.07
```

---

### ✅ Phase 1c — Airflow Orchestration (Complete)

Apache Airflow 2.10.4 running in Docker, accessible at `http://<WSL_IP>:8081`.  
3 DAGs active and scheduled. No manual data collection needed anymore.

---

### ✅ Phase 3 — ML Training (Complete — May 22, 2026)

- **Script:** `src/models/train.py`
- **Run:** `make train`
- **MLflow experiment:** Experiment #1 — all runs tracked at `http://<WSL_IP>:5000`

**Model Leaderboard:**

| Model | R² | RMSE | MAE | CV R² |
|---|---|---|---|---|
| **XGBoost** ✅ | **0.942** | **4.74** | **3.31** | **0.948** |
| LightGBM | 0.940 | 4.81 | 3.32 | 0.948 |
| Stacking Ensemble | 0.940 | 4.82 | 3.49 | 0.938 |
| Random Forest | 0.933 | 5.09 | 3.67 | 0.930 |
| Ridge (baseline) | 0.878 | 6.88 | 5.53 | 0.868 |

**Winner: XGBoost** (Optuna-tuned, 30 trials) — predictions typically within ±5 points on the 0–100 success score.

**Top SHAP Features (XGBoost):**

| Feature | Importance | Insight |
|---|---|---|
| `type_avg_score` | 10.87 | District avg score for this business type — strongest signal |
| `above_type_avg` | 7.40 | Whether this place beats its local competition |
| `price_level_encoded` | 2.34 | Price positioning matters |
| `dist_nearest_supermarket` | 0.53 | Proximity to anchor stores |
| `longitude` | 0.42 | Geographic clustering effect |
| `type_count_in_district` | 0.40 | Market saturation |

> **Key insight:** Competitive positioning within a business type (relative to district peers) is by far the most predictive factor — more than raw proximity, density, or weather.

**MLflow Run IDs:**
- Ridge: `aa279b6ff44f4218b15761f0b0eb97c4`
- Random Forest: `e1fbdb6f91ee424498888c1d5c8449b7`
- XGBoost: `3cc6339815ab438dba72f63059b52065`
- LightGBM: `8749fa6680304e58928e28af0f218801`
- Stacking: `8b4ac8d43c4f41b3b148c34a800533fc`

---

### 🔄 Phase 3b — Time Series Forecasting (Next Step)

Add Prophet model to predict seasonal success patterns:
- Which months are best for each district/type?
- Is this marketplace growing or declining?
- **Script to create:** `src/models/train_prophet.py`

---

### ⬜ Phase 4 — CI/CD Pipeline (Pending)

- Write Dockerfiles for API + ML worker
- Configure Terraform for Oracle Cloud free tier
- Add model performance tests to GitHub Actions
- Set up Prometheus + Grafana monitoring
- Implement blue-green deployment

---

### ⬜ Phase 5 — LangGraph AI Agents (Pending)

4 specialized agents connected in a LangGraph state graph:

```
User: "Où ouvrir un café à Agdal?"
        ↓
Agent 1: Query Understanding  → {type: café, location: Agdal, task: recommend}
        ↓
Agent 2: Data Retrieval       → queries PostGIS for Agdal data
        ↓
Agent 3: ML Analysis          → runs stacking model + SHAP
        ↓
Agent 4: Insight Generation   → French/English/Darija explanation + map
```

**Files to create:**
- `src/agents/query_agent.py`
- `src/agents/retrieval_agent.py`
- `src/agents/analysis_agent.py`
- `src/agents/insight_agent.py`
- `src/agents/graph.py` ← LangGraph state machine

---

### ⬜ Phase 6 — Testing & Documentation (Pending)

- Achieve >80% test coverage
- OpenAPI/Swagger docs (auto-generated by FastAPI)
- Architecture diagrams
- Demo video
- Portfolio README

---

## Database Tables

| Table | Rows | Purpose |
|---|---|---|
| `raw_pois` | 2,677 | OSM points of interest with GPS coordinates |
| `weather_observations` | 731 | Daily weather for Rabat (2023–2024) |
| `places_ratings` | 589 | Google Places ratings + success scores |
| `features` | 589 | 47-column ML-ready feature matrix |

---

## Environment Setup (Fresh Machine or After Reboot)

```bash
# ── Prerequisites (one-time) ───────────────────────────────────────────────
# WSL2 + Ubuntu installed
# Docker Desktop running with WSL integration enabled for Ubuntu

# ── Every session ─────────────────────────────────────────────────────────
cd ~/marketmind

# Start Docker services (PostgreSQL, Redis, Adminer, MLflow, Airflow x3)
make db-up

# Verify all 7 containers are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Get WSL IP (changes on reboot sometimes)
hostname -I

# ── If containers are unhealthy ────────────────────────────────────────────
make db-down
make db-up

# ── If poetry environment is broken ───────────────────────────────────────
poetry install

# ── Run tests to verify everything works ──────────────────────────────────
make test
```

---

## .env File (Required Values)

```bash
# Database
DATABASE_URL=postgresql://marketmind:marketmind_dev@localhost:5432/marketmind
POSTGRES_DB=marketmind
POSTGRES_USER=marketmind
POSTGRES_PASSWORD=marketmind_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# MLflow — USE WSL IP NOT localhost
MLFLOW_TRACKING_URI=http://<YOUR_WSL_IP>:5000

# Google Places API
GOOGLE_PLACES_API_KEY=<your_key>

# LLM (needed for Phase 5)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Service URLs

> **Note:** Use your WSL IP (run `hostname -I`) not `localhost`.  
> WSL IP can change after Windows reboot — always check with `hostname -I`.

| Service | URL | Credentials |
|---|---|---|
| Adminer (DB GUI) | `http://<WSL_IP>:8080` | server=`db`, user=`marketmind`, pass=`marketmind_dev` |
| MLflow | `http://<WSL_IP>:5000` | no login |
| Airflow | `http://<WSL_IP>:8081` | user=`admin`, pass=`admin` |
| FastAPI docs | `http://<WSL_IP>:8000/docs` | (Phase 5) |

---

## Git Workflow

```bash
# Check current status
git status
git log --oneline -5

# Commit and push
git add -A
git commit -m "feat: description of what you did"
git push
```

**Commit history:**
```
ec2e79e feat: Phase 3 complete — XGBoost R²=0.942, all models trained, SHAP analysis done
b2cc0e0 (previous)
5831dfa chore: remove cache and coverage files from git tracking
b5719e9 feat: Phase 1 complete — OSM + weather data collection working
f662b68 chore: initial project scaffold — Phase 0 complete
```

---

## Immediate Next Actions

```
1. make db-up                              ← start services
2. hostname -I                             ← get WSL IP
3. Create src/models/train_prophet.py      ← Phase 3b: seasonal forecasting
4. make train-prophet                      ← run Prophet model
5. Phase 4 or Phase 5                      ← CI/CD or LangGraph agents
6. git add -A && git push                  ← save progress
```

---

*Last updated: May 22, 2026 | Phase 3 complete — XGBoost R²=0.942 | Phase 3b next*
