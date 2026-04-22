"""
MarketMind — Main Data Pipeline DAG
Orchestrates the full data collection and feature engineering pipeline.

Schedule:
- Daily   : weather data
- Weekly  : Google Places ratings  
- Monthly : OSM POIs + full feature rebuild
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── Default args applied to all tasks ─────────────────────────────────────
default_args = {
    "owner":            "marketmind",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ─────────────────────────────────────────────────────────────────────────
# DAG 1 — Daily weather update
# ─────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="daily_weather_collection",
    default_args=default_args,
    description="Collect daily weather data for Rabat",
    schedule_interval="0 6 * * *",  # every day at 6am
    start_date=days_ago(1),
    catchup=False,
    tags=["marketmind", "collection", "daily"],
) as dag_weather:

    collect_weather = BashOperator(
        task_id="collect_weather",
        bash_command="cd /opt/airflow/project && python src/collection/collect_weather.py",
    )

    collect_weather

# ─────────────────────────────────────────────────────────────────────────
# DAG 2 — Weekly Places ratings update
# ─────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="weekly_places_collection",
    default_args=default_args,
    description="Collect Google Places ratings for Rabat marketplaces",
    schedule_interval="0 7 * * 1",  # every Monday at 7am
    start_date=days_ago(1),
    catchup=False,
    tags=["marketmind", "collection", "weekly"],
) as dag_places:

    collect_places = BashOperator(
        task_id="collect_places",
        bash_command="cd /opt/airflow/project && python src/collection/collect_places.py",
    )

    rebuild_features = BashOperator(
        task_id="rebuild_features",
        bash_command="cd /opt/airflow/project && python src/features/feature_pipeline.py",
    )

    # Places must finish before features rebuild
    collect_places >> rebuild_features

# ─────────────────────────────────────────────────────────────────────────
# DAG 3 — Monthly full pipeline
# ─────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="monthly_full_pipeline",
    default_args=default_args,
    description="Full pipeline: OSM + Places + Weather + Features",
    schedule_interval="0 5 1 * *",  # 1st of every month at 5am
    start_date=days_ago(1),
    catchup=False,
    tags=["marketmind", "collection", "monthly"],
) as dag_monthly:

    collect_osm = BashOperator(
        task_id="collect_osm",
        bash_command="cd /opt/airflow/project && python src/collection/collect_osm.py",
    )

    collect_weather_monthly = BashOperator(
        task_id="collect_weather",
        bash_command="cd /opt/airflow/project && python src/collection/collect_weather.py",
    )

    collect_places_monthly = BashOperator(
        task_id="collect_places",
        bash_command="cd /opt/airflow/project && python src/collection/collect_places.py",
    )

    rebuild_features_monthly = BashOperator(
        task_id="rebuild_features",
        bash_command="cd /opt/airflow/project && python src/features/feature_pipeline.py",
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="""
            cd /opt/airflow/project && python -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.getenv('DATABASE_URL', 'postgresql://marketmind:marketmind_dev@db:5432/marketmind'))
with engine.connect() as conn:
    pois    = conn.execute(text('SELECT COUNT(*) FROM raw_pois')).scalar()
    places  = conn.execute(text('SELECT COUNT(*) FROM places_ratings')).scalar()
    weather = conn.execute(text('SELECT COUNT(*) FROM weather_observations')).scalar()
    print(f'Validation: POIs={pois}, Places={places}, Weather={weather}')
    assert pois > 0,    'raw_pois is empty!'
    assert places > 0,  'places_ratings is empty!'
    assert weather > 0, 'weather_observations is empty!'
    print('All checks passed.')
"
        """,
    )

    # Task dependencies — defines the execution order
    [collect_osm, collect_weather_monthly, collect_places_monthly] >> rebuild_features_monthly >> validate_data
