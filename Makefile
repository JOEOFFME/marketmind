.PHONY: help install db-up db-down collect-osm collect-weather test lint format clean

help:
	@echo "MarketMind — available commands"
	@echo ""
	@echo "  make install        Install Python dependencies"
	@echo "  make db-up          Start PostgreSQL + Redis (Docker)"
	@echo "  make db-down        Stop containers"
	@echo "  make collect-osm    Collect OSM POIs for Rabat"
	@echo "  make collect-weather Collect weather data"
	@echo "  make test           Run test suite"
	@echo "  make lint           Run flake8 + mypy"
	@echo "  make format         Run black + isort"
	@echo "  make clean          Remove __pycache__ and .pyc files"

install:
	pip install poetry
	poetry install

db-up:
	docker compose -f infrastructure/docker/docker-compose.yml up -d
	@echo "✓ DB running — Adminer at http://localhost:8080"

db-down:
	docker compose -f infrastructure/docker/docker-compose.yml down

collect-osm:
	poetry run python src/collection/collect_osm.py

collect-weather:
	poetry run python src/collection/collect_weather.py

collect-places:
	poetry run python src/collection/collect_places.py

test:
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
