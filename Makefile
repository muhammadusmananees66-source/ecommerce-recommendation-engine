.PHONY: help install install-dev test lint run seed-data train evaluate docker-build compose-up compose-down clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install core (lightweight) dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies (tests, lint)
	pip install -r requirements-dev.txt

test: ## Run the full test suite
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linters
	ruff check src tests
	mypy src --ignore-missing-imports

run: ## Run the API locally (needs data/ generated first)
	uvicorn src.api.app:create_app --factory --reload --port 8000

seed-data: ## Generate synthetic sample data for local dev/CI
	python scripts/generate_sample_data.py

train: ## Train the recommender on data/interactions.parquet + data/items.parquet
	python scripts/train_model.py \
		--interactions data/interactions.parquet \
		--items data/items.parquet \
		--text-cols title description \
		--output models/hybrid_v1

evaluate: ## Run the RAG evaluation gate locally
	python scripts/evaluate_rag.py --test-data data/rag_test.parquet --seed-items data/items.parquet

docker-build: ## Build the Docker image
	docker build -t rag-recommendation:latest -f docker/Dockerfile .

compose-up: ## Start the app + Redis locally via docker-compose
	docker compose up --build

compose-down: ## Stop docker-compose services
	docker compose down

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml rag-evaluation.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true