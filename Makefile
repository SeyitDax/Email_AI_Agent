# Email AI Agent Makefile - BOILERPLATE
# Development and deployment automation

.PHONY: help install dev test lint format clean build run docker-build docker-run docker-stop logs

# Default target
help: ## Show this help message
	@echo "Email AI Agent - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

setup-env: ## Setup environment file
	cp .env.example .env
	@echo "Please edit .env file with your actual configuration values"

# Development
dev: ## Run development server with hot reload
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-worker: ## Run Celery worker for development
	celery -A src.workers.celery_app worker --loglevel=info --reload

dev-flower: ## Run Celery Flower for task monitoring
	celery -A src.workers.celery_app flower --port=5555

# Testing
test: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-coverage: ## Generate coverage report
	pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Code Quality
lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/

security-check: ## Run security checks
	bandit -r src/
	safety check

# Database
db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migration
	alembic downgrade -1

db-revision: ## Create new migration
	@read -p "Enter migration message: " message; \
	alembic revision --autogenerate -m "$$message"

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "WARNING: This will delete all database data!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		alembic downgrade base && alembic upgrade head; \
	fi

# Docker Commands
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t email-ai-agent:latest .

docker-build-dev: ## Build Docker image for development
	docker build -f docker/Dockerfile.dev -t email-ai-agent:dev .

docker-run: ## Run with Docker Compose
	docker-compose -f docker/docker-compose.yml up -d

docker-run-dev: ## Run development environment with Docker Compose
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

docker-stop: ## Stop Docker containers
	docker-compose -f docker/docker-compose.yml down

docker-logs: ## Show Docker logs
	docker-compose -f docker/docker-compose.yml logs -f

docker-shell: ## Get shell in running container
	docker-compose -f docker/docker-compose.yml exec email-ai-agent /bin/bash

# Production
build: ## Build for production
	docker build -f docker/Dockerfile -t email-ai-agent:$(shell git rev-parse --short HEAD) .
	docker tag email-ai-agent:$(shell git rev-parse --short HEAD) email-ai-agent:latest

deploy: ## Deploy to production (placeholder)
	@echo "Deployment commands would go here"
	@echo "This would typically involve:"
	@echo "- Building production images"
	@echo "- Pushing to container registry"
	@echo "- Updating kubernetes/docker swarm configs"
	@echo "- Rolling out updates"

# Monitoring & Logs
logs: ## Show application logs
	tail -f logs/email_agent.log

logs-error: ## Show error logs only
	tail -f logs/email_agent.log | grep ERROR

metrics: ## Show basic metrics
	curl http://localhost:8000/health/metrics

# Data Management
seed-data: ## Seed database with sample data
	python scripts/seed_knowledge.py

backup-db: ## Backup database
	@echo "Creating database backup..."
	pg_dump $(DATABASE_URL) > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db: ## Restore database from backup
	@echo "Available backups:"
	@ls -la backups/
	@read -p "Enter backup filename: " backup; \
	psql $(DATABASE_URL) < backups/$$backup

# Cleaning
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-docker: ## Clean up Docker resources
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f
	docker volume prune -f

# Documentation
docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"
	@echo "Would generate API docs, architecture docs, etc."

# Health Checks
health: ## Check application health
	curl -f http://localhost:8000/health || exit 1

health-detailed: ## Detailed health check
	curl -s http://localhost:8000/health/detailed | jq '.'

# Performance
benchmark: ## Run performance benchmarks
	@echo "Benchmark tests not implemented yet"
	@echo "Would run load tests against the API"

profile: ## Run performance profiling
	@echo "Performance profiling not implemented yet"
	@echo "Would analyze application performance"

# Git Hooks
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Environment Info
info: ## Show environment information
	@echo "Python Version: $(shell python --version)"
	@echo "Pip Version: $(shell pip --version)"
	@echo "Docker Version: $(shell docker --version)"
	@echo "Docker Compose Version: $(shell docker-compose --version)"
	@echo "Git Branch: $(shell git branch --show-current)"
	@echo "Git Commit: $(shell git rev-parse --short HEAD)"