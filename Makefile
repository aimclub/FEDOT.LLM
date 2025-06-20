# FedotLLM Project Makefile
# Comprehensive build and development automation

# Docker configuration
DC = docker compose
EXEC = docker exec -it
APP_FILE = docker/docker-compose.yml
APP_CONTAINER = fdlm-app
CHROMA_CONTAINER = fdlm-chroma
ENV = --env-file .env
SH = /bin/bash

# Project configuration
PYTHON = python3
UV = uv
PROJECT_NAME = fedotllm
STREAMLIT_PORT = 8080
STREAMLIT_HOST = 0.0.0.0

# Colors for output
BLUE = \033[34m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
NC = \033[0m # No Color

.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)FedotLLM Project Commands$(NC)"
	@echo "$(BLUE)========================$(NC)"
	@echo ""
	@echo "$(GREEN)Docker Operations:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^docker-/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development Workflow:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 !~ /^docker-/ && $$1 !~ /^help$$/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

.PHONY: docker-build
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	${DC} ${ENV} -f ${APP_FILE} build

.PHONY: docker-build-dev
docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build -t ${PROJECT_NAME}-dev -f docker/dev.Dockerfile .

.PHONY: docker-build-prod
docker-build-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(NC)"
	docker build -t ${PROJECT_NAME}-prod -f docker/run.Dockerfile .

.PHONY: docker-run
docker-run: ## Run with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	${DC} ${ENV} -f ${APP_FILE} up

.PHONY: docker-run-detached
docker-run-detached: ## Run with docker-compose in detached mode
	@echo "$(BLUE)Starting services in detached mode...$(NC)"
	${DC} ${ENV} -f ${APP_FILE} up -d

.PHONY: docker-dev
docker-dev: ## Run development environment with watch mode
	@echo "$(BLUE)Starting development environment with watch mode...$(NC)"
	${DC} ${ENV} -f ${APP_FILE} up --watch

.PHONY: docker-dev-build
docker-dev-build: ## Run development environment with build and watch
	@echo "$(BLUE)Building and starting development environment...$(NC)"
	${DC} ${ENV} -f ${APP_FILE} up --build --watch

.PHONY: docker-stop
docker-stop: ## Stop containers
	@echo "$(BLUE)Stopping containers...$(NC)"
	${DC} -f ${APP_FILE} down

.PHONY: docker-restart
docker-restart: ## Restart containers
	@echo "$(BLUE)Restarting containers...$(NC)"
	${DC} -f ${APP_FILE} restart

.PHONY: docker-clean
docker-clean: ## Clean up containers and images
	@echo "$(BLUE)Stopping and removing containers...$(NC)"
	${DC} -f ${APP_FILE} down -v --remove-orphans
	@echo "$(BLUE)Removing project images...$(NC)"
	-docker rmi ${PROJECT_NAME}-dev ${PROJECT_NAME}-prod 2>/dev/null || true
	@echo "$(BLUE)Cleaning up unused Docker resources...$(NC)"
	docker system prune -f

.PHONY: docker-clean-all
docker-clean-all: ## Clean up everything including volumes
	@echo "$(RED)WARNING: This will remove all containers, images, and volumes!$(NC)"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	${DC} -f ${APP_FILE} down -v --remove-orphans
	docker system prune -af --volumes

.PHONY: docker-logs
docker-logs: ## View container logs
	@echo "$(BLUE)Showing container logs...$(NC)"
	${DC} -f ${APP_FILE} logs -f

.PHONY: docker-logs-app
docker-logs-app: ## View app container logs
	@echo "$(BLUE)Showing app container logs...$(NC)"
	${DC} -f ${APP_FILE} logs -f app

.PHONY: docker-logs-chroma
docker-logs-chroma: ## View ChromaDB container logs
	@echo "$(BLUE)Showing ChromaDB container logs...$(NC)"
	${DC} -f ${APP_FILE} logs -f chromadb

.PHONY: docker-shell
docker-shell: ## Access app container shell
	@echo "$(BLUE)Accessing app container shell...$(NC)"
	${EXEC} ${APP_CONTAINER} ${SH}

.PHONY: docker-shell-chroma
docker-shell-chroma: ## Access ChromaDB container shell
	@echo "$(BLUE)Accessing ChromaDB container shell...$(NC)"
	${EXEC} ${CHROMA_CONTAINER} ${SH}

.PHONY: docker-ps
docker-ps: ## Show running containers
	@echo "$(BLUE)Running containers:$(NC)"
	${DC} -f ${APP_FILE} ps

# =============================================================================
# LOCAL DEVELOPMENT WORKFLOW
# =============================================================================

.PHONY: install
install: ## Install project dependencies locally
	@echo "$(BLUE)Installing dependencies with uv...$(NC)"
	${UV} sync --dev
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

.PHONY: install-prod
install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	${UV} sync --no-dev
	@echo "$(GREEN)Production dependencies installed!$(NC)"

.PHONY: update
update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	${UV} sync --upgrade
	@echo "$(GREEN)Dependencies updated!$(NC)"

.PHONY: venv
venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	${UV} venv
	@echo "$(GREEN)Virtual environment created!$(NC)"

# =============================================================================
# TESTING
# =============================================================================

.PHONY: test
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	${UV} run pytest

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	${UV} run pytest -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	${UV} run pytest --cov=${PROJECT_NAME} --cov-report=html --cov-report=term

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	${UV} run pytest tests/unit/

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	${UV} run pytest-watch

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: lint
lint: ## Run linting with ruff
	@echo "$(BLUE)Running linter...$(NC)"
	${UV} run ruff check .

.PHONY: lint-fix
lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	${UV} run ruff check --fix .

.PHONY: format
format: ## Format code and organize imports with ruff
	@echo "$(BLUE)Formatting code and organizing imports...$(NC)"
	${UV} run ruff check --select I --fix .
	${UV} run ruff format .

.PHONY: format-check
format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	${UV} run ruff format --check .

.PHONY: quality
quality: lint format-check test ## Run all quality checks
	@echo "$(GREEN)All quality checks completed!$(NC)"

# =============================================================================
# APPLICATION COMMANDS
# =============================================================================

.PHONY: streamlit
streamlit: ## Run Streamlit app locally
	@echo "$(BLUE)Starting Streamlit app...$(NC)"
	${UV} run python -m streamlit run ${PROJECT_NAME}/web/streamlit-app.py --server.port=${STREAMLIT_PORT} --server.address=${STREAMLIT_HOST}

.PHONY: streamlit-dev
streamlit-dev: ## Run Streamlit app in development mode
	@echo "$(BLUE)Starting Streamlit app in development mode...$(NC)"
	${UV} run python -m streamlit run ${PROJECT_NAME}/web/streamlit-app.py --server.port=${STREAMLIT_PORT} --server.address=${STREAMLIT_HOST} --server.runOnSave=true

.PHONY: jupyter
jupyter: ## Start Jupyter notebook
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	${UV} run jupyter notebook

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

.PHONY: clean
clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(NC)"

.PHONY: clean-logs
clean-logs: ## Clean log files
	@echo "$(BLUE)Cleaning log files...$(NC)"
	find . -name "*.log" -type f -delete
	@echo "$(GREEN)Log files cleaned!$(NC)"

.PHONY: env-check
env-check: ## Check environment and dependencies
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Python version: $$(${PYTHON} --version)"
	@echo "UV version: $$(${UV} --version 2>/dev/null || echo 'UV not installed')"
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Docker not installed')"
	@echo "Docker Compose version: $$(docker compose version 2>/dev/null || echo 'Docker Compose not installed')"

.PHONY: reset
reset: clean docker-clean install ## Reset everything and reinstall
	@echo "$(GREEN)Project reset completed!$(NC)"

.PHONY: status
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "$(GREEN)Docker containers:$(NC)"
	@${DC} -f ${APP_FILE} ps 2>/dev/null || echo "No containers running"
	@echo ""
	@echo "$(GREEN)Python environment:$(NC)"
	@${UV} run python --version 2>/dev/null || echo "No UV environment found"

# =============================================================================
# DEVELOPMENT SHORTCUTS
# =============================================================================

.PHONY: dev
dev: install docker-dev ## Quick start development environment
	@echo "$(GREEN)Development environment ready!$(NC)"

.PHONY: quick-test
quick-test: lint test ## Quick quality check
	@echo "$(GREEN)Quick test completed!$(NC)"

.PHONY: full-check
full-check: quality test-coverage ## Full project check
	@echo "$(GREEN)Full project check completed!$(NC)"