# Project variables
APP_NAME = fastapi-musicgen
PYTHON_INTERPRETER = python3
ENV ?= dev
DOCKER_IMAGE = $(APP_NAME):latest
PORT ?= 8888

# Help target (prints help for each target)
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install dependencies"
	@echo "  run           Run the FastAPI application"
	@echo "  lint          Run linters for the codebase"
	@echo "  format        Auto-format the codebase using black"
	@echo "  test          Run unit tests"
	@echo "  clean         Remove Python cache and temporary files"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-run    Run the application in Docker"
	@echo "  docker-push   Push Docker image to registry"
	@echo "  docker-clean  Clean up Docker containers and images"

.PHONY: install
install:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

.PHONY: dev
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

.PHONY: format
format:
	@echo "Formatting code..."
	black .
	isort .

.PHONY: test
test:
	@echo "Running tests..."
	pytest

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache


.PHONY: build
build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .

.PHONY: run
run:
	@echo "Running Docker container..."
	docker run -it --rm -p ${PORT}:${PORT} $(DOCKER_IMAGE)

.PHONY: push
push:
	@echo "Pushing Docker image to registry..."
	docker push $(DOCKER_IMAGE)

.PHONY: prune
prune:
	@echo "Cleaning Docker images and containers..."
	docker system prune -f
	docker images -q | xargs docker rmi -f || true