# ========================================================================================
# CloakPivot Project Makefile
# ========================================================================================
# Single command hub for all development and CI/CD operations
# All tool configurations are centralized in pyproject.toml
# ========================================================================================

.PHONY: help dev all check format lint type test clean
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PACKAGE := cloakpivot
TEST_PATH := tests

# Color codes for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# ========================================================================================
# Primary Targets
# ========================================================================================

help: ## Show this help message
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║                  CloakPivot Development Tools                 ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make dev       - Setup development environment"
	@echo "  make check     - Run quick validation (format, lint)"
	@echo "  make test      - Run test suite with coverage"
	@echo "  make all       - Run full CI/CD pipeline"
	@echo ""

dev: ## Setup development environment with all dependencies
	@echo "$(GREEN)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Development environment ready$(NC)"

all: format lint type test ## CI/CD entry point - run all checks and tests
	@echo "$(GREEN)════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ All checks passed successfully!$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════════════════════$(NC)"

check: format lint ## Quick pre-commit validation (format + lint)
	@echo "$(GREEN)✓ Quick check passed$(NC)"

# ========================================================================================
# Code Quality Targets
# ========================================================================================

format: ## Format code with Black
	@echo "$(CYAN)Running Black formatter...$(NC)"
	@$(PYTHON) -m black $(PACKAGE) $(TEST_PATH) --config pyproject.toml
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Lint code with Ruff
	@echo "$(CYAN)Running Ruff linter...$(NC)"
	@$(PYTHON) -m ruff check $(PACKAGE) $(TEST_PATH) --fix
	@echo "$(GREEN)✓ Linting complete$(NC)"

type: ## Type check with MyPy
	@echo "$(CYAN)Running MyPy type checker...$(NC)"
	@$(PYTHON) -m mypy --config-file pyproject.toml
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# ========================================================================================
# Testing Targets
# ========================================================================================

test: ## Run tests with coverage report
	@echo "$(CYAN)Running test suite with coverage...$(NC)"
	@$(PYTHON) -m pytest
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-unit: ## Run only unit tests
	@echo "$(CYAN)Running unit tests...$(NC)"
	@$(PYTHON) -m pytest -m unit

test-integration: ## Run only integration tests
	@echo "$(CYAN)Running integration tests...$(NC)"
	@$(PYTHON) -m pytest -m integration

test-e2e: ## Run only end-to-end tests
	@echo "$(CYAN)Running end-to-end tests...$(NC)"
	@$(PYTHON) -m pytest -m e2e

test-verbose: ## Run tests with verbose output
	@echo "$(CYAN)Running tests (verbose)...$(NC)"
	@$(PYTHON) -m pytest -vv

test-fast: ## Run tests without coverage
	@echo "$(CYAN)Running tests (no coverage)...$(NC)"
	@$(PYTHON) -m pytest --no-cov

coverage-html: ## Generate HTML coverage report
	@echo "$(CYAN)Generating HTML coverage report...$(NC)"
	@$(PYTHON) -m pytest --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

# ========================================================================================
# Utility Targets
# ========================================================================================

clean: ## Remove all build artifacts and caches
	@echo "$(YELLOW)Cleaning project...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*~" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

install: ## Install package in production mode
	@echo "$(CYAN)Installing package...$(NC)"
	$(PIP) install .
	@echo "$(GREEN)✓ Package installed$(NC)"

install-dev: dev ## Alias for 'make dev'
	@echo "$(GREEN)✓ Development installation complete$(NC)"

uninstall: ## Uninstall package
	@echo "$(YELLOW)Uninstalling package...$(NC)"
	$(PIP) uninstall -y $(PACKAGE)
	@echo "$(GREEN)✓ Package uninstalled$(NC)"

# ========================================================================================
# Development Workflow Targets
# ========================================================================================

watch: ## Watch for changes and run tests (requires pytest-watch)
	@echo "$(CYAN)Watching for changes...$(NC)"
	@command -v ptw >/dev/null 2>&1 || (echo "$(RED)Error: pytest-watch not installed. Run: pip install pytest-watch$(NC)" && exit 1)
	@ptw -- -x

build: clean ## Build distribution packages
	@echo "$(CYAN)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Build complete$(NC)"

publish-test: build ## Publish to TestPyPI (requires credentials)
	@echo "$(CYAN)Publishing to TestPyPI...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Published to TestPyPI$(NC)"

publish: build ## Publish to PyPI (requires credentials)
	@echo "$(RED)Publishing to PyPI...$(NC)"
	@echo "$(YELLOW)Are you sure? [y/N]$(NC)" && read ans && [ $${ans:-N} = y ]
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)✓ Published to PyPI$(NC)"

# ========================================================================================
# Dependency Management
# ========================================================================================

deps-update: ## Update all dependencies to latest versions
	@echo "$(CYAN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev]"
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

deps-check: ## Check for outdated dependencies
	@echo "$(CYAN)Checking dependencies...$(NC)"
	$(PIP) list --outdated
	@echo "$(GREEN)✓ Dependency check complete$(NC)"

# ========================================================================================
# Documentation Targets
# ========================================================================================

docs-serve: ## Serve documentation locally (if docs exist)
	@echo "$(CYAN)Serving documentation...$(NC)"
	@if [ -d "docs" ]; then \
		cd docs && $(PYTHON) -m http.server 8000; \
	else \
		echo "$(YELLOW)No docs directory found$(NC)"; \
	fi

# ========================================================================================
# CI/CD Helper Targets
# ========================================================================================

ci-install: ## Install package for CI environment
	@echo "$(CYAN)Setting up CI environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ CI environment ready$(NC)"

ci-test: ## Run tests for CI (with coverage)
	@echo "$(CYAN)Running CI tests...$(NC)"
	$(PYTHON) -m pytest --cov-report=xml --cov-report=term

# ========================================================================================
# Git Workflow Helpers
# ========================================================================================

pre-commit: check ## Pre-commit hook validation
	@echo "$(GREEN)✓ Pre-commit checks passed$(NC)"

pre-push: all ## Pre-push hook validation (full pipeline)
	@echo "$(GREEN)✓ Pre-push validation complete$(NC)"

# ========================================================================================
# Debug and Diagnostic Targets
# ========================================================================================

info: ## Show project information
	@echo "$(CYAN)Project Information:$(NC)"
	@echo "  Package: $(PACKAGE)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Location: $(shell pwd)"
	@echo ""
	@echo "$(CYAN)Installed Tools:$(NC)"
	@echo "  Black: $(shell $(PYTHON) -m black --version 2>/dev/null || echo 'Not installed')"
	@echo "  Ruff: $(shell $(PYTHON) -m ruff --version 2>/dev/null || echo 'Not installed')"
	@echo "  MyPy: $(shell $(PYTHON) -m mypy --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pytest: $(shell $(PYTHON) -m pytest --version 2>/dev/null || echo 'Not installed')"

verify: ## Verify all tools are properly configured
	@echo "$(CYAN)Verifying tool configuration...$(NC)"
	@$(PYTHON) -c "import black" && echo "$(GREEN)✓ Black installed$(NC)" || echo "$(RED)✗ Black not installed$(NC)"
	@$(PYTHON) -c "import ruff" && echo "$(GREEN)✓ Ruff installed$(NC)" || echo "$(RED)✗ Ruff not installed$(NC)"
	@$(PYTHON) -c "import mypy" && echo "$(GREEN)✓ MyPy installed$(NC)" || echo "$(RED)✗ MyPy not installed$(NC)"
	@$(PYTHON) -c "import pytest" && echo "$(GREEN)✓ Pytest installed$(NC)" || echo "$(RED)✗ Pytest not installed$(NC)"
	@$(PYTHON) -c "import pytest_cov" && echo "$(GREEN)✓ Coverage installed$(NC)" || echo "$(RED)✗ Coverage not installed$(NC)"
	@echo ""
	@echo "$(CYAN)Configuration files:$(NC)"
	@test -f pyproject.toml && echo "$(GREEN)✓ pyproject.toml found$(NC)" || echo "$(RED)✗ pyproject.toml missing$(NC)"
	@echo ""
	@echo "$(GREEN)Verification complete$(NC)"