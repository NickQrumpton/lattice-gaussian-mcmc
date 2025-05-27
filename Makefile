# Makefile for Lattice Gaussian MCMC Project
# Provides convenient commands for testing, development, and deployment

.PHONY: help install test test-fast test-slow test-unit test-integration test-statistical test-performance
.PHONY: coverage coverage-html coverage-xml lint format clean docs build check-deps security
.PHONY: pre-commit setup-dev benchmark profile golden-update reproducibility-check

# Default target
help:
	@echo "Lattice Gaussian MCMC - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install         Install package and dependencies"
	@echo "  setup-dev       Setup development environment"
	@echo "  check-deps      Check dependency compatibility"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            Run all tests"
	@echo "  test-fast       Run fast tests only (excludes slow marker)"
	@echo "  test-slow       Run slow tests only"
	@echo "  test-unit       Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-statistical Run statistical tests"
	@echo "  test-performance Run performance tests"
	@echo "  test-edge       Run edge case tests"
	@echo "  test-repro      Run reproducibility tests"
	@echo ""
	@echo "Coverage and Reports:"
	@echo "  coverage        Generate coverage report"
	@echo "  coverage-html   Generate HTML coverage report"
	@echo "  coverage-xml    Generate XML coverage report"
	@echo "  benchmark       Run performance benchmarks"
	@echo "  profile         Profile test execution"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run code linting"
	@echo "  format          Format code"
	@echo "  security        Run security checks"
	@echo "  pre-commit      Run pre-commit hooks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean           Clean build artifacts"
	@echo "  golden-update   Update golden reference files"
	@echo "  reproducibility-check Verify reproducibility"
	@echo ""

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
SOURCE_DIR := src
TEST_DIR := tests
COVERAGE_DIR := htmlcov
DOCS_DIR := docs

# Installation
install:
	@echo "Installing package and dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✓ Installation complete"

setup-dev: install
	@echo "Setting up development environment..."
	$(PIP) install pytest pytest-cov pytest-xdist pytest-benchmark pytest-mock
	$(PIP) install pytest-html pytest-json-report pytest-timeout pytest-sugar
	$(PIP) install black isort flake8 mypy bandit interrogate
	$(PIP) install pre-commit sphinx sphinx-rtd-theme
	pre-commit install
	@echo "✓ Development environment ready"

check-deps:
	@echo "Checking dependency compatibility..."
	$(PIP) check
	$(PYTHON) -c "import src; print('✓ Package imports successfully')"
	@echo "✓ Dependencies compatible"

# Testing Commands
test:
	@echo "Running complete test suite..."
	$(PYTEST) $(TEST_DIR) --cov=$(SOURCE_DIR) --cov-report=term-missing -v

test-fast:
	@echo "Running fast tests..."
	$(PYTEST) $(TEST_DIR) -m "not slow" --cov=$(SOURCE_DIR) --cov-report=term-missing -n auto

test-slow:
	@echo "Running slow tests..."
	$(PYTEST) $(TEST_DIR) -m "slow" -v --timeout=600

test-unit:
	@echo "Running unit tests..."
	$(PYTEST) $(TEST_DIR)/unit/ --cov=$(SOURCE_DIR) --cov-report=term-missing -v

test-integration:
	@echo "Running integration tests..."
	$(PYTEST) $(TEST_DIR)/integration/ --cov=$(SOURCE_DIR) --cov-report=term-missing -v

test-statistical:
	@echo "Running statistical tests..."
	$(PYTEST) $(TEST_DIR) -m "statistical" --cov=$(SOURCE_DIR) --cov-report=term-missing -v

test-performance:
	@echo "Running performance tests..."
	$(PYTEST) $(TEST_DIR) -m "performance" --benchmark-only --benchmark-sort=mean -v

test-edge:
	@echo "Running edge case tests..."
	$(PYTEST) $(TEST_DIR) -m "edge_case" -v

test-repro:
	@echo "Running reproducibility tests..."
	$(PYTEST) $(TEST_DIR) -m "reproducibility" -v

# Coverage Reports
coverage:
	@echo "Generating coverage report..."
	$(PYTEST) $(TEST_DIR) --cov=$(SOURCE_DIR) --cov-report=term-missing --cov-report=html

coverage-html: coverage
	@echo "HTML coverage report generated in $(COVERAGE_DIR)/"
	@echo "Open $(COVERAGE_DIR)/index.html in browser"

coverage-xml:
	@echo "Generating XML coverage report..."
	$(PYTEST) $(TEST_DIR) --cov=$(SOURCE_DIR) --cov-report=xml

# Performance and Profiling
benchmark:
	@echo "Running performance benchmarks..."
	$(PYTEST) $(TEST_DIR) -m "performance" --benchmark-only --benchmark-json=benchmark.json
	@echo "Benchmark results saved to benchmark.json"

profile:
	@echo "Profiling test execution..."
	$(PYTEST) $(TEST_DIR) --profile-svg --timeout=300
	@echo "Profile saved as prof/combined.svg"

# Code Quality
lint:
	@echo "Running code linting..."
	flake8 $(SOURCE_DIR) --count --statistics
	flake8 $(TEST_DIR) --count --statistics --extend-ignore=E501
	@echo "✓ Linting passed"

format:
	@echo "Formatting code..."
	black $(SOURCE_DIR) $(TEST_DIR) --line-length=100
	isort $(SOURCE_DIR) $(TEST_DIR) --profile=black --line-length=100
	@echo "✓ Code formatted"

security:
	@echo "Running security checks..."
	bandit -r $(SOURCE_DIR) -ll
	safety check
	@echo "✓ Security checks passed"

pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# Documentation
docs:
	@echo "Building documentation..."
	cd $(DOCS_DIR) && $(PYTHON) -m sphinx -b html . _build/html
	@echo "Documentation built in $(DOCS_DIR)/_build/html/"

docs-serve: docs
	@echo "Serving documentation locally..."
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

# Maintenance
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage $(COVERAGE_DIR)/
	rm -rf .pytest_cache/ .mypy_cache/ .tox/
	rm -rf prof/ benchmark.json coverage.xml
	@echo "✓ Cleanup complete"

golden-update:
	@echo "Updating golden reference files..."
	rm -rf $(TEST_DIR)/golden/*
	$(PYTEST) $(TEST_DIR) -k "golden" --tb=short
	@echo "✓ Golden files updated"

reproducibility-check:
	@echo "Verifying reproducibility..."
	@echo "Run 1..."
	$(PYTEST) $(TEST_DIR) -m "reproducibility" --tb=short > /tmp/run1.log 2>&1
	@echo "Run 2..."  
	$(PYTEST) $(TEST_DIR) -m "reproducibility" --tb=short > /tmp/run2.log 2>&1
	@echo "Comparing results..."
	diff /tmp/run1.log /tmp/run2.log && echo "✓ Reproducibility verified" || (echo "✗ Reproducibility failed" && exit 1)

# Continuous Integration Targets
ci-test:
	@echo "Running CI test suite..."
	$(PYTEST) $(TEST_DIR) -m "not slow" --cov=$(SOURCE_DIR) --cov-report=xml --cov-report=term-missing -n auto --timeout=300

ci-slow:
	@echo "Running CI slow tests..."
	$(PYTEST) $(TEST_DIR) -m "slow" --cov=$(SOURCE_DIR) --cov-append --cov-report=xml --timeout=600

ci-integration:
	@echo "Running CI integration tests..."
	$(PYTEST) $(TEST_DIR)/integration/ --cov=$(SOURCE_DIR) --cov-append --cov-report=xml --timeout=600

ci-coverage-check:
	@echo "Checking coverage threshold..."
	coverage report --fail-under=80

# Development Workflow Targets
dev-test: test-fast lint
	@echo "✓ Development test cycle complete"

dev-commit: format lint test-fast
	@echo "✓ Ready for commit"

dev-push: format lint test security
	@echo "✓ Ready for push"

# Release Targets
release-check: clean test coverage security docs
	@echo "✓ Release checks passed"

release-build: release-check
	@echo "Building release..."
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "✓ Release built in dist/"

# Help for specific test categories
help-markers:
	@echo "Available pytest markers:"
	@echo "  unit           - Unit tests for individual functions"
	@echo "  integration    - Integration tests for module interactions"
	@echo "  end_to_end     - Full pipeline tests"
	@echo "  slow           - Tests taking more than 5 seconds"
	@echo "  statistical    - Tests verifying statistical properties"
	@echo "  numerical      - Tests for numerical accuracy"
	@echo "  edge_case      - Edge cases and error conditions"
	@echo "  reproducibility - Deterministic behavior tests"
	@echo "  performance    - Performance and timing tests"

# Development convenience targets
watch-test:
	@echo "Watching for changes and running fast tests..."
	watchmedo shell-command --patterns="*.py" --recursive --command="make test-fast" .

interactive-test:
	@echo "Starting interactive test session..."
	$(PYTEST) $(TEST_DIR) --pdb -v

debug-test:
	@echo "Running tests in debug mode..."
	$(PYTEST) $(TEST_DIR) -vv -s --tb=long

# Statistics and reporting
test-stats:
	@echo "Test statistics:"
	@echo "Total tests: $$($(PYTEST) --collect-only -q | grep -c test_)"
	@echo "Unit tests: $$($(PYTEST) $(TEST_DIR)/unit/ --collect-only -q | grep -c test_)"
	@echo "Integration tests: $$($(PYTEST) $(TEST_DIR)/integration/ --collect-only -q | grep -c test_)"
	@echo "Slow tests: $$($(PYTEST) $(TEST_DIR) -m slow --collect-only -q | grep -c test_)"
	@echo "Fast tests: $$($(PYTEST) $(TEST_DIR) -m "not slow" --collect-only -q | grep -c test_)"

# Quality gates
quality-gate: lint security test-fast coverage-xml
	@echo "Running quality gate checks..."
	coverage report --fail-under=80
	@echo "✓ Quality gate passed"

# Project health check
health-check: check-deps quality-gate docs
	@echo "✓ Project health check passed"