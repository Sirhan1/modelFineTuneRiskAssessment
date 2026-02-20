VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
PYTEST=$(VENV)/bin/pytest
RUFF=$(VENV)/bin/ruff
MYPY=$(VENV)/bin/mypy
TWINE=$(VENV)/bin/twine

.PHONY: setup install test demo lint typecheck build check-dist clean

setup:
	./scripts/setup_macos_apple_silicon.sh

install:
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(PIP) install -e ".[dev]"

test:
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(PYTEST)

demo:
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(VENV)/bin/alignment-risk demo --output-dir artifacts

lint:
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(RUFF) check src tests

typecheck:
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(MYPY) src/alignment_risk

build:
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	rm -rf build dist
	$(PYTHON) -m build

check-dist: build
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(TWINE) check dist/*

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache artifacts build dist
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
