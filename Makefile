.PHONY: install lint format format-check test data train evaluate predict clean

VENV := .venv
VENV_BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
endif
PYTHON := $(VENV_BIN)/python

install:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) -m churn.data.synthetic

train:
	$(PYTHON) -m churn.modeling.train

predict:
	$(PYTHON) -m churn.modeling.predict

evaluate: train

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns

clean:
	rm -rf data/raw data/scored.parquet models reports mlruns .pytest_cache .ruff_cache
