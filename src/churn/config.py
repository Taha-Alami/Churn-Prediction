"""Project configuration.

All settings are plain dataclasses with environment-variable overrides. There
are no real secrets to manage here: the shipped data connector
(``LocalSyntheticConnector``) needs no credentials because it generates data
locally rather than calling a warehouse. The env-var override mechanism is
kept anyway so a future warehouse-backed connector (see
``src/churn/data/connector.py``) can be configured the same way in a real
deployment (e.g. ``CHURN_DATA_PATH``, ``CHURN_RANDOM_SEED``, connection
strings, etc.) without changing any calling code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings, overridable via environment variables."""

    random_seed: int = field(default_factory=lambda: _env_int("CHURN_RANDOM_SEED", 42))
    data_dir: Path = field(default_factory=lambda: _env_path("CHURN_DATA_DIR", REPO_ROOT / "data"))
    raw_data_path: Path = field(init=False)
    models_dir: Path = field(
        default_factory=lambda: _env_path("CHURN_MODELS_DIR", REPO_ROOT / "models")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.environ.get(
            "MLFLOW_TRACKING_URI", f"file:{(REPO_ROOT / 'mlruns').as_posix()}"
        )
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: os.environ.get("CHURN_MLFLOW_EXPERIMENT", "subscription-churn")
    )
    n_customers: int = field(default_factory=lambda: _env_int("CHURN_N_CUSTOMERS", 20_000))
    config_path: Path = field(
        default_factory=lambda: _env_path(
            "CHURN_MODEL_CONFIG", REPO_ROOT / "configs" / "model.yaml"
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_data_path", self.data_dir / "raw" / "customers.parquet")


settings = Settings()
