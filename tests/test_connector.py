"""Tests for LocalSyntheticConnector's generate-then-cache behavior."""

from __future__ import annotations

from unittest.mock import patch

from churn.data.connector import LocalSyntheticConnector
from churn.data.synthetic import generate_synthetic_customers


def test_first_call_generates_and_writes_cache(tmp_path):
    cache_path = tmp_path / "customers.parquet"
    connector = LocalSyntheticConnector(cache_path=cache_path, n_customers=200, seed=1)

    assert not cache_path.exists()
    df = connector.load_customers()

    assert cache_path.exists()
    assert len(df) == 200


def test_second_call_reads_cache_without_regenerating(tmp_path):
    cache_path = tmp_path / "customers.parquet"
    connector = LocalSyntheticConnector(cache_path=cache_path, n_customers=200, seed=1)
    connector.load_customers()

    with patch("churn.data.connector.generate_synthetic_customers") as mock_generate:
        df = connector.load_customers()

    mock_generate.assert_not_called()
    assert len(df) == 200


def test_force_regenerate_ignores_existing_cache(tmp_path):
    cache_path = tmp_path / "customers.parquet"
    connector = LocalSyntheticConnector(cache_path=cache_path, n_customers=200, seed=1)
    connector.load_customers()

    forced = LocalSyntheticConnector(
        cache_path=cache_path, n_customers=200, seed=1, force_regenerate=True
    )
    with patch(
        "churn.data.connector.generate_synthetic_customers",
        wraps=generate_synthetic_customers,
    ) as mock_generate:
        forced.load_customers()

    mock_generate.assert_called_once()


def test_cached_data_round_trips_correctly(tmp_path):
    cache_path = tmp_path / "customers.parquet"
    connector = LocalSyntheticConnector(cache_path=cache_path, n_customers=300, seed=5)
    first = connector.load_customers()
    second = connector.load_customers()

    assert list(first.columns) == list(second.columns)
    assert first["churn"].sum() == second["churn"].sum()
