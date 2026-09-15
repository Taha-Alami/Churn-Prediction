"""Tests for the synthetic data generator (churn.data.synthetic)."""

from __future__ import annotations

import pandas as pd

from churn.data.synthetic import COLUMNS, generate_synthetic_customers


def test_generation_is_deterministic_given_a_seed():
    df1 = generate_synthetic_customers(n_customers=500, seed=7)
    df2 = generate_synthetic_customers(n_customers=500, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    df1 = generate_synthetic_customers(n_customers=500, seed=1)
    df2 = generate_synthetic_customers(n_customers=500, seed=2)
    assert not df1["churn"].equals(df2["churn"])


def test_expected_columns_and_row_count():
    df = generate_synthetic_customers(n_customers=1_000, seed=42)
    assert list(df.columns) == COLUMNS
    assert len(df) == 1_000


def test_dtypes():
    df = generate_synthetic_customers(n_customers=1_000, seed=42)
    assert df["tenure_months"].dtype.kind in "iu"
    assert df["monthly_charges"].dtype.kind == "f"
    assert df["autopay"].dtype == bool
    assert df["has_multiple_lines"].dtype == bool
    assert df["is_senior"].dtype == bool
    assert set(df["churn"].unique()).issubset({0, 1})
    assert set(df["contract_type"].unique()).issubset({"month_to_month", "one_year", "two_year"})


def test_churn_rate_is_in_a_sane_range():
    df = generate_synthetic_customers(n_customers=20_000, seed=42)
    churn_rate = df["churn"].mean()
    assert 0.10 < churn_rate < 0.40


def test_no_fully_null_columns():
    df = generate_synthetic_customers(n_customers=2_000, seed=42)
    assert not df.isna().all().any()


def test_total_charges_is_null_only_for_zero_tenure_customers():
    df = generate_synthetic_customers(n_customers=5_000, seed=42)
    null_mask = df["total_charges"].isna()
    assert null_mask.sum() > 0, "expected at least some brand-new customers in a 5k sample"
    assert (df.loc[null_mask, "tenure_months"] == 0).all()
    assert (df.loc[~null_mask, "tenure_months"] > 0).all()
