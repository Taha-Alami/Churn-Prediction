"""Tests for feature engineering (churn.features)."""

from __future__ import annotations

import numpy as np
import pytest

from churn.data.synthetic import generate_synthetic_customers
from churn.features import add_derived_features, build_feature_matrix


@pytest.fixture
def sample_df():
    return generate_synthetic_customers(n_customers=500, seed=3)


def test_add_derived_features_fills_total_charges_null(sample_df):
    assert sample_df["total_charges"].isna().any()
    out = add_derived_features(sample_df)
    assert not out["total_charges"].isna().any()
    # brand-new customers get total_charges == 0.0, not an arbitrary fill
    new_customers = out[out["is_new_customer"]]
    assert (new_customers["total_charges"] == 0.0).all()
    assert (new_customers["avg_monthly_spend"] == 0.0).all()


def test_add_derived_features_is_new_customer_flag_matches_tenure(sample_df):
    out = add_derived_features(sample_df)
    assert (out["is_new_customer"] == (sample_df["tenure_months"] == 0)).all()


def test_build_feature_matrix_shape_and_no_nulls(sample_df):
    X, y = build_feature_matrix(sample_df)
    assert len(X) == len(sample_df)
    assert not X.isna().any().any()
    assert y.isin([0, 1]).all()


def test_build_feature_matrix_one_hot_encodes_categoricals(sample_df):
    X, _ = build_feature_matrix(sample_df)
    expected_cols = {
        "contract_type_month_to_month",
        "contract_type_one_year",
        "contract_type_two_year",
        "payment_method_electronic_check",
        "payment_method_mailed_check",
        "payment_method_bank_transfer",
        "payment_method_credit_card",
    }
    assert expected_cols.issubset(set(X.columns))
    # one-hot columns are 0/1 integers
    for col in expected_cols:
        assert set(X[col].unique()).issubset({0, 1})


def test_build_feature_matrix_boolean_columns_cast_to_int(sample_df):
    X, _ = build_feature_matrix(sample_df)
    for col in ["autopay", "has_multiple_lines", "is_senior"]:
        assert X[col].dtype.kind in "iu"


def test_build_feature_matrix_without_target_column():
    df = generate_synthetic_customers(n_customers=50, seed=9).drop(columns=["churn"])
    X, y = build_feature_matrix(df)
    assert y is None
    assert len(X) == 50


def test_build_feature_matrix_raises_on_unexpected_null():
    df = generate_synthetic_customers(n_customers=50, seed=9)
    df = df.copy()
    df.loc[0, "monthly_charges"] = np.nan
    with pytest.raises(ValueError):
        build_feature_matrix(df)
