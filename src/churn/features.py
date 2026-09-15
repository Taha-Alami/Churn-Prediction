"""Feature engineering for the churn model.

Transforms the raw subscription-telecom customer table (as produced by
``churn.data.synthetic`` / ``churn.data.connector``) into a numeric feature
matrix suitable for scikit-learn / XGBoost.

Handling of the ``total_charges`` null:
    Brand-new customers (``tenure_months == 0``) have a null
    ``total_charges`` in the raw data because they have not completed a
    billing cycle yet. We do not drop these rows or mean-impute blindly
    (that would fabricate billing history). Instead we:
      1. Add a boolean flag ``is_new_customer`` (``tenure_months == 0``),
         which is itself a useful churn signal on its own.
      2. Impute the null ``total_charges`` to ``0.0``, which is the
         factually correct value for a customer who has not been billed
         yet, rather than an arbitrary fill.

No other columns contain nulls in the synthetic generator, but
``build_feature_matrix`` will raise if it encounters unexpected nulls in a
numeric column, rather than silently propagating them into the model.
"""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"

CATEGORICAL_COLUMNS = ["contract_type", "payment_method"]
BOOLEAN_COLUMNS = ["autopay", "has_multiple_lines", "is_senior"]
NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets_90d",
    "avg_monthly_usage_gb",
]
DERIVED_NUMERIC_COLUMNS = ["avg_monthly_spend", "charges_per_gb"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns and handle the documented ``total_charges`` null.

    Adds:
      - ``is_new_customer``: bool, ``tenure_months == 0``.
      - ``total_charges`` nulls filled with 0.0 (see module docstring).
      - ``avg_monthly_spend``: ``total_charges / max(tenure_months, 1)``,
        i.e. lifetime spend normalized by tenure (using 1 instead of 0 in
        the denominator for brand-new customers, whose total_charges is
        already 0, giving avg_monthly_spend == 0 for them too).
      - ``charges_per_gb``: ``monthly_charges / avg_monthly_usage_gb``, a
        proxy for price sensitivity per unit of usage.
    """
    out = df.copy()
    out["is_new_customer"] = out["tenure_months"].eq(0)
    out["total_charges"] = out["total_charges"].fillna(0.0)

    tenure_denominator = out["tenure_months"].clip(lower=1)
    out["avg_monthly_spend"] = out["total_charges"] / tenure_denominator
    out["charges_per_gb"] = out["monthly_charges"] / out["avg_monthly_usage_gb"].clip(lower=0.01)
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the numeric feature matrix ``X`` and target vector ``y``.

    Categorical columns are one-hot encoded (``pandas.get_dummies``, no
    dropped level -- tree models handle the redundant column fine and it
    keeps SHAP output interpretable per category). Boolean columns are cast
    to int. Returns ``(X, y)`` with ``y`` as int 0/1.
    """
    engineered = add_derived_features(df)

    numeric_cols = NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS + ["is_new_customer"]
    numeric_block = engineered[numeric_cols].copy()
    numeric_block["is_new_customer"] = numeric_block["is_new_customer"].astype(int)

    if numeric_block.isna().any().any():
        bad_cols = numeric_block.columns[numeric_block.isna().any()].tolist()
        raise ValueError(f"Unexpected nulls in numeric feature columns: {bad_cols}")

    bool_block = engineered[BOOLEAN_COLUMNS].astype(int)

    categorical_block = pd.get_dummies(
        engineered[CATEGORICAL_COLUMNS].astype("category"), prefix=CATEGORICAL_COLUMNS
    ).astype(int)

    X = pd.concat([numeric_block, bool_block, categorical_block], axis=1)

    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].astype(int)

    return X, y


def feature_names(X: pd.DataFrame) -> list[str]:
    """Convenience accessor used by evaluation/SHAP code."""
    return list(X.columns)
