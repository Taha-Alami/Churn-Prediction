"""Batch scoring on new customer data using a saved model.

Run as a script::

    python -m churn.modeling.predict \\
        --input data/raw/customers.parquet \\
        --output data/scored.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn.config import settings
from churn.features import build_feature_matrix


def load_model(model_path: Path) -> dict:
    return joblib.load(model_path)


def align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Reindex a feature matrix to the training-time column set.

    Columns present at training time but missing in new data (e.g. a
    one-hot category that didn't appear in a small batch) are filled with 0.
    Columns present in new data but not seen at training time are dropped,
    since the model has no coefficient/split for them.
    """
    return X.reindex(columns=feature_columns, fill_value=0)


def predict_batch(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """Score a raw customer dataframe, returning ids + churn probability + predicted label."""
    X, _ = build_feature_matrix(df)
    X = align_features(X, model_bundle["feature_columns"])

    probabilities = model_bundle["model"].predict_proba(X)[:, 1]
    threshold = model_bundle["threshold"]

    out = pd.DataFrame(
        {
            "customer_id": df["customer_id"] if "customer_id" in df.columns else df.index,
            "churn_probability": probabilities,
            "churn_predicted": (probabilities >= threshold).astype(int),
        }
    )
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.raw_data_path)
    parser.add_argument("--output", type=Path, default=settings.data_dir / "scored.parquet")
    parser.add_argument("--model-path", type=Path, default=settings.models_dir / "model.joblib")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input)
    model_bundle = load_model(args.model_path)
    scored = predict_batch(df, model_bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    print(f"Scored {len(scored)} rows -> {args.output}")
    return scored


if __name__ == "__main__":
    main()
