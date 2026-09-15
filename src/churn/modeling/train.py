"""Train, cross-validate, calibrate, and log the churn model.

Run as a script: ``python -m churn.modeling.train``. Reads hyperparameters
and thresholds from ``configs/model.yaml`` (no magic numbers here), trains a
baseline logistic regression alongside the main XGBoost model, runs
stratified k-fold CV on the main model, calibrates it, evaluates both models
on a held-out test set, logs everything to a local file-based MLflow
tracking store, and saves the final model to ``models/model.joblib`` for
``churn.modeling.predict``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import yaml
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churn.config import settings
from churn.data.connector import LocalSyntheticConnector
from churn.features import build_feature_matrix
from churn.modeling.evaluate import (
    brier_score,
    compute_metrics,
    plot_calibration_curve,
    plot_shap_summary,
    select_f1_optimal_threshold,
)


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_baseline_pipeline(cfg: dict[str, Any]) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("oversample", RandomOverSampler(random_state=cfg["imbalance"]["random_seed"])),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=cfg["baseline_model"]["max_iter"],
                    random_state=cfg["baseline_model"]["random_seed"],
                ),
            ),
        ]
    )


def build_main_pipeline(cfg: dict[str, Any]) -> ImbPipeline:
    main_cfg = cfg["main_model"]
    return ImbPipeline(
        steps=[
            ("oversample", RandomOverSampler(random_state=cfg["imbalance"]["random_seed"])),
            (
                "clf",
                XGBClassifier(
                    n_estimators=main_cfg["n_estimators"],
                    max_depth=main_cfg["max_depth"],
                    learning_rate=main_cfg["learning_rate"],
                    subsample=main_cfg["subsample"],
                    colsample_bytree=main_cfg["colsample_bytree"],
                    min_child_weight=main_cfg["min_child_weight"],
                    reg_lambda=main_cfg["reg_lambda"],
                    eval_metric=main_cfg["eval_metric"],
                    random_state=main_cfg["random_seed"],
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_cv_splitter(y, cfg: dict[str, Any]) -> StratifiedKFold:
    n_splits = min(cfg["cv"]["n_splits"], int(np.bincount(y).min()))
    n_splits = max(n_splits, 2)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["cv"]["random_seed"])


def run_cv(pipeline: ImbPipeline, X, y, skf: StratifiedKFold) -> dict[str, Any]:
    scoring = ["roc_auc", "average_precision", "f1"]
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=None)
    return {
        "n_splits": skf.get_n_splits(),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "pr_auc_mean": float(np.mean(scores["test_average_precision"])),
        "pr_auc_std": float(np.std(scores["test_average_precision"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=settings.config_path)
    parser.add_argument("--n-customers", type=int, default=None)
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--reports-dir", type=Path, default=settings.data_dir.parent / "reports")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    n_customers = args.n_customers or settings.n_customers

    connector = LocalSyntheticConnector(
        cache_path=settings.raw_data_path,
        n_customers=n_customers,
        seed=settings.random_seed,
    )
    df = connector.load_customers()
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_seed"],
        stratify=y,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    results: dict[str, Any] = {"n_customers": len(df), "churn_rate": float(df["churn"].mean())}

    skf = make_cv_splitter(y_train, cfg)

    # Model selection (raw vs. calibrated) and decision-threshold tuning are
    # both done from out-of-fold (OOF) predictions on the TRAINING split only
    # -- the test set is touched exactly once, at the very end, for the
    # final reported metrics. This avoids the common leakage bug of picking
    # a threshold or a calibration strategy by peeking at test performance.
    #
    # cross_val_predict on a fresh CalibratedClassifierCV(main_pipeline, ...)
    # performs nested CV: for each outer fold it fits calibration (with its
    # own inner CV) on the other folds and predicts on the held-out fold, so
    # the resulting OOF calibrated probabilities are genuinely out-of-sample.
    oof_prob_raw = cross_val_predict(
        build_main_pipeline(cfg), X_train, y_train, cv=skf, method="predict_proba"
    )[:, 1]
    oof_prob_calibrated = cross_val_predict(
        CalibratedClassifierCV(
            build_main_pipeline(cfg),
            method=cfg["calibration"]["method"],
            cv=cfg["calibration"]["cv"],
        ),
        X_train,
        y_train,
        cv=skf,
        method="predict_proba",
    )[:, 1]

    oof_brier_raw = brier_score(y_train, oof_prob_raw)
    oof_brier_calibrated = brier_score(y_train, oof_prob_calibrated)
    use_calibrated = oof_brier_calibrated < oof_brier_raw
    oof_prob_final = oof_prob_calibrated if use_calibrated else oof_prob_raw
    threshold = select_f1_optimal_threshold(y_train, oof_prob_final)

    results["decision_threshold"] = threshold
    results["oof_brier_raw"] = oof_brier_raw
    results["oof_brier_calibrated"] = oof_brier_calibrated
    results["used_calibrated_model"] = use_calibrated

    with mlflow.start_run(run_name="baseline_logistic_regression") as baseline_run:
        baseline_pipeline = build_baseline_pipeline(cfg)
        baseline_pipeline.fit(X_train, y_train)
        baseline_prob = baseline_pipeline.predict_proba(X_test)[:, 1]
        baseline_metrics = compute_metrics(y_test, baseline_prob, threshold=threshold)

        mlflow.log_params({"model_type": "logistic_regression", **cfg["baseline_model"]})
        mlflow.log_metrics(
            {k: v for k, v in baseline_metrics.items() if isinstance(v, int | float)}
        )
        mlflow.sklearn.log_model(baseline_pipeline, artifact_path="model")
        results["baseline"] = baseline_metrics
        results["baseline_run_id"] = baseline_run.info.run_id

    with mlflow.start_run(run_name="main_xgboost") as main_run:
        main_pipeline = build_main_pipeline(cfg)

        cv_results = run_cv(main_pipeline, X_train, y_train, skf)
        cv_results["decision_threshold"] = threshold
        mlflow.log_metrics(
            {f"cv_{k}": v for k, v in cv_results.items() if isinstance(v, int | float)}
        )

        main_pipeline.fit(X_train, y_train)
        main_prob_raw = main_pipeline.predict_proba(X_test)[:, 1]
        main_metrics_raw = compute_metrics(y_test, main_prob_raw, threshold=threshold)
        brier_raw = brier_score(y_test, main_prob_raw)

        calibrated = CalibratedClassifierCV(
            main_pipeline,
            method=cfg["calibration"]["method"],
            cv=cfg["calibration"]["cv"],
        )
        calibrated.fit(X_train, y_train)
        main_prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
        main_metrics_calibrated = compute_metrics(y_test, main_prob_calibrated, threshold=threshold)
        brier_calibrated = brier_score(y_test, main_prob_calibrated)

        final_model = calibrated if use_calibrated else main_pipeline
        final_prob = main_prob_calibrated if use_calibrated else main_prob_raw
        final_metrics = main_metrics_calibrated if use_calibrated else main_metrics_raw

        mlflow.log_params({"model_type": "xgboost", **cfg["main_model"], **cfg["imbalance"]})
        mlflow.log_metrics(
            {k: v for k, v in main_metrics_raw.items() if isinstance(v, int | float)}
        )
        mlflow.log_metrics(
            {
                "oof_brier_raw": oof_brier_raw,
                "oof_brier_calibrated": oof_brier_calibrated,
                "test_brier_raw": brier_raw,
                "test_brier_calibrated": brier_calibrated,
                "used_calibrated_model": float(use_calibrated),
            }
        )
        mlflow.sklearn.log_model(final_model, artifact_path="model")

        reports_dir = Path(args.reports_dir)
        calibration_path = plot_calibration_curve(
            y_test, final_prob, reports_dir / "calibration_curve.png", label="xgboost (final)"
        )
        mlflow.log_artifact(str(calibration_path))

        try:
            fitted_xgb = main_pipeline.named_steps["clf"]
            shap_path = plot_shap_summary(fitted_xgb, X_test, reports_dir / "shap_summary.png")
            mlflow.log_artifact(str(shap_path))
            results["shap_summary_path"] = str(shap_path)
        except Exception as exc:  # pragma: no cover - SHAP plotting is best-effort
            print(f"SHAP summary plot failed: {exc}", file=sys.stderr)

        results["cv"] = cv_results
        results["main_raw"] = main_metrics_raw
        results["main_calibrated"] = main_metrics_calibrated
        results["test_brier_raw"] = brier_raw
        results["test_brier_calibrated"] = brier_calibrated
        results["main_final"] = final_metrics
        results["main_run_id"] = main_run.info.run_id
        results["calibration_curve_path"] = str(calibration_path)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(X.columns),
            "threshold": threshold,
            "used_calibrated_model": use_calibrated,
        },
        model_path,
    )
    results["model_path"] = str(model_path)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
