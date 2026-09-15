"""Metric computation, calibration curve, and SHAP plotting.

The metric functions here are thin, well-tested wrappers around scikit-learn
so training and CI-smoke code share one source of truth for "what does a
model's performance look like", and so we have direct pytest coverage on a
hand-constructed example with a known expected answer (see
``tests/test_evaluate.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, F1 (at ``threshold``), and a confusion matrix.

    Parameters
    ----------
    y_true: array of 0/1 ground truth labels.
    y_prob: array of predicted positive-class probabilities.
    threshold: decision threshold used to turn ``y_prob`` into hard labels
        for F1 and the confusion matrix. ROC-AUC and PR-AUC are threshold
        independent.

    Returns
    -------
    dict with keys ``roc_auc``, ``pr_auc``, ``f1``, ``threshold``, and
    ``confusion_matrix`` (a list of lists: ``[[tn, fp], [fn, tp]]``).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }


def select_f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the decision threshold that maximizes F1 over a precision-recall sweep.

    Intended to be called on out-of-fold (cross-validated) training
    predictions, never on the test set, so the chosen threshold does not
    leak test-set information. ``precision_recall_curve`` returns
    ``len(thresholds) == len(precision) - 1``; the last precision/recall
    point (precision=1, recall=0) has no corresponding threshold and is
    dropped before taking the argmax.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    f1_scores = np.where(
        (precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    label: str = "model",
) -> Path:
    """Plot and save a reliability diagram comparing predicted vs observed churn rate."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predictions and outcomes (lower is better calibrated)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def plot_shap_summary(model: Any, X: Any, output_path: Path, max_display: int = 15) -> Path:
    """Compute SHAP values for a tree model and save a summary (beeswarm) plot."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
