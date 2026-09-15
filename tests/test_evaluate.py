"""Tests for churn.modeling.evaluate metric functions on hand-constructed examples."""

from __future__ import annotations

import numpy as np
import pytest

from churn.modeling.evaluate import brier_score, compute_metrics, select_f1_optimal_threshold

# Hand-constructed example with perfect rank separation:
# negatives = [0.1, 0.4, 0.6], positives = [0.7, 0.9].
# Every positive score exceeds every negative score, so ROC-AUC and PR-AUC
# (average precision) are both exactly 1.0 regardless of threshold.
Y_TRUE = np.array([0, 0, 0, 1, 1])
Y_PROB = np.array([0.1, 0.4, 0.6, 0.7, 0.9])


def test_compute_metrics_perfect_separation_auc():
    metrics = compute_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_compute_metrics_confusion_matrix_and_f1_at_threshold():
    # threshold=0.5 -> predictions [0, 0, 1, 1, 1]
    # tn=2 (indices 0,1), fp=1 (index 2), fn=0, tp=2 (indices 3,4)
    # precision = 2/3, recall = 1.0, f1 = 2*(2/3*1)/(2/3+1) = 0.8
    metrics = compute_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    assert metrics["confusion_matrix"] == [[2, 1], [0, 2]]
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["threshold"] == 0.5


def test_compute_metrics_higher_threshold_changes_confusion_matrix():
    # threshold=0.65 -> predictions [0, 0, 0, 1, 1]: perfect classification
    metrics = compute_metrics(Y_TRUE, Y_PROB, threshold=0.65)
    assert metrics["confusion_matrix"] == [[3, 0], [0, 2]]
    assert metrics["f1"] == pytest.approx(1.0)


def test_brier_score_known_value():
    # (0.2-0)^2 = 0.04, (0.8-1)^2 = 0.04 -> mean = 0.04
    score = brier_score(np.array([0, 1]), np.array([0.2, 0.8]))
    assert score == pytest.approx(0.04)


def test_brier_score_perfect_predictions_is_zero():
    score = brier_score(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert score == pytest.approx(0.0)


def test_select_f1_optimal_threshold_beats_or_ties_grid_search():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=200)
    y_prob = np.clip(y_true * 0.5 + rng.normal(0.3, 0.25, size=200), 0, 1)

    chosen_threshold = select_f1_optimal_threshold(y_true, y_prob)
    chosen_f1 = compute_metrics(y_true, y_prob, threshold=chosen_threshold)["f1"]

    grid_f1_scores = [
        compute_metrics(y_true, y_prob, threshold=t)["f1"] for t in np.linspace(0.01, 0.99, 99)
    ]
    assert chosen_f1 >= max(grid_f1_scores) - 1e-9


def test_select_f1_optimal_threshold_on_perfectly_separable_data():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = select_f1_optimal_threshold(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    assert metrics["f1"] == pytest.approx(1.0)
