"""Evaluation helpers shared across the prediction notebooks.

Includes:

* :func:`build_pipeline` -- standardize numeric columns then fit a model
* :func:`compute_test_metrics` -- ROC-AUC, PR-AUC, F1, precision, recall,
  Brier, confusion matrices at multiple thresholds
* :func:`tune_threshold` -- pick a probability threshold that maximizes F1
  (or recall at a fixed precision floor)
* :func:`recall_at_precision` -- summary statistic used in the report
* :func:`plot_pr_curves`, :func:`plot_calibration` -- figures for the
  prediction track
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Pipeline construction
# -----------------------------------------------------------------------------


def build_pipeline(
    estimator,
    numeric_columns: Sequence[str],
    binary_columns: Sequence[str],
    *,
    scale_for_estimator: bool,
) -> Pipeline:
    """Wrap ``estimator`` in a ``Pipeline`` with optional standardization.

    For linear and kernel-based models we standardize the numeric columns;
    binary and one-hot columns pass through.  For tree-based models we let
    everything pass through unchanged.
    """
    if scale_for_estimator:
        preproc = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), list(numeric_columns)),
                ("bin", "passthrough", list(binary_columns)),
            ],
            remainder="drop",
        )
    else:
        preproc = ColumnTransformer(
            transformers=[
                ("num", "passthrough", list(numeric_columns)),
                ("bin", "passthrough", list(binary_columns)),
            ],
            remainder="drop",
        )
    return Pipeline(steps=[("preproc", preproc), ("clf", estimator)])


# -----------------------------------------------------------------------------
# Threshold and metric helpers
# -----------------------------------------------------------------------------


def recall_at_precision(y_true, y_proba, min_precision: float) -> float:
    """Largest recall at which precision is at least ``min_precision``."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    mask = precisions[:-1] >= min_precision
    if not mask.any():
        return 0.0
    return float(recalls[:-1][mask].max())


def tune_threshold(
    y_true,
    y_proba,
    objective: str = "f1",
    min_precision: float = 0.5,
) -> float:
    """Choose a decision threshold that maximizes the chosen objective.

    objective:
        * ``'f1'`` -- maximizes F1 at the chosen threshold
        * ``'recall_at_precision'`` -- largest threshold that keeps
          precision >= ``min_precision``; if no such threshold exists,
          falls back to the F1-optimal threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if objective == "f1":
        # F1 over all thresholds (precision_recall_curve returns one extra point).
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
        best = int(np.argmax(f1s))
        return float(thresholds[best])
    elif objective == "recall_at_precision":
        mask = precisions[:-1] >= min_precision
        if not mask.any():
            return tune_threshold(y_true, y_proba, objective="f1")
        # Pick the threshold with the largest recall under the precision floor.
        candidates = np.where(mask)[0]
        best = candidates[np.argmax(recalls[:-1][candidates])]
        return float(thresholds[best])
    else:
        raise ValueError(f"unknown objective {objective!r}")


@dataclass
class TestMetrics:
    name: str
    threshold: float
    roc_auc: float
    pr_auc: float
    brier: float
    precision: float
    recall: float
    f1: float
    recall_at_p50: float
    recall_at_p70: float
    accuracy: float
    confusion: np.ndarray  # 2x2

    def as_row(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "threshold": round(self.threshold, 4),
            "ROC_AUC": round(self.roc_auc, 4),
            "PR_AUC": round(self.pr_auc, 4),
            "Brier": round(self.brier, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "recall@P=0.50": round(self.recall_at_p50, 4),
            "recall@P=0.70": round(self.recall_at_p70, 4),
            "accuracy": round(self.accuracy, 4),
        }


def compute_test_metrics(
    name: str,
    y_true,
    y_proba,
    threshold: float,
) -> TestMetrics:
    y_pred = (y_proba >= threshold).astype(int)
    return TestMetrics(
        name=name,
        threshold=threshold,
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        brier=float(brier_score_loss(y_true, y_proba)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        recall_at_p50=float(recall_at_precision(y_true, y_proba, 0.50)),
        recall_at_p70=float(recall_at_precision(y_true, y_proba, 0.70)),
        accuracy=float((y_pred == np.asarray(y_true)).mean()),
        confusion=confusion_matrix(y_true, y_pred),
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def plot_pr_curves(
    curves: Iterable[tuple[str, np.ndarray, np.ndarray]],
    ax: Optional[plt.Axes] = None,
    title: str = "Precision-Recall curves",
    baseline_positive_rate: Optional[float] = None,
) -> plt.Axes:
    """``curves`` is an iterable of ``(name, y_true, y_proba)``.

    Each curve gets the model's PR-AUC printed in the legend.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, y_true, y_proba in curves:
        precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recalls, precisions, label=f"{name} (AP = {ap:.3f})", linewidth=1.7)
    if baseline_positive_rate is not None:
        ax.axhline(baseline_positive_rate, linestyle="--", color="gray", linewidth=1,
                   label=f"Random baseline ({baseline_positive_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    return ax


def plot_calibration(
    y_true,
    y_proba,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 10,
    name: str = "model",
) -> plt.Axes:
    """Reliability diagram with quantile-binned predicted probabilities."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.5))
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    # Use quantile-based bins so each bin has a similar count.
    quantiles = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = 0.0, 1.0
    bin_idx = np.clip(np.searchsorted(quantiles, y_proba, side="right") - 1, 0, n_bins - 1)
    bin_means_pred, bin_means_true, bin_counts = [], [], []
    for k in range(n_bins):
        mask = bin_idx == k
        if mask.sum() == 0:
            continue
        bin_means_pred.append(y_proba[mask].mean())
        bin_means_true.append(y_true[mask].mean())
        bin_counts.append(int(mask.sum()))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(bin_means_pred, bin_means_true, "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Calibration: {name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def confusion_to_frame(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=["actual=0", "actual=1"],
        columns=["pred=0", "pred=1"],
    )


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
