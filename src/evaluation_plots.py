"""Evaluation figures for the Bank Marketing ensemble project.

All functions consume actual results from the frozen, held-out test set.
Model selection and tuning must be finished using training-only CV before use.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


METRICS = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")


def plot_model_comparison(results: pd.DataFrame):
    """Compare 3 base models with 1 selected ensemble on a 0–1 metric scale."""
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("Provide a non-empty evaluation results DataFrame.")
    missing = set(METRICS).difference(results.columns)
    if missing:
        raise ValueError(f"Missing evaluation metric columns: {sorted(missing)}")
    if len(results) != 4:
        raise ValueError("Expected 3 base models and 1 selected ensemble (4 rows).")
    values = results.loc[:, list(METRICS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("All five evaluation metrics must contain numeric values.")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("All evaluation metric values must be finite.")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(values))
    width = 0.15
    for metric_index, metric in enumerate(METRICS):
        offset = (metric_index - (len(METRICS) - 1) / 2) * width
        ax.bar(positions + offset, values[metric], width, label=metric)
    ax.set_xticks(positions, values.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Held-out test performance: 3 base models vs selected ensemble")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(confusion_matrices: Mapping[str, np.ndarray]):
    """Plot count-based confusion matrices; rows are true and columns predicted."""
    if not confusion_matrices:
        raise ValueError("No confusion matrices supplied.")
    if len(confusion_matrices) != 4:
        raise ValueError("Expected 4 confusion matrices (3 base models + ensemble).")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, matrix) in zip(axes.flat, confusion_matrices.items()):
        cm = np.asarray(matrix)
        if cm.shape != (2, 2):
            raise ValueError(f"Expected a 2x2 confusion matrix for {name!r}.")
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
    fig.suptitle("Confusion matrices — held-out test set", fontsize=13)
    fig.tight_layout()
    return fig


def plot_roc_curves(fitted_models: Mapping[str, object], X_test, y_test):
    """Display ROC curves from fitted final models on the untouched test split."""
    if len(fitted_models) != 4:
        raise ValueError("Expected 3 fitted base models and 1 selected ensemble.")
    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError("ROC curves require test labels containing both 0 and 1.")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax, name=name, pos_label=1,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Chance")
    ax.set_title("ROC curves — held-out test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
