"""Ensemble evaluation helpers for the 6001CMD Bank Marketing project.

Integration requirements:
- Ben supplies one leakage-safe train/test split and shared preprocessing interface.
- Zyen supplies three sklearn-compatible (preferably Pipeline) classifiers.
- Model selection uses TRAINING cross-validation scores, not the final test set.
- The target must be binary: 0 = no, 1 = yes.

Do not fit feature transformations on the entire dataset before splitting.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def choose_two_models(validation_f1: dict[str, float]) -> tuple[str, str]:
    """Rank exactly three base models using F1 measured WITHOUT test data.

    Review prediction diversity as well before accepting this initial pair.
    """
    if len(validation_f1) != 3:
        raise ValueError("Supply validation F1 scores for exactly three base models.")
    if any(not np.isfinite(float(score)) for score in validation_f1.values()):
        raise ValueError("All validation F1 scores must be finite numbers.")
    ranked = sorted(validation_f1, key=validation_f1.get, reverse=True)
    return ranked[0], ranked[1]


def make_ensembles(
    base_models: dict,
    selected: tuple[str, str],
    *,
    random_state: int = 42,
) -> dict:
    """Build Voting and Stacking from two unfitted or cloneable base estimators.

    Each estimator should include any preprocessing inside its Pipeline.
    Both estimators must support predict_proba for soft voting.
    """
    if len(selected) != 2 or selected[0] == selected[1]:
        raise ValueError("Select two distinct base models.")
    if not all(name in base_models for name in selected):
        raise KeyError("Selected names must appear in base_models.")
    chosen = [(name, clone(base_models[name])) for name in selected]
    if not all(hasattr(est, "predict_proba") for _, est in chosen):
        raise ValueError("Soft voting requires both base estimators to support predict_proba.")

    return {
        "Ensemble - Soft Voting": VotingClassifier(
            estimators=chosen, voting="soft", n_jobs=None
        ),
        "Ensemble - Stacking": StackingClassifier(
            estimators=chosen,
            final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            stack_method="predict_proba",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=None,
            passthrough=False,
        ),
    }


def _positive_score(fitted_model, X):
    """Obtain a positive-class score for ROC-AUC, if the API permits."""
    if hasattr(fitted_model, "predict_proba"):
        probability = fitted_model.predict_proba(X)
        class_list = list(fitted_model.classes_)
        return np.asarray(probability)[:, class_list.index(1)]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X)
    return None


def evaluate_candidates(
    base_models: dict,
    selected: tuple[str, str],
    X_train,
    y_train,
    X_test,
    y_test,
    ensemble_method: str = "Ensemble - Soft Voting",
) -> tuple[pd.DataFrame, dict, dict]:
    """Evaluate exactly 3 base models and ONE validation-selected ensemble.

    Run only after model/hyperparameter and ensemble technique selection
    have been completed using training folds, not final test outcomes.
    """
    if len(base_models) != 3:
        raise ValueError("Exactly three base models are required.")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training labels must contain both 0 and 1.")
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("Test labels must be encoded as 0/1.")

    ensembles = make_ensembles(base_models, selected)
    if ensemble_method not in ensembles:
        raise ValueError(f"Choose an ensemble from {list(ensembles)}")
    candidates = {**base_models, ensemble_method: ensembles[ensemble_method]}
    rows, fitted_models, confusion_matrices = [], {}, {}

    for name, candidate in candidates.items():
        start = perf_counter()
        model = clone(candidate)
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start

        predicted = model.predict(X_test)
        positive_score = _positive_score(model, X_test)
        auc = (
            roc_auc_score(y_test, positive_score)
            if positive_score is not None and len(np.unique(y_test)) == 2
            else np.nan
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(y_test, predicted, zero_division=0),
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "F1": f1_score(y_test, predicted, zero_division=0),
                "ROC-AUC": auc,
                "Fit Seconds": fit_seconds,
                "Train F1": f1_score(y_train, model.predict(X_train), zero_division=0),
            }
        )
        fitted_models[name] = model
        confusion_matrices[name] = confusion_matrix(y_test, predicted, labels=[0, 1])

    return pd.DataFrame(rows).set_index("Model"), fitted_models, confusion_matrices
