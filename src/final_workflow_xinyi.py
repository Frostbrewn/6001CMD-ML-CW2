"""Integrate Benjamin's preprocessing, Zyen's tuned models and Xinyi's ensemble.

Run from the repo root:
  python -u src/final_workflow_xinyi.py --check-only
  python -u src/final_workflow_xinyi.py --full

--check-only does not train models.
--full uses Zyen's actual 5-fold grid search and can be computationally expensive.
Outputs are local: data/raw/xinyi_final_local/ (ignored by Git).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import cross_val_score

from src.shared_preprocessing import (
    PRIMARY_SCORING, build_preprocessor, get_split, make_cv,
)
from src.ensemble import choose_two_models, make_ensembles, evaluate_candidates
from src.evaluation_plots import (
    plot_model_comparison, plot_confusion_matrices, plot_roc_curves,
)


def zyen_runner():
    """Use the approved Zyen notebook definitions, without running its final cell.

    Only execute these two known, team-reviewed code cells from the repo.
    """
    path = ROOT / "notebooks" / "02_base_models.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    definitions = next((s for s in cells if "models = {" in s and "LogisticRegression" in s), None)
    runner = next((s for s in cells if "def run_base_models(" in s), None)
    if not definitions or not runner:
        raise RuntimeError("Zyen's notebook model definitions or runner are missing")
    context = {"__name__": "_zyen_models", "__file__": str(path)}
    exec(compile(definitions, str(path), "exec"), context)
    exec(compile(runner, str(path), "exec"), context)
    expected = {"Logistic Regression", "Decision Tree", "Random Forest"}
    if set(context["models"]) != expected:
        raise RuntimeError("Unexpected models in Zyen's notebook")
    return context["run_base_models"]


def main(full=False):
    X_train, X_test, y_train, y_test = get_split()
    preprocessing = build_preprocessor(X_train)
    train_models = zyen_runner()
    print(f"Shared split: {len(X_train):,} train / {len(X_test):,} test", flush=True)
    print(f"Shared CV folds: {make_cv().get_n_splits()}", flush=True)
    if not full:
        transformer = preprocessing.fit(X_train.iloc[:200])
        train_shape = transformer.transform(X_train.iloc[:200]).shape
        test_shape = transformer.transform(X_test.iloc[:20]).shape
        assert train_shape[1] == test_shape[1]
        print("INTEGRATION CHECK PASSED. No models trained.", flush=True)
        return

    print("Starting Zyen's final 5-fold tuning (long-running)...", flush=True)
    cv_table, zyen_test_table, _, tuned = train_models(
        preprocessing, X_train, X_test, y_train, y_test, scoring=PRIMARY_SCORING,
    )
    if len(tuned) != 3:
        raise RuntimeError("Expected exactly three tuned Zyen models")
    cv_values = dict(zip(cv_table["model"], cv_table["tuned_cv_mean"]))
    selected = choose_two_models(cv_values)
    print("Training-CV selected models:", selected, flush=True)

    ensemble_scores = {}
    for name, estimator in make_ensembles(tuned, selected).items():
        print("Validating ensemble:", name, flush=True)
        scores = cross_val_score(
            estimator, X_train, y_train, scoring=PRIMARY_SCORING,
            cv=make_cv(), n_jobs=1, error_score="raise",
        )
        ensemble_scores[name] = float(np.mean(scores))
        print(f"{name} training CV F1: {ensemble_scores[name]:.4f}", flush=True)
    chosen = max(ensemble_scores, key=ensemble_scores.get)
    print("Selected ensemble:", chosen, flush=True)
    comparison, fitted, matrices = evaluate_candidates(
        tuned, selected, X_train, y_train, X_test, y_test,
        ensemble_method=chosen,
    )

    output = ROOT / "data/raw/xinyi_final_local"
    output.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(output / "training_cv.csv", index=False)
    zyen_test_table.to_csv(output / "zyen_base_test.csv", index=False)
    comparison.to_csv(output / "test_comparison.csv")
    (output / "experiment_notes.json").write_text(json.dumps({
        "status": "FINAL INTEGRATION CANDIDATE - review before main merge",
        "shared_preprocessing": "src/shared_preprocessing.py",
        "zyen_models": "notebooks/02_base_models.ipynb",
        "cv_folds": make_cv().get_n_splits(),
        "selected_models": selected,
        "ensemble_cv_f1": ensemble_scores,
        "chosen_ensemble": chosen,
    }, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = (
        ("model_comparison.png", plot_model_comparison(comparison)),
        ("confusion_matrices.png", plot_confusion_matrices(matrices)),
        ("roc_curves.png", plot_roc_curves(fitted, X_test, y_test)),
    )
    for filename, fig in figures:
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("\nFINAL INTEGRATION RESULTS (not yet team-approved):", flush=True)
    print(comparison.round(4).to_string(), flush=True)
    print("Saved to:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    main(full=args.full)
