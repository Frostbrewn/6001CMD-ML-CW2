"""Independent integration prototype for Xinyi's 6001CMD ensemble work.

This does NOT replace Ben's approved split or Zyen's official tuning.
It is an end-to-end, reproducible fallback to unblock development.
Run: python src/independent_workflow_xinyi.py --data data/raw/bank-full.csv
Outputs are written locally and should not be represented as final team results
until the group agrees on preprocessing, splits, and model selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.ensemble import choose_two_models, evaluate_candidates, make_ensembles
    from src.evaluation_plots import plot_confusion_matrices, plot_model_comparison, plot_roc_curves
except ModuleNotFoundError:
    # Supports running this file directly from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.ensemble import choose_two_models, evaluate_candidates, make_ensembles
    from src.evaluation_plots import plot_confusion_matrices, plot_model_comparison, plot_roc_curves


SEED = 42


def load_split(csv_path: Path, sample_size: int | None = None):
    """Use one stratified split; intentionally exclude post-call duration for deployment."""
    df = pd.read_csv(csv_path, sep=";")
    if "y" not in df.columns or "duration" not in df.columns:
        raise ValueError("Expected the original UCI bank-full.csv with 'y' and 'duration'.")
    if sample_size is not None and sample_size < len(df):
        # Pilot only; stratified sample preserves the rare positive class.
        df, _ = train_test_split(df, train_size=sample_size, stratify=df["y"], random_state=SEED)
    y = df["y"].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected target values in 'y'.")
    X = df.drop(columns=["y", "duration"])
    # Categorical 'unknown' and numerical pdays=-1 are meaningful sentinels;
    # leave them in the data unless the team chooses an agreed treatment.
    return train_test_split(X, y.astype(int), test_size=0.2, stratify=y, random_state=SEED)


def make_preprocessor(X_train):
    number_cols = list(X_train.select_dtypes(include="number").columns)
    category_cols = list(X_train.columns.difference(number_cols))
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, number_cols),
        ("categorical", categorical, category_cols),
    ])


def train_three_models(X_train, y_train, preprocessor, cv):
    """Small, documented grid searches based on Zyen's three algorithm families."""
    specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"model__max_depth": [5, 10], "model__min_samples_leaf": [10],
             "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=60, n_jobs=1, random_state=SEED),
            {"model__max_depth": [10, None], "model__min_samples_leaf": [5],
             "model__class_weight": [None, "balanced"]},
        ),
    }
    tuned, report = {}, []
    for name, (estimator, grid) in specs.items():
        search = GridSearchCV(
            Pipeline([("preprocess", preprocessor), ("model", estimator)]),
            param_grid=grid, cv=cv, scoring="f1", n_jobs=1, refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        report.append({"Model": name, "Best Train-CV F1": search.best_score_,
                       "Best Parameters": search.best_params_})
        print(f"{name}: CV F1 = {search.best_score_:.4f} | {search.best_params_}", flush=True)
    return tuned, pd.DataFrame(report).set_index("Model")


def run(csv_path: Path, output_dir: Path, sample_size: int | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = load_split(csv_path, sample_size)
    print(f"Train={len(X_train):,}, test={len(X_test):,}, positive class fraction={y_train.mean():.3f}", flush=True)
    preprocessor = make_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    tuned, cv_report = train_three_models(X_train, y_train, preprocessor, cv)
    cv_scores = cv_report["Best Train-CV F1"].to_dict()
    selected = choose_two_models(cv_scores)
    print("Candidate pair:", selected, flush=True)
    alternative_scores = {}
    for name, model in make_ensembles(tuned, selected).items():
        folds = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        alternative_scores[name] = float(np.mean(folds))
        print(f"{name}: CV F1 = {alternative_scores[name]:.4f}", flush=True)
    method = max(alternative_scores, key=alternative_scores.get)
    print("Validation-selected ensemble:", method, flush=True)

    # Model/ensemble selection is frozen here. Test data are used only below.
    test_report, final_models, confusion_matrices = evaluate_candidates(
        tuned, selected, X_train, y_train, X_test, y_test, ensemble_method=method
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_report.to_csv(output_dir / "training_cv.csv")
    test_report.to_csv(output_dir / "test_comparison.csv")
    metadata = {
        "status": "INDEPENDENT PROTOTYPE — not approved team experiment",
        "seed": SEED, "sample_size": sample_size or "full dataset",
        "test_fraction": 0.2, "dropped_features": ["duration"],
        "selected_models": list(selected),
        "excluded_model": next(name for name in tuned if name not in selected),
        "ensemble_cv_f1": alternative_scores,
        "selected_ensemble": method,
    }
    (output_dir / "experiment_notes.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, fig in (
        ("model_comparison.png", plot_model_comparison(test_report)),
        ("confusion_matrices.png", plot_confusion_matrices(confusion_matrices)),
        ("roc_curves.png", plot_roc_curves(final_models, X_test, y_test)),
    ):
        fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("\nFinal held-out test results (independent prototype only):", flush=True)
    print(test_report.round(4).to_string(), flush=True)
    print(f"Saved outputs under: {output_dir.resolve()}", flush=True)
    return cv_report, test_report, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/bank-full.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/xinyi_local_prototype"))
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Optional stratified pilot sample; do not report as full-dataset findings")
    args = parser.parse_args()
    run(args.data, args.output, args.sample_size)
