
"""Shared preprocessing utilities for 6001CMD Bank Marketing CW2."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Shared experiment configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
PRIMARY_SCORING = "f1"


def load_raw_data(data_path=None):
    """Load the original semicolon-separated Bank Marketing dataset."""

    if data_path is None:
        data_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "raw"
            / "bank-full.csv"
        )

    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    return pd.read_csv(data_path, sep=";")


def prepare_features_target(raw_df):
    """Prepare predictors for pre-call subscription prediction."""

    required = {"y", "duration", "campaign", "pdays", "previous"}
    missing = required.difference(raw_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if raw_df["y"].isna().any():
        raise ValueError("Target contains missing values.")

    if set(raw_df["y"].unique()) != {"no", "yes"}:
        raise ValueError("Unexpected target labels.")

    # Encode the target: no = 0, yes = 1
    y = raw_df["y"].map({"no": 0, "yes": 1}).astype("int8")

    # Exclude the post-contact duration and original target
    X = raw_df.drop(columns=["y", "duration"]).copy()

    # Original campaign count includes the current contact
    campaign = pd.to_numeric(X.pop("campaign"), errors="raise")

    if campaign.isna().any() or (campaign < 1).any():
        raise ValueError("Invalid campaign values.")

    X["campaign_prior_contacts"] = campaign - 1

    # -1 means no previous campaign contact.
    # Retain previous as the contact-count indicator.
    pdays = pd.to_numeric(X.pop("pdays"), errors="raise")

    if pdays.isna().any() or (pdays < -1).any():
        raise ValueError("Invalid pdays values.")

    X["days_since_previous_contact"] = (
        pdays.replace(-1, np.nan).astype(float)
    )

    if not X.columns.is_unique:
        raise ValueError("Duplicate predictor column names found.")

    return X, y


def get_split(data_path=None):
    """Return the common stratified 80:20 training/test split."""

    raw_df = load_raw_data(data_path)
    X, y = prepare_features_target(raw_df)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor(X_train):
    """Build an unfitted preprocessing transformer."""

    numeric_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def make_cv():
    """Return the shared five-fold stratified CV strategy."""

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
