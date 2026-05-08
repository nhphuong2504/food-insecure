"""Shared data loading, cleaning, encoding, and train/test splitting helpers.

Used by the EDA, prediction, and segmentation notebooks so that all three
operate on identical preprocessed inputs.

Design choices (see report Section "Data and Preprocessing"):

* The outcome is ``food_insecure_flag_adult`` (binary). The 4-level
  ``adltfscat`` and the household-reported ``foodsufficient`` are excluded
  from the feature matrix because they are direct constructions of, or
  proxies for, the outcome.
* Sentinel survey codes (``-996``, ``-997``, ``-998``) are recoded to NaN.
  ``caraccess`` carries a sentinel in 92% of rows and is dropped entirely.
  ``fah_storetype_unique`` is essentially constant and is also dropped.
* String-coded categorical heads (``head_hispanic``, ``head_racecat``) carry
  the response code ``R`` for "refused"; we recode ``R`` to NaN and one-hot
  encode the remainder.
* Distance variables are heavy-right-skewed; we add a ``log1p`` transform
  alongside the raw distance so that linear models see a near-symmetric
  signal without losing the original scale for tree models.
* Missing numeric values are median-imputed; missing categorical values get
  their own ``"unknown"`` level so the imputation decision does not bleed
  signal into the tree-based model.

The cleaned matrix returned by :func:`build_feature_matrix` is purely
numeric (post one-hot encoding) and ready to feed into a scikit-learn
``Pipeline`` whose first step performs ``StandardScaler`` for the linear
and kernel-based models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

RANDOM_STATE: int = 42

OUTCOME: str = "food_insecure_flag_adult"

# Columns dropped before modeling.  Each entry is annotated with the reason.
LEAK_COLUMNS: tuple[str, ...] = (
    "adltfscat",      # 4-level outcome from which the binary flag is derived
    "foodsufficient", # household-reported food sufficiency: a near-duplicate label
)
ID_COLUMNS: tuple[str, ...] = ("hhnum",)
LOW_QUALITY_COLUMNS: tuple[str, ...] = (
    "caraccess",            # ~92% sentinel-coded; effectively missing
    "fah_storetype_unique", # near-zero variance (mean ~0.002)
)
SURVEY_WEIGHT_COLUMNS: tuple[str, ...] = ("hhwgt",)

# Variables that may carry survey sentinel codes (-996, -997, -998).  We
# recode these to NaN before any analysis.
SENTINEL_COLUMNS: tuple[str, ...] = (
    "foodpantry",
    "anyvehicle",
    "vehiclenum",
)
SENTINEL_VALUES: tuple[int, ...] = (-996, -997, -998)

# Distance columns that are heavy-right-skewed.  We keep both the raw value
# and a ``log1p`` transform.
DISTANCE_COLUMNS: tuple[str, ...] = (
    "dist_sm",
    "dist_cs",
    "dist_walmart",
    "nearsnap_dist",
    "nearff_dist",
    "nearnonff_dist",
)

# Categorical columns to one-hot encode.  ``poverty_band`` is stored as a
# free-text label in the raw data; we treat it as nominal here.  ``targetgroup``
# is a sampling indicator with 4 nominal levels.
NOMINAL_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "head_sex",
    "head_hispanic",
    "head_racecat",
    "head_employment",
    "region",
    "poverty_band",
    "targetgroup",
)

# Variables that look ordinal and are used as numeric features after
# imputation.  ``head_educcat`` runs from 1 (lowest) to 6 (highest).
ORDINAL_NUMERIC_COLUMNS: tuple[str, ...] = (
    "head_educcat",
    "head_age",
)

# Binary / count columns that pass through unchanged after sentinel cleanup.
NUMERIC_PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "snap_any",
    "foodpantry",
    "household_size",
    "num_adults",
    "num_children",
    "num_elderly",
    "employed_adults",
    "any_employed_adult",
    "rural",
    "nonmetro",
    "anyvehicle",
    "vehiclenum",
    "fah_event_count",
    "fafh_event_count",
    "fafh_schoolmeal_events",
)

# -----------------------------------------------------------------------------
# Loading and cleaning
# -----------------------------------------------------------------------------


def load_raw(data_path: str | Path = "data/Case_dataset.csv") -> pd.DataFrame:
    """Load the raw provided CSV without modification."""
    df = pd.read_csv(data_path)
    return df


def _recode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SENTINEL_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = out[col].where(~out[col].isin(SENTINEL_VALUES), other=np.nan)
    return out


def _recode_string_refusals(df: pd.DataFrame) -> pd.DataFrame:
    """Recode the literal string ``"R"`` (refused) in head categoricals to NaN."""
    out = df.copy()
    for col in ("head_hispanic", "head_racecat"):
        if col in out.columns:
            out[col] = out[col].replace({"R": np.nan})
    return out


def _add_distance_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DISTANCE_COLUMNS:
        if col in out.columns:
            out[f"{col}_log1p"] = np.log1p(out[col].clip(lower=0))
    return out


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of derived household-composition features."""
    out = df.copy()
    eps = 1e-3
    out["children_per_adult"] = out["num_children"] / (out["num_adults"] + eps)
    out["elderly_share"] = out["num_elderly"] / (out["household_size"] + eps)
    out["children_share"] = out["num_children"] / (out["household_size"] + eps)
    out["employed_adult_share"] = out["employed_adults"] / (out["num_adults"] + eps)
    return out


DERIVED_FEATURE_COLUMNS: tuple[str, ...] = (
    "children_per_adult",
    "elderly_share",
    "children_share",
    "employed_adult_share",
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all sentinel and refusal recodings, derived features, and log
    transforms.  Returns a DataFrame still containing the outcome column.
    """
    out = df.copy()
    out = _recode_sentinels(out)
    out = _recode_string_refusals(out)
    # Cast bool snap_any to int so downstream encoding sees a single dtype.
    if "snap_any" in out.columns:
        out["snap_any"] = out["snap_any"].astype(int)
    out = _add_distance_logs(out)
    out = _add_derived_features(out)
    return out


# -----------------------------------------------------------------------------
# Feature matrix construction
# -----------------------------------------------------------------------------


@dataclass
class FeatureMatrix:
    """Return value of :func:`build_feature_matrix`."""

    X: pd.DataFrame              # numeric, post one-hot encoding, with NaN imputed
    y: pd.Series                 # binary outcome
    numeric_columns: list[str]   # columns to standardize (continuous + log + derived + ordinal)
    binary_columns: list[str]    # columns that should not be standardized
    feature_groups: dict[str, list[str]]  # human-readable groupings for plots


def _build_numeric_imputed(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    out = df[cols].copy()
    for c in cols:
        median = out[c].median()
        out[c] = out[c].fillna(median)
    return out


def _build_one_hot(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in columns if c in df.columns]
    work = df[cols].copy()
    # Promote all to string so NaN becomes its own category named "unknown".
    for c in cols:
        work[c] = work[c].astype("object").where(~work[c].isna(), other="unknown")
        work[c] = work[c].astype(str)
    dummies = pd.get_dummies(work, prefix=cols, prefix_sep="=", dtype=int)
    return dummies, list(dummies.columns)


def build_feature_matrix(df_clean: pd.DataFrame) -> FeatureMatrix:
    """Construct ``X`` and ``y`` from a cleaned dataframe.

    The returned ``X`` is numeric and contains no NaNs.  Continuous numeric
    columns and ordinal columns are recorded in ``numeric_columns`` so that
    downstream pipelines can apply a ``StandardScaler`` only where it
    helps (linear, kernel, PCA, K-Means models).
    """
    if OUTCOME not in df_clean.columns:
        raise KeyError(f"Outcome column '{OUTCOME}' missing from cleaned dataframe")

    y = df_clean[OUTCOME].astype(int).rename(OUTCOME)

    drop = set(LEAK_COLUMNS) | set(ID_COLUMNS) | set(LOW_QUALITY_COLUMNS) \
        | set(SURVEY_WEIGHT_COLUMNS) | {OUTCOME}

    df = df_clean.drop(columns=[c for c in drop if c in df_clean.columns])

    # Numeric block -----------------------------------------------------------
    numeric_continuous = [
        c for c in (
            list(DISTANCE_COLUMNS)
            + [f"{c}_log1p" for c in DISTANCE_COLUMNS]
            + list(DERIVED_FEATURE_COLUMNS)
            + list(ORDINAL_NUMERIC_COLUMNS)
        )
        if c in df.columns
    ]
    numeric_block = _build_numeric_imputed(df, numeric_continuous)

    binary_passthrough = [c for c in NUMERIC_PASSTHROUGH_COLUMNS if c in df.columns]
    binary_block = _build_numeric_imputed(df, binary_passthrough)

    onehot_block, onehot_cols = _build_one_hot(df, NOMINAL_CATEGORICAL_COLUMNS)

    X = pd.concat([numeric_block, binary_block, onehot_block], axis=1)

    feature_groups = {
        "numeric_continuous": numeric_continuous,
        "binary_or_count": binary_passthrough,
        "one_hot": onehot_cols,
    }
    numeric_columns = numeric_continuous  # ones to standardize
    binary_columns = binary_passthrough + onehot_cols

    return FeatureMatrix(
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        binary_columns=binary_columns,
        feature_groups=feature_groups,
    )


def make_train_test_split(
    fm: FeatureMatrix,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Stratified 80/20 split returning a dict of named splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        fm.X, fm.y,
        test_size=test_size,
        stratify=fm.y,
        random_state=random_state,
    )
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def save_artifacts(splits: dict, out_dir: str | Path = "artifacts") -> None:
    """Persist cleaned splits to ``artifacts/`` for the modeling notebooks.

    CSV is used to avoid an extra parquet dependency; the cleaned splits
    are small (~4k rows, ~60 columns).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in splits.items():
        path = out / f"{name}.csv"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False)
        else:
            obj.to_frame().to_csv(path, index=False)


def load_artifacts(in_dir: str | Path = "artifacts") -> dict[str, pd.DataFrame | pd.Series]:
    """Load cleaned splits previously saved by :func:`save_artifacts`."""
    in_dir = Path(in_dir)
    out: dict[str, pd.DataFrame | pd.Series] = {}
    for name in ("X_train", "X_test", "y_train", "y_test"):
        df = pd.read_csv(in_dir / f"{name}.csv")
        if name.startswith("y_"):
            out[name] = df.iloc[:, 0]
        else:
            out[name] = df
    return out
