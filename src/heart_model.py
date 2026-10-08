"""
Heart Attack Risk Prediction - data loading, preprocessing, Logistic Regression training & explanation.

Pipeline:  raw CSV -> validation / missing values -> One-Hot encoding (cp, restecg, ca)
           -> stratified 80/20 split -> StandardScaler -> LogisticRegression
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "heart_data.csv"
DATA_FILE_STEM = "Project Heart attack data"

TARGET = "target"
NUMERIC = ["age", "trtbps", "chol", "thalachh"]
BINARY = ["sex", "fbs", "exng"]
# Multi-class variables: categories are fixed so a single new patient gets exactly the same columns.
CATEGORICAL = {"cp": [0, 1, 2, 3], "restecg": [0, 1, 2], "ca": [0, 1, 2, 3]}
RAW_FEATURES = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "ca"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
SIMILAR_K = 25        # "similar patients" used for the empirical reasonability check

# A realistic new patient (assignment: "invent a new realistic observation")
SAMPLE_PATIENT = {"age": 58, "sex": 1, "cp": 2, "trtbps": 140, "chol": 260, "fbs": 0,
                  "restecg": 1, "thalachh": 155, "exng": 0, "ca": 0}

# Clinical units used to express odds ratios in meaningful steps
CLINICAL_STEP = {"age": 10, "trtbps": 10, "chol": 50, "thalachh": 10}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    data: pd.DataFrame                    # clean raw data
    X_test: pd.DataFrame
    y_test: pd.Series
    proba_test: np.ndarray
    metrics: dict
    coefficients: pd.DataFrame
    train_raw_means: pd.Series
    group_of: dict = field(default_factory=dict)   # encoded column -> original feature
    neighbors: NearestNeighbors | None = None      # on scaled features of the whole dataset


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def find_data_file(folder: str | Path | None = None) -> Path:
    """Find 'Project Heart attack data.csv' in the given folder (or its parent), else use data/heart_data.csv."""
    if folder:
        folder = Path(folder)
        for base in (folder, folder.parent):
            for ext in (".csv", ".xlsx"):
                p = base / f"{DATA_FILE_STEM}{ext}"
                if p.exists():
                    return p
        if folder.is_file():
            return folder
        for ext in (".csv", ".xlsx"):                 # path given without its extension / one folder down
            for p in (folder.with_name(folder.name + ext), *folder.parent.glob(f"*/{DATA_FILE_STEM}{ext}")):
                if p.exists():
                    return p
        raise FileNotFoundError(f"'{DATA_FILE_STEM}' not found in {folder}")
    return DEFAULT_DATA


def load_data(path: str | Path = DEFAULT_DATA) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"exang": "exng", "thalach": "thalachh", "rest_ecg": "restecg"})

    info = {"n_rows_raw": len(df), "missing_values_raw": int(df.isna().sum().sum()),
            "duplicates_raw": int(df.duplicated().sum())}
    df = df[RAW_FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")

    # Missing values: median for numeric, mode for categorical/binary (the provided file has none)
    for c in NUMERIC:
        df[c] = df[c].fillna(df[c].median())
    for c in BINARY + list(CATEGORICAL):
        df[c] = df[c].fillna(df[c].mode()[0])
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)

    # Keep only valid codes
    valid = df[TARGET].isin([0, 1]) & df[BINARY].isin([0, 1]).all(axis=1)
    for c, cats in CATEGORICAL.items():
        valid &= df[c].isin(cats)
    df = df[valid].astype(int).reset_index(drop=True)
    info["n_rows"] = len(df)
    return df, info


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot encode the multi-class variables (drop_first -> baseline = category 0); binaries stay as-is."""
    X = df[RAW_FEATURES].copy()
    for c, cats in CATEGORICAL.items():
        X[c] = pd.Categorical(X[c], categories=cats)
    return pd.get_dummies(X, columns=list(CATEGORICAL), drop_first=True, dtype=int)


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, proba=None) -> dict:
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1": f1_score(y_true, y_pred, zero_division=0)}
    if proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
               specificity=tn / (tn + fp) if tn + fp else 0.0)
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}


def train_model(path: str | Path = DEFAULT_DATA) -> ModelBundle:
    df, info = load_data(path)
    X = encode(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    proba_train = pipe.predict_proba(X_train)[:, 1]
    test = classification_metrics(y_test, (proba_test >= 0.5).astype(int), proba_test)
    train = classification_metrics(y_train, (proba_train >= 0.5).astype(int), proba_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_recall = cross_val_score(pipe, X, y, cv=cv, scoring="recall")
    cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    # Threshold trade-off (medical screening often lowers the threshold to catch more patients)
    thresholds = []
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        m = classification_metrics(y_test, (proba_test >= th).astype(int))
        thresholds.append({"threshold": th, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "fn", "fp")}})

    majority = int(y_train.mode()[0])
    baseline = classification_metrics(y_test, np.full(len(y_test), majority))

    metrics = {**info, "n_train": len(X_train), "n_test": len(X_test),
               "positive_rate": float(y.mean()), "n_features": X.shape[1],
               "train": train, "test": test, "baseline_majority": baseline,
               "cv_recall_mean": float(cv_recall.mean()), "cv_recall_std": float(cv_recall.std()),
               "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
               "thresholds": thresholds}

    scaler: StandardScaler = pipe.named_steps["scaler"]
    lr: LogisticRegression = pipe.named_steps["model"]
    names = list(X.columns)
    group_of = {n: n.split("_")[0] if n.split("_")[0] in CATEGORICAL else n for n in names}

    coef_std = lr.coef_[0]
    coef_real = coef_std / scaler.scale_          # change in log-odds per 1 original unit
    step = np.array([CLINICAL_STEP.get(n, 1) for n in names])
    coefs = pd.DataFrame({
        "feature": names, "group": [group_of[n] for n in names],
        "coef_std": coef_std, "odds_ratio_std": np.exp(coef_std),
        "coef_real": coef_real, "step": step, "odds_ratio_step": np.exp(coef_real * step),
    }).assign(abs_std=lambda d: d["coef_std"].abs()).sort_values("abs_std", ascending=False)
    coefs = coefs.drop(columns="abs_std").reset_index(drop=True)
    metrics["intercept"] = float(lr.intercept_[0])

    neighbors = NearestNeighbors(n_neighbors=SIMILAR_K).fit(scaler.transform(X))

    return ModelBundle(pipeline=pipe, feature_names=names, data=df, X_test=X_test, y_test=y_test,
                       proba_test=proba_test, metrics=metrics, coefficients=coefs,
                       train_raw_means=df.loc[X_train.index, RAW_FEATURES].mean(), group_of=group_of,
                       neighbors=neighbors)


# --------------------------------------------------------------------------- #
# Prediction with explanation
# --------------------------------------------------------------------------- #
def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def predict_with_explanation(bundle: ModelBundle, patient: dict, threshold: float = 0.5) -> dict:
    """Predict risk and split the log-odds into per-feature contributions vs. the average training patient.

    After StandardScaler the average patient sits at z = 0, so  logit = intercept + Σ coef_j · z_j
    and coef_j · z_j is exactly how much feature j moves this patient away from the average.
    """
    X = encode(pd.DataFrame([patient]))[bundle.feature_names]
    scaler = bundle.pipeline.named_steps["scaler"]
    lr = bundle.pipeline.named_steps["model"]
    z = scaler.transform(X)[0]
    parts = pd.Series(lr.coef_[0] * z, index=bundle.feature_names)
    contrib = parts.groupby(bundle.group_of).sum()
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

    proba = float(bundle.pipeline.predict_proba(X)[0, 1])
    intercept = float(lr.intercept_[0])
    _, idx = bundle.neighbors.kneighbors(z.reshape(1, -1))
    similar = bundle.data.iloc[idx[0]]
    return {"proba": proba, "prediction": int(proba >= threshold), "threshold": threshold,
            "logit": intercept + float(parts.sum()), "baseline_logit": intercept,
            "baseline_proba": _sigmoid(intercept), "contributions": contrib,
            "population_rate": bundle.metrics["positive_rate"],
            "similar_rate": float(similar[TARGET].mean()), "similar_count": len(similar)}
