"""Baseline SVM (StandardScaler + LinearSVC) training and validation utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_svm_pipeline(
    c_val: float = 0.001,
    class_weight: str | None = "balanced",
    max_iter: int = 5000,
    random_state: int = 42,
) -> Pipeline:
    """Create a leakage-free StandardScaler + LinearSVC classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=float(c_val),
                    class_weight=class_weight,
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute binary solar-cell classification metrics with class 1 = DEFECTIVE."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "defective_f1": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "healthy_f1": float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "defective_precision": float(
            precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        ),
        "defective_recall": float(
            recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_cv_on_train(
    pipeline: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """Run Stratified K-Fold CV strictly on the training split and summarize fold metrics."""
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    macro_f1_scores: list[float] = []
    defective_f1_scores: list[float] = []
    defective_recall_scores: list[float] = []
    accuracy_scores: list[float] = []

    for tr_idx, va_idx in skf.split(x_train, y_train):
        fold_pipe = clone(pipeline)
        fold_pipe.fit(x_train[tr_idx], y_train[tr_idx])
        fold_pred = fold_pipe.predict(x_train[va_idx])
        metrics = compute_classification_metrics(y_train[va_idx], fold_pred)
        macro_f1_scores.append(metrics["macro_f1"])
        defective_f1_scores.append(metrics["defective_f1"])
        defective_recall_scores.append(metrics["defective_recall"])
        accuracy_scores.append(metrics["accuracy"])

    return {
        "cv_macro_f1_mean": float(np.mean(macro_f1_scores)),
        "cv_macro_f1_std": float(np.std(macro_f1_scores)),
        "cv_defective_f1_mean": float(np.mean(defective_f1_scores)),
        "cv_defective_recall_mean": float(np.mean(defective_recall_scores)),
        "cv_accuracy_mean": float(np.mean(accuracy_scores)),
    }


def run_validation_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float],
    class_weights: list[str | None],
    max_iter: int = 5000,
    cv_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate a grid of LinearSVC C and class_weight values on train CV and val splits."""
    rows: list[dict[str, Any]] = []
    for cw in class_weights:
        for c_val in c_values:
            base_pipe = build_svm_pipeline(
                c_val=float(c_val),
                class_weight=cw,
                max_iter=max_iter,
                random_state=seed,
            )
            cv_summary = evaluate_cv_on_train(
                base_pipe,
                x_train=x_train,
                y_train=y_train,
                n_splits=cv_folds,
                seed=seed,
            )

            fitted_pipe = clone(base_pipe)
            t0 = time.perf_counter()
            fitted_pipe.fit(x_train, y_train)
            train_time_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            val_pred = fitted_pipe.predict(x_val)
            val_infer_ms = ((time.perf_counter() - t1) * 1000.0) / float(len(x_val))

            val_metrics = compute_classification_metrics(y_val, val_pred)
            rows.append(
                {
                    "class_weight": "none" if cw is None else str(cw),
                    "C": float(c_val),
                    "cv_macro_f1_mean": round(cv_summary["cv_macro_f1_mean"], 4),
                    "cv_macro_f1_std": round(cv_summary["cv_macro_f1_std"], 4),
                    "val_macro_f1": round(val_metrics["macro_f1"], 4),
                    "val_defective_f1": round(val_metrics["defective_f1"], 4),
                    "val_defective_precision": round(val_metrics["defective_precision"], 4),
                    "val_defective_recall": round(val_metrics["defective_recall"], 4),
                    "val_accuracy": round(val_metrics["accuracy"], 4),
                    "train_time_s": round(train_time_s, 4),
                    "val_infer_ms_per_image": round(val_infer_ms, 5),
                }
            )

    df_grid = pd.DataFrame(rows)
    df_grid = df_grid.sort_values(
        by=["val_macro_f1", "val_defective_recall"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return df_grid


def train_and_save_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_val: float,
    class_weight: str | None,
    max_iter: int,
    seed: int,
    model_path: str | Path = "models/baseline.joblib",
) -> tuple[Pipeline, float]:
    """Fit the frozen E1 baseline SVM pipeline on training data and persist it with joblib."""
    pipeline = build_svm_pipeline(
        c_val=c_val,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )
    t0 = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_time_s = time.perf_counter() - t0

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return pipeline, float(train_time_s)
