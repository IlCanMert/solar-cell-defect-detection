"""Train and validate E1 Baseline SVM (StandardScaler + LinearSVC) on all 334 HOG+LBP features."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import (
    compute_classification_metrics,
    evaluate_cv_on_train,
    run_validation_grid,
    train_and_save_baseline,
)
from src.dataset import load_config


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    svm_cfg = cfg["baseline_svm"]
    cv_folds = int(cfg["ga"]["cv_folds"])
    seed = int(svm_cfg["random_state"])
    max_iter = int(svm_cfg["max_iter"])

    features_path = PROJECT_ROOT / "data" / "features_hog_lbp.npz"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {features_path}. Run `python -m src.features` first."
        )

    with np.load(features_path, allow_pickle=False) as data:
        x_all = data["X"]
        y_all = data["y"]
        splits = data["splits"]

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    # Strict leakage guard: test split is never used during tuning or baseline fitting
    assert int(train_mask.sum()) + int(val_mask.sum()) + int(test_mask.sum()) == len(y_all)
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]
    del test_mask

    print(f"Loaded features: train={x_train.shape}, val={x_val.shape} (test split locked)")

    c_grid = [float(c) for c in svm_cfg.get("c_grid", [0.001, 0.01, 0.1, 1.0, 10.0])]
    cw_grid = list(svm_cfg.get("class_weight_grid", [None, "balanced"]))

    print("\n=== Running Leakage-Free Validation Grid (Train CV + Val Split) ===")
    df_grid = run_validation_grid(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        c_values=c_grid,
        class_weights=cw_grid,
        max_iter=max_iter,
        cv_folds=cv_folds,
        seed=seed,
    )

    grid_path = PROJECT_ROOT / "results" / "baseline_validation_grid.csv"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(grid_path, index=False)
    print(df_grid.to_string(index=False))
    print(f"\nSaved validation grid to: {grid_path}")

    # Verify frozen config selection against best validation configuration
    best_row = df_grid.iloc[0]
    frozen_c = float(svm_cfg["C"])
    frozen_cw = svm_cfg["class_weight"]

    print(
        f"\nBest validation row : class_weight={best_row['class_weight']}, C={best_row['C']} "
        f"(Val Macro F1={best_row['val_macro_f1']:.4f}, Def Recall={best_row['val_defective_recall']:.4f})"
    )
    print(f"Frozen config setting: class_weight={frozen_cw}, C={frozen_c}")

    model_path = PROJECT_ROOT / "models" / "baseline.joblib"
    pipeline, train_time_s = train_and_save_baseline(
        x_train=x_train,
        y_train=y_train,
        c_val=frozen_c,
        class_weight=frozen_cw,
        max_iter=max_iter,
        seed=seed,
        model_path=model_path,
    )

    cv_metrics = evaluate_cv_on_train(
        pipeline,
        x_train=x_train,
        y_train=y_train,
        n_splits=cv_folds,
        seed=seed,
    )
    train_pred = pipeline.predict(x_train)
    train_resub_metrics = compute_classification_metrics(y_train, train_pred)

    t0 = time.perf_counter()
    val_pred = pipeline.predict(x_val)
    val_infer_ms_per_image = ((time.perf_counter() - t0) * 1000.0) / float(len(x_val))
    val_metrics = compute_classification_metrics(y_val, val_pred)

    summary = {
        "model_name": "E1_Baseline_LinearSVC_All334",
        "n_features": int(x_train.shape[1]),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "hyperparameters": {
            "C": frozen_c,
            "class_weight": frozen_cw,
            "max_iter": max_iter,
            "random_state": seed,
        },
        "train_time_s": round(train_time_s, 4),
        "val_infer_ms_per_image": round(val_infer_ms_per_image, 5),
        "train_cv_3fold": {k: round(v, 4) for k, v in cv_metrics.items()},
        "train_resubstitution": train_resub_metrics,
        "validation": val_metrics,
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "grid_csv": str(grid_path.relative_to(PROJECT_ROOT)),
    }

    summary_path = PROJECT_ROOT / "results" / "baseline_validation.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Frozen E1 Baseline Summary ===")
    print(f"Model saved          : {model_path}")
    print(f"Train time           : {train_time_s:.4f} s")
    print(f"Train 3-fold CV F1   : {cv_metrics['cv_macro_f1_mean']:.4f} +/- {cv_metrics['cv_macro_f1_std']:.4f}")
    print(f"Val Macro F1         : {val_metrics['macro_f1']:.4f}")
    print(f"Val Defective F1     : {val_metrics['defective_f1']:.4f}")
    print(f"Val Defective Prec.  : {val_metrics['defective_precision']:.4f}")
    print(f"Val Defective Recall : {val_metrics['defective_recall']:.4f}")
    print(f"Val Accuracy         : {val_metrics['accuracy']:.4f}")
    print(f"Val Confusion Matrix : {val_metrics['confusion_matrix']}  ([[TN, FP], [FN, TP]])")
    print(f"Summary saved        : {summary_path}")


if __name__ == "__main__":
    main()
