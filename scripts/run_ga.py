"""Run Genetic Algorithm (E2) feature selection on training data and save GA-SVM model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_svm_pipeline, compute_classification_metrics
from src.dataset import load_config
from src.ga_selection import run_genetic_algorithm


def _save_convergence_plot(
    history: list[dict[str, Any]],
    total_features: int,
    best_eval: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a 1x2 plot of GA fitness convergence and selected feature count across generations."""
    gens = [h["generation"] for h in history]
    best_fit = [h["best_fitness"] for h in history]
    mean_fit = [h["mean_fitness"] for h in history]
    best_f1 = [h["best_cv_macro_f1"] for h in history]
    best_feats = [h["best_selected_count"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(
        "Genetic Algorithm Feature Selection Convergence (Training 3-Fold CV)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    ax1 = axes[0]
    ax1.plot(gens, best_f1, marker="^", linewidth=2.0, color="#1f77b4", label="Best 3-Fold CV Macro F1")
    ax1.plot(gens, best_fit, marker="o", linewidth=2.0, color="#2ca02c", label="Best Fitness (F1 - penalty)")
    ax1.plot(gens, mean_fit, marker="s", linestyle="--", linewidth=1.6, color="#ff7f0e", label="Mean Pop. Fitness")
    ax1.set_title("Fitness & CV Macro F1 Across Generations", fontsize=11)
    ax1.set_xlabel("Generation", fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_xticks(gens)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=9)

    ax2 = axes[1]
    ax2.plot(gens, best_feats, marker="D", linewidth=2.2, color="#8e44ad", label="Best Mask Selected Features")
    ax2.axhline(
        y=total_features,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Full Feature Set ({total_features})",
    )
    ax2.set_title("Selected Feature Count Across Generations", fontsize=11)
    ax2.set_xlabel("Generation", fontsize=10)
    ax2.set_ylabel("Number of Selected Features", fontsize=10)
    ax2.set_xticks(gens)
    ax2.set_ylim(0, total_features + 30)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="right", fontsize=9)

    final_count = int(best_eval["selected_count"])
    reduction_pct = 100.0 * (1.0 - float(final_count) / float(total_features))
    ax2.annotate(
        f"Final: {final_count}/{total_features} (-{reduction_pct:.1f}%)\n"
        f"HOG: {best_eval['hog_selected']}/324 | LBP: {best_eval['lbp_selected']}/10",
        xy=(gens[-1], best_feats[-1]),
        xytext=(max(0.5, gens[-1] - 4.2), min(total_features - 25, best_feats[-1] + 55)),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9f3fc", edgecolor="#8e44ad", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2),
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    ga_cfg = cfg["ga"]
    svm_cfg = cfg["baseline_svm"]
    seed = int(ga_cfg["random_state"])

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
    assert int(train_mask.sum()) + int(val_mask.sum()) + int(test_mask.sum()) == len(y_all)

    # Strict leakage guard: test split is never touched during GA search or model fitting
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]
    del test_mask

    n_total_features = int(x_train.shape[1])
    print("=== Running E2 Genetic Algorithm Feature Selection (Training Data Only) ===")
    print(
        f"Train={x_train.shape}, Val={x_val.shape} | "
        f"pop={ga_cfg['population_size']}, gens={ga_cfg['generations']}, "
        f"lambda={ga_cfg['lambda_penalty']}, seed={seed}"
    )

    ga_result = run_genetic_algorithm(
        x_train=x_train,
        y_train=y_train,
        cfg=cfg,
        seed=seed,
        verbose=True,
    )

    best_mask_uint8 = ga_result["best_mask"]
    best_mask_bool = best_mask_uint8.astype(bool)
    best_eval = ga_result["best_eval"]

    mask_path = PROJECT_ROOT / "models" / "selected_mask.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, best_mask_bool)

    # Fit final E2 GA-SVM pipeline using the exact frozen baseline SVM settings
    ga_svm_pipe = build_svm_pipeline(
        c_val=float(svm_cfg["C"]),
        class_weight=svm_cfg["class_weight"],
        max_iter=int(svm_cfg["max_iter"]),
        random_state=int(svm_cfg["random_state"]),
    )
    t_fit0 = time.perf_counter()
    ga_svm_pipe.fit(x_train[:, best_mask_bool], y_train)
    final_fit_time_s = time.perf_counter() - t_fit0

    model_path = PROJECT_ROOT / "models" / "ga_svm.joblib"
    joblib.dump(ga_svm_pipe, model_path)

    # Evaluate on validation split for E1 vs E2 sanity check (test remains locked until Week 3)
    t_inf0 = time.perf_counter()
    val_pred = ga_svm_pipe.predict(x_val[:, best_mask_bool])
    val_infer_ms_per_image = ((time.perf_counter() - t_inf0) * 1000.0) / float(len(x_val))
    val_metrics = compute_classification_metrics(y_val, val_pred)

    baseline_val_path = PROJECT_ROOT / "results" / "baseline_validation.json"
    baseline_val_comparison: dict[str, Any] = {}
    if baseline_val_path.exists():
        with baseline_val_path.open("r", encoding="utf-8") as handle:
            base_json = json.load(handle)
        baseline_val_comparison = {
            "baseline_n_features": base_json["n_features"],
            "baseline_cv_macro_f1": base_json["train_cv_3fold"]["cv_macro_f1_mean"],
            "baseline_val_macro_f1": round(base_json["validation"]["macro_f1"], 4),
            "baseline_val_defective_recall": round(base_json["validation"]["defective_recall"], 4),
            "baseline_val_accuracy": round(base_json["validation"]["accuracy"], 4),
        }

    fig_path = PROJECT_ROOT / "results" / "figures" / "ga_convergence.png"
    _save_convergence_plot(
        history=ga_result["history"],
        total_features=n_total_features,
        best_eval=best_eval,
        output_path=fig_path,
    )

    hog_dim = int(cfg["features"]["hog"]["expected_length"])
    selected_indices = np.flatnonzero(best_mask_bool).tolist()
    hog_selected_indices = [idx for idx in selected_indices if idx < hog_dim]
    lbp_selected_indices = [idx - hog_dim for idx in selected_indices if idx >= hog_dim]
    reduction_pct = 100.0 * (1.0 - float(len(selected_indices)) / float(n_total_features))

    history_payload = {
        "model_name": "E2_GA_LinearSVC",
        "seed": seed,
        "ga_params": {
            "population_size": int(ga_cfg["population_size"]),
            "generations": int(ga_cfg["generations"]),
            "crossover_prob": float(ga_cfg["crossover_prob"]),
            "mutation_prob": float(ga_cfg["mutation_prob"]),
            "elitism_count": int(ga_cfg["elitism_count"]),
            "tournament_size": int(ga_cfg["tournament_size"]),
            "cv_folds": int(ga_cfg["cv_folds"]),
            "lambda_penalty": float(ga_cfg["lambda_penalty"]),
        },
        "svm_params": {
            "C": float(svm_cfg["C"]),
            "class_weight": svm_cfg["class_weight"],
            "max_iter": int(svm_cfg["max_iter"]),
            "random_state": int(svm_cfg["random_state"]),
        },
        "search_time_s": round(float(ga_result["search_time_s"]), 4),
        "final_fit_time_s": round(float(final_fit_time_s), 4),
        "val_infer_ms_per_image": round(float(val_infer_ms_per_image), 5),
        "n_unique_evaluations": int(ga_result["n_evaluations"]),
        "n_cache_hits": int(ga_result["n_cache_hits"]),
        "total_features": n_total_features,
        "selected_feature_count": int(len(selected_indices)),
        "feature_reduction_pct": round(float(reduction_pct), 2),
        "hog_selected_count": int(len(hog_selected_indices)),
        "lbp_selected_count": int(len(lbp_selected_indices)),
        "best_eval": best_eval,
        "selected_feature_indices": selected_indices,
        "hog_selected_indices": hog_selected_indices,
        "lbp_selected_indices": lbp_selected_indices,
        "validation": val_metrics,
        "baseline_validation_comparison": baseline_val_comparison,
        "history": ga_result["history"],
        "artifacts": {
            "selected_mask": str(mask_path.relative_to(PROJECT_ROOT)),
            "ga_svm_model": str(model_path.relative_to(PROJECT_ROOT)),
            "convergence_figure": str(fig_path.relative_to(PROJECT_ROOT)),
        },
    }

    history_path = PROJECT_ROOT / "results" / "ga_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history_payload, handle, indent=2)

    print("\n=== E2 GA-SVM Summary ===")
    print(
        f"Selected features   : {len(selected_indices)}/{n_total_features} "
        f"(-{reduction_pct:.2f}% | HOG: {len(hog_selected_indices)}/324, LBP: {len(lbp_selected_indices)}/10)"
    )
    print(f"Best Train CV F1    : {best_eval['cv_macro_f1_mean']:.4f} +/- {best_eval['cv_macro_f1_std']:.4f}")
    print(f"Best GA Fitness     : {best_eval['fitness']:.4f} (penalty={best_eval['penalty']:.4f})")
    print(
        f"GA Search Time      : {ga_result['search_time_s']:.4f} s "
        f"(unique evals={ga_result['n_evaluations']}, cache hits={ga_result['n_cache_hits']})"
    )
    print(f"Final Fit Time      : {final_fit_time_s:.4f} s")
    print(
        f"Val GA-SVM          : Macro F1={val_metrics['macro_f1']:.4f} | "
        f"Def F1={val_metrics['defective_f1']:.4f} | "
        f"Def Prec={val_metrics['defective_precision']:.4f} | "
        f"Def Rec={val_metrics['defective_recall']:.4f} | "
        f"Acc={val_metrics['accuracy']:.4f}"
    )
    if baseline_val_comparison:
        print(
            f"Val Baseline (334)  : Macro F1={baseline_val_comparison['baseline_val_macro_f1']:.4f} | "
            f"Def Rec={baseline_val_comparison['baseline_val_defective_recall']:.4f} | "
            f"Acc={baseline_val_comparison['baseline_val_accuracy']:.4f}"
        )
    print(f"Mask saved          : {mask_path}")
    print(f"GA-SVM model saved  : {model_path}")
    print(f"Convergence plot    : {fig_path}")
    print(f"GA history JSON     : {history_path}")


if __name__ == "__main__":
    main()
