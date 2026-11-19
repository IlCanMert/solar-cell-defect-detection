"""Evaluate E1-E5 models and multi-seed GA on the locked test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config, load_raw_elpv
from src.evaluation import (
    evaluate_fitted_model,
    run_ablation_experiment,
    run_error_analysis_and_save_samples,
    run_multiseed_ga_evaluation,
    run_random_subset_baseline,
    save_confusion_matrices_figure,
    save_selected_features_breakdown_figure,
)


def _format_metric_row(
    exp_id: str,
    model_name: str,
    selected_features: int | float,
    hog_features: int | float,
    lbp_features: int | float,
    reduction_pct: float,
    train_cv_f1: float,
    val_f1: float,
    test_metrics: dict[str, Any],
    search_or_train_s: float,
    fit_time_s: float,
    clf_infer_ms: float,
    total_infer_ms: float,
) -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "model_name": model_name,
        "selected_features": round(float(selected_features), 2),
        "hog_features": round(float(hog_features), 2),
        "lbp_features": round(float(lbp_features), 2),
        "feature_reduction_pct": round(float(reduction_pct), 2),
        "train_cv_macro_f1": round(float(train_cv_f1), 4),
        "val_macro_f1": round(float(val_f1), 4),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), 4),
        "test_defective_f1": round(float(test_metrics["defective_f1"]), 4),
        "test_defective_precision": round(float(test_metrics["defective_precision"]), 4),
        "test_defective_recall": round(float(test_metrics["defective_recall"]), 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "train_or_search_time_s": round(float(search_or_train_s), 4),
        "fit_time_s": round(float(fit_time_s), 4),
        "classifier_infer_ms": round(float(clf_infer_ms), 5),
        "total_infer_ms": round(float(total_infer_ms), 5),
    }


def _generate_plots_and_error_analysis(
    cfg: dict[str, Any],
    evaluation_summary: dict[str, Any],
    df_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Generate Adim 3.3 figures and qualitative error analysis artifacts."""
    figures_dir = PROJECT_ROOT / "results" / "figures"
    samples_dir = PROJECT_ROOT / "results" / "sample_predictions"
    splits_csv = PROJECT_ROOT / "data" / "splits.csv"

    images, _, _, _ = load_raw_elpv()
    df_splits = pd.read_csv(splits_csv)

    e1_exp = evaluation_summary["experiments"]["E1"]
    e2_exp = evaluation_summary["experiments"]["E2_seed42"]
    hog_dim = int(cfg["features"]["hog"]["expected_length"])
    lbp_dim = int(cfg["features"]["lbp"]["expected_length"])

    cm_fig = save_confusion_matrices_figure(
        e1_cm=e1_exp["confusion_matrix"],
        e2_cm=e2_exp["confusion_matrix"],
        e1_row=e1_exp["row"],
        e2_row=e2_exp["row"],
        output_path=figures_dir / "confusion_matrices.png",
    )
    breakdown_fig = save_selected_features_breakdown_figure(
        e2_hog=int(e2_exp["row"]["hog_features"]),
        e2_lbp=int(e2_exp["row"]["lbp_features"]),
        hog_total=hog_dim,
        lbp_total=lbp_dim,
        df_metrics=df_metrics,
        output_path=figures_dir / "selected_features_breakdown.png",
    )
    error_summary = run_error_analysis_and_save_samples(
        images=images,
        df_splits=df_splits,
        eval_summary=evaluation_summary,
        output_dir=samples_dir,
        figures_dir=figures_dir,
    )

    print("\n=== Adim 3.3 Figures & Qualitative Error Analysis ===")
    print(f"Saved confusion matrices        : {cm_fig}")
    print(f"Saved features breakdown        : {breakdown_fig}")
    print(f"Saved error analysis grid       : {error_summary['saved_files']['error_analysis_grid']}")
    print(f"Saved E1 vs E2 disagreements    : {error_summary['saved_files']['e1_vs_e2_disagreements']}")
    print(f"Saved error analysis JSON       : {error_summary['saved_files']['error_analysis_summary']}")
    disagree = error_summary["e1_vs_e2_disagreements"]
    print(
        "E2 Outcomes (TP/TN/FP/FN)       : "
        f"{error_summary['e2_outcome_counts']['TP']}/"
        f"{error_summary['e2_outcome_counts']['TN']}/"
        f"{error_summary['e2_outcome_counts']['FP']}/"
        f"{error_summary['e2_outcome_counts']['FN']}"
    )
    print(
        "E1 vs E2 Disagreements          : "
        f"total={disagree['total_disagreements']}, "
        f"GA fixed={disagree['e2_fixed_e1_error']}, "
        f"GA regressed={disagree['e2_regressed_vs_e1']}, "
        f"net={disagree['net_improvement']:+d}"
    )
    return error_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate E1-E5 models and generate visualization / error-analysis artifacts."
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip model re-training/evaluation and generate Adim 3.3 figures from saved JSON/CSV.",
    )
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    metrics_csv = PROJECT_ROOT / "results" / "metrics.csv"
    eval_summary_path = PROJECT_ROOT / "results" / "evaluation_summary.json"

    if args.plots_only:
        if not metrics_csv.exists() or not eval_summary_path.exists():
            raise FileNotFoundError(
                "Missing saved results/metrics.csv or results/evaluation_summary.json. "
                "Run without --plots-only first."
            )
        df_metrics = pd.read_csv(metrics_csv)
        with eval_summary_path.open("r", encoding="utf-8") as handle:
            evaluation_summary = json.load(handle)
        _generate_plots_and_error_analysis(cfg, evaluation_summary, df_metrics)
        return

    svm_cfg = cfg["baseline_svm"]
    cv_folds = int(cfg["ga"]["cv_folds"])
    hog_dim = int(cfg["features"]["hog"]["expected_length"])
    lbp_dim = int(cfg["features"]["lbp"]["expected_length"])
    total_dim = int(cfg["features"]["total_expected_length"])

    features_path = PROJECT_ROOT / "data" / "features_hog_lbp.npz"
    with np.load(features_path, allow_pickle=False) as data:
        x_all = data["X"]
        y_all = data["y"]
        splits = data["splits"]

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]
    x_test, y_test = x_all[test_mask], y_all[test_mask]

    # Load feature extraction latency (ms/image) to compute end-to-end inference time
    feat_summary_path = PROJECT_ROOT / "results" / "feature_extraction_summary.json"
    feat_extract_ms = 0.0
    if feat_summary_path.exists():
        with feat_summary_path.open("r", encoding="utf-8") as handle:
            feat_extract_ms = float(json.load(handle)["ms_per_image"])

    # Load saved artifacts from Week 2
    baseline_pipe = joblib.load(PROJECT_ROOT / "models" / "baseline.joblib")
    ga_pipe = joblib.load(PROJECT_ROOT / "models" / "ga_svm.joblib")
    ga_mask_bool = np.load(PROJECT_ROOT / "models" / "selected_mask.npy").astype(bool)

    with (PROJECT_ROOT / "results" / "baseline_validation.json").open("r", encoding="utf-8") as h:
        base_val_json = json.load(h)
    with (PROJECT_ROOT / "results" / "ga_history.json").open("r", encoding="utf-8") as h:
        ga_hist_json = json.load(h)

    print("=== 1. Evaluating E1 Baseline (All 334 Features) on Locked Test Set ===")
    e1_val_eval = evaluate_fitted_model(baseline_pipe, x_val, y_val)
    e1_test_eval = evaluate_fitted_model(baseline_pipe, x_test, y_test)
    e1_clf_ms = float(e1_test_eval["classifier_infer_ms_per_image"])
    e1_row = _format_metric_row(
        exp_id="E1",
        model_name="Baseline SVM (All HOG+LBP)",
        selected_features=total_dim,
        hog_features=hog_dim,
        lbp_features=lbp_dim,
        reduction_pct=0.0,
        train_cv_f1=float(base_val_json["train_cv_3fold"]["cv_macro_f1_mean"]),
        val_f1=float(e1_val_eval["macro_f1"]),
        test_metrics=e1_test_eval,
        search_or_train_s=float(base_val_json["train_time_s"]),
        fit_time_s=float(base_val_json["train_time_s"]),
        clf_infer_ms=e1_clf_ms,
        total_infer_ms=feat_extract_ms + e1_clf_ms,
    )

    print("=== 2. Evaluating E2 GA-SVM (seed=42) on Locked Test Set ===")
    e2_val_eval = evaluate_fitted_model(ga_pipe, x_val[:, ga_mask_bool], y_val)
    e2_test_eval = evaluate_fitted_model(ga_pipe, x_test[:, ga_mask_bool], y_test)
    e2_clf_ms = float(e2_test_eval["classifier_infer_ms_per_image"])
    e2_selected = int(ga_mask_bool.sum())
    e2_hog = int(ga_mask_bool[:hog_dim].sum())
    e2_lbp = int(ga_mask_bool[hog_dim:].sum())
    e2_reduction = 100.0 * (1.0 - float(e2_selected) / float(total_dim))
    e2_row = _format_metric_row(
        exp_id="E2_seed42",
        model_name="GA-SVM (seed=42)",
        selected_features=e2_selected,
        hog_features=e2_hog,
        lbp_features=e2_lbp,
        reduction_pct=e2_reduction,
        train_cv_f1=float(ga_hist_json["best_eval"]["cv_macro_f1_mean"]),
        val_f1=float(e2_val_eval["macro_f1"]),
        test_metrics=e2_test_eval,
        search_or_train_s=float(ga_hist_json["search_time_s"]),
        fit_time_s=float(ga_hist_json["final_fit_time_s"]),
        clf_infer_ms=e2_clf_ms,
        total_infer_ms=feat_extract_ms + e2_clf_ms,
    )

    print("=== 3. Running Multi-Seed GA Stability Evaluation (seeds=[42, 123, 999]) ===")
    df_multiseed, multiseed_summary = run_multiseed_ga_evaluation(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        cfg=cfg,
        feat_extract_ms=feat_extract_ms,
    )
    multiseed_csv = PROJECT_ROOT / "results" / "ga_multiseed_runs.csv"
    df_multiseed.to_csv(multiseed_csv, index=False)
    multiseed_json = PROJECT_ROOT / "results" / "ga_multiseed_summary.json"
    with multiseed_json.open("w", encoding="utf-8") as handle:
        json.dump(multiseed_summary, handle, indent=2)

    ms_stats = multiseed_summary["mean_std"]
    e2_mean_row = {
        "experiment_id": "E2_3seed_mean",
        "model_name": "GA-SVM (3-Seed Mean)",
        "selected_features": ms_stats["selected_features"]["mean"],
        "hog_features": ms_stats["hog_features"]["mean"],
        "lbp_features": ms_stats["lbp_features"]["mean"],
        "feature_reduction_pct": ms_stats["feature_reduction_pct"]["mean"],
        "train_cv_macro_f1": ms_stats["train_cv_macro_f1"]["mean"],
        "val_macro_f1": ms_stats["val_macro_f1"]["mean"],
        "test_macro_f1": ms_stats["test_macro_f1"]["mean"],
        "test_defective_f1": ms_stats["test_defective_f1"]["mean"],
        "test_defective_precision": ms_stats["test_defective_precision"]["mean"],
        "test_defective_recall": ms_stats["test_defective_recall"]["mean"],
        "test_accuracy": ms_stats["test_accuracy"]["mean"],
        "train_or_search_time_s": ms_stats["search_time_s"]["mean"],
        "fit_time_s": ms_stats["fit_time_s"]["mean"],
        "classifier_infer_ms": round(float(ms_stats["classifier_infer_ms"]["mean"]), 5),
        "total_infer_ms": round(float(ms_stats["total_infer_ms"]["mean"]), 5),
    }

    print("\n=== 4. Running Ablation Experiments (E3: HOG-only, E4: LBP-only, E5: Random-208) ===")
    hog_mask = np.zeros(total_dim, dtype=bool)
    hog_mask[:hog_dim] = True
    e3_res = run_ablation_experiment(
        name="E3_HOG_Only",
        feature_mask=hog_mask,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        svm_cfg=svm_cfg,
        cv_folds=cv_folds,
        hog_dim=hog_dim,
        feat_extract_ms=feat_extract_ms,
    )
    e3_row = _format_metric_row(
        exp_id="E3",
        model_name="HOG-Only + SVM",
        selected_features=e3_res["selected_features"],
        hog_features=e3_res["hog_features"],
        lbp_features=e3_res["lbp_features"],
        reduction_pct=e3_res["feature_reduction_pct"],
        train_cv_f1=e3_res["train_cv_macro_f1"],
        val_f1=e3_res["val"]["macro_f1"],
        test_metrics=e3_res["test"],
        search_or_train_s=e3_res["fit_time_s"],
        fit_time_s=e3_res["fit_time_s"],
        clf_infer_ms=e3_res["classifier_infer_ms"],
        total_infer_ms=e3_res["total_infer_ms"],
    )

    lbp_mask = np.zeros(total_dim, dtype=bool)
    lbp_mask[hog_dim:] = True
    e4_res = run_ablation_experiment(
        name="E4_LBP_Only",
        feature_mask=lbp_mask,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        svm_cfg=svm_cfg,
        cv_folds=cv_folds,
        hog_dim=hog_dim,
        feat_extract_ms=feat_extract_ms,
    )
    e4_row = _format_metric_row(
        exp_id="E4",
        model_name="LBP-Only + SVM",
        selected_features=e4_res["selected_features"],
        hog_features=e4_res["hog_features"],
        lbp_features=e4_res["lbp_features"],
        reduction_pct=e4_res["feature_reduction_pct"],
        train_cv_f1=e4_res["train_cv_macro_f1"],
        val_f1=e4_res["val"]["macro_f1"],
        test_metrics=e4_res["test"],
        search_or_train_s=e4_res["fit_time_s"],
        fit_time_s=e4_res["fit_time_s"],
        clf_infer_ms=e4_res["classifier_infer_ms"],
        total_infer_ms=e4_res["total_infer_ms"],
    )

    e5_bundle = run_random_subset_baseline(
        target_count=e2_selected,
        total_features=total_dim,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        svm_cfg=svm_cfg,
        seeds=[42, 123, 999, 7, 2026],
        cv_folds=cv_folds,
        hog_dim=hog_dim,
        feat_extract_ms=feat_extract_ms,
    )
    e5_s42 = e5_bundle["seed_42"]
    e5_s42_row = _format_metric_row(
        exp_id="E5_seed42",
        model_name=f"Random Subset ({e2_selected}, seed=42)",
        selected_features=e5_s42["selected_features"],
        hog_features=e5_s42["hog_features"],
        lbp_features=e5_s42["lbp_features"],
        reduction_pct=e5_s42["feature_reduction_pct"],
        train_cv_f1=e5_s42["train_cv_macro_f1"],
        val_f1=e5_s42["val"]["macro_f1"],
        test_metrics=e5_s42["test"],
        search_or_train_s=e5_s42["fit_time_s"],
        fit_time_s=e5_s42["fit_time_s"],
        clf_infer_ms=e5_s42["classifier_infer_ms"],
        total_infer_ms=e5_s42["total_infer_ms"],
    )

    e5_agg = e5_bundle["summary_mean_std"]
    e5_mean_row = {
        "experiment_id": "E5_5seed_mean",
        "model_name": f"Random Subset ({e2_selected}, 5-Seed Mean)",
        "selected_features": float(e2_selected),
        "hog_features": round(e5_agg["hog_features"]["mean"], 2),
        "lbp_features": round(e5_agg["lbp_features"]["mean"], 2),
        "feature_reduction_pct": round(e2_reduction, 2),
        "train_cv_macro_f1": round(e5_agg["train_cv_macro_f1"]["mean"], 4),
        "val_macro_f1": round(e5_agg["val_macro_f1"]["mean"], 4),
        "test_macro_f1": round(e5_agg["test_macro_f1"]["mean"], 4),
        "test_defective_f1": round(e5_agg["test_defective_f1"]["mean"], 4),
        "test_defective_precision": round(e5_agg["test_defective_precision"]["mean"], 4),
        "test_defective_recall": round(e5_agg["test_defective_recall"]["mean"], 4),
        "test_accuracy": round(e5_agg["test_accuracy"]["mean"], 4),
        "train_or_search_time_s": round(e5_agg["fit_time_s"]["mean"], 4),
        "fit_time_s": round(e5_agg["fit_time_s"]["mean"], 4),
        "classifier_infer_ms": round(e5_agg["classifier_infer_ms"]["mean"], 5),
        "total_infer_ms": round(e5_agg["total_infer_ms"]["mean"], 5),
    }

    df_metrics = pd.DataFrame(
        [e1_row, e2_row, e2_mean_row, e3_row, e4_row, e5_s42_row, e5_mean_row]
    )
    metrics_csv = PROJECT_ROOT / "results" / "metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    evaluation_summary = {
        "test_samples": int(len(y_test)),
        "test_healthy": int((y_test == 0).sum()),
        "test_defective": int((y_test == 1).sum()),
        "feature_extraction_ms_per_image": round(feat_extract_ms, 4),
        "experiments": {
            "E1": {
                "row": e1_row,
                "confusion_matrix": e1_test_eval["confusion_matrix"],
                "val_confusion_matrix": e1_val_eval["confusion_matrix"],
            },
            "E2_seed42": {
                "row": e2_row,
                "confusion_matrix": e2_test_eval["confusion_matrix"],
                "val_confusion_matrix": e2_val_eval["confusion_matrix"],
            },
            "E3_HOG_Only": e3_res,
            "E4_LBP_Only": e4_res,
            "E5_Random": {
                "seed_42": e5_s42,
                "summary_mean_std": e5_agg,
            },
        },
        "multiseed_ga": multiseed_summary,
        "test_predictions": {
            "y_true": [int(v) for v in y_test],
            "e1_pred": e1_test_eval["y_pred"],
            "e1_scores": e1_test_eval["decision_scores"],
            "e2_pred": e2_test_eval["y_pred"],
            "e2_scores": e2_test_eval["decision_scores"],
        },
    }

    eval_summary_path = PROJECT_ROOT / "results" / "evaluation_summary.json"
    with eval_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_summary, handle, indent=2)

    print("\n=== Locked Test Set Comparison (E1-E5) ===")
    display_cols = [
        "experiment_id",
        "selected_features",
        "feature_reduction_pct",
        "train_cv_macro_f1",
        "val_macro_f1",
        "test_macro_f1",
        "test_defective_f1",
        "test_defective_recall",
        "test_accuracy",
        "train_or_search_time_s",
    ]
    print(df_metrics[display_cols].to_string(index=False))

    print("\n=== Multi-Seed GA Runs (seeds=[42, 123, 999]) ===")
    print(
        df_multiseed[
            [
                "seed",
                "selected_features",
                "hog_features",
                "lbp_features",
                "train_cv_macro_f1",
                "test_macro_f1",
                "test_defective_recall",
                "test_accuracy",
                "search_time_s",
            ]
        ].to_string(index=False)
    )
    print("\nMulti-Seed GA Mean +/- Std:")
    for key in [
        "selected_features",
        "feature_reduction_pct",
        "train_cv_macro_f1",
        "test_macro_f1",
        "test_defective_recall",
        "test_accuracy",
        "search_time_s",
    ]:
        print(f"  {key:<24}: {ms_stats[key]['mean']:.4f} +/- {ms_stats[key]['std']:.4f}")

    print(f"\nSaved metrics CSV        : {metrics_csv}")
    print(f"Saved multi-seed CSV     : {multiseed_csv}")
    print(f"Saved multi-seed JSON    : {multiseed_json}")
    print(f"Saved evaluation summary : {eval_summary_path}")

    _generate_plots_and_error_analysis(cfg, evaluation_summary, df_metrics)


if __name__ == "__main__":
    main()
