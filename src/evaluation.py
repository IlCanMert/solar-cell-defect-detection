"""Model evaluation, multi-seed GA stability, and ablation experiments (E1-E5)."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.baseline import (
    build_svm_pipeline,
    compute_classification_metrics,
    evaluate_cv_on_train,
)
from src.ga_selection import run_genetic_algorithm


def evaluate_fitted_model(
    pipeline: Pipeline,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    n_repeats_timing: int = 20,
) -> dict[str, Any]:
    """Evaluate a fitted SVM pipeline on a split and measure classifier inference latency."""
    y_pred = pipeline.predict(x_eval)
    scores = pipeline.decision_function(x_eval)

    t0 = time.perf_counter()
    for _ in range(int(n_repeats_timing)):
        _ = pipeline.predict(x_eval)
    elapsed_s = time.perf_counter() - t0
    classifier_infer_ms = (elapsed_s * 1000.0) / float(int(n_repeats_timing) * len(x_eval))

    metrics = compute_classification_metrics(y_eval, y_pred)
    metrics["classifier_infer_ms_per_image"] = float(classifier_infer_ms)
    metrics["y_pred"] = [int(v) for v in y_pred]
    metrics["decision_scores"] = [float(v) for v in scores]
    return metrics


def run_ablation_experiment(
    name: str,
    feature_mask: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    svm_cfg: dict[str, Any],
    cv_folds: int = 3,
    search_time_s: float = 0.0,
    hog_dim: int = 324,
    feat_extract_ms: float = 0.0,
) -> dict[str, Any]:
    """Train and evaluate a fresh LinearSVC pipeline on a specified binary feature mask."""
    bool_mask = np.asarray(feature_mask, dtype=bool).ravel()
    total_features = int(len(bool_mask))
    selected_count = int(bool_mask.sum())
    if selected_count == 0:
        raise ValueError(f"Feature mask for {name} selected 0 features")

    hog_selected = int(bool_mask[:hog_dim].sum())
    lbp_selected = int(bool_mask[hog_dim:].sum())
    reduction_pct = 100.0 * (1.0 - float(selected_count) / float(total_features))

    c_val = float(svm_cfg["C"])
    class_weight = svm_cfg["class_weight"]
    max_iter = int(svm_cfg["max_iter"])
    seed = int(svm_cfg["random_state"])

    base_pipe = build_svm_pipeline(
        c_val=c_val,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )
    cv_summary = evaluate_cv_on_train(
        base_pipe,
        x_train=x_train[:, bool_mask],
        y_train=y_train,
        n_splits=cv_folds,
        seed=seed,
    )

    fitted_pipe = build_svm_pipeline(
        c_val=c_val,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )
    t_fit0 = time.perf_counter()
    fitted_pipe.fit(x_train[:, bool_mask], y_train)
    fit_time_s = time.perf_counter() - t_fit0

    val_eval = evaluate_fitted_model(fitted_pipe, x_val[:, bool_mask], y_val)
    test_eval = evaluate_fitted_model(fitted_pipe, x_test[:, bool_mask], y_test)
    clf_infer_ms = float(test_eval["classifier_infer_ms_per_image"])

    return {
        "experiment_name": name,
        "selected_features": selected_count,
        "hog_features": hog_selected,
        "lbp_features": lbp_selected,
        "feature_reduction_pct": float(reduction_pct),
        "train_cv_macro_f1": float(cv_summary["cv_macro_f1_mean"]),
        "train_cv_macro_f1_std": float(cv_summary["cv_macro_f1_std"]),
        "val": {k: v for k, v in val_eval.items() if k not in ("y_pred", "decision_scores")},
        "test": {k: v for k, v in test_eval.items() if k not in ("y_pred", "decision_scores")},
        "search_time_s": float(search_time_s),
        "fit_time_s": float(fit_time_s),
        "classifier_infer_ms": clf_infer_ms,
        "total_infer_ms": float(feat_extract_ms + clf_infer_ms),
    }


def run_random_subset_baseline(
    target_count: int,
    total_features: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    svm_cfg: dict[str, Any],
    seeds: list[int] | None = None,
    cv_folds: int = 3,
    hog_dim: int = 324,
    feat_extract_ms: float = 0.0,
) -> dict[str, Any]:
    """Evaluate random feature subsets of size target_count across multiple random seeds."""
    if seeds is None:
        seeds = [42, 123, 999, 7, 2026]

    trials: list[dict[str, Any]] = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        chosen_idx = rng.choice(total_features, size=int(target_count), replace=False)
        mask = np.zeros(total_features, dtype=bool)
        mask[chosen_idx] = True

        res = run_ablation_experiment(
            name=f"E5_Random_{target_count}_seed{s}",
            feature_mask=mask,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            svm_cfg=svm_cfg,
            cv_folds=cv_folds,
            search_time_s=0.0,
            hog_dim=hog_dim,
            feat_extract_ms=feat_extract_ms,
        )
        res["seed"] = int(s)
        trials.append(res)

    def _agg(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    summary = {
        "n_trials": len(trials),
        "seeds": [int(s) for s in seeds],
        "target_count": int(target_count),
        "hog_features": _agg([t["hog_features"] for t in trials]),
        "lbp_features": _agg([t["lbp_features"] for t in trials]),
        "train_cv_macro_f1": _agg([t["train_cv_macro_f1"] for t in trials]),
        "val_macro_f1": _agg([t["val"]["macro_f1"] for t in trials]),
        "test_macro_f1": _agg([t["test"]["macro_f1"] for t in trials]),
        "test_defective_f1": _agg([t["test"]["defective_f1"] for t in trials]),
        "test_defective_precision": _agg([t["test"]["defective_precision"] for t in trials]),
        "test_defective_recall": _agg([t["test"]["defective_recall"] for t in trials]),
        "test_accuracy": _agg([t["test"]["accuracy"] for t in trials]),
        "fit_time_s": _agg([t["fit_time_s"] for t in trials]),
        "classifier_infer_ms": _agg([t["classifier_infer_ms"] for t in trials]),
        "total_infer_ms": _agg([t["total_infer_ms"] for t in trials]),
    }

    return {
        "seed_42": trials[0],
        "trials": trials,
        "summary_mean_std": summary,
    }


def run_multiseed_ga_evaluation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
    feat_extract_ms: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run GA feature selection across multiple seeds and summarize test/feature stability."""
    ga_cfg = cfg["ga"]
    svm_cfg = cfg["baseline_svm"]
    hog_dim = int(cfg["features"]["hog"]["expected_length"])
    total_features = int(x_train.shape[1])
    seeds = [int(s) for s in ga_cfg["multi_seeds"]]

    run_rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n--- Multi-Seed GA Run (seed={seed}) ---")
        ga_res = run_genetic_algorithm(
            x_train=x_train,
            y_train=y_train,
            cfg=cfg,
            seed=seed,
            verbose=True,
        )
        bool_mask = ga_res["best_mask"].astype(bool)
        best_eval = ga_res["best_eval"]

        pipe = build_svm_pipeline(
            c_val=float(svm_cfg["C"]),
            class_weight=svm_cfg["class_weight"],
            max_iter=int(svm_cfg["max_iter"]),
            random_state=int(svm_cfg["random_state"]),
        )
        t_fit0 = time.perf_counter()
        pipe.fit(x_train[:, bool_mask], y_train)
        fit_time_s = time.perf_counter() - t_fit0

        val_eval = evaluate_fitted_model(pipe, x_val[:, bool_mask], y_val)
        test_eval = evaluate_fitted_model(pipe, x_test[:, bool_mask], y_test)
        clf_infer_ms = float(test_eval["classifier_infer_ms_per_image"])
        selected_count = int(bool_mask.sum())
        reduction_pct = 100.0 * (1.0 - float(selected_count) / float(total_features))

        run_rows.append(
            {
                "seed": seed,
                "selected_features": selected_count,
                "hog_features": int(best_eval["hog_selected"]),
                "lbp_features": int(best_eval["lbp_selected"]),
                "feature_reduction_pct": round(reduction_pct, 2),
                "best_fitness": round(float(best_eval["fitness"]), 4),
                "train_cv_macro_f1": round(float(best_eval["cv_macro_f1_mean"]), 4),
                "val_macro_f1": round(float(val_eval["macro_f1"]), 4),
                "test_macro_f1": round(float(test_eval["macro_f1"]), 4),
                "test_defective_f1": round(float(test_eval["defective_f1"]), 4),
                "test_defective_precision": round(float(test_eval["defective_precision"]), 4),
                "test_defective_recall": round(float(test_eval["defective_recall"]), 4),
                "test_accuracy": round(float(test_eval["accuracy"]), 4),
                "search_time_s": round(float(ga_res["search_time_s"]), 4),
                "fit_time_s": round(float(fit_time_s), 4),
                "classifier_infer_ms": round(clf_infer_ms, 5),
                "total_infer_ms": round(float(feat_extract_ms + clf_infer_ms), 5),
                "n_unique_evals": int(ga_res["n_evaluations"]),
                "n_cache_hits": int(ga_res["n_cache_hits"]),
            }
        )

    df_runs = pd.DataFrame(run_rows)
    metric_cols = [
        "selected_features",
        "hog_features",
        "lbp_features",
        "feature_reduction_pct",
        "best_fitness",
        "train_cv_macro_f1",
        "val_macro_f1",
        "test_macro_f1",
        "test_defective_f1",
        "test_defective_precision",
        "test_defective_recall",
        "test_accuracy",
        "search_time_s",
        "fit_time_s",
        "classifier_infer_ms",
        "total_infer_ms",
    ]
    agg_stats: dict[str, dict[str, float]] = {}
    for col in metric_cols:
        vals = df_runs[col].to_numpy(dtype=np.float64)
        agg_stats[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
        }

    summary = {
        "seeds": seeds,
        "n_runs": len(seeds),
        "std_ddof": 1,
        "runs": run_rows,
        "mean_std": agg_stats,
    }
    return df_runs, summary


def _draw_single_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    title: str,
    vmax: int,
) -> Any:
    """Render an annotated 2x2 confusion matrix heatmap onto an axis."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    cell_tags = [["TN", "FP"], ["FN", "TP"]]
    row_sums = cm.sum(axis=1, keepdims=True)
    thresh = vmax * 0.55

    for r in range(2):
        for c in range(2):
            count = int(cm[r, c])
            pct = (100.0 * count / float(row_sums[r, 0])) if row_sums[r, 0] > 0 else 0.0
            color = "white" if count > thresh else "#111827"
            ax.text(
                c,
                r,
                f"{cell_tags[r][c]}\n{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
                linespacing=1.35,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Healthy (0)", "Pred: Defective (1)"], fontsize=10)
    ax.set_yticklabels(["True: Healthy (0)", "True: Defective (1)"], fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    return im


def save_confusion_matrices_figure(
    e1_cm: list[list[int]] | np.ndarray,
    e2_cm: list[list[int]] | np.ndarray,
    e1_row: dict[str, Any],
    e2_row: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot side-by-side locked test-set confusion matrices for E1 and E2."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm1 = np.asarray(e1_cm, dtype=int)
    cm2 = np.asarray(e2_cm, dtype=int)
    vmax = int(max(cm1.max(), cm2.max()))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), dpi=160)
    e1_title = (
        f"E1: Baseline LinearSVC (334 Features)\n"
        f"Macro F1={float(e1_row['test_macro_f1']):.4f} | "
        f"Def. Recall={float(e1_row['test_defective_recall']):.4f} | "
        f"Acc={float(e1_row['test_accuracy']):.4f}"
    )
    e2_title = (
        f"E2: GA-Selected LinearSVC (seed=42, {int(e2_row['selected_features'])} Features)\n"
        f"Macro F1={float(e2_row['test_macro_f1']):.4f} | "
        f"Def. Recall={float(e2_row['test_defective_recall']):.4f} | "
        f"Acc={float(e2_row['test_accuracy']):.4f}"
    )

    _draw_single_confusion_matrix(axes[0], cm1, e1_title, vmax=vmax)
    im = _draw_single_confusion_matrix(axes[1], cm2, e2_title, vmax=vmax)

    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.10, right=0.88, wspace=0.38)
    cbar_ax = fig.add_axes([0.90, 0.16, 0.02, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Test Sample Count", fontsize=10)

    fig.suptitle(
        "Locked Test Set Confusion Matrices (n = 334: 226 Healthy, 108 Defective)",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


def save_selected_features_breakdown_figure(
    e2_hog: int,
    e2_lbp: int,
    hog_total: int,
    lbp_total: int,
    df_metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot GA feature retention breakdown (HOG vs. LBP) and E1-E5 test comparison."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    e2_total = e2_hog + e2_lbp
    full_total = hog_total + lbp_total

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.2, 5.9), dpi=160)

    # Left panel: Retained vs. Pruned features per modality
    categories = [
        f"HOG\n({hog_total} dim)",
        f"Uniform LBP\n({lbp_total} dim)",
        f"Combined\n({full_total} dim)",
    ]
    retained = np.array([e2_hog, e2_lbp, e2_total], dtype=float)
    totals = np.array([hog_total, lbp_total, full_total], dtype=float)
    pruned = totals - retained
    x = np.arange(len(categories))
    width = 0.50

    ax_left.bar(x, retained, width=width, color="#1d4ed8", label="Retained by GA (seed=42)")
    ax_left.bar(
        x,
        pruned,
        width=width,
        bottom=retained,
        color="#cbd5e1",
        edgecolor="#64748b",
        linestyle="--",
        linewidth=0.8,
        label="Pruned by GA",
    )

    for idx, (ret, tot) in enumerate(zip(retained, totals)):
        pct = 100.0 * ret / tot if tot > 0 else 0.0
        ax_left.text(
            idx,
            tot + 9.0,
            f"{int(ret)}/{int(tot)}\n({pct:.1f}% kept)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(categories, fontsize=10)
    ax_left.set_ylabel("Feature Count", fontsize=10)
    ax_left.set_ylim(0, full_total + 115)
    ax_left.grid(axis="y", linestyle=":", alpha=0.6)
    ax_left.legend(loc="upper center", ncol=2, fontsize=8.8)
    ax_left.set_title(
        "GA Modality Breakdown: Edge (HOG) vs. Texture (LBP)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    # Right panel: E1-E5 test Macro F1 vs Defective Recall
    order_specs = [
        ("E1", "E1: Baseline\n(334 feat)"),
        ("E2_seed42", f"E2: GA s=42\n({e2_total} feat)"),
        ("E2_3seed_mean", "E2: 3-Seed\n(192.7 avg)"),
        ("E5_5seed_mean", f"E5: Rand-{e2_total}\n(5-seed avg)"),
        ("E3", f"E3: HOG-Only\n({hog_total} feat)"),
        ("E4", f"E4: LBP-Only\n({lbp_total} feat)"),
    ]
    rows_by_id = df_metrics.set_index("experiment_id").to_dict(orient="index")
    labels: list[str] = []
    macro_f1s: list[float] = []
    def_recalls: list[float] = []
    for exp_id, label in order_specs:
        if exp_id in rows_by_id:
            labels.append(label)
            macro_f1s.append(float(rows_by_id[exp_id]["test_macro_f1"]))
            def_recalls.append(float(rows_by_id[exp_id]["test_defective_recall"]))

    xr = np.arange(len(labels))
    bar_w = 0.36
    bars1 = ax_right.bar(
        xr - bar_w / 2, macro_f1s, width=bar_w, color="#0f766e", label="Test Macro F1"
    )
    bars2 = ax_right.bar(
        xr + bar_w / 2, def_recalls, width=bar_w, color="#d97706", label="Test Defective Recall"
    )

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax_right.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.006,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                color="#1e293b",
            )

    ax_right.set_xticks(xr)
    ax_right.set_xticklabels(labels, fontsize=8.6)
    ax_right.set_ylabel("Score", fontsize=10)
    ax_right.set_ylim(0.55, 0.80)
    ax_right.grid(axis="y", linestyle=":", alpha=0.6)
    ax_right.legend(loc="upper right", fontsize=9)
    ax_right.set_title(
        "Locked Test Performance Across Feature Subsets (E1-E5)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.06, right=0.98, wspace=0.22)
    fig.suptitle(
        "GA Feature Selection Analysis: Selected Modalities and Test Comparison",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


def _classify_outcome(y_true: int, y_pred: int) -> str:
    """Map ground-truth and predicted binary labels to TP, TN, FP, or FN."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def _select_diverse_samples(
    df_cat: pd.DataFrame,
    max_samples: int = 4,
) -> pd.DataFrame:
    """Pick top-confidence examples balancing monocrystalline and polycrystalline cells."""
    if df_cat.empty:
        return df_cat
    ranked = df_cat.sort_values("e2_abs_score", ascending=False)
    mono = ranked[ranked["cell_type"] == "mono"].head(2)
    poly = ranked[ranked["cell_type"] == "poly"].head(2)
    chosen = pd.concat([mono, poly], axis=0)
    if len(chosen) < max_samples:
        remaining = ranked.loc[~ranked.index.isin(chosen.index)].head(max_samples - len(chosen))
        chosen = pd.concat([chosen, remaining], axis=0)
    return chosen.head(max_samples)


def run_error_analysis_and_save_samples(
    images: np.ndarray,
    df_splits: pd.DataFrame,
    eval_summary: dict[str, Any],
    output_dir: str | Path,
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Perform qualitative and subgroup error analysis on test predictions and save sample grids."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir is not None:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

    df_test = (
        df_splits.sort_values("subset_index")
        .loc[lambda d: d["split"] == "test"]
        .reset_index(drop=True)
        .copy()
    )

    preds = eval_summary["test_predictions"]
    y_true = np.asarray(preds["y_true"], dtype=int)
    e1_pred = np.asarray(preds["e1_pred"], dtype=int)
    e1_scores = np.asarray(preds["e1_scores"], dtype=float)
    e2_pred = np.asarray(preds["e2_pred"], dtype=int)
    e2_scores = np.asarray(preds["e2_scores"], dtype=float)

    if len(df_test) != len(y_true):
        raise ValueError(
            f"Test split length mismatch: df_test={len(df_test)}, predictions={len(y_true)}"
        )
    if not np.array_equal(df_test["label"].to_numpy(dtype=int), y_true):
        raise ValueError("Aligned test split labels do not match eval_summary test_predictions y_true.")

    df_test["y_true"] = y_true
    df_test["e1_pred"] = e1_pred
    df_test["e1_score"] = e1_scores
    df_test["e2_pred"] = e2_pred
    df_test["e2_score"] = e2_scores
    df_test["e2_abs_score"] = np.abs(e2_scores)
    df_test["e1_outcome"] = [
        _classify_outcome(int(yt), int(yp)) for yt, yp in zip(y_true, e1_pred)
    ]
    df_test["e2_outcome"] = [
        _classify_outcome(int(yt), int(yp)) for yt, yp in zip(y_true, e2_pred)
    ]
    df_test["e1_correct"] = df_test["e1_pred"] == df_test["y_true"]
    df_test["e2_correct"] = df_test["e2_pred"] == df_test["y_true"]

    # Subgroup breakdown by crystal type (mono vs poly)
    cell_type_breakdown: dict[str, dict[str, Any]] = {}
    for ctype, group in df_test.groupby("cell_type"):
        n_group = int(len(group))
        e1_err = int((~group["e1_correct"]).sum())
        e2_err = int((~group["e2_correct"]).sum())
        outcomes = group["e2_outcome"].value_counts().to_dict()
        cell_type_breakdown[str(ctype)] = {
            "total": n_group,
            "true_healthy": int((group["y_true"] == 0).sum()),
            "true_defective": int((group["y_true"] == 1).sum()),
            "e1_errors": e1_err,
            "e1_error_rate": round(e1_err / n_group, 4),
            "e2_errors": e2_err,
            "e2_error_rate": round(e2_err / n_group, 4),
            "e2_TP": int(outcomes.get("TP", 0)),
            "e2_TN": int(outcomes.get("TN", 0)),
            "e2_FP": int(outcomes.get("FP", 0)),
            "e2_FN": int(outcomes.get("FN", 0)),
        }

    # Disagreements between E1 and E2
    df_disagree = df_test[df_test["e1_pred"] != df_test["e2_pred"]].copy()
    df_fixed = df_disagree[(~df_disagree["e1_correct"]) & df_disagree["e2_correct"]]
    df_regressed = df_disagree[df_disagree["e1_correct"] & (~df_disagree["e2_correct"])]

    # Render 4x4 qualitative error analysis grid (TP, TN, FP, FN for E2)
    categories = [
        ("TP", "TP: Defective Caught\n(True=1, Pred=1)", "#15803d"),
        ("TN", "TN: Healthy Confirmed\n(True=0, Pred=0)", "#1d4ed8"),
        ("FP", "FP: False Alarm\n(True=0, Pred=1)", "#b45309"),
        ("FN", "FN: Missed Defect\n(True=1, Pred=0)", "#b91c1c"),
    ]
    n_cols = 4
    fig, axes = plt.subplots(len(categories), n_cols, figsize=(13.5, 14.0), dpi=160)
    showcased_samples: dict[str, list[dict[str, Any]]] = {}

    for row_idx, (cat_code, cat_title, title_color) in enumerate(categories):
        df_cat = _select_diverse_samples(
            df_test[df_test["e2_outcome"] == cat_code], max_samples=n_cols
        )
        showcased_samples[cat_code] = []
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx < len(df_cat):
                rec = df_cat.iloc[col_idx]
                orig_idx = int(rec["original_index"])
                ax.imshow(images[orig_idx], cmap="gray", vmin=0, vmax=255)
                true_str = "DEF" if int(rec["y_true"]) == 1 else "HLTH"
                e2_str = "DEF" if int(rec["e2_pred"]) == 1 else "HLTH"
                e1_str = "DEF" if int(rec["e1_pred"]) == 1 else "HLTH"
                ax.set_title(
                    f"#{int(rec['subset_index'])} ({rec['cell_type']}) | "
                    f"T:{true_str} P2:{e2_str} (P1:{e1_str})\n"
                    f"E2 margin={float(rec['e2_score']):+.2f} | E1={float(rec['e1_score']):+.2f}",
                    fontsize=8.5,
                    color=title_color if col_idx == 0 else "#1e293b",
                    fontweight="bold" if col_idx == 0 else "normal",
                    pad=6,
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor(title_color)
                    spine.set_linewidth(1.5)
                showcased_samples[cat_code].append(
                    {
                        "subset_index": int(rec["subset_index"]),
                        "original_index": orig_idx,
                        "image_path": str(rec["image_path"]),
                        "cell_type": str(rec["cell_type"]),
                        "y_true": int(rec["y_true"]),
                        "e1_pred": int(rec["e1_pred"]),
                        "e1_score": round(float(rec["e1_score"]), 4),
                        "e2_pred": int(rec["e2_pred"]),
                        "e2_score": round(float(rec["e2_score"]), 4),
                    }
                )
            else:
                ax.axis("off")

        axes[row_idx, 0].set_ylabel(
            cat_title,
            fontsize=9.5,
            fontweight="bold",
            color=title_color,
            labelpad=10,
        )

    fig.subplots_adjust(top=0.91, bottom=0.03, left=0.06, right=0.98, hspace=0.34, wspace=0.14)
    fig.suptitle(
        "E2 (GA-SVM, seed=42) Qualitative Test Set Error Analysis (TP / TN / FP / FN)",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )
    grid_path = out_dir / "error_analysis_grid.png"
    fig.savefig(grid_path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    if figures_dir is not None:
        shutil.copy2(grid_path, Path(figures_dir) / "error_analysis_grid.png")

    # Render E1 vs E2 disagreement gallery (up to 6 fixed + up to 6 regressed)
    fixed_sel = df_fixed.sort_values("e2_abs_score", ascending=False).head(6)
    reg_sel = df_regressed.sort_values("e2_abs_score", ascending=False).head(6)
    disagree_cols = max(len(fixed_sel), len(reg_sel), 1)
    fig_d, axes_d = plt.subplots(
        2, disagree_cols, figsize=(max(9.0, 2.7 * disagree_cols), 6.6), dpi=160
    )
    if disagree_cols == 1:
        axes_d = np.asarray(axes_d).reshape(2, 1)

    disagree_groups = [
        (fixed_sel, f"GA Fixed Baseline Error (n={len(df_fixed)})", "#15803d"),
        (reg_sel, f"GA Regressed vs Baseline (n={len(df_regressed)})", "#b91c1c"),
    ]
    for r_idx, (grp_df, grp_label, grp_color) in enumerate(disagree_groups):
        for c_idx in range(disagree_cols):
            ax = axes_d[r_idx, c_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if c_idx < len(grp_df):
                rec = grp_df.iloc[c_idx]
                orig_idx = int(rec["original_index"])
                ax.imshow(images[orig_idx], cmap="gray", vmin=0, vmax=255)
                true_str = "DEF" if int(rec["y_true"]) == 1 else "HLTH"
                e1_str = "DEF" if int(rec["e1_pred"]) == 1 else "HLTH"
                e2_str = "DEF" if int(rec["e2_pred"]) == 1 else "HLTH"
                ax.set_title(
                    f"#{int(rec['subset_index'])} ({rec['cell_type']}) | T:{true_str}\n"
                    f"E1:{e1_str} ({float(rec['e1_score']):+.2f}) -> E2:{e2_str} ({float(rec['e2_score']):+.2f})",
                    fontsize=8.2,
                    color="#1e293b",
                    pad=5,
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor(grp_color)
                    spine.set_linewidth(1.5)
            else:
                ax.axis("off")
        axes_d[r_idx, 0].set_ylabel(grp_label, fontsize=9.5, fontweight="bold", color=grp_color)

    fig_d.subplots_adjust(top=0.84, bottom=0.05, left=0.08, right=0.98, hspace=0.40, wspace=0.16)
    fig_d.suptitle(
        f"E1 (Baseline) vs. E2 (GA-SVM) Test Disagreements: "
        f"{len(df_disagree)}/{len(df_test)} samples ({len(df_fixed)} fixed, {len(df_regressed)} regressed)",
        fontsize=12.5,
        fontweight="bold",
        y=0.96,
    )
    disagree_path = out_dir / "e1_vs_e2_disagreements.png"
    fig_d.savefig(disagree_path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig_d)

    outcome_counts = df_test["e2_outcome"].value_counts().to_dict()
    e1_outcome_counts = df_test["e1_outcome"].value_counts().to_dict()
    analysis_summary: dict[str, Any] = {
        "test_total": int(len(df_test)),
        "e1_outcome_counts": {k: int(e1_outcome_counts.get(k, 0)) for k in ("TP", "TN", "FP", "FN")},
        "e2_outcome_counts": {k: int(outcome_counts.get(k, 0)) for k in ("TP", "TN", "FP", "FN")},
        "e1_total_errors": int((~df_test["e1_correct"]).sum()),
        "e2_total_errors": int((~df_test["e2_correct"]).sum()),
        "cell_type_breakdown": cell_type_breakdown,
        "e1_vs_e2_disagreements": {
            "total_disagreements": int(len(df_disagree)),
            "e2_fixed_e1_error": int(len(df_fixed)),
            "e2_regressed_vs_e1": int(len(df_regressed)),
            "net_improvement": int(len(df_fixed) - len(df_regressed)),
            "fixed_subset_indices": [int(v) for v in df_fixed["subset_index"].tolist()],
            "regressed_subset_indices": [int(v) for v in df_regressed["subset_index"].tolist()],
        },
        "showcased_samples": showcased_samples,
        "saved_files": {
            "error_analysis_grid": str(grid_path),
            "e1_vs_e2_disagreements": str(disagree_path),
        },
    }
    if figures_dir is not None:
        analysis_summary["saved_files"]["figures_error_analysis_grid"] = str(
            Path(figures_dir) / "error_analysis_grid.png"
        )

    summary_json_path = out_dir / "error_analysis_summary.json"
    summary_json_path.write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")
    analysis_summary["saved_files"]["error_analysis_summary"] = str(summary_json_path)
    return analysis_summary
