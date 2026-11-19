"""CLI single-image and locked-test showcase demo for Solar Cell Defect Classification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    load_image_from_path,
    load_inference_bundle,
    predict_single_image,
    save_prediction_card,
)

DEFAULT_SHOWCASE_INDICES: list[int] = [470, 1180, 1576]


def _lookup_split_metadata(
    df_splits: pd.DataFrame,
    *,
    subset_index: int | None = None,
    image_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find matching metadata row from data/splits.csv by subset_index or image path."""
    if df_splits.empty:
        return None

    if subset_index is not None:
        matched = df_splits[df_splits["subset_index"] == int(subset_index)]
        if not matched.empty:
            return matched.iloc[0].to_dict()
        return None

    if image_path is not None:
        norm = str(image_path).replace("\\", "/")
        matched = df_splits[df_splits["image_path"] == norm]
        if matched.empty:
            filename = Path(norm).name
            matched = df_splits[df_splits["image_path"].str.endswith("/" + filename)]
        if not matched.empty:
            return matched.iloc[0].to_dict()
    return None


def _build_meta_from_row(row: dict[str, Any] | None, fallback_name: str) -> dict[str, Any]:
    """Build clean sample metadata dictionary for printing and figure titles."""
    if row is None:
        return {
            "sample_tag": fallback_name,
            "subset_index": None,
            "original_index": None,
            "image_path": fallback_name,
            "split": None,
            "cell_type": None,
            "true_label": None,
            "true_label_name": None,
        }
    return {
        "sample_tag": f"#{int(row['subset_index'])} ({Path(str(row['image_path'])).name})",
        "subset_index": int(row["subset_index"]),
        "original_index": int(row["original_index"]),
        "image_path": str(row["image_path"]),
        "split": str(row["split"]),
        "cell_type": str(row["cell_type"]),
        "true_label": int(row["label"]),
        "true_label_name": str(row["label_name"]),
    }


def _print_prediction_block(pred: dict[str, Any], input_desc: str) -> None:
    """Print formatted single-sample comparison between Baseline SVM and GA-SVM."""
    meta = pred.get("meta", {})
    timings = pred["timing_ms"]
    print(f"Input image         : {input_desc}")
    if meta.get("cell_type") or meta.get("true_label_name"):
        print(
            f"Metadata            : split={meta.get('split', 'N/A')} | "
            f"cell_type={meta.get('cell_type', 'N/A')} | "
            f"ground_truth={meta.get('true_label_name', 'Unknown')}"
        )
    print(
        f"Baseline prediction : {pred['baseline_name']} "
        f"(margin={pred['baseline_margin']:+.4f}, clf_time={timings['baseline_clf_ms']:.4f} ms)"
    )
    print(
        f"GA-SVM prediction   : {pred['ga_name']} "
        f"(margin={pred['ga_margin']:+.4f}, clf_time={timings['ga_clf_ms']:.4f} ms)"
    )
    print(
        f"Selected features   : {pred['selected_features']} / {pred['total_features']} "
        f"(HOG={pred['selected_hog']}/324, LBP={pred['selected_lbp']}/10, "
        f"-{pred['feature_reduction_pct']:.2f}%)"
    )
    print(
        f"Inference timing    : feature_extract={timings['feature_extraction_ms']:.3f} ms | "
        f"baseline_total={timings['baseline_total_ms']:.3f} ms | "
        f"ga_total={timings['ga_total_ms']:.3f} ms"
    )
    print("-" * 72)


def _strip_visuals_for_json(pred: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a prediction record without numpy arrays."""
    return {k: v for k, v in pred.items() if k != "visuals"}


def _maybe_open_file(path: Path) -> None:
    """Open saved artifact visually when --show is requested on Windows."""
    if os.name == "nt":
        os.startfile(str(path))


def _print_reference_summary(root: Path) -> None:
    """Point the presenter/user to locked-test summary artifacts on disk."""
    metrics_path = root / "results" / "metrics.csv"
    cm_path = root / "results" / "figures" / "confusion_matrices.png"
    breakdown_path = root / "results" / "figures" / "selected_features_breakdown.png"
    print("\nDecision score: above zero predicts DEFECTIVE. At or below zero predicts HEALTHY.")
    print("Scores are uncalibrated.")
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path)
        e1 = df_m[df_m["experiment_id"] == "E1"].iloc[0]
        e2 = df_m[df_m["experiment_id"] == "E2_seed42"].iloc[0]
        print(
            "\nLocked Test Reference (n=334): "
            f"E1 Macro F1={float(e1['test_macro_f1']):.4f} (334 feats) | "
            f"E2 GA-SVM Macro F1={float(e2['test_macro_f1']):.4f} "
            f"({int(e2['selected_features'])} feats, -{float(e2['feature_reduction_pct']):.2f}%)"
        )
    print(f"Metrics table       : {metrics_path}")
    print(f"Confusion matrices  : {cm_path}")
    print(f"Feature breakdown   : {breakdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-image or showcase ELPV defect inference (Baseline vs. GA-SVM)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a cell image (disk path or ELPV relative path, e.g., images/cell0527.png).",
    )
    group.add_argument(
        "--subset-index",
        type=int,
        default=None,
        help="ELPV subset_index from data/splits.csv (e.g., 1576).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path for the diagnostic card.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open generated prediction image(s) after saving.",
    )
    args = parser.parse_args()

    bundle = load_inference_bundle(PROJECT_ROOT)
    splits_path = PROJECT_ROOT / "data" / "splits.csv"
    df_splits = pd.read_csv(splits_path) if splits_path.exists() else pd.DataFrame()
    samples_dir = PROJECT_ROOT / "results" / "sample_predictions"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Single-sample CLI mode
    if args.image is not None or args.subset_index is not None:
        if args.subset_index is not None:
            row = _lookup_split_metadata(df_splits, subset_index=args.subset_index)
            if row is None:
                raise KeyError(f"subset_index={args.subset_index} not found in {splits_path}")
            img, resolved_path = load_image_from_path(row["image_path"], PROJECT_ROOT)
            meta = _build_meta_from_row(row, fallback_name=resolved_path.name)
            input_desc = f"subset_index={args.subset_index} ({row['image_path']})"
        else:
            img, resolved_path = load_image_from_path(args.image, PROJECT_ROOT)
            row = _lookup_split_metadata(df_splits, image_path=args.image)
            meta = _build_meta_from_row(row, fallback_name=resolved_path.name)
            input_desc = f"{args.image} -> {resolved_path}"

        pred = predict_single_image(img, bundle, include_visuals=True)
        pred["meta"] = meta

        out_png = (
            Path(args.output)
            if args.output is not None
            else samples_dir / "demo_single_prediction.png"
        )
        save_prediction_card([pred], out_png)

        print("=== Solar Cell Defect Classification — Single Image Inference ===")
        _print_prediction_block(pred, input_desc=input_desc)
        print(f"Saved prediction card: {out_png}")
        _print_reference_summary(PROJECT_ROOT)

        if args.show:
            _maybe_open_file(out_png)
        return

    # Default showcase mode: 3 representative locked-test cells (#470 TP, #1180 TN, #1576 GA fix)
    print("=== Solar Cell Defect Classification — Locked Test Showcase (E1 vs. E2 GA-SVM) ===")
    showcase_results: list[dict[str, Any]] = []
    for idx in DEFAULT_SHOWCASE_INDICES:
        row = _lookup_split_metadata(df_splits, subset_index=idx)
        if row is None:
            raise KeyError(f"Showcase subset_index={idx} not found in {splits_path}")
        img, resolved_path = load_image_from_path(row["image_path"], PROJECT_ROOT)
        pred = predict_single_image(img, bundle, include_visuals=True)
        pred["meta"] = _build_meta_from_row(row, fallback_name=resolved_path.name)
        pred["meta"]["resolved_path"] = str(resolved_path)
        showcase_results.append(pred)
        _print_prediction_block(pred, input_desc=f"#{idx} ({row['image_path']})")

    showcase_png = (
        Path(args.output) if args.output is not None else samples_dir / "demo_showcase.png"
    )
    save_prediction_card(
        showcase_results,
        showcase_png,
        suptitle=(
            "Locked Test Showcase: Defective (#470), Healthy (#1180), "
            "and GA-Fixed Poly False Alarm (#1576)"
        ),
    )

    single_png = samples_dir / "demo_single_prediction.png"
    save_prediction_card(
        [showcase_results[-1]],
        single_png,
        suptitle=(
            "Single-Cell Diagnostic (#1576, poly Healthy): "
            "Baseline False Alarm vs. GA-SVM Correction"
        ),
    )

    demo_json = samples_dir / "demo_output.json"
    json_payload = {
        "mode": "default_showcase",
        "showcase_indices": DEFAULT_SHOWCASE_INDICES,
        "selected_features": bundle["selected_features"],
        "selected_hog": bundle["selected_hog"],
        "selected_lbp": bundle["selected_lbp"],
        "total_features": bundle["total_features"],
        "feature_reduction_pct": bundle["feature_reduction_pct"],
        "predictions": [_strip_visuals_for_json(p) for p in showcase_results],
        "saved_files": {
            "demo_showcase": str(showcase_png),
            "demo_single_prediction": str(single_png),
            "demo_output": str(demo_json),
        },
    }
    demo_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Saved showcase card   : {showcase_png}")
    print(f"Saved single card     : {single_png}")
    print(f"Saved demo summary    : {demo_json}")
    _print_reference_summary(PROJECT_ROOT)

    if args.show:
        _maybe_open_file(showcase_png)


if __name__ == "__main__":
    main()
