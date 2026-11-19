"""Prepare ELPV binary dataset, verify distributions, freeze splits.csv, and export sample grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    build_subset_dataframe,
    create_or_load_splits,
    filter_binary_subset,
    inspect_elpv_metadata,
    load_config,
    load_raw_elpv,
)


def _save_dataset_samples_figure(
    images: np.ndarray,
    df_splits: pd.DataFrame,
    output_path: Path,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Save a 2x4 figure showing 4 HEALTHY (2 mono, 2 poly) and 4 DEFECTIVE (2 mono, 2 poly) cells."""
    rng = np.random.default_rng(seed)
    selection_groups = [
        ("HEALTHY", "mono"),
        ("HEALTHY", "poly"),
        ("DEFECTIVE", "mono"),
        ("DEFECTIVE", "poly"),
    ]

    chosen_rows: list[pd.Series] = []
    for label_name, cell_type in selection_groups:
        candidates = df_splits[
            (df_splits["label_name"] == label_name) & (df_splits["cell_type"] == cell_type)
        ]
        if len(candidates) < 2:
            raise ValueError(f"Not enough samples for group ({label_name}, {cell_type})")
        picked_positions = np.sort(rng.choice(len(candidates), size=2, replace=False))
        for pos in picked_positions:
            chosen_rows.append(candidates.iloc[int(pos)])

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    fig.suptitle(
        "ELPV Binary Dataset Samples (HEALTHY vs. DEFECTIVE across Mono/Poly Cells)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    recorded_samples: list[dict[str, str]] = []
    for ax, row in zip(axes.flat, chosen_rows):
        orig_idx = int(row["original_index"])
        cell_id = Path(str(row["image_path"])).stem
        ax.imshow(images[orig_idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{cell_id} | {row['label_name']}\n{row['cell_type']} | split={row['split']}",
            fontsize=10,
        )
        ax.axis("off")
        recorded_samples.append(
            {
                "cell_id": cell_id,
                "original_index": str(orig_idx),
                "label_name": str(row["label_name"]),
                "cell_type": str(row["cell_type"]),
                "split": str(row["split"]),
            }
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return recorded_samples


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    seed = int(cfg["project"]["seed"])
    dataset_cfg = cfg["dataset"]
    ratios = dataset_cfg["split_ratios"]
    splits_path = PROJECT_ROOT / dataset_cfg["splits_file"]

    print("=== 1. Inspecting ELPV Metadata ===")
    meta_info = inspect_elpv_metadata()
    print(f"Data directory : {meta_info['data_dir']}")
    print(f"Entries        : {meta_info['entries']}")
    print(f"Has module_id  : {meta_info['has_module_id']}")
    print(f"Sample rows    : {meta_info['sample_rows']}")

    print("\n=== 2. Loading Raw ELPV Dataset ===")
    images, proba, types, image_paths = load_raw_elpv()
    print(f"Total images   : {len(images)}")
    print(f"Image shape    : {images.shape}, dtype={images.dtype}")
    print(f"Pixel range    : [{int(images.min())}, {int(images.max())}]")

    unique_probs, prob_counts = np.unique(np.round(proba, 4), return_counts=True)
    prob_distribution = {str(float(p)): int(c) for p, c in zip(unique_probs, prob_counts)}
    print("Probability label distribution:")
    for p_val, cnt in prob_distribution.items():
        print(f"  proba={p_val:>6}: {cnt}")

    unique_types, type_counts = np.unique(types, return_counts=True)
    type_distribution = {str(t): int(c) for t, c in zip(unique_types, type_counts)}
    print(f"Cell types     : {type_distribution}")

    print("\n=== 3. Filtering Definite Binary Subset (0.0 & 1.0) ===")
    subset = filter_binary_subset(
        images=images,
        proba=proba,
        types=types,
        paths=image_paths,
        healthy_val=float(dataset_cfg["included_labels"]["healthy"]),
        defective_val=float(dataset_cfg["included_labels"]["defective"]),
    )
    df_meta = build_subset_dataframe(subset)
    n_total_binary = len(df_meta)
    n_healthy = int((df_meta["label"] == 0).sum())
    n_defective = int((df_meta["label"] == 1).sum())
    print(f"Binary subset  : {n_total_binary} images (HEALTHY={n_healthy}, DEFECTIVE={n_defective})")

    print("\n=== 4. Creating Stratified Splits (70% / 15% / 15%) ===")
    df_splits = create_or_load_splits(
        df_meta=df_meta,
        train_ratio=float(ratios["train"]),
        val_ratio=float(ratios["val"]),
        test_ratio=float(ratios["test"]),
        seed=seed,
        splits_path=splits_path,
        force_recreate=True,
    )
    print(f"Saved splits to: {splits_path}")

    split_summary: dict[str, dict[str, int | float]] = {}
    print("\nSplit Breakdown:")
    print(f"{'Split':<8} {'Total':>6} {'Pct':>7} {'HEALTHY':>9} {'DEFECTIVE':>11} {'DefRate':>9} {'mono':>7} {'poly':>7}")
    for split_name in ["train", "val", "test"]:
        sub = df_splits[df_splits["split"] == split_name]
        total = int(len(sub))
        healthy = int((sub["label"] == 0).sum())
        defective = int((sub["label"] == 1).sum())
        mono = int((sub["cell_type"] == "mono").sum())
        poly = int((sub["cell_type"] == "poly").sum())
        pct = 100.0 * total / n_total_binary
        def_rate = 100.0 * defective / total
        split_summary[split_name] = {
            "total": total,
            "percentage": round(pct, 2),
            "healthy": healthy,
            "defective": defective,
            "defective_rate_pct": round(def_rate, 2),
            "mono": mono,
            "poly": poly,
        }
        print(
            f"{split_name:<8} {total:>6} {pct:>6.2f}% {healthy:>9} {defective:>11} "
            f"{def_rate:>8.2f}% {mono:>7} {poly:>7}"
        )

    figures_path = PROJECT_ROOT / "results" / "figures" / "dataset_samples.png"
    sample_records = _save_dataset_samples_figure(
        images=images,
        df_splits=df_splits,
        output_path=figures_path,
        seed=seed,
    )
    print(f"\nSaved sample grid to: {figures_path}")

    summary_payload = {
        "seed": seed,
        "raw_total_images": int(len(images)),
        "image_shape": list(images.shape[1:]),
        "image_dtype": str(images.dtype),
        "pixel_min": int(images.min()),
        "pixel_max": int(images.max()),
        "has_module_id": bool(meta_info["has_module_id"]),
        "split_strategy": "stratified_joint_label_and_cell_type",
        "raw_probability_distribution": prob_distribution,
        "raw_cell_type_distribution": type_distribution,
        "binary_subset_total": n_total_binary,
        "binary_healthy": n_healthy,
        "binary_defective": n_defective,
        "excluded_intermediate_count": int(len(images) - n_total_binary),
        "splits": split_summary,
        "sample_figure": str(figures_path.relative_to(PROJECT_ROOT)),
        "sample_cells": sample_records,
    }
    summary_path = PROJECT_ROOT / "results" / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    print(f"Saved dataset summary to: {summary_path}")


if __name__ == "__main__":
    main()
