"""HOG and LBP feature extraction pipeline for ELPV solar cell images."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.feature import hog, local_binary_pattern

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config, load_raw_elpv
from src.preprocessing import preprocess_for_features


def extract_hog(
    image_uint8: np.ndarray,
    hog_cfg: dict[str, Any],
    visualize: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract HOG descriptor vector (and optional gradient image) from a grayscale image."""
    result = hog(
        image_uint8,
        orientations=int(hog_cfg["orientations"]),
        pixels_per_cell=tuple(int(v) for v in hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(int(v) for v in hog_cfg["cells_per_block"]),
        block_norm=str(hog_cfg["block_norm"]),
        feature_vector=True,
        visualize=visualize,
    )

    expected_len = int(hog_cfg["expected_length"])
    if visualize:
        feat, hog_image = result
        feat_arr = np.asarray(feat, dtype=np.float64)
        if feat_arr.shape != (expected_len,):
            raise ValueError(f"Expected HOG vector length {expected_len}, got {feat_arr.shape}")
        return feat_arr, np.asarray(hog_image, dtype=np.float64)

    feat_arr = np.asarray(result, dtype=np.float64)
    if feat_arr.shape != (expected_len,):
        raise ValueError(f"Expected HOG vector length {expected_len}, got {feat_arr.shape}")
    return feat_arr


def extract_lbp(
    image_uint8: np.ndarray,
    lbp_cfg: dict[str, Any],
    return_map: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract normalized uniform LBP histogram (and optional LBP code map)."""
    p_neighbors = int(lbp_cfg["P"])
    radius = float(lbp_cfg["R"])
    method = str(lbp_cfg["method"])
    n_bins = int(lbp_cfg["n_bins"])
    expected_len = int(lbp_cfg["expected_length"])

    lbp_map = local_binary_pattern(image_uint8, P=p_neighbors, R=radius, method=method)
    bin_edges = np.arange(0, n_bins + 1, dtype=np.float64)
    counts, _ = np.histogram(lbp_map.ravel(), bins=bin_edges)
    hist = counts.astype(np.float64) / float(lbp_map.size)

    if hist.shape != (expected_len,):
        raise ValueError(f"Expected LBP vector length {expected_len}, got {hist.shape}")
    if not np.isclose(hist.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Normalized LBP histogram does not sum to 1.0 (sum={hist.sum()})")

    if return_map:
        return hist, lbp_map
    return hist


def extract_single_image_features(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Preprocess one EL image and extract its concatenated HOG + LBP feature vector."""
    img_hog, img_lbp = preprocess_for_features(image, cfg)
    feat_cfg = cfg["features"]

    hog_vec = extract_hog(img_hog, feat_cfg["hog"], visualize=False)
    lbp_vec = extract_lbp(img_lbp, feat_cfg["lbp"], return_map=False)
    combined = np.concatenate([hog_vec, lbp_vec]).astype(np.float64)

    expected_total = int(feat_cfg["total_expected_length"])
    if combined.shape != (expected_total,):
        raise ValueError(f"Expected combined feature length {expected_total}, got {combined.shape}")
    if not np.isfinite(combined).all():
        raise ValueError("Extracted feature vector contains NaN or Inf values")
    return combined


def extract_or_load_feature_matrix(
    images: np.ndarray,
    df_splits: pd.DataFrame,
    cfg: dict[str, Any],
    cache_path: str | Path = "data/features_hog_lbp.npz",
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Extract (or load from .npz cache) HOG+LBP features aligned with data/splits.csv."""
    cache_file = Path(cache_path)
    if cache_file.exists() and not force_recompute:
        with np.load(cache_file, allow_pickle=False) as cached:
            return {
                "X": cached["X"],
                "X_hog": cached["X_hog"],
                "X_lbp": cached["X_lbp"],
                "y": cached["y"],
                "splits": cached["splits"],
                "subset_indices": cached["subset_indices"],
                "original_indices": cached["original_indices"],
                "extraction_time_s": float(cached["extraction_time_s"]),
                "ms_per_image": float(cached["ms_per_image"]),
                "from_cache": True,
            }

    ordered_splits = df_splits.sort_values("subset_index").reset_index(drop=True)
    n_samples = len(ordered_splits)
    hog_dim = int(cfg["features"]["hog"]["expected_length"])
    lbp_dim = int(cfg["features"]["lbp"]["expected_length"])
    total_dim = int(cfg["features"]["total_expected_length"])
    if hog_dim + lbp_dim != total_dim:
        raise ValueError(f"Feature dimensions mismatch: {hog_dim} + {lbp_dim} != {total_dim}")

    x_mat = np.empty((n_samples, total_dim), dtype=np.float64)
    start_time = time.perf_counter()
    for row_idx, orig_idx in enumerate(ordered_splits["original_index"].to_numpy()):
        x_mat[row_idx] = extract_single_image_features(images[int(orig_idx)], cfg)
    elapsed_s = time.perf_counter() - start_time
    ms_per_image = (elapsed_s * 1000.0) / float(n_samples)

    if not np.isfinite(x_mat).all():
        raise ValueError("Feature matrix contains NaN or Inf values")

    x_hog = x_mat[:, :hog_dim]
    x_lbp = x_mat[:, hog_dim:]
    y = ordered_splits["label"].to_numpy(dtype=np.int64)
    splits = ordered_splits["split"].to_numpy(dtype=str)
    subset_indices = ordered_splits["subset_index"].to_numpy(dtype=np.int64)
    original_indices = ordered_splits["original_index"].to_numpy(dtype=np.int64)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        X=x_mat,
        X_hog=x_hog,
        X_lbp=x_lbp,
        y=y,
        splits=splits,
        subset_indices=subset_indices,
        original_indices=original_indices,
        extraction_time_s=np.float64(elapsed_s),
        ms_per_image=np.float64(ms_per_image),
    )

    return {
        "X": x_mat,
        "X_hog": x_hog,
        "X_lbp": x_lbp,
        "y": y,
        "splits": splits,
        "subset_indices": subset_indices,
        "original_indices": original_indices,
        "extraction_time_s": float(elapsed_s),
        "ms_per_image": float(ms_per_image),
        "from_cache": False,
    }


def save_feature_visualization(
    images: np.ndarray,
    df_splits: pd.DataFrame,
    cfg: dict[str, Any],
    output_path: str | Path = "results/figures/feature_visualization.png",
) -> list[dict[str, str]]:
    """Save a 2x4 visualization comparing Original, HOG, LBP map, and LBP histogram."""
    train_df = df_splits[df_splits["split"] == "train"]
    healthy_row = train_df[train_df["label"] == 0].iloc[0]
    defective_row = train_df[train_df["label"] == 1].iloc[0]
    selected_rows = [healthy_row, defective_row]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    fig.suptitle(
        "Preprocessing, HOG Gradients, and Uniform LBP Features (HEALTHY vs. DEFECTIVE)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    recorded: list[dict[str, str]] = []
    for row_idx, row in enumerate(selected_rows):
        orig_idx = int(row["original_index"])
        cell_id = Path(str(row["image_path"])).stem
        orig_img = images[orig_idx]
        img_hog, img_lbp = preprocess_for_features(orig_img, cfg)

        _, hog_vis = extract_hog(img_hog, cfg["features"]["hog"], visualize=True)
        hog_vis_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, hog_vis.max() or 1.0))
        lbp_hist, lbp_map = extract_lbp(img_lbp, cfg["features"]["lbp"], return_map=True)

        # Column 1: Original 300x300 grayscale image
        ax0 = axes[row_idx, 0]
        ax0.imshow(orig_img, cmap="gray", vmin=0, vmax=255)
        ax0.set_title(f"{cell_id} ({row['label_name']}, {row['cell_type']})\nOriginal 300x300", fontsize=9.5)
        ax0.axis("off")

        # Column 2: Resized 64x64 HOG visualization
        ax1 = axes[row_idx, 1]
        ax1.imshow(hog_vis_rescaled, cmap="gray")
        ax1.set_title("HOG Visualization (64x64)\n324 Features (9 ori, 16x16 cell)", fontsize=9.5)
        ax1.axis("off")

        # Column 3: LBP uniform code map on 64x64
        ax2 = axes[row_idx, 2]
        ax2.imshow(lbp_map, cmap="viridis", vmin=0, vmax=9)
        ax2.set_title("Uniform LBP Map (64x64)\nP=8, R=1 (10 pattern codes)", fontsize=9.5)
        ax2.axis("off")

        # Column 4: Normalized 10-bin LBP histogram
        ax3 = axes[row_idx, 3]
        bins = np.arange(len(lbp_hist))
        color = "#2a7b88" if int(row["label"]) == 0 else "#c0392b"
        ax3.bar(bins, lbp_hist, color=color, edgecolor="black", linewidth=0.6)
        ax3.set_xticks(bins)
        ax3.set_ylim(0.0, max(0.35, float(lbp_hist.max()) * 1.15))
        ax3.set_xlabel("Uniform LBP Bin (0-9)", fontsize=8.5)
        ax3.set_ylabel("Normalized Freq.", fontsize=8.5)
        ax3.set_title("LBP 10-Bin Histogram\n(Sum = 1.00)", fontsize=9.5)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

        recorded.append(
            {
                "cell_id": cell_id,
                "original_index": str(orig_idx),
                "label_name": str(row["label_name"]),
                "cell_type": str(row["cell_type"]),
                "split": str(row["split"]),
            }
        )

    plt.subplots_adjust(top=0.87, hspace=0.36, wspace=0.26)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return recorded


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    splits_path = PROJECT_ROOT / cfg["dataset"]["splits_file"]
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Frozen splits file not found at {splits_path}. Run scripts/prepare_data.py first."
        )

    df_splits = pd.read_csv(splits_path)
    images, _, _, _ = load_raw_elpv()

    cache_path = PROJECT_ROOT / "data" / "features_hog_lbp.npz"
    result = extract_or_load_feature_matrix(
        images=images,
        df_splits=df_splits,
        cfg=cfg,
        cache_path=cache_path,
        force_recompute=True,
    )

    x_mat = result["X"]
    x_hog = result["X_hog"]
    x_lbp = result["X_lbp"]
    y = result["y"]

    fig_path = PROJECT_ROOT / "results" / "figures" / "feature_visualization.png"
    visualized_cells = save_feature_visualization(
        images=images,
        df_splits=df_splits,
        cfg=cfg,
        output_path=fig_path,
    )

    summary = {
        "total_samples": int(x_mat.shape[0]),
        "hog_dim": int(x_hog.shape[1]),
        "lbp_dim": int(x_lbp.shape[1]),
        "total_dim": int(x_mat.shape[1]),
        "matrix_shape": list(x_mat.shape),
        "matrix_dtype": str(x_mat.dtype),
        "label_shape": list(y.shape),
        "has_nan": bool(np.isnan(x_mat).any()),
        "has_inf": bool(np.isinf(x_mat).any()),
        "lbp_row_sum_min": float(x_lbp.sum(axis=1).min()),
        "lbp_row_sum_max": float(x_lbp.sum(axis=1).max()),
        "extraction_time_s": round(float(result["extraction_time_s"]), 4),
        "ms_per_image": round(float(result["ms_per_image"]), 4),
        "cache_file": str(cache_path.relative_to(PROJECT_ROOT)),
        "figure_file": str(fig_path.relative_to(PROJECT_ROOT)),
        "sample_cells_visualized": visualized_cells,
    }

    summary_path = PROJECT_ROOT / "results" / "feature_extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== HOG + LBP Feature Extraction Complete ===")
    print(f"X shape          : {x_mat.shape} ({x_mat.dtype})")
    print(f"X_hog shape      : {x_hog.shape}")
    print(f"X_lbp shape      : {x_lbp.shape}")
    print(f"NaN / Inf check  : has_nan={summary['has_nan']}, has_inf={summary['has_inf']}")
    print(f"LBP row sum range: [{summary['lbp_row_sum_min']:.6f}, {summary['lbp_row_sum_max']:.6f}]")
    print(f"Extraction time  : {summary['extraction_time_s']:.4f} s ({summary['ms_per_image']:.4f} ms/image)")
    print(f"Cached matrix    : {cache_path}")
    print(f"Feature visual   : {fig_path}")
    print(f"Summary JSON     : {summary_path}")


if __name__ == "__main__":
    main()
