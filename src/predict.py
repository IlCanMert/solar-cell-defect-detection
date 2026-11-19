"""Single-image inference and diagnostic visualization utilities for ELPV solar cells."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LABEL_NAMES, _get_elpv_data_dir, load_config
from src.features import extract_hog, extract_lbp
from src.preprocessing import preprocess_for_features, validate_grayscale_image

LABEL_MAP: dict[int, str] = LABEL_NAMES


def load_inference_bundle(project_root: str | Path | None = None) -> dict[str, Any]:
    """Load config, fitted Baseline & GA SVM pipelines, and the GA feature mask."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    cfg_path = root / "configs" / "experiment.yaml"
    baseline_path = root / "models" / "baseline.joblib"
    ga_model_path = root / "models" / "ga_svm.joblib"
    mask_path = root / "models" / "selected_mask.npy"

    for required in (cfg_path, baseline_path, ga_model_path, mask_path):
        if not required.exists():
            raise FileNotFoundError(f"Required inference artifact not found: {required}")

    cfg = load_config(cfg_path)
    hog_dim = int(cfg["features"]["hog"]["expected_length"])
    lbp_dim = int(cfg["features"]["lbp"]["expected_length"])
    total_dim = int(cfg["features"]["total_expected_length"])
    if hog_dim + lbp_dim != total_dim:
        raise ValueError(f"Invalid feature dimensions in config: {hog_dim} + {lbp_dim} != {total_dim}")

    selected_mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    if selected_mask.ndim != 1 or selected_mask.shape[0] != total_dim:
        raise ValueError(
            f"Expected 1D boolean selected_mask of length {total_dim}, got shape {selected_mask.shape}"
        )
    if int(selected_mask.sum()) == 0:
        raise ValueError("Loaded GA selected_mask contains zero selected features")

    selected_hog = int(selected_mask[:hog_dim].sum())
    selected_lbp = int(selected_mask[hog_dim:].sum())
    selected_features = int(selected_mask.sum())
    reduction_pct = round(100.0 * (1.0 - selected_features / float(total_dim)), 2)

    baseline_model = joblib.load(baseline_path)
    ga_model = joblib.load(ga_model_path)

    return {
        "project_root": root,
        "cfg": cfg,
        "baseline_model": baseline_model,
        "ga_model": ga_model,
        "selected_mask": selected_mask,
        "hog_dim": hog_dim,
        "lbp_dim": lbp_dim,
        "total_features": total_dim,
        "selected_features": selected_features,
        "selected_hog": selected_hog,
        "selected_lbp": selected_lbp,
        "feature_reduction_pct": reduction_pct,
    }


def load_image_from_path(
    image_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Resolve an EL image path (filesystem or ELPV relative path) and load as uint8 grayscale.

    Uses np.fromfile + cv2.imdecode rather than cv2.imread because Windows paths
    containing non-ASCII characters (such as 'İmage_Proc_Project') cause cv2.imread
    to return None.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    raw_input = Path(image_path)
    elpv_dir = _get_elpv_data_dir()

    candidates: list[Path] = [raw_input]
    if not raw_input.is_absolute():
        candidates.append(root / raw_input)
        candidates.append(elpv_dir / raw_input)
        candidates.append(elpv_dir / "images" / raw_input.name)

    resolved: Path | None = None
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            break

    if resolved is None:
        raise FileNotFoundError(
            f"Could not find image '{image_path}' on disk, under workspace '{root}', "
            f"or inside ELPV dataset directory '{elpv_dir}'."
        )

    raw_bytes = np.fromfile(str(resolved), dtype=np.uint8)
    decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"Failed to decode image file: {resolved}")

    return validate_grayscale_image(decoded), resolved


def predict_single_image(
    image: np.ndarray,
    bundle: dict[str, Any],
    include_visuals: bool = False,
) -> dict[str, Any]:
    """Run Baseline (334-dim) and GA-SVM (208-dim) inference on a single solar cell image."""
    cfg = bundle["cfg"]
    feat_cfg = cfg["features"]
    mask = bundle["selected_mask"]
    baseline_model = bundle["baseline_model"]
    ga_model = bundle["ga_model"]

    img_gray = validate_grayscale_image(image)

    # Timed extraction without visualization rendering overhead
    t0 = time.perf_counter()
    img_hog, img_lbp = preprocess_for_features(img_gray, cfg)
    hog_vec = extract_hog(img_hog, feat_cfg["hog"], visualize=False)
    lbp_vec = extract_lbp(img_lbp, feat_cfg["lbp"], return_map=False)
    feat_vec = np.concatenate([hog_vec, lbp_vec]).astype(np.float64)
    feat_ms = (time.perf_counter() - t0) * 1000.0

    x_full = feat_vec.reshape(1, -1)
    x_ga = x_full[:, mask]

    t1 = time.perf_counter()
    e1_pred = int(baseline_model.predict(x_full)[0])
    e1_margin = float(baseline_model.decision_function(x_full)[0])
    e1_clf_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    e2_pred = int(ga_model.predict(x_ga)[0])
    e2_margin = float(ga_model.decision_function(x_ga)[0])
    e2_clf_ms = (time.perf_counter() - t2) * 1000.0

    result: dict[str, Any] = {
        "image_shape": [int(img_gray.shape[0]), int(img_gray.shape[1])],
        "baseline_label": e1_pred,
        "baseline_name": LABEL_MAP[e1_pred],
        "baseline_margin": round(e1_margin, 6),
        "ga_label": e2_pred,
        "ga_name": LABEL_MAP[e2_pred],
        "ga_margin": round(e2_margin, 6),
        "models_agree": bool(e1_pred == e2_pred),
        "total_features": int(bundle["total_features"]),
        "selected_features": int(bundle["selected_features"]),
        "selected_hog": int(bundle["selected_hog"]),
        "selected_lbp": int(bundle["selected_lbp"]),
        "feature_reduction_pct": float(bundle["feature_reduction_pct"]),
        "timing_ms": {
            "feature_extraction_ms": round(feat_ms, 4),
            "baseline_clf_ms": round(e1_clf_ms, 4),
            "baseline_total_ms": round(feat_ms + e1_clf_ms, 4),
            "ga_clf_ms": round(e2_clf_ms, 4),
            "ga_total_ms": round(feat_ms + e2_clf_ms, 4),
        },
        "score_note": (
            "LinearSVC reports an uncalibrated decision score. Scores above zero indicate "
            "DEFECTIVE; scores at or below zero indicate HEALTHY."
        ),
    }

    if include_visuals:
        _, hog_vis = extract_hog(img_hog, feat_cfg["hog"], visualize=True)
        max_val = float(hog_vis.max())
        hog_vis_rescaled = exposure.rescale_intensity(
            hog_vis, in_range=(0.0, max_val if max_val > 0.0 else 1.0)
        )
        lbp_hist, lbp_map = extract_lbp(img_lbp, feat_cfg["lbp"], return_map=True)
        result["visuals"] = {
            "img_gray": img_gray,
            "img_hog": img_hog,
            "hog_vis_rescaled": hog_vis_rescaled,
            "lbp_map": lbp_map,
            "lbp_hist": lbp_hist,
        }

    return result


def save_prediction_card(
    pred_results: list[dict[str, Any]],
    output_path: str | Path,
    suptitle: str | None = None,
) -> Path:
    """Write a diagnostic prediction card to disk and return its path."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    render_prediction_card(pred_results, out_file, suptitle)
    return out_file


def render_prediction_card(
    pred_results: list[dict[str, Any]],
    output: Path | BinaryIO,
    suptitle: str | None = None,
) -> None:
    """Render an N x 4 diagnostic prediction card (Original, HOG, LBP map, Histogram+Decision)."""
    if not pred_results:
        raise ValueError("pred_results cannot be empty")
    for item in pred_results:
        if "visuals" not in item:
            raise ValueError("Each item in pred_results must be run with include_visuals=True")

    n_rows = len(pred_results)
    fig_height = 4.6 if n_rows == 1 else 4.1 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, 4, figsize=(15.5, fig_height), dpi=160)
    if n_rows == 1:
        axes = np.asarray(axes).reshape(1, 4)

    for row_idx, item in enumerate(pred_results):
        vis = item["visuals"]
        meta = item.get("meta", {})
        sample_tag = str(meta.get("sample_tag", f"Sample {row_idx + 1}"))
        cell_type = meta.get("cell_type")
        true_name = meta.get("true_label_name")

        # Column 1: Original grayscale EL image
        ax0 = axes[row_idx, 0]
        ax0.imshow(vis["img_gray"], cmap="gray", vmin=0, vmax=255)
        subtitle_parts = []
        if cell_type:
            subtitle_parts.append(f"Type: {cell_type}")
        if true_name:
            subtitle_parts.append(f"True: {true_name}")
        line2 = " | ".join(subtitle_parts) if subtitle_parts else f"Size: {item['image_shape'][0]}x{item['image_shape'][1]}"
        ax0.set_title(f"{sample_tag}\n{line2}", fontsize=9.8, fontweight="bold", pad=8)
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Column 2: HOG gradient visualization
        ax1 = axes[row_idx, 1]
        ax1.imshow(vis["hog_vis_rescaled"], cmap="gray")
        ax1.set_title(
            f"HOG Gradients (64x64)\nSelected: {item['selected_hog']}/324 HOG dims",
            fontsize=9.5,
            pad=8,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Column 3: Uniform LBP pattern code map
        ax2 = axes[row_idx, 2]
        ax2.imshow(vis["lbp_map"], cmap="viridis", vmin=0, vmax=9)
        ax2.set_title(
            f"Uniform LBP Map (64x64)\nSelected: {item['selected_lbp']}/10 LBP dims",
            fontsize=9.5,
            pad=8,
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Column 4: LBP histogram + decision summary box
        ax3 = axes[row_idx, 3]
        lbp_hist = vis["lbp_hist"]
        bins = np.arange(len(lbp_hist))
        ga_is_defective = item["ga_label"] == 1
        bar_color = "#dc2626" if ga_is_defective else "#0f766e"
        border_color = "#b91c1c" if ga_is_defective else "#15803d"

        ax3.bar(bins, lbp_hist, color=bar_color, edgecolor="#1e293b", linewidth=0.6, alpha=0.88)
        ax3.set_xticks(bins)
        y_top = max(0.45, float(lbp_hist.max()) * 1.95)
        ax3.set_ylim(0.0, y_top)
        ax3.set_xlabel("Uniform LBP Bin (0-9)", fontsize=8.5)
        ax3.set_ylabel("Norm. Freq.", fontsize=8.5)
        ax3.grid(axis="y", linestyle=":", alpha=0.5)

        timings = item["timing_ms"]
        box_text = (
            f"Baseline (334/334): {item['baseline_name']} ({item['baseline_margin']:+.3f})\n"
            f"GA-SVM ({item['selected_features']}/334, -{item['feature_reduction_pct']:.1f}%): "
            f"{item['ga_name']} ({item['ga_margin']:+.3f})\n"
            f"Time: {timings['ga_total_ms']:.2f} ms (feat {timings['feature_extraction_ms']:.2f} ms)"
        )
        ax3.text(
            0.03,
            0.96,
            box_text,
            transform=ax3.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#f8fafc",
                "edgecolor": border_color,
                "linewidth": 1.2,
                "alpha": 0.96,
            },
        )
        ax3.set_title(
            f"LBP Histogram & Prediction -> GA: {item['ga_name']}",
            fontsize=9.5,
            fontweight="bold",
            color=border_color,
            pad=8,
        )

    top_margin = 0.80 if n_rows == 1 else 0.90
    fig.subplots_adjust(
        top=top_margin,
        bottom=0.12 if n_rows == 1 else 0.05,
        left=0.04,
        right=0.98,
        wspace=0.25,
        hspace=0.40,
    )
    title_str = (
        suptitle
        or "Solar Cell EL Inference: Baseline (334 dims) vs. GA-SVM (208 dims, margin threshold=0.0)"
    )
    fig.suptitle(title_str, fontsize=12.5, fontweight="bold", y=0.97 if n_rows == 1 else 0.975)
    try:
        fmt = None if isinstance(output, Path) else "png"
        fig.savefig(output, format=fmt, dpi=160, bbox_inches="tight", pad_inches=0.25)
    finally:
        plt.close(fig)
