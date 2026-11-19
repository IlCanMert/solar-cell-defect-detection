"""Dataset loading, binary filtering, and reproducible stratified splitting for ELPV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from elpv_dataset import utils as elpv_utils
from sklearn.model_selection import train_test_split


LABEL_NAMES: dict[int, str] = {
    0: "HEALTHY",
    1: "DEFECTIVE",
}


def load_config(config_path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    """Load the YAML experiment configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _get_elpv_data_dir() -> Path:
    """Return the data directory bundled inside the installed elpv_dataset package."""
    return Path(elpv_utils.__file__).resolve().parent / "data"


def inspect_elpv_metadata() -> dict[str, Any]:
    """Inspect bundled ELPV metadata files to check whether module_id mapping exists."""
    data_dir = _get_elpv_data_dir()
    labels_path = data_dir / "labels.csv"
    files_in_data = sorted(p.name for p in data_dir.iterdir())
    sample_lines: list[str] = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as handle:
            for _ in range(3):
                line = handle.readline()
                if not line:
                    break
                sample_lines.append(line.strip())

    # labels.csv contains 3 whitespace-separated fields: path, probability, type (no module_id)
    has_module_id = False
    return {
        "data_dir": str(data_dir),
        "entries": files_in_data,
        "labels_file": str(labels_path),
        "sample_rows": sample_lines,
        "has_module_id": has_module_id,
    }


def load_raw_elpv() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load raw ELPV images, defect probabilities, crystal types, and relative image paths."""
    images, proba, types = elpv_utils.load_dataset()

    labels_path = _get_elpv_data_dir() / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"ELPV labels.csv not found at: {labels_path}")

    raw_records = np.genfromtxt(
        str(labels_path),
        dtype=["|S32", "<f8", "|S8"],
        names=["path", "probability", "type"],
    )
    image_paths = [p.decode("utf-8").strip() for p in raw_records["path"]]

    # Validate array integrity
    if images.ndim != 3 or images.shape[1:] != (300, 300):
        raise ValueError(f"Unexpected ELPV image shape: {images.shape}, expected (N, 300, 300)")
    if images.dtype != np.uint8:
        raise ValueError(f"Unexpected ELPV image dtype: {images.dtype}, expected uint8")
    if not (len(images) == len(proba) == len(types) == len(image_paths)):
        raise ValueError("Mismatch in lengths across images, proba, types, and image_paths")
    if np.isnan(images).any() or np.isnan(proba).any():
        raise ValueError("NaN values detected in ELPV images or probability labels")

    types_str = np.asarray([str(t).strip() for t in types], dtype=object)
    return images, proba, types_str, image_paths


def filter_binary_subset(
    images: np.ndarray,
    proba: np.ndarray,
    types: np.ndarray,
    paths: list[str],
    healthy_val: float = 0.0,
    defective_val: float = 1.0,
) -> dict[str, Any]:
    """Filter the ELPV dataset to definite binary labels (0.0 -> 0 HEALTHY, 1.0 -> 1 DEFECTIVE)."""
    is_healthy = np.isclose(proba, healthy_val, atol=1e-8)
    is_defective = np.isclose(proba, defective_val, atol=1e-8)
    keep_mask = is_healthy | is_defective

    original_indices = np.flatnonzero(keep_mask)
    filtered_images = images[keep_mask]
    filtered_proba = proba[keep_mask]
    filtered_types = np.asarray(types)[keep_mask]
    filtered_paths = [paths[idx] for idx in original_indices]
    binary_labels = np.where(np.isclose(filtered_proba, defective_val, atol=1e-8), 1, 0).astype(np.int64)

    return {
        "images": filtered_images,
        "labels": binary_labels,
        "proba": filtered_proba,
        "types": filtered_types,
        "paths": filtered_paths,
        "original_indices": original_indices,
    }


def build_subset_dataframe(subset: dict[str, Any]) -> pd.DataFrame:
    """Construct a metadata DataFrame from the filtered binary subset dictionary."""
    labels = subset["labels"]
    return pd.DataFrame(
        {
            "subset_index": np.arange(len(labels), dtype=np.int64),
            "original_index": subset["original_indices"].astype(np.int64),
            "image_path": subset["paths"],
            "proba": subset["proba"].astype(float),
            "label": labels.astype(np.int64),
            "label_name": [LABEL_NAMES[int(lbl)] for lbl in labels],
            "cell_type": [str(t) for t in subset["types"]],
        }
    )


def create_or_load_splits(
    df_meta: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    splits_path: str | Path = "data/splits.csv",
    force_recreate: bool = False,
) -> pd.DataFrame:
    """Create reproducible stratified train/val/test splits or load an existing splits CSV."""
    path = Path(splits_path)
    if path.exists() and not force_recreate:
        existing = pd.read_csv(path)
        if len(existing) != len(df_meta):
            raise ValueError(
                f"Existing splits file {path} has {len(existing)} rows, expected {len(df_meta)}"
            )
        return existing

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    strat_key = df_meta["label"].astype(str) + "_" + df_meta["cell_type"].astype(str)
    all_indices = df_meta.index.to_numpy()

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=(1.0 - train_ratio),
        stratify=strat_key.loc[all_indices],
        random_state=seed,
    )

    val_share_of_temp = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_share_of_temp),
        stratify=strat_key.loc[temp_idx],
        random_state=seed,
    )

    split_labels = pd.Series(index=df_meta.index, dtype="object")
    split_labels.loc[train_idx] = "train"
    split_labels.loc[val_idx] = "val"
    split_labels.loc[test_idx] = "test"

    if split_labels.isna().any():
        raise RuntimeError("Some samples were not assigned to any split")
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise RuntimeError("Overlap detected between train, val, and test splits")

    df_splits = df_meta.copy()
    df_splits["split"] = split_labels
    columns_order = [
        "subset_index",
        "original_index",
        "image_path",
        "proba",
        "label",
        "label_name",
        "cell_type",
        "split",
    ]
    df_splits = df_splits[columns_order]

    path.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(path, index=False)
    return df_splits
