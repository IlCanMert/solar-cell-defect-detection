"""Genetic Algorithm (GA) wrapper for combinatorial HOG + LBP feature selection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.baseline import build_svm_pipeline, compute_classification_metrics


def repair_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ensure a binary chromosome is 1D uint8 and selects at least one feature."""
    arr = (np.asarray(mask, dtype=np.uint8).ravel() > 0).astype(np.uint8)
    if arr.size == 0:
        raise ValueError("Feature mask cannot be empty")
    if int(arr.sum()) == 0:
        arr[int(rng.integers(0, arr.size))] = 1
    return arr


def initialize_population(
    pop_size: int,
    n_features: int,
    rng: np.random.Generator,
    min_density: float = 0.20,
    max_density: float = 0.80,
) -> list[np.ndarray]:
    """Initialize a diverse population of binary masks spanning [min_density, max_density]."""
    if pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {pop_size}")
    if pop_size == 1:
        densities = np.array([0.5 * (min_density + max_density)])
    else:
        densities = np.linspace(min_density, max_density, pop_size)

    population: list[np.ndarray] = []
    for density in densities:
        raw = (rng.random(n_features) < float(density)).astype(np.uint8)
        population.append(repair_mask(raw, rng))
    return population


class FitnessEvaluator:
    """Evaluates binary feature masks using training-only 3-fold Stratified CV and sparsity penalty."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        c_val: float = 0.001,
        class_weight: str | None = "balanced",
        max_iter: int = 5000,
        cv_folds: int = 3,
        lambda_penalty: float = 0.05,
        seed: int = 42,
        hog_dim: int = 324,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.n_features = int(x_train.shape[1])
        self.c_val = float(c_val)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.cv_folds = int(cv_folds)
        self.lambda_penalty = float(lambda_penalty)
        self.seed = int(seed)
        self.hog_dim = int(hog_dim)

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        self.folds = list(skf.split(x_train, y_train))

        self.cache: dict[bytes, dict[str, Any]] = {}
        self.n_evaluations = 0
        self.n_cache_hits = 0

    def evaluate(self, mask: np.ndarray) -> dict[str, Any]:
        """Return cached or freshly computed CV metrics and penalized fitness for a feature mask."""
        mask_bin = (np.asarray(mask, dtype=np.uint8).ravel() > 0).astype(np.uint8)
        selected_count = int(mask_bin.sum())
        if selected_count == 0:
            raise ValueError("Cannot evaluate an all-zero feature mask")

        key = mask_bin.tobytes()
        if key in self.cache:
            self.n_cache_hits += 1
            return self.cache[key]

        self.n_evaluations += 1
        bool_mask = mask_bin.astype(bool)
        x_sub = self.x_train[:, bool_mask]

        macro_f1_scores: list[float] = []
        defective_f1_scores: list[float] = []
        defective_recall_scores: list[float] = []
        accuracy_scores: list[float] = []

        for tr_idx, va_idx in self.folds:
            pipe = build_svm_pipeline(
                c_val=self.c_val,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            pipe.fit(x_sub[tr_idx], self.y_train[tr_idx])
            pred = pipe.predict(x_sub[va_idx])
            m = compute_classification_metrics(self.y_train[va_idx], pred)
            macro_f1_scores.append(m["macro_f1"])
            defective_f1_scores.append(m["defective_f1"])
            defective_recall_scores.append(m["defective_recall"])
            accuracy_scores.append(m["accuracy"])

        cv_macro_f1_mean = float(np.mean(macro_f1_scores))
        cv_macro_f1_std = float(np.std(macro_f1_scores))
        hog_selected = int(mask_bin[: self.hog_dim].sum())
        lbp_selected = int(mask_bin[self.hog_dim :].sum())
        feature_ratio = float(selected_count) / float(self.n_features)
        penalty = self.lambda_penalty * feature_ratio
        fitness = cv_macro_f1_mean - penalty

        result = {
            "fitness": float(fitness),
            "cv_macro_f1_mean": cv_macro_f1_mean,
            "cv_macro_f1_std": cv_macro_f1_std,
            "cv_defective_f1_mean": float(np.mean(defective_f1_scores)),
            "cv_defective_recall_mean": float(np.mean(defective_recall_scores)),
            "cv_accuracy_mean": float(np.mean(accuracy_scores)),
            "selected_count": selected_count,
            "hog_selected": hog_selected,
            "lbp_selected": lbp_selected,
            "feature_ratio": float(feature_ratio),
            "penalty": float(penalty),
        }
        self.cache[key] = result
        return result


def tournament_selection(
    population: list[np.ndarray],
    evals: list[dict[str, Any]],
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select one parent mask via deterministic-tiebroken tournament selection."""
    k = min(int(tournament_size), len(population))
    candidates = rng.choice(len(population), size=k, replace=False)
    winner_idx = max(
        (int(i) for i in candidates),
        key=lambda idx: (evals[idx]["fitness"], -evals[idx]["selected_count"]),
    )
    return population[winner_idx].copy()


def two_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform two-point crossover between two binary parent masks."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    n_features = len(parent1)
    if n_features >= 3 and rng.random() < float(crossover_prob):
        pt1, pt2 = sorted(rng.choice(np.arange(1, n_features), size=2, replace=False).tolist())
        child1[pt1:pt2] = parent2[pt1:pt2]
        child2[pt1:pt2] = parent1[pt1:pt2]
    return child1, child2


def bit_flip_mutation(
    mask: np.ndarray,
    mutation_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip each bit independently with probability mutation_prob and repair empty masks."""
    mutated = mask.copy()
    flips = rng.random(len(mutated)) < float(mutation_prob)
    mutated[flips] = 1 - mutated[flips]
    return repair_mask(mutated, rng)


def run_genetic_algorithm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run evolutionary feature selection on training data only and return best mask & history."""
    ga_cfg = cfg["ga"]
    svm_cfg = cfg["baseline_svm"]
    hog_dim = int(cfg["features"]["hog"]["expected_length"])

    run_seed = int(ga_cfg["random_state"]) if seed is None else int(seed)
    pop_size = int(ga_cfg["population_size"])
    generations = int(ga_cfg["generations"])
    crossover_prob = float(ga_cfg["crossover_prob"])
    mutation_prob = float(ga_cfg["mutation_prob"])
    elitism_count = min(int(ga_cfg["elitism_count"]), pop_size)
    tournament_size = int(ga_cfg["tournament_size"])
    cv_folds = int(ga_cfg["cv_folds"])
    lambda_penalty = float(ga_cfg["lambda_penalty"])

    rng = np.random.default_rng(run_seed)
    n_features = int(x_train.shape[1])

    evaluator = FitnessEvaluator(
        x_train=x_train,
        y_train=y_train,
        c_val=float(svm_cfg["C"]),
        class_weight=svm_cfg["class_weight"],
        max_iter=int(svm_cfg["max_iter"]),
        cv_folds=cv_folds,
        lambda_penalty=lambda_penalty,
        seed=run_seed,
        hog_dim=hog_dim,
    )

    t_start = time.perf_counter()
    population = initialize_population(pop_size=pop_size, n_features=n_features, rng=rng)
    evals = [evaluator.evaluate(ind) for ind in population]

    best_mask: np.ndarray | None = None
    best_eval: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    def _record_generation(gen_idx: int) -> None:
        nonlocal best_mask, best_eval
        ranked_indices = sorted(
            range(len(population)),
            key=lambda i: (evals[i]["fitness"], -evals[i]["selected_count"]),
            reverse=True,
        )
        gen_best_idx = ranked_indices[0]
        gen_best_eval = evals[gen_best_idx]
        gen_best_mask = population[gen_best_idx]

        if best_eval is None or (
            gen_best_eval["fitness"],
            -gen_best_eval["selected_count"],
        ) > (best_eval["fitness"], -best_eval["selected_count"]):
            best_eval = dict(gen_best_eval)
            best_mask = gen_best_mask.copy()

        fitness_vals = [e["fitness"] for e in evals]
        selected_counts = [e["selected_count"] for e in evals]
        elapsed_s = time.perf_counter() - t_start

        entry = {
            "generation": int(gen_idx),
            "best_fitness": float(best_eval["fitness"]),
            "gen_best_fitness": float(gen_best_eval["fitness"]),
            "mean_fitness": float(np.mean(fitness_vals)),
            "std_fitness": float(np.std(fitness_vals)),
            "best_cv_macro_f1": float(best_eval["cv_macro_f1_mean"]),
            "best_cv_macro_f1_std": float(best_eval["cv_macro_f1_std"]),
            "best_selected_count": int(best_eval["selected_count"]),
            "best_hog_selected": int(best_eval["hog_selected"]),
            "best_lbp_selected": int(best_eval["lbp_selected"]),
            "mean_selected_count": float(np.mean(selected_counts)),
            "cumulative_evals": int(evaluator.n_evaluations),
            "cumulative_cache_hits": int(evaluator.n_cache_hits),
            "elapsed_s": round(float(elapsed_s), 4),
        }
        history.append(entry)

        if verbose:
            print(
                f"Gen {gen_idx:>2}/{generations} | "
                f"best_fit={entry['best_fitness']:.4f} | "
                f"mean_fit={entry['mean_fitness']:.4f} | "
                f"best_CV_F1={entry['best_cv_macro_f1']:.4f} | "
                f"feat={entry['best_selected_count']:>3}/{n_features} "
                f"(HOG:{entry['best_hog_selected']}, LBP:{entry['best_lbp_selected']}) | "
                f"evals={entry['cumulative_evals']} hits={entry['cumulative_cache_hits']} | "
                f"t={entry['elapsed_s']:.2f}s"
            )

    _record_generation(0)

    for gen in range(1, generations + 1):
        ranked_indices = sorted(
            range(len(population)),
            key=lambda i: (evals[i]["fitness"], -evals[i]["selected_count"]),
            reverse=True,
        )
        next_population = [population[i].copy() for i in ranked_indices[:elitism_count]]

        while len(next_population) < pop_size:
            p1 = tournament_selection(population, evals, tournament_size, rng)
            p2 = tournament_selection(population, evals, tournament_size, rng)
            c1, c2 = two_point_crossover(p1, p2, crossover_prob, rng)
            next_population.append(bit_flip_mutation(c1, mutation_prob, rng))
            if len(next_population) < pop_size:
                next_population.append(bit_flip_mutation(c2, mutation_prob, rng))

        population = next_population
        evals = [evaluator.evaluate(ind) for ind in population]
        _record_generation(gen)

    search_time_s = time.perf_counter() - t_start
    assert best_mask is not None and best_eval is not None

    return {
        "seed": run_seed,
        "best_mask": best_mask,
        "best_eval": best_eval,
        "history": history,
        "search_time_s": float(search_time_s),
        "n_evaluations": int(evaluator.n_evaluations),
        "n_cache_hits": int(evaluator.n_cache_hits),
    }
