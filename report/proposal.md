# Project Proposal Form

**Course:** SENG435 — Introduction to Digital Image Processing  
**Deadline:** 20.11.2026  

| Student information | Details |
|---|---|
| **Name(s) and student ID(s)** | `Mertcan İLHAN — 22050911018` |
| **Project title** | **Intelligent Solar Cell Defect Classification Using Genetic Algorithm-Based Feature Selection** |
| **Submission date** | 20.11.2026 |

---

## 1 Problem and objective

This project addresses automated binary defect classification (`HEALTHY` vs. `DEFECTIVE`) of individual photovoltaic (PV) solar cells using `300 x 300` grayscale electroluminescence (EL) images. Structural micro-cracks, inactive dark cell regions, and material degradation in solar cells are often invisible under ordinary RGB photography, whereas EL imaging reveals electrical and structural anomalies clearly. However, manual visual inspection of thousands of EL cell images by human experts is slow, repetitive, and prone to subjectivity.

The core objective is to build a transparent, reproducible image classification pipeline that extracts complementary shape/edge and local texture features from EL cell images and investigates whether a Genetic Algorithm (GA) can select a compact subset of those features while maintaining or improving classification performance compared to using the entire feature vector. Given an input EL solar cell image, the working application will output: (1) the predicted class label (`HEALTHY` or `DEFECTIVE`), (2) the classifier decision score, (3) a side-by-side comparison between the full-feature Baseline SVM and the GA-SVM model, and (4) quantitative evaluation summaries including confusion matrices, retained feature ratio, and runtime measurements.

---

## 2 Dataset

This project uses the publicly available **ELPV (*Electroluminescence Photovoltaic*)** dataset (`https://github.com/zae-bayern/elpv-dataset`; Jülich DATA DOI: `10.26165/JUELICH-DATA/GCBNMA`), released for academic use under the `CC BY-NC-SA 4.0` license (with loader code under `Apache 2.0`). The full dataset contains **2,624** 8-bit grayscale EL images (`300 x 300` pixels) collected from 44 solar modules, comprising `1,074` monocrystalline (`mono`) and `1,550` polycrystalline (`poly`) cells labeled with expert-estimated defect probabilities (`0.0`, `0.3333`, `0.6667`, and `1.0`).

To establish a well-defined binary ground truth within the four-week project scope, the main experiment uses only the definitely labeled images: **1,508** `HEALTHY` (`0.0`) and **715** `DEFECTIVE` (`1.0`), totaling **2,223 images** (`901` monocrystalline and `1,322` polycrystalline). The `401` intermediate uncertainty samples (`295` at `0.3333` and `106` at `0.6667`) are excluded from the binary study and will be explicitly reported as a scope limitation.

Inspection of the bundled `labels.csv` metadata confirmed that `module_id` annotations are not included (`path`, `probability`, and `type` only). Therefore, the 2,223 binary-labeled images have been partitioned using **joint stratification on both class label and crystal type (`mono`/`poly`)** with fixed `seed = 42` (`data/splits.csv`), and the potential similarity of cells from the same PV module will be transparently acknowledged as a limitation in the final report:

- **Training split (70.00%):** 1,556 images (`1,056` healthy, `500` defective; `631` mono, `925` poly)
- **Validation split (14.98%):** 333 images (`226` healthy, `107` defective; `135` mono, `198` poly)
- **Held-out Test split (15.02%):** 334 images (`226` healthy, `108` defective; `135` mono, `199` poly)

---

## 3 Planned intelligent technique

The system combines two complementary intelligent techniques:

1. **Genetic Algorithm (GA) for Feature Selection (Primary Optimization Technique):**  
   Each EL image is first resized to `64 x 64` pixels using area resampling (`cv2.INTER_AREA`) and represented by **334 handcrafted descriptors**: **324 Histogram of Oriented Gradients (HOG)** features (`orientations=9`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)`, `block_norm="L2-Hys"`) capturing directional cracks and edge gradients, concatenated with **10 normalized uniform Local Binary Pattern (LBP)** histogram features (`P=8`, `R=1`, `method="uniform"`) capturing fine surface texture. Not all 334 features contribute equally across monocrystalline and polycrystalline cells, and redundant dimensions can degrade generalization or slow down inference. Each GA chromosome encodes a candidate feature subset as a 334-bit binary mask $m \in \{0,1\}^{334}$ (repaired if all zeros). Fitness is computed exclusively on the training split using stratified 3-fold cross-validation by balancing classification quality against feature count:

   $$\text{fitness}(m) = \text{mean\_macro\_F1}_{\text{CV}}(m) - \lambda \cdot \left(\frac{\|m\|_1}{334}\right), \quad \lambda = 0.05$$

   The evolutionary search uses tournament selection, two-point crossover ($p_c = 0.7$), bit-flip mutation ($p_m \approx 1/334$), elitism (preserving the top two candidates), and a fitness cache under a bounded budget (initial setup: population size `12`, `8` generations).

2. **Linear Support Vector Machine (`LinearSVC`) Classifier:**  
   The `Pipeline(StandardScaler, LinearSVC)` fits scaling parameters within each training fold. The linear SVM uses balanced class weights for the 1,508 healthy and 715 defective samples. Its fast training supports the repeated cross-validation runs during GA search.

---

## 4 Baseline and evaluation plan

- **Baseline Method (E1 — Full-Feature SVM):** A `Pipeline(StandardScaler, LinearSVC)` model trained on all **334** HOG + LBP features without GA feature selection.
- **Proposed Method (E2 — GA-SVM):** The exact same `Pipeline(StandardScaler, LinearSVC)` classifier trained only on the GA-selected feature subset.
- **Additional Ablations (E3–E5, time permitting):** HOG-only (`324` features), LBP-only (`10` features), and a random feature subset matching the GA subset size to compare GA with arbitrary dimensionality reduction.
- **Metrics & Visual Evaluation:** Because the dataset is class-imbalanced (`32.16%` defective), **Macro F1-score** is chosen as the primary evaluation metric alongside defective-class Precision, Recall, and F1-score, overall Accuracy, Confusion Matrices, selected feature count, percentage dimensionality reduction, GA search/training time, and single-image inference time. Qualitative visual evaluation will inspect True Positive, True Negative, False Positive, and False Negative EL predictions.
- **Fairness & Leakage Prevention:** Both Baseline and GA-SVM use identical frozen splits (`data/splits.csv`), identical preprocessing and feature extraction code, identical `StandardScaler` scaling fitted strictly inside training folds, and identical classifier settings. The `test` split (`334` images) remains strictly locked until final evaluation and is never accessed by the GA fitness function or validation tuning. Any negative result (e.g., if GA does not outperform the full-feature baseline) will be reported honestly as a scientific cost–benefit finding.

---

## 5 Implementation and expected contribution

- **Tools and Libraries:** Python 3.11+ (`3.14` verified), `numpy`, `scipy`, `opencv-python`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib`, `joblib`, `pyyaml`, and `elpv-dataset`.
- **Hardware:** Standard CPU laptop (no GPU, paid cloud API, or external camera hardware required).
- **Deliverables and Custom Implementation:** A modular Python project (`src/` for dataset loading, preprocessing, HOG+LBP feature extraction, baseline SVM, custom GA feature selection with caching, evaluation, and prediction; `scripts/` for reproducible execution and a CLI single-image demonstration `demo.py`), frozen split indices (`data/splits.csv`), saved model artifacts (`models/*.joblib`, `models/selected_mask.npy`), evaluation tables/figures (`results/`), presentation slides, and an English academic report in PDF format.
- **Expected Contribution:** A clean, fully reproducible EL solar-cell classification pipeline and a rigorous empirical analysis of the trade-off between classification quality (Macro F1 / Defective Recall), feature dimensionality reduction, and computational cost.

---

## 6 Work plan and responsibilities

- **Week 1 (15–20 Nov 2026):** Project skeleton and configuration setup, ELPV dataset audit and frozen stratified split (`data/splits.csv`), `64 x 64` preprocessing, `334`-dimensional HOG + LBP feature extraction pipeline, and Project Proposal submission.
- **Week 2 (21–27 Nov 2026):** Validation and freezing of the `LinearSVC` Baseline (`E1`), implementation of the binary-mask GA with 3-fold CV fitness and caching, and execution of GA feature selection (`E2`).
- **Week 3 (28 Nov–4 Dec 2026):** Single-shot evaluation on the locked test set (`E1` vs. `E2`), ablation experiments (`E3–E5`) and multi-seed stability checks, confusion matrix plotting, and qualitative FP/FN error analysis.
- **Week 4 (5–11 Dec 2026; final course deadline: 31 Dec 2026):** Interactive CLI single-image demo (`scripts/demo.py`), clean installation and reproducibility verification, English academic PDF report writing, and presentation slide/demo preparation.
- **Responsibilities:** Individual project — the student is responsible for all implementation, experiment execution, analysis, academic reporting, and live demonstration (if conducted as a 2-student team, explicit per-module contributions will be documented in the final report).

---

*Assoc. Prof. Tuba ÇAĞLIKANTAR*  
*Image Processing | Course Project Guide*
