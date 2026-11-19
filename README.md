# Intelligent Solar Cell Defect Classification Using Genetic Algorithm-Based Feature Selection

A classical image processing and evolutionary feature-selection pipeline for binary defect detection (**HEALTHY** vs. **DEFECTIVE**) in electroluminescence (EL) photovoltaic solar cell imagery (**SENG435 — Digital Image Processing** course project).

---

## 1. Project Overview & Research Question

Electroluminescence (EL) imaging reveals micro-cracks, inactive/dead areas, finger interruptions, and crystal defects that are invisible in standard RGB photographs. While high-dimensional local descriptors capture rich structural and textural cues, irrelevant or redundant feature dimensions increase classifier complexity and degrade generalization under polycrystalline background noise.

**Research Question:**  
*Can Genetic Algorithm (GA) feature selection over combined HOG (shape/edge) and Uniform LBP (texture) features reduce feature dimensionality relative to an all-feature Linear SVM baseline while maintaining or improving locked test-set defect classification performance?*

### Methodology Summary
- **Image Representation:** `300 × 300` 8-bit grayscale EL cell images resized via `cv2.INTER_AREA` to `64 × 64`.
- **Feature Extraction (`334` total features):**
  - **HOG (Histogram of Oriented Gradients):** `orientations=9`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)`, `block_norm="L2-Hys"` → **`324`** edge/gradient dimensions.
  - **Uniform LBP (Local Binary Patterns):** `P=8`, `R=1`, `method="uniform"`, 10 normalized histogram bins → **`10`** micro-texture dimensions.
- **Classifier Pipeline:** `StandardScaler` + `LinearSVC(C=0.001, class_weight="balanced", max_iter=5000)`, fitted strictly inside `sklearn.pipeline.Pipeline` on training/CV folds to prevent scaling leakage.
- **Wrapper Genetic Algorithm:** Binary chromosome $m \in \{0, 1\}^{334}$ evolved with tournament selection ($k=3$), two-point crossover ($p=0.7$), bit-flip mutation ($p \approx 1/334 = 0.002994$), and elitism ($2$) over `12` individuals × `8` generations using 3-fold stratified CV on `train` only:
  $$\text{Fitness}(m) = F_{1,\text{CV}}(m) - \lambda \cdot \frac{\|m\|_1}{334}, \qquad \lambda = 0.05$$

---

## 2. Repository Structure

```text
İmage_Proc_Project/
├── configs/
│   └── experiment.yaml             # Locked hyperparameters, paths, and random seeds
├── data/
│   ├── README.md                   # ELPV dataset licensing and citation guidelines
│   ├── splits.csv                  # Frozen stratified 70/15/15 split indices (2,223 rows)
│   └── features_hog_lbp.npz        # Cached (2223, 334) HOG+LBP feature matrix
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # ELPV loader, binary filter, stratified split inspection
│   ├── preprocessing.py            # Grayscale normalization and INTER_AREA resizing
│   ├── features.py                 # HOG (324) + Uniform LBP (10) extraction & visualization
│   ├── baseline.py                 # E1 StandardScaler + LinearSVC validation search & training
│   ├── ga_selection.py             # Wrapper GA bit-mask evolution & convergence tracking
│   ├── evaluation.py               # Locked test E1–E5 evaluation, plots, and error analysis
│   └── predict.py                  # Single-image inference & 4-panel diagnostic cards
├── scripts/
│   ├── prepare_data.py             # Step 1: Filter ELPV & freeze stratified data/splits.csv
│   ├── train_baseline.py           # Step 3: Tune & fit E1 Baseline SVM
│   ├── run_ga.py                   # Step 4: Run E2 GA feature selection & fit GA-SVM
│   ├── evaluate_models.py          # Step 5: Evaluate E1–E5 + multi-seed GA + figures/errors
│   ├── demo.py                     # Step 6: Live CLI single-image & showcase inference
│   ├── app.py                      # Interactive Flask Web Dashboard (http://127.0.0.1:5000)
│   ├── generate_report.py          # Step 7: Build academic PDF/DOCX final report
│   └── generate_slides.py          # Step 8: Build 10-slide landscape presentation PDF
├── models/
│   ├── baseline.joblib             # Fitted E1 pipeline (334 features)
│   ├── ga_svm.joblib               # Fitted E2 pipeline (208 GA-selected features)
│   └── selected_mask.npy           # Boolean mask (length 334) from GA seed=42
├── results/
│   ├── metrics.csv                 # Master E1–E5 comparison table
│   ├── evaluation_summary.json     # Detailed test metrics, confusion matrices, predictions
│   ├── ga_multiseed_runs.csv       # Multi-seed GA stability table (seeds 42, 123, 999)
│   ├── ga_multiseed_summary.json   # Mean +/- std aggregation across GA seeds
│   ├── figures/                    # Dataset, feature, GA convergence, CM, breakdown, error grids
│   └── sample_predictions/         # Error analysis summary, disagreements, demo diagnostic cards
├── report/
│   ├── proposal.md                 # Course project proposal (Markdown source)
│   ├── Project_Proposal_Form.docx  # Filled SENG435 project proposal form (Word format)
│   ├── final_report_draft.md       # Academic final report Markdown draft
│   ├── final_report.pdf            # Compiled 10-page academic report (ReportLab)
│   └── final_report.docx           # Compiled academic report (Word format)
├── presentation/
│   ├── slides_outline.md           # 10-slide speaker notes, live demo script & Q&A guide
│   └── slides.pdf                  # Compiled 10-slide landscape presentation deck (ReportLab)
├── PROJE_PLANI.md                  # 4-week project plan & step-by-step progress log
├── requirements.txt
└── README.md
```

---

## 3. Environment & Installation

Tested and verified on **Python 3.14.3** (Windows) with **Python >= 3.10** supported.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The results were reproduced on a CPU using Python 3.14.3 on Windows with these package versions.

| Package | Version | Package | Version |
| --- | --- | --- | --- |
| numpy | 2.4.3 | scipy | 1.17.1 |
| scikit-image | 0.26.0 | scikit-learn | 1.8.0 |
| opencv-python | 5.0.0.93 | pandas | 3.0.1 |
| matplotlib | 3.10.8 | joblib | 1.5.3 |
| pyyaml | 6.0.3 | elpv-dataset | 1.0.0.post1 |
| pillow | 12.2.0 | flask | 3.1.3 |
| reportlab | 5.1.0 | python-docx | 1.2.0 |

---

## 4. Dataset and Experimental Splits

- **Dataset:** Public **ELPV Dataset** loaded programmatically via `elpv_dataset.utils.load_dataset` (raw images are not re-distributed in Git; see [data/README.md](data/README.md)).
- **Binary Subset Filtering:** From `2,624` raw `300 × 300` grayscale EL cell images, uncertain defect probabilities (`0.3333` [295] and `0.6667` [106], total `401`) are excluded. Experiments use the remaining **`2,223`** high-confidence images:
  - `0.0` (`HEALTHY`): **`1,508`** (`67.84%`)
  - `1.0` (`DEFECTIVE`): **`715`** (`32.16%`)
- **Stratified Split (`data/splits.csv`):** Frozen `70% / 15% / 15%` joint stratification on `(label, cell_type)`:
  - `train`: `1,556` | `val`: `333` | `test`: `334` (`226` Healthy, `108` Defective; `135` mono, `199` poly)
- **Known Limitation (`module_id`):** Programmatic inspection (`results/dataset_summary.json`, `has_module_id=false`) confirmed that the bundled `elpv-dataset` metadata contains no PV-module identifier (`module_id`), preventing module-grouped splitting. Cell-level joint stratification across `(label, cell_type)` is applied and documented as a limitation.
- **Model fitting and evaluation.** `StandardScaler` is fit exclusively on `train` / CV training folds inside `Pipeline`; Baseline hyperparameter search uses `val`; GA fitness uses 3-fold stratified CV on `train` only; `test` (`n = 334`) is reserved for evaluation. This protocol measures performance on held-out cells. Cells from one PV module may appear in more than one partition.

---

## 5. End-to-End Execution

### Step-by-Step Pipeline Commands

```bash
# 1. Inspect ELPV dataset, filter binary subset (2,223), freeze train/val/test splits
python scripts/prepare_data.py

# 2. Extract 334 HOG+LBP features across all 2,223 images & visualize sample features
python -m src.features

# 3. Validate & train E1 Baseline Pipeline (StandardScaler + LinearSVC)
python scripts/train_baseline.py

# 4. Run Wrapper Genetic Algorithm (E2, seed=42) & save selected_mask.npy + ga_svm.joblib
python scripts/run_ga.py

# 5. Evaluate E1-E5 on locked test set, run 3-seed GA stability, generate figures & error analysis
python scripts/evaluate_models.py

#    (Optional fast re-plotting from saved JSON/CSV without retraining)
python scripts/evaluate_models.py --plots-only

# 6. Run live showcase & save diagnostic cards
python scripts/demo.py

# 7. Compile academic final report & 10-slide presentation deck (pass your name/ID)
python scripts/generate_report.py --author "Mertcan İLHAN" --student-id "22050911018"
python scripts/generate_slides.py --author "Mertcan İLHAN" --student-id "22050911018"
```

### Single-Command Full Reproducibility Run

**Windows PowerShell:**
```powershell
python scripts/prepare_data.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; python -m src.features; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; python scripts/train_baseline.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; python scripts/run_ga.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; python scripts/evaluate_models.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; python scripts/demo.py
```

**Linux / macOS Bash:**
```bash
python scripts/prepare_data.py && python -m src.features && python scripts/train_baseline.py && python scripts/run_ga.py && python scripts/evaluate_models.py && python scripts/demo.py
```

---

## 6. Live Demo (`scripts/demo.py`) & Interactive Web UI (`scripts/app.py`)

### Interactive Web Dashboard (`scripts/app.py`)
```powershell
# Launch the interactive Flask Web UI (opens http://127.0.0.1:5000 automatically)
python scripts/app.py

# Or specify custom port without auto-opening a browser tab
python scripts/app.py --port 5050 --no-browser

# Run automated 7-endpoint verification smoke test
python scripts/app.py --smoke-test
```

The Turkish interface opens on **Hücreyi incele**. Select **Örnek hücreyi incele** to compare both predictions with the known label of ELPV cell #1576. Use **EL görüntüsü yükle** for your own single-cell EL image. **Başka bir örnek seç** opens more examples, including a missed defect and all 334 test cells.

Below the predictions, **Karar puanı, süre ve görüntü özellikleri** explains scores and shows the HOG and LBP views. **Deney sonuçları** explains Macro F1, defective recall, accuracy, and feature reduction using the saved comparison. **Görseller** opens the experimental plots.

### CLI Demo (`scripts/demo.py`)

```bash
# Default 3-sample locked-test showcase (#470 TP, #1180 TN, #1576 GA poly fix)
python scripts/demo.py

# Predict on a specific ELPV relative path or any external image file path
python scripts/demo.py --image scripts/static/cell1857.png

# Predict by subset_index in data/splits.csv and save to a custom PNG path
python scripts/demo.py --subset-index 1576 --output results/sample_predictions/cell1576.png

# Optionally open the saved PNG after rendering (Windows image viewer)
python scripts/demo.py --subset-index 1576 --show
```

> **Decision scores.** LinearSVC reports an uncalibrated decision score. A score above zero indicates a defective prediction. A score at or below zero indicates a healthy prediction.

---

## 7. Locked Test Experimental Results (`E1`–`E5`, `n = 334`)

Exact locked test metrics recorded in [results/metrics.csv](results/metrics.csv):

| Experiment ID | Model / Feature Subset | Selected Features | Feature Reduction | Train CV Macro F1 | Val Macro F1 | **Test Macro F1** | Test Defective F1 | Test Defective Recall | **Test Accuracy** | Search / Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Baseline SVM (All HOG+LBP) | `334` (`324` HOG, `10` LBP) | `0.00%` | `0.7333` | `0.7635` | **`0.7258`** | `0.6372` | `0.6667` | **`0.7545`** | `0.1055 s` fit |
| **E2 (`seed=42`)** | **GA-SVM (`seed=42`)** | **`208`** (`198` HOG, `10` LBP) | **`37.72%`** | `0.7434` | `0.7404` | **`0.7325`** | **`0.6460`** | **`0.6759`** | **`0.7605`** | `8.1770 s` GA / `0.0618 s` fit |
| **E2 (3-Seed Mean)** | GA-SVM (`seeds=[42,123,999]`) | `192.67` (`± 32.81`) | `42.31%` | `0.7471` | `0.7391` | `0.7232` (`± 0.0081`) | `0.6331` | `0.6605` | `0.7525` | `6.2938 s` avg GA |
| **E3** | HOG-Only + SVM | `324` (`324` HOG, `0` LBP) | `2.99%` | `0.7286` | `0.7105` | `0.7045` | `0.6071` | `0.6296` | `0.7365` | `0.0722 s` fit |
| **E4** | LBP-Only + SVM | `10` (`0` HOG, `10` LBP) | `97.01%` | `0.7021` | `0.7035` | `0.6949` | `0.6154` | `0.7037` | `0.7156` | `0.0024 s` fit |
| **E5 (`seed=42`)** | Random Subset (`208`, `seed=42`) | `208` (`201` HOG, `7` LBP) | `37.72%` | `0.7394` | `0.7517` | `0.7188` | `0.6320` | `0.6759` | `0.7455` | `0.0353 s` fit |
| **E5 (5-Seed Mean)** | Random Subset (`208`, 5-seed mean) | `208` (`202.0` HOG, `6.0` LBP) | `37.72%` | `0.7246` | `0.7353` | `0.7151` | `0.6266` | `0.6685` | `0.7425` | `0.0376 s` fit |

### Key Findings
- **Feature reduction on the cell-level test split.** Across seeds 42, 123, and 999, GA used 42.31% fewer features on average and yielded Macro F1 of 0.7232 ± 0.0081 versus 0.7258 for the full-feature baseline. Seed 42 retained 208 features and made two fewer errors than the baseline. The other two GA runs scored below the baseline.
- **Complementarity of Edge & Texture (`E3`, `E4`):** Combining HOG and LBP outperforms both HOG-only (`0.7045`) and LBP-only (`0.6949`).
- **Random feature selection (E5).** At 208 features, GA seed 42 scored 0.7325 on the test cells. Five random subsets of the same size averaged 0.7151.
- **Subgroup errors.** On the test split, seed 42 GA made 56 polycrystalline errors versus 62 for the baseline, and 24 monocrystalline errors versus 20 for the baseline.

---

## 8. References & Licensing

- **ELPV Dataset License:** Images **CC BY-NC-SA 4.0**; `elpv-dataset` loader code **Apache 2.0**.
- **Dataset Citations:**
  1. Buerhop-Lutz, C., Deitsch, S., Maier, A., Gallwitz, F., Berger, S., Doll, B., Hauch, J., Camus, C., & Brabec, C. J. (2018). *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery*. EU PVSEC.
  2. Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., & Riess, C. (2019). *Automatic classification of defective photovoltaic module cells in electroluminescence images*. Solar Energy, 185, 455–468.
