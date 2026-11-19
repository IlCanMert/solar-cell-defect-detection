# SENG435 — Solar Cell Defect Classification (HOG + LBP + SVM + GA)
## 4 Haftalık Uygulama ve İlerleme Planı

> **Proje Başlığı:** *Intelligent Solar Cell Defect Classification Using Genetic Algorithm-Based Feature Selection*  
> **Ders:** SENG435 — Introduction to Digital Image Processing  
> **Başlangıç Tarihi:** 15 Kasım 2026  
> **Teknik Bitiş Hedefi (4 Hafta):** 11 Aralık 2026 *(Resmî son teslim: 31 Aralık 2026)*  
> **Referans Doküman:** [SENG435_Solar_Cell_Defect_Classification_Proje_Dokumani.md](../SENG435_Solar_Cell_Defect_Classification_Proje_Dokumani.md)

---

## 🎯 Projenin Çekirdek Kapsamı ve Kuralları

Bu plan boyunca **tamamen sadık kalacağımız** temel ilkeler:
1. **Veri Seti (ELPV):** 2.624 adet $300 \times 300$ gri tonlamalı hücre görüntüsünden yalnızca kesin etiketli olanlar (`0.0` = `HEALTHY` [1.508 adet] ve `1.0` = `DEFECTIVE` [715 adet], toplam **2.223 görüntü**) kullanılacaktır. Ara olasılıklar (`0.33`, `0.66`/`0.67`) kapsam dışıdır ve raporda gerekçesiyle açıklanacaktır.
2. **Veri Sızıntısı (Data Leakage) Önlemi:**
   - Veri `%70 Train / %15 Validation / %15 Test` olarak ayrılacak ve `data/splits.csv` dosyasına kaydedilecektir.
   - Modül bilgisi (`module_id`) depoda/üst veride mevcutsa **group-aware split** uygulanacak; yoksa **stratified split** uygulanıp modül benzerliği raporda sınırlılık (limitation) olarak yazılacaktır.
   - `StandardScaler` her zaman `Pipeline` içinde yalnızca eğitim katlarında (`train` / CV fold) fit edilecektir.
   - **Test kümesi** GA özellik seçimi veya hiperparametre ayarında **kesinlikle kullanılmayacaktır**.
3. **Özellik Çıkarımı (Feature Extraction):**
   - **HOG:** $64 \times 64$ yeniden boyutlandırılmış görüntü, `orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, `block_norm='L2-Hys'` $\rightarrow$ **324 özellik**.
   - **LBP:** `P=8`, `R=1`, `method='uniform'` $\rightarrow$ **10 bölmeli (bin)** normalize histogram $\rightarrow$ **10 özellik**.
   - **Toplam:** $324 + 10 = 334$ özellik.
4. **Modeller ve Adil Karşılaştırma:**
   - **E1 (Baseline):** Tüm 334 özellik + `Pipeline(StandardScaler, LinearSVC)`.
   - **E2 (GA-SVM):** GA ile seçilen alt küme + aynı `Pipeline(StandardScaler, LinearSVC)`.
   - **GA Fitness Fonksiyonu:** Yalnızca eğitim verisi üzerinde 3-katlı Stratified CV ile:
     $$\text{fitness}(\text{mask}) = \text{mean\_macro\_F1\_CV}(\text{mask}) - \lambda \cdot \left(\frac{\text{selected\_features}}{\text{total\_features}}\right), \quad \lambda = 0.05$$

---

## 📁 Hedeflenen Proje Klasör Yapısı

İlerledikçe bu çalışma dizininde (`İmage_Proc_Project`) aşağıdaki yapıyı adım adım inşa edeceğiz:

```text
İmage_Proc_Project/
├── PROJE_PLANI.md               # Bu yol haritası ve görev takip dosyası
├── README.md                    # Kurulum, veri indirme ve çalıştırma kılavuzu
├── requirements.txt             # Sabitlenmiş Python paket sürümleri
├── .gitignore                   # Veri, sanal ortam ve geçici dosyalar
├── configs/
│   └── experiment.yaml          # HOG, LBP, SVM, GA ve seed parametreleri
├── data/
│   ├── README.md                # ELPV veri seti lisans (CC BY-NC-SA 4.0) ve bilgi notu
│   └── splits.csv               # Tekrar üretilebilir train/val/test indeksleri
├── src/
│   ├── __init__.py
│   ├── dataset.py               # ELPV yükleme, 0.0/1.0 filtreleme, split oluşturma
│   ├── preprocessing.py         # Görüntü doğrulama ve boyutlandırma (64x64 / 300x300)
│   ├── features.py              # HOG (324) + LBP (10) çıkarımı ve birleştirme
│   ├── baseline.py              # StandardScaler + LinearSVC eğitim ve doğrulama
│   ├── ga_selection.py          # GA kromozom, fitness (3-fold CV + ceza), cache, evrim
│   ├── evaluation.py            # Metrikler, confusion matrix, hata/süre analizi
│   └── predict.py               # Tek görüntü üzerinde Baseline vs GA-SVM tahmini
├── scripts/
│   ├── prepare_data.py          # Veriyi kontrol eder ve data/splits.csv üretir
│   ├── train_baseline.py        # E1 Baseline modelini eğitir ve kaydeder
│   ├── run_ga.py                # E2 GA özellik seçimini çalıştırır ve maskeyi kaydeder
│   ├── evaluate_models.py       # E1, E2 (ve opsiyonel E3-E5) test sonuçlarını üretir
│   └── demo.py                  # Canlı sunum için komut satırı / görsel demo
├── models/                      # baseline.joblib, ga_svm.joblib, selected_mask.npy
├── results/
│   ├── metrics.csv              # Tüm deneylerin karşılaştırmalı sayısal tablosu
│   ├── ga_history.json          # Nesiller boyunca en iyi/ortalama fitness kaydı
│   ├── figures/                 # Confusion matrix, GA yakınsama grafiği, örnek görseller
│   └── sample_predictions/      # Doğru ve hatalı tahmin örnekleri (FP / FN analizi)
├── report/
│   ├── proposal.md              # 20 Kasım teslimi için proje önerisi metni
│   └── final_report_draft.md    # İngilizce akademik rapor taslağı
└── presentation/
    └── slides_outline.md        # Sunum akışı ve demo senaryosu
```

---

## 🗓️ Hafta Hafta Uygulama Planı (Toplam 4 Hafta)

İlerleyişimizi **4 ana faza (haftaya)** ve her haftayı **somut adımlara** böldük. Her adımı tamamladıkça buradaki kutucukları (`[x]`) işaretleyerek ilerleyeceğiz.

---

### 📌 HAFTA 1 (15 Kasım – 20 Kasım 2026)
**Hedef:** Proje İskeleti, ELPV Veri Seti Hazırlığı, Özellik Çıkarımı (HOG+LBP) ve **Proposal Teslimi (20 Kasım)**

#### Adım 1.1 — Ortam Kurulumu ve Proje İskeleti (1–2. Gün)
- [x] Klasör yapısının (`configs/`, `data/`, `src/`, `scripts/`, `models/`, `results/`, `report/`) oluşturulması.
- [x] `requirements.txt` (`numpy`, `scipy`, `scikit-image`, `scikit-learn`, `opencv-python`, `pandas`, `matplotlib`, `joblib`, `pyyaml`, `elpv-dataset`) ve `.gitignore` dosyalarının hazırlanması.
- [x] `configs/experiment.yaml` içinde tüm sabitlerin (`seed: 42`, HOG/LBP parametreleri, split oranları, GA bütçesi) tanımlanması.

#### Adım 1.2 — ELPV Veri Seti İnceleme ve Bölünme (`splits.csv`) (2–3. Gün)
- [x] `elpv-dataset` paketinin kurulması ve `load_dataset()` ile `images`, `proba`, `types` dizilerinin doğrulanması.
- [x] Etiket dağılımının kontrol edilmesi (`0.0`: 1.508, `1.0`: 715 $\rightarrow$ 2.223 görüntü).
- [x] Üst veride modül bilgisi olup olmadığının incelenmesi; uygun stratejiyle `%70 Train / %15 Val / %15 Test` ayrımının yapılıp `data/splits.csv` olarak dondurulması (`src/dataset.py` ve `scripts/prepare_data.py`).
- [x] Veri setinden örnek `HEALTHY` ve `DEFECTIVE` (monokristal/polikristal) görüntülerin rapor için `results/figures/dataset_samples.png` olarak kaydedilmesi.

#### Adım 1.3 — Ön İşleme ve HOG + LBP Özellik Çıkarımı (4–5. Gün)
- [x] `src/preprocessing.py`: Görüntü boyut/tip doğrulaması ve HOG için $64 \times 64$ yeniden boyutlandırma fonksiyonunun yazılması.
- [x] `src/features.py`:
  - HOG özellik çıkarımı ($324$ boyut) ve doğrulama testi.
  - LBP (`P=8, R=1, uniform`) 10-bin normalize histogram çıkarımı ($10$ boyut) ve doğrulama testi.
  - $334$ boyutlu birleşik özellik vektörü üretimi ve örnek bir görüntü üzerinde HOG görselleştirmesi (`results/figures/feature_visualization.png`).

#### Adım 1.4 — Proje Önerisi (Proposal) Teslimi (6. Gün — **Son Tarih: 20 Kasım 2026**)
- [x] `report/proposal.md` ve `report/Project_Proposal_Form.docx` dosyalarının hocanın formuna birebir uygun şekilde hazırlanması *(öğrencinin yalnızca ad/numara alanını doldurup 20 Kasım'a kadar teslim etmesi kaldı)*.

> ✅ **1. Hafta Çıktısı (Milestone 1 — Teknik Olarak Tamamlandı):** Hazır Proposal (`report/proposal.md` & `report/Project_Proposal_Form.docx`) + `data/splits.csv` + 2.223 görüntünün tamamından hatasız $334$ özellik çıkaran çalışan boru hattı.

---

### 📌 HAFTA 2 (21 Kasım – 27 Kasım 2026)
**Hedef:** SVM Baseline Modeli (E1) ve Genetik Algoritma (GA) ile Özellik Seçimi (E2)

#### Adım 2.1 — E1: SVM Baseline Modeli (1–2. Gün)
- [x] `src/baseline.py` ve `scripts/train_baseline.py` dosyalarının yazılması.
- [x] `Pipeline([('scaler', StandardScaler()), ('svm', LinearSVC(random_state=42))])` yapısının kurulması.
- [x] Doğrulama (`validation`) kümesi üzerinde `C` parametresi ve `class_weight` (`None` vs `'balanced'`) etkisinin sızıntısız incelenip temel SVM ayarının sabitlenmesi (`C=0.001`, `class_weight="balanced"`).
- [x] Baseline modelinin eğitilip `models/baseline.joblib` olarak kaydedilmesi; eğitim süresi ve doğrulama F1 skorunun loglanması.

#### Adım 2.2 — E2: Genetik Algoritma (GA) Tasarımı ve Kodlanması (3–5. Gün)
- [x] `src/ga_selection.py` modülünün geliştirilmesi (temiz, bağımlılıksız ve tam kontrol sağlayan Python/NumPy tabanlı GA):
  - **Birey (Chromosome):** $334$ uzunluğunda ikili (binary) maske (`0`/`1`), en az 1 özelliğin seçili olmasını garanti eden onarım (repair) kuralı.
  - **Fitness Fonksiyonu:** Yalnızca `train` kümesi üzerinde `StratifiedKFold(n_splits=3)` ile `Pipeline(StandardScaler, LinearSVC)` eğitip `macro F1` hesaplayan ve $\lambda \cdot (\text{seçilen}/334)$ cezasını düşen fonksiyon.
  - **Fitness Cache:** Aynı bit maskesi tekrar üretildiğinde CV hesabını atlayıp önbellekten döndüren sözlük (`dict`) yapısı.
  - **Operatörler:** Turnuva seçimi (Tournament Selection), iki noktalı çaprazlama (`p_cx = 0.7`), bit-flip mutasyon (`p_mut = 1/334`) ve elitizm (en iyi 2 bireyi koruma).

#### Adım 2.3 — GA Koşumu ve Yakınsama Kaydı (6–7. Gün)
- [x] `scripts/run_ga.py` ile başlangıç bütçesinde (`pop_size=12`, `generations=8`, `lambda=0.05`, `seed=42`) GA'nın çalıştırılması ve sürenin ölçülmesi.
- [x] Nesiller boyunca en iyi fitness, ortalama fitness ve seçilen özellik sayısının `results/ga_history.json` dosyasına kaydedilmesi.
- [x] En iyi özellik maskesinin (`models/selected_mask.npy`) ve bu maskeyle eğitilmiş nihai GA-SVM modelinin (`models/ga_svm.joblib`) kaydedilmesi.

> ✅ **2. Hafta Çıktısı (Milestone 2 — Tamamlandı):** Kayıtlı `baseline.joblib`, kayıtlı `selected_mask.npy` ve `ga_svm.joblib`, ayrıca GA'nın nesiller boyu yakınsama grafiği (`results/figures/ga_convergence.png`).

---

### 📌 HAFTA 3 (28 Kasım – 4 Aralık 2026)
**Hedef:** Kilitli Test Kümesinde Adil Karşılaştırma, Ablation Deneyleri ve Hata Analizi

#### Adım 3.1 — Nihai Test Kümesi Değerlendirmesi (E1 vs E2) (1–2. Gün)
- [x] `src/evaluation.py` ve `scripts/evaluate_models.py` dosyalarının yazılması.
- [x] Şimdiye kadar hiç dokunulmamış **Test Kümesi (%15)** üzerinde E1 (Baseline) ve E2 (GA-SVM) modellerinin tek seferde değerlendirilmesi.
- [x] Tüm zorunlu metriklerin hesaplanması ve `results/metrics.csv` dosyasına yazılması:
  - `Macro F1`, `Defective F1`, `Defective Precision`, `Defective Recall`, `Accuracy`
  - `Selected Feature Count`, `Feature Reduction (%)`
  - `Training / Search Time (s)`, `Single-Image Inference Time (ms)`

#### Adım 3.2 — İstatistiksel Güvenilirlik ve Ablation Çalışmaları (3–4. Gün)
- [x] **Çoklu Tohum (Multi-seed) Kontrolü:** GA'nın 3 farklı seed (`42, 123, 999`) ile çalıştırılıp test Macro F1 ve seçilen özellik sayısı için `Ortalama ± Standart Sapma` hesaplanması.
- [x] **Ek Deneyler (Ablation):**
  - **E3:** Yalnızca HOG (324 özellik) + SVM.
  - **E4:** Yalnızca LBP (10 özellik) + SVM.
  - **E5:** GA'nın seçtiği sayıda **rastgele** özellik (208 özellik, `seed=42` ve 5-seed ortalama) + SVM.

#### Adım 3.3 — Görselleştirme ve Nitel Hata Analizi (5–7. Gün)
- [x] E1 ve E2 için yan yana **Confusion Matrix** çizimi (`results/figures/confusion_matrices.png`).
- [x] Seçilen özelliklerin dağılım analizi: GA 324 HOG özelliğinin kaçını, 10 LBP özelliğinin kaçını seçti? (`results/figures/selected_features_breakdown.png`).
- [x] **Hata Analizi (Error Analysis):** Test kümesinde doğru bilinen (TP, TN) ve yanlış sınıflandırılan (FP — sağlıklıya kusurlu denmesi, FN — kusurlunun kaçırılması) hücre görüntülerinin incelenmesi ve `results/sample_predictions/` altına kaydedilmesi (`error_analysis_grid.png`, `e1_vs_e2_disagreements.png`, `error_analysis_summary.json`).

> ✅ **3. Hafta Çıktısı (Milestone 3):** Doldurulmuş gerçek deney sonuçları tablosu (`metrics.csv`), tüm grafikler (`results/figures/`) ve hata analizi görselleri. [Tamamlandı]

---

### 📌 HAFTA 4 (5 Aralık – 11 Aralık 2026)
**Hedef:** Çalışan Demo Uygulaması, Tekrar Üretilebilirlik Testi, İngilizce Akademik Rapor ve Sunum

#### Adım 4.1 — Tek Görüntü Tahmin ve Canlı Demo (`demo.py`) (1–2. Gün)
- [x] `src/predict.py` ve `scripts/demo.py` dosyalarının tamamlanması.
- [x] Komut satırından bir görüntü yolu veya test örneği verildiğinde:
  1. Görüntüyü okuyup HOG+LBP çıkaran (`load_image_from_path`, `predict_single_image`),
  2. Hem Baseline hem GA-SVM tahminini, karar skorunu (kalibre edilmemiş hiperdüzlem marjini), seçilen özellik sayısını (`208 / 334`, `-37.72%`) ve tahmin süresini ekrana yazdıran,
  3. İsteğe bağlı olarak (`--output`, `--show`) görüntüyü ve tahmin özetini görsel tanı kartına (`demo_showcase.png`, `demo_single_prediction.png`) ve yapısal JSON dosyasına (`demo_output.json`) kaydeden demo akışının hazırlanması.

#### Adım 4.2 — Dokümantasyon ve Temiz Kurulum Testi (3. Gün)
- [x] `README.md` dosyasının sıfırdan bir kullanıcının projeyi kurup tüm deneyleri ve demoyu tek komutla çalıştırabileceği şekilde yazılması.
- [x] Baştan sona bütün scriptlerin (`prepare_data.py` $\rightarrow$ `src.features` $\rightarrow$ `train_baseline.py` $\rightarrow$ `run_ga.py` $\rightarrow$ `evaluate_models.py` $\rightarrow$ `demo.py`) temiz bir akışta test edilmesi (`exit 0`, deterministik metrik doğrulaması).

#### Adım 4.3 — İngilizce Akademik Raporun Yazımı (4–6. Gün)
- [x] `report/final_report_draft.md`, `report/final_report.pdf` ve `report/final_report.docx` (`scripts/generate_report.py` ile otomatik derlenen) içinde kılavuzun istediği tüm bölümlerin **gerçek deney sonuçlarıyla** yazılması:
  1. *Title, Authors, Abstract, Keywords*
  2. *Introduction & Problem Definition*
  3. *Related Work* (Buerhop-Lutz 2018, Deitsch 2019/2021 vb.)
  4. *Dataset & Preprocessing* (ELPV, 2.223 görüntü, veri sızıntısı ve modül tartışması)
  5. *Methodology* (HOG, LBP, LinearSVC, GA kromozom ve fitness formülü)
  6. *Experimental Results & Discussion* (E1–E5 tabloları, confusion matrix, FP/FN görsel analizi, süre/performans ödünleşimi)
  7. *Conclusion & Future Work*
  8. *References, Contribution Statement, AI Tool Use Statement*

#### Adım 4.4 — Sunum ve Canlı Demo Hazırlığı (7. Gün — **11 Aralık 2026**)
- [x] `presentation/slides_outline.md` ile 3–5 dakikalık sunum slaytlarının ve canlı demo senaryosunun hazırlanması.
- [x] `scripts/generate_slides.py` ile tüm grafik/tablo ve demo kartlarını gömen 10 sayfalık `presentation/slides.pdf` sunum PDF'inin derlenip görsel olarak doğrulanması.

> ✅ **4. Hafta Çıktısı (Milestone 4):** Teknik olarak %100 tamamlanmış, çalışan, raporlanmış ve sunuma hazır proje paketi (11 Aralık – 31 Aralık arası yalnızca son okumalar ve resmî teslim için tampon süredir). — **TAMAMLANDI**

---

## ⚡ Acil Durum / Süre Daralması Öncelik Tablosu (Fallback Planı)

Eğer herhangi bir haftada beklenmedik bir gecikme yaşanırsa aşağıdaki öncelik sırası uygulanacaktır:

| Öncelik | Kapsam Kalemi | Karar |
| :--- | :--- | :--- |
| **P0 (Zorunlu Çekirdek)** | ELPV 2.223 görüntü yükleme, `splits.csv`, HOG+LBP (334 özellik), E1 Baseline SVM, E2 GA-SVM, Test karşılaştırması, `demo.py`, İngilizce Rapor | **Kesinlikle yapılacak** |
| **P1 (Yüksek Değerli)** | 3 farklı seed ile GA tekrarı, E3 (HOG-only), E4 (LBP-only), E5 (Random subset) ablation deneyleri | Zaman elverdiğince eklenecek (kod mimarisi buna hazır olacak) |
| **P2 (Opsiyonel)** | RBF çekirdekli SVM veya CLAHE ön işleme denemesi | Yalnızca P0 ve P1 erken biterse |
| **Kapsam Dışı** | Dört sınıflı sınıflandırma, sıfırdan CNN, piksel segmentasyonu, Web/Mobil arayüz | **Yapılmayacak** |

---

## 🚀 Nasıl İlerleyeceğiz?

Bu plan bizim **ana kontrol listemizdir**. Bir sonraki mesajında:
- **"Hafta 1 Adım 1.1 ve 1.2 ile başlayalım"** diyebilirsin; hemen proje klasör yapısını, `requirements.txt`, `configs/experiment.yaml` ve veri yükleme/bölme kodlarını oluşturup çalıştırabiliriz.
- Ya da önce **20 Kasım teslimli Proposal** metnini (`report/proposal.md`) hazır hâle getirmemi isteyebilirsin.

---

## 📋 Adım Adım İlerleme Günlüğü (Yapılanlar ve Gerekçeleri)

### Adım 1.1 — Ortam Kurulumu ve Proje İskeleti
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅
- **Oluşturulan Klasörler:** `configs/`, `data/`, `src/`, `scripts/`, `models/`, `results/`, `results/figures/`, `results/sample_predictions/`, `report/`, `presentation/`
- **Oluşturulan Dosyalar:**
  - `src/__init__.py`
  - `requirements.txt`
  - `.gitignore`
  - `configs/experiment.yaml`
  - `data/README.md`
- **Doğrulanan Ortam ve Paket Sürümleri (Python 3.14.3):**
  - `numpy` `2.4.3`, `scipy` `1.17.1`, `scikit-image` `0.26.0`, `scikit-learn` `1.8.0`
  - `opencv-python` (`cv2`) `5.0.0`, `pandas` `3.0.1`, `matplotlib` `3.10.8`, `joblib` `1.5.3`
  - `pyyaml` `6.0.3`, `pillow` `12.2.0`, `elpv-dataset` `1.0.0.post1` (`load_dataset` import edildi)
  - `configs/experiment.yaml` ayrıştırma doğrulaması geçti (`project: solar-cell-defect-ga`, `seed: 42`, `total_expected_length: 334`).
- **Teknik Gerekçeler:**
  - **Tek Merkezî Yapılandırma (`configs/experiment.yaml`):** HOG ($324$), LBP ($10$), toplam özellik ($334$), `seed: 42`, veri bölme oranları (`0.70 / 0.15 / 0.15`), SVM (`LinearSVC`, `C=1.0`, `class_weight='balanced'`) ve GA bütçesi tek dosyada sabitlenmiştir. Böylece Baseline (E1) ve GA-SVM (E2) deneyleri arasında parametre kayması (parameter drift) yaşanması önlenir ve bilimsel karşılaştırma adil kalır.
  - **Modüler Paket Yapısı (`src/__init__.py` ve `scripts/`):** Çekirdek görüntü işleme/modelleme mantığı (`src/`) ile çalıştırılabilir deney scriptleri (`scripts/`) birbirinden ayrılarak kod tekrarı önlenir ve kılavuzdaki temiz mimari/tekrarlanabilirlik şartı sağlanır.
  - **Bağımlılık ve Sürüm Kontrolü (`requirements.txt`, `.gitignore`):** Projeyi başka bir bilgisayarda veya değerlendirici ortamında tek komutla yeniden kurabilmek için minimum uyumlu sürümler sabitlenmiştir; geçici Python önbellekleri, sanal ortamlar ve ham veri klonları `.gitignore` ile hariç tutulmuştur.
  - **Lisans ve Akademik Atıf Uyumluluğu (`data/README.md`):** ELPV veri setinin `CC BY-NC-SA 4.0` (görüntüler) ve `Apache 2.0` (kod) lisansları, ham veriyi Git'e toplu yüklemek yerine `elpv-dataset` paketiyle programatik yükleme yaklaşımı ve zorunlu akademik atıflar (Buerhop-Lutz ve ark., 2018; Deitsch ve ark., 2019) belgelenmiştir.

### Adım 1.2 — ELPV Veri Seti İnceleme ve Bölünme (`splits.csv`)
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅
- **Oluşturulan / Güncellenen Dosyalar:**
  - `src/dataset.py` (ELPV yükleme, bütünlük kontrolleri, `0.0`/`1.0` filtreleme ve katmanlı bölme fonksiyonları)
  - `scripts/prepare_data.py` (veri hazırlama, istatistik doğrulama ve örnek görsel oluşturma scripti)
  - `data/splits.csv` (2.223 örneğin dondurulmuş `train`/`val`/`test` indeksleri)
  - `results/dataset_summary.json` (doğrulanmış ham ve bölünmüş veri sayıları)
  - `results/figures/dataset_samples.png` (`2 x 4` HEALTHY/DEFECTIVE ve mono/poly örnek gridi)
- **Doğrulanan Veri İstatistikleri:**
  - **Ham ELPV:** 2.624 görüntü, `(2624, 300, 300)`, `uint8`, piksel aralığı `[0, 255]`, NaN yok; monokristal (`mono`) 1.074, polikristal (`poly`) 1.550.
  - **Ham Olasılık Etiketleri:** `0.0` (1.508), `0.3333` (295 — hariç), `0.6667` (106 — hariç), `1.0` (715).
  - **İkili Sınıflandırma Alt Kümesi:** 2.223 görüntü (`HEALTHY`: 1.508, `DEFECTIVE`: 715; 401 belirsiz ara etiket hariç tutuldu).
  - **Modül Üst Verisi Kontrolü:** `elpv_dataset/data/labels.csv` içinde yalnızca `path`, `probability` ve `type` sütunları bulunduğu; `module_id` bilgisinin yer almadığı doğrulandı (`has_module_id: false`).
  - **Dondurulmuş Bölünme (`seed=42`, `label` + `cell_type` ortak katmanlama):**
    - `train`: 1.556 (%70.00) — `HEALTHY`: 1.056, `DEFECTIVE`: 500 (%32.13), `mono`: 631, `poly`: 925
    - `val`: 333 (%14.98) — `HEALTHY`: 226, `DEFECTIVE`: 107 (%32.13), `mono`: 135, `poly`: 198
    - `test`: 334 (%15.02) — `HEALTHY`: 226, `DEFECTIVE`: 108 (%32.34), `mono`: 135, `poly`: 199
- **Teknik Gerekçeler:**
  - **Ara Etiketlerin (`0.33` / `0.67`) Dışarıda Tutulması:** 4 haftalık kapsamda net tanımlı bir ikili (`HEALTHY` vs `DEFECTIVE`) sınıflandırma problemi kurmak için yalnızca kesin uzman etiketleri (`0.0` ve `1.0`) seçilmiştir; hariç tutulan 401 görüntü raporda kapsam sınırı olarak açıklanacaktır.
  - **Ortak Katmanlı Bölme (`label` + `cell_type`):** `labels.csv` dosyasında modül kimliği bulunmadığı için grup bazlı bölme yerine, hem kusur oranının (~%32) hem de kristal türlerinin (`mono`/`poly`) üç kümede eşit dağılmasını sağlayan ortak katmanlı bölme uygulanmıştır; modül kaynaklı benzerlik olasılığı akademik raporda sınırlılık (limitation) olarak belirtilecektir.
  - **Bölünmenin `data/splits.csv` İle Dondurulması:** Tüm gelecek deneylerin (E1 Baseline, E2 GA-SVM ve E3–E5 ablation) birebir aynı `train`/`val`/`test` örneklerini kullanmasını garanti eder ve test kümesine veri sızıntısını önler.

### Adım 1.3 — Ön İşleme ve HOG + LBP Özellik Çıkarımı
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅
- **Oluşturulan / Güncellenen Dosyalar:**
  - `src/preprocessing.py` (gri tonlama/tip doğrulaması ve `cv2.INTER_AREA` ile $64 \times 64$ yeniden boyutlandırma)
  - `src/features.py` (HOG + uniform LBP çıkarımı, matris önbellekleme ve görselleştirme boru hattı)
  - `data/features_hog_lbp.npz` (2.223 görüntü için sıkıştırılmış özellik matrisi ve split/etiket dizileri)
  - `results/feature_extraction_summary.json` (boyut, bütünlük ve süre doğrulama özeti)
  - `results/figures/feature_visualization.png` (`cell0004` HEALTHY ve `cell0001` DEFECTIVE için Orijinal / HOG / LBP Haritası / 10-bin Histogram görselleştirmesi)
- **Doğrulanan Çıktılar:**
  - **Birleşik Özellik Matrisi (`X`):** `(2223, 334)` (`float64`) — `X_hog`: `(2223, 324)`, `X_lbp`: `(2223, 10)`
  - **Sayısal Bütünlük:** `has_nan = false`, `has_inf = false`; tüm satırlarda LBP histogram toplamı `[1.000000, 1.000000]`
  - **Ölçülen Çıkarım Süresi:** 2.223 görüntü için toplam **2.2893 s** (görüntü başına ortalama **1.0298 ms**)
- **Teknik Gerekçeler:**
  - **`cv2.INTER_AREA` ile $64 \times 64$ Ölçekleme:** $300 \times 300$ görüntüden $64 \times 64$ boyutuna küçültürken piksel alanı ortalaması alarak aliasing/harelenme artefaktlarını azaltır ve GA arama uzayını 334 özellikte kompakt tutar.
  - **HOG (`orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, `L2-Hys`):** $64 \times 64$ görüntüde $4 \times 4$ hücre ızgarası üzerinden $3 \times 3$ örtüşen blok oluşturur ($3 \times 3 \times 2 \times 2 \times 9 = 324$ özellik); çatlak ve koyu çizgi kenar yönlerini yakalarken `L2-Hys` blok normalizasyonu hücreler arası parlaklık farklarına karşı dayanıklılık sağlar.
  - **Uniform LBP (`P=8, R=1`, 10-bin normalize histogram):** $P + 2 = 10$ dağılım bölmesi (9 uniform yerel doku deseni + 1 uniform olmayan) üretir ve piksel sayısına bölerek ölçekten bağımsız yoğunluk dağılımı çıkarır.
  - **Özellik Matrisinin `.npz` Olarak Önbelleklenmesi:** Tüm modellerin (E1–E5) birebir aynı deterministik özellik vektörlerini kullanmasını garanti eder ve GA'nın 3-fold CV değerlendirmelerini disk/G-İ tekrarından kurtararak hızlandırır.

### Adım 1.4 — Proje Önerisi (Proposal) Hazırlığı (Hafta 1 Kapanışı / Milestone 1)
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅ *(Yalnızca ad/numara alanının doldurulup 20.11.2026 tarihine kadar teslim edilmesi gerekmektedir)*
- **Oluşturulan Dosyalar:**
  - `report/proposal.md` (kılavuzun 6. sayfasındaki *Project Proposal Form* yapısına birebir uygun İngilizce öneri metni)
  - `report/Project_Proposal_Form.docx` (öğrencinin kendi adını/numarasını yazıp doğrudan PDF olarak dışa aktarabileceği düzenlenebilir Word formu)
- **Teknik Gerekçeler:**
  - **Resmî Kılavuz Formuyla Birebir Uyum:** `image processing course project guide.pdf` (Sayfa 6, Doç. Dr. Tuba ÇAĞLIKANTAR) içindeki *Student information* tablosu ve 6 numaralı zorunlu başlık yapısı (*Problem and objective, Dataset, Planned intelligent technique, Baseline and evaluation plan, Implementation and expected contribution, Work plan and responsibilities*) eksiksiz korunmuştur.
  - **Doğrulanmış Sayıların Doğrudan Kullanılması:** Tahmini sayılar yerine Adım 1.2 ve 1.3'te fiilen doğrulanan sayılar (`2.624` ham, `2.223` ikili alt küme, `1.556 / 333 / 334` split sayıları, `module_id` yokluğu, `324 HOG + 10 LBP = 334` özellik) eklenerek akademik tutarlılık ve güvenilirlik sağlanmıştır.
  - **Hafta 1 Durumu:** **Milestone 1** (Ortam kurulumu, ELPV yükleme/bölme, HOG+LBP boru hattı ve Proposal hazırlığı) %100 tamamlanmıştır.

### Adım 2.1 — E1: SVM Baseline Modeli Eğitimi ve Doğrulaması
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅
- **Oluşturulan / Güncellenen Dosyalar:**
  - `src/baseline.py` (sızıntısız `Pipeline(StandardScaler, LinearSVC)`, CV değerlendirme, validation grid ve model kaydetme)
  - `scripts/train_baseline.py` (`train`/`val` üzerinde baseline grid araması ve sabitlenmiş E1 eğitimi; `test` kümesi kilitli)
  - `configs/experiment.yaml` (`C: 0.001`, `class_weight: "balanced"`, `c_grid`, `class_weight_grid` eklendi/güncellendi)
  - `models/baseline.joblib` (334 özellikli eğitilmiş E1 Baseline modeli)
  - `results/baseline_validation_grid.csv` (10 konfigürasyonlu `C` × `class_weight` validation karşılaştırma tablosu)
  - `results/baseline_validation.json` (sabitlenmiş E1 modelinin CV, eğitim ve doğrulama metrikleri)
- **Doğrulanan Metrikler (`C=0.001`, `class_weight="balanced"`, tüm 334 özellik):**
  - **Train 3-Fold Stratified CV Macro F1:** `0.7333 ± 0.0038`
  - **Validation Macro F1:** `0.7635` (önceki varsayılan `C=1.0` değerinde `0.6848` idi)
  - **Validation Defective F1 / Precision / Recall:** `0.6847` / `0.6609` / `0.7103`
  - **Validation Accuracy:** `0.7898`
  - **Validation Confusion Matrix:** `[[187, 39], [31, 76]]` (`TN=187, FP=39, FN=31, TP=76`)
  - **Eğitim Süresi / Öngörü Hızı:** `0.0757 s` eğitim, görüntü başı `0.00256 ms` doğrulama tahmini
- **Teknik Gerekçeler:**
  - **Sızıntısız `Pipeline` Disiplini:** `StandardScaler` her çapraz doğrulama katında ve nihai eğitimde yalnızca ilgili eğitim verisine `fit` edilerek ölçekleme sızıntısı tamamen önlenmiştir.
  - **Sınıf Dengesizliği ve `class_weight="balanced"`:** `1.508 : 715` sınıf dengesizliği nedeniyle `class_weight=None` modelleri kusurlu hücrelerin neredeyse yarısını kaçırırken (`Val Defective Recall = 0.4860–0.5514`), `balanced` ağırlıklandırma kusurlu hücre yakalama oranını `0.7103` seviyesine çıkarmıştır.
  - **Güçlü L2 Düzenlileştirme (`C=0.001`):** 334 özellik ve 1.556 eğitim örneğinde yüksek `C` (`1.0` veya `10.0`) aşırı uyum (overfitting) yaparak validation başarısını düşürürken (`0.6848` ve `0.6652`), güçlü düzenlileştirme (`C=0.001`) en yüksek genelleme skorunu (`0.7635`) vermiştir. Bu ayar sabitlenerek GA deneylerinde de aynı sınıflandırıcı koşullarının korunması sağlanmıştır.

### Adım 2.2 ve 2.3 — E2: Genetik Algoritma (GA) Tasarımı, Koşumu ve Yakınsama (Hafta 2 Kapanışı / Milestone 2)
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅
- **Oluşturulan Dosyalar:**
  - `src/ga_selection.py` (çeşitli başlangıç popülasyonu, onarım kuralı, `bytes` anahtarlı fitness cache, 3-fold CV `FitnessEvaluator`, turnuva seçimi, iki noktalı çaprazlama, bit-flip mutasyon ve elitizm)
  - `scripts/run_ga.py` (`train` verisinde GA araması, en iyi maskeyle `GA-SVM` eğitimi, `val` kontrolü ve yakınsama grafiği)
  - `models/selected_mask.npy` (334 boyutlu seçilen özellik maskesi)
  - `models/ga_svm.joblib` (208 seçili özellik üzerinde eğitilmiş E2 modeli)
  - `results/ga_history.json` (nesil 0–8 geçmişi, seçilen özellik indeksleri ve metrikler)
  - `results/figures/ga_convergence.png` (nesiller boyu fitness/CV F1 ve seçilen özellik sayısı grafiği)
- **Doğrulanan GA Çıktıları (`pop=12`, `gens=8`, `lambda=0.05`, `seed=42`):**
  - **Seçilen Özellik Sayısı:** **208 / 334** (`%37.72` boyut azaltma) — **HOG: 198 / 324**, **LBP: 10 / 10** (LBP özelliklerinin tamamı GA tarafından korunmuştur)
  - **En İyi 3-Fold Train CV Macro F1:** `0.7434 ± 0.0117` (E1 Baseline Train CV `0.7333` değerinden yüksek)
  - **En İyi Fitness / Ceza:** `0.7123` (`penalty = 0.0311`) — Nesil 0'daki ortalama fitness `0.6895` iken Nesil 8'de `0.7102` seviyesine yükselmiştir
  - **Arama ve Eğitim Süresi:** GA araması **8.1885 s** (`82` benzersiz CV değerlendirmesi, `26` önbellek isabeti), nihai model eğitimi **0.0316 s** (Baseline'dan 2 kattan fazla hızlı)
  - **Validation Kümesi Kontrolü (`333` görüntü, `test` kümesi hâlâ kilitli):** `GA-SVM` Macro F1 = `0.7404` (`Def F1 = 0.6547`, `Def Recall = 0.6822`, `Acc = 0.7688`, CM = `[[183, 43], [34, 73]]`) — Baseline `0.7635` karşısında ilginç ve dürüst bir özellik-sayısı / genelleme ödünleşimi göstermektedir (nihai karar Hafta 3'te kilitli test kümesinde ve çoklu seed deneylerinde ölçülecektir)
- **Teknik Gerekçeler:**
  - **Çeşitli Yoğunluklu Başlangıç (`0.20–0.80`):** Başlangıç popülasyonunu sabit `%50` yerine `%20–%80` aralığına yayarak GA'nın hem çok az hem de orta-yoğunluklu özellik alt kümelerini ilk nesilden itibaren keşfetmesini sağlar.
  - **Sıfır-Maske Onarımı ve Önbellek (`dict`):** Mutasyon sonrası boş maske oluşumunu önler; elitizm ve aynı maskeler tekrarlandığında 3-fold CV hesabını `bytes` anahtarlı önbellekten okuyarak arama süresini %24 civarında kısaltır (`26` cache hit).
  - **Hafta 2 Durumu:** **Milestone 2** (`baseline.joblib`, `selected_mask.npy`, `ga_svm.joblib`, `ga_convergence.png`) %100 tamamlanmıştır.

### Adım 3.1 ve 3.2 — Kilitli Test Kümesi Değerlendirmesi, Çoklu Tohum ve Ablation (E1–E5)
- **Tarih:** 15 Kasım 2026
- **Durum:** Tamamlandı ✅
- **Oluşturulan Dosyalar:**
  - `src/evaluation.py` (sınıflandırıcı tahmin/gecikme ölçümü, E3–E5 ablation ve çoklu tohum GA değerlendirme modülü)
  - `scripts/evaluate_models.py` (kilitli 334 test görüntüsü üzerinde tüm deneyleri tek seferde çalıştıran script)
  - `results/metrics.csv` (E1–E5 tam karşılaştırma tablosu)
  - `results/ga_multiseed_runs.csv` ve `results/ga_multiseed_summary.json` (`seeds=[42, 123, 999]` ayrıntıları ve ortalama ± standart sapma)
  - `results/evaluation_summary.json` (karışıklık matrisleri ve hata analizi için test tahmin/skor kayıtları)
- **Doğrulanan Kilitli Test Kümesi Sonuçları (`334` görüntü: `226` HEALTHY, `108` DEFECTIVE):**
  - **E1 — Baseline (334 özellik):** `Test Macro F1 = 0.7258`, `Def F1 = 0.6372`, `Def Prec = 0.6102`, `Def Recall = 0.6667`, `Accuracy = 0.7545`, CM = `[[180, 46], [36, 72]]`, eğitim `0.0757 s`, sınıflandırıcı öngörü `0.00127 ms` (`toplam: 1.03107 ms/görüntü`).
  - **E2 — GA-SVM (`seed=42`, 208 özellik, `-%37.72`):** `Test Macro F1 = 0.7325`, `Def F1 = 0.6460`, `Def Prec = 0.6186`, `Def Recall = 0.6759`, `Accuracy = 0.7605`, CM = `[[181, 45], [35, 73]]`, GA arama `8.1885 s` / eğitim `0.0316 s`, sınıflandırıcı öngörü `0.00088 ms` (`toplam: 1.03068 ms/görüntü`).
  - **E2 — Çoklu Tohum GA (`seeds=[42, 123, 999]`, Ort ± SS):**
    - Seçilen Özellik: `192.67 ± 32.81` (`%42.31 ± %9.82` azaltma; `seed=999` yalnızca `155` özellikle `-%53.59` indirim sağladı)
    - Train 3-Fold CV Macro F1: `0.7471 ± 0.0043`
    - Test Macro F1: `0.7232 ± 0.0081` (`0.7325`, `0.7179`, `0.7191`)
    - Test Defective Recall: `0.6605 ± 0.0142` | Test Accuracy: `0.7525 ± 0.0069`
    - GA Arama Süresi: `7.2631 ± 1.7875 s`
  - **E3 — Yalnızca HOG (324 özellik):** `Test Macro F1 = 0.7045`, `Def Recall = 0.6296`, `Accuracy = 0.7365`.
  - **E4 — Yalnızca LBP (10 özellik):** `Test Macro F1 = 0.6949`, `Def Recall = 0.7037`, `Accuracy = 0.7156`.
  - **E5 — Rastgele 208 Özellik (`seed=42` / 5-seed ortalama):** `Test Macro F1 = 0.7188` (`seed=42`) ve `0.7151` (`5-seed ort.`), `Accuracy = 0.7455` / `0.7425`.
- **Teknik Gerekçeler ve Bilimsel Bulgular:**
  - **Ana Araştırma Sorusu Yanıtı:** `E2_seed42` modeli `%37.72` daha az özellikle (`208/334`) kilitli test kümesinde `E1` Baseline'a kıyasla hem daha yüksek Macro F1 (`0.7325` vs `0.7258`) hem daha yüksek Defective Recall (`0.6759` vs `0.6667`) elde etmiş; 3 tohum ortalamasında ise ortalama **%42.31** daha az özellik (`192.67`) ile Baseline ile pratikte eşdeğer test başarımı (`0.7232 ± 0.0081`) korumuştur.
  - **HOG ve LBP Tamamlayıcılığı (E3 & E4):** Tek başına HOG (`0.7045`) ve tek başına LBP (`0.6949`), ikisinin birleşiminin (`0.7258` / `0.7325`) altında kalarak kenar ve doku bilgisinin birlikte gerekliliğini doğrulamıştır.
  - **GA vs. Rastgele Boyut Azaltma (E5):** Aynı özellik sayısında (`208`) rastgele seçim yalnızca `0.7151` (5-seed ort.) ve `0.7188` (`seed=42`) F1 skoruna ulaşırken GA'nın `0.7325` (`seed=42`) ve `0.7232` (3-seed ort.) elde etmesi, GA'nın rastgele elemeden üstün, anlamlı özellik alt kümeleri seçtiğini kanıtlamıştır.

### Hafta 3 — Adım 3.3: Görselleştirme, Özellik Dağılımı ve Nitel Hata Analizi Tamamlandı
- **Yapılan İşlemler:**
  - `src/evaluation.py` içerisine `save_confusion_matrices_figure`, `save_selected_features_breakdown_figure` ve `run_error_analysis_and_save_samples` fonksiyonları eklendi; `scripts/evaluate_models.py` dosyasına `--plots-only` bayrağı ve tam değerlendirme sonu otomatik görselleştirme/hata analizi akışı entegre edildi.
  - `python scripts/evaluate_models.py --plots-only` çalıştırıldı (`exit 0`) ve tüm çıktı grafikleri (`read_file` ile görsel olarak doğrulandı):
    - `results/figures/confusion_matrices.png`: E1 (`TN=180 [%79.6], FP=46 [%20.4], FN=36 [%33.3], TP=72 [%66.7]`) ve E2 (`TN=181 [%80.1], FP=45 [%19.9], FN=35 [%32.4], TP=73 [%67.6]`) kilitli test confusion matrix karşılaştırması.
    - `results/figures/selected_features_breakdown.png`: GA (`seed=42`) modalite dağılımı (**HOG `198/324` [%61.1 korundu]**, **Uniform LBP `10/10` [%100.0 korundu]**, **Toplam `208/334` [%37.72 azaltma]**) ve E1–E5 kilitli test Macro F1 / Defective Recall karşılaştırması.
    - `results/sample_predictions/error_analysis_grid.png` (ayrıca `results/figures/error_analysis_grid.png` kopyası): E2 için seçilmiş `TP` (`#470`, `#467`, `#1614`, `#1979`), `TN` (`#1180`, `#832`, `#1873`, `#1880`), `FP` (`#8`, `#937`, `#123`, `#1709`) ve `FN` (`#850`, `#2070`, `#1550`, `#2155`) nitel örnek ızgarası.
    - `results/sample_predictions/e1_vs_e2_disagreements.png` ve `results/sample_predictions/error_analysis_summary.json`: E1 vs E2 test ayrışmaları ve `mono`/`poly` alt grup istatistikleri.
- **Nitel ve Alt Grup Hata Analizi Bulguları:**
  - **Kristal Tipi Kırılımı (`mono` vs. `poly`):** E2 test kümesinde `mono` hücrelerde (`n=135`) hata oranı `%17.78` (`24` hata: `FP=11, FN=13`), `poly` hücrelerde (`n=199`) ise `%28.14` (`56` hata: `FP=34, FN=22`) olmuştur. GA özellik seçimi özellikle `poly` hücrelerdeki hata sayısını E1'e kıyasla `62`'den (`%31.16`) `56`'ya (`%28.14`) düşürmüştür.
  - **E1 vs. E2 Ayrışması:** 334 test örneğinin `20`'sinde iki model farklı tahmin üretmiş; E2, E1'in yaptığı **11** hatayı düzeltirken **9** örnekte gerilemiş (net `+2` doğru tahmin, toplam hata `82 -> 80`).
  - **Görsel Başarı ve Hata Örüntüleri:** E2 belirgin çatlakları (`#470`, `#467`) ve tamamen karanlık/ölü hücreleri (`#1614`, `#1979`) yüksek güvenle (`+1.64` ile `+2.69` karar skoru) yakalamıştır. False Positive hataları ağırlıklı olarak `poly` kristal tane/dislokasyon sınırlarının (`#123`, `#1709`) ve kenar kararmasının (`#8`, `#937`) kusura benzemesinden; False Negative hataları ise ince/lokal kenar mikro-çatlaklarının (`#2070`) ve düşük kontrastlı hataların (`#850`, `#1550`, `#2155`) `16x16` HOG + global LBP özniteliklerinde sönümlenmesinden kaynaklanmaktadır.
- **Milestone 3 Tamamlandı:** Hafta 3 kapsamındaki tüm nicel deneyler, çoklu tohum analizleri, grafikler ve nitel hata analizi eksiksiz tamamlandı. Sıradaki adım: **Hafta 4 — Adım 4.1 (`src/predict.py` ve `scripts/demo.py`)**.

### Hafta 4 — Adım 4.1: Tek Görüntü Tahmin Modülü ve Canlı Demo Tamamlandı
- **Yapılan İşlemler:**
  - `src/predict.py` oluşturuldu: `load_inference_bundle`, Unicode Windows dosya yollarında `cv2.imread` hatasını önlemek için `np.fromfile` + `cv2.imdecode` kullanan `load_image_from_path`, HOG (`324`) + Uniform LBP (`10`) özniteliklerini çıkarıp Baseline (`334/334`) ve GA-SVM (`208/334`, `-37.72%`, HOG `198/324` + LBP `10/10`) modellerini zamanlayarak çalıştıran `predict_single_image`, ve `N x 4` görsel tanı kartları (Orijinal EL, `64x64` HOG gradyanı, `64x64` Uniform LBP haritası, LBP histogramı + model karar kutusu) üreten `save_prediction_card` eklendi.
  - `scripts/demo.py` oluşturuldu: Varsayılan vitrin modu (kilitli test kümesinden `#470`, `#1180`, `#1576` örnekleri), `--image <path>` tekil dosya/ELPV göreli yol modu, `--subset-index <int>` split indeks modu, `--output` ve Windows görüntüleyici destekli `--show` bayrağı eklendi. Terminal çıktısında ve grafiklerde `LinearSVC` skorlarının kalibre edilmiş olasılık değil, işaretli hiperdüzlem marjini (`threshold = 0.0`) olduğu açıkça belirtildi.
- **Doğrulama Komutları ve Gözlenen Sonuçlar:**
  - `python scripts/demo.py` (`exit 0`):
    - **`#470` (`images/cell0527.png`, `mono`, True=`DEFECTIVE`):** Baseline `DEFECTIVE` (`margin=+2.1236`), GA-SVM `DEFECTIVE` (`margin=+1.6783`), toplam `5.852 ms` (`feat=5.514 ms`).
    - **`#1180` (`images/cell1356.png`, `mono`, True=`HEALTHY`):** Baseline `HEALTHY` (`margin=-1.3478`), GA-SVM `HEALTHY` (`margin=-1.2470`), toplam `1.508 ms` (`feat=1.196 ms`).
    - **`#1576` (`images/cell1857.png`, `poly`, True=`HEALTHY`):** Baseline yanlış alarm `DEFECTIVE` (`margin=+0.0135`), GA-SVM doğru düzeltme `HEALTHY` (`margin=-0.2085`), toplam `2.074 ms` (`feat=1.604 ms`).
  - `python scripts/demo.py --image images/cell0527.png` (`exit 0`) ve `python scripts/demo.py --subset-index 1576` (`exit 0`) çalıştırılarak tekil komut satırı modları doğrulandı.
  - Kaydedilen çıktılar (`results/sample_predictions/demo_showcase.png`, `results/sample_predictions/demo_single_prediction.png`, `results/sample_predictions/demo_output.json`) okunup görsel olarak incelendi; marjin değerlerinin `evaluation_summary.json` ile birebir uyumlu olduğu doğrulandı.
- **Teknik Gerekçe:** Sunum ve teslim sırasında hem tek bir görüntü yoluyla canlı tahmin gösterimi hem de Baseline'ın polikristal dokuda yaptığı yanlış alarmı GA-SVM'in düzelttiği somut örneğin (`#1576`) görsel öznitelik haritalarıyla birlikte izlenebilmesi sağlandı. Sıradaki adım: **Adım 4.2 (`README.md` ve uçtan uca temiz kurulum/tekrar üretilebilirlik testi)**.

### Hafta 4 — Adım 4.2: Dokümantasyon (`README.md`) ve Uçtan Uca Tekrar Üretilebilirlik Testi Tamamlandı
- **Yapılan İşlemler:**
  - Proje kök dizininde kapsamlı `README.md` oluşturuldu: problem tanımı ve araştırma sorusu, HOG (`324`) + Uniform LBP (`10`) + `LinearSVC` (`C=0.001, balanced`) + GA fitness formülü, güncel depo yapısı, Python `>=3.10` / doğrulanmış `3.14.3` kurulum adımları, ELPV ikili alt küme ve veri sızıntısı önlemleri (`has_module_id=false` sınırlaması ve lisans bilgileri dahil), adım adım + tek komut (PowerShell ve Bash) çalıştırma yönergeleri, `scripts/demo.py` kullanım örnekleri ve E1–E5 kilitli test sonuç tablosu dokümante edildi.
  - Baştan sona bütün pipeline (`python scripts/prepare_data.py` $\rightarrow$ `python -m src.features` $\rightarrow$ `python scripts/train_baseline.py` $\rightarrow$ `python scripts/run_ga.py` $\rightarrow$ `python scripts/evaluate_models.py` $\rightarrow$ `python scripts/demo.py`) sırayla ve tek zincirde çalıştırılarak test edildi (`exit 0`).
- **Doğrulama ve Tekrar Üretilebilirlik Bulguları:**
  - Tüm veriler (`data/splits.csv`, `data/features_hog_lbp.npz` — `2.8554 s`, `1.2845 ms/image`), modeller (`models/baseline.joblib`, `models/ga_svm.joblib`, `models/selected_mask.npy`), metrikler (`results/metrics.csv`), çoklu tohum sonuçları (`seeds=[42, 123, 999]`), grafikler ve demo kartları sıfırdan yeniden üretildi.
  - `results/metrics.csv` okunarak tüm sınıflandırma metriklerinin, seçilen özellik sayılarının (`E1=334`, `E2_seed42=208`, `E2_3seed_mean=192.6667`, `E3=324`, `E4=10`, `E5=208`), test Macro F1 skorlarının (`0.7258`, `0.7325`, `0.7232`, `0.7045`, `0.6949`, `0.7188`, `0.7151`) ve E2/E1 hata sayıların (`TP=73, TN=181, FP=45, FN=35`, ayrışma `20`: `11` düzeltme / `9` gerileme) önceki koşumlarla **tam deterministik biçimde birebir aynı** kaldığı doğrulandı (yalnızca duvar saati işlem süreleri doğal olarak birkaç milisaniye/saniye oynamıştır).
- **Teknik Gerekçe:** Projeyi sıfırdan devralacak veya değerlendirecek bir kişinin (`hocamız` / ekip üyesi) tek doküman üzerinden kurulumu tamamlayıp tüm deneyleri deterministik biçimde baştan üretmesi ve canlı demoyu çalıştırabilmesi garanti altına alındı. Sıradaki adım: **Adım 4.3 (İngilizce Akademik Raporun Yazımı: `report/final_report_draft.md`)**.

### Hafta 4 — Adım 4.3: İngilizce Akademik Rapor (`final_report.pdf`, `.docx`, `.md`) Tamamlandı
- **Yapılan İşlemler:**
  - Ders kılavuzunun 8. ve 11. bölümlerindeki tüm akademik gereksinimleri karşılayan İngilizce makale taslağı (`report/final_report_draft.md`) yazıldı: Title/Author, Abstract, Keywords, 1. Introduction, 2. Related Work, 3. Dataset & Preprocessing, 4. Methodology (HOG+LBP, Pipeline, Validation Grid, GA Fitness), 5. Experimental Results & Discussion (E1–E5, 3-seed stability, Confusion Matrices, mono/poly kırılımı, nitel TP/TN/FP/FN hata analizi, maliyet/süre analizi), 6. Conclusion & Limitations, 7. Doğrulanmış 9 Akademik Referans (`Buerhop-Lutz 2018`, `Deitsch 2019/2021`, `Dalal & Triggs 2005`, `Ojala 2002`, `Cortes & Vapnik 1995`, `Siedlecki & Sklansky 1989`, `van der Walt 2014`, `Pedregosa 2011`), 8. Contribution Statement ve 9. AI Tool Use Statement.
  - Raporu doğrudan doğrulanmış `results/*.json` ve `results/*.csv` dosyalarından otomatik derleyip tüm 5 tabloyu ve 7 şekli gömen `scripts/generate_report.py` scripti (`reportlab 5.1.0` + `python-docx`, isteğe bağlı `--author` ve `--student-id` bayraklı) yazıldı ve `requirements.txt` uyumlu PDF çıktı oluşturuldu (`python scripts/generate_report.py`, `exit 0`).
  - Üretilen 10 sayfalık `report/final_report.pdf` (`5.483.725` bayt) ve `report/final_report.docx` (`3.317.991` bayt) görsel olarak sayfalar hâlinde okunarak doğrulandı; ilk derleme kontrolünde tespit edilen `∈` eksik glif kutusu, Tablo 4/5 başlık sıkışması ve Şekil 2 hücre etiketi (`cell0004` / `cell0001`) düzeltilip PDF yeniden derlenerek temizliği onaylandı.
- **Teknik Gerekçe:** Rapor metninin yalnızca kâğıt üzerinde statik kalması yerine doğrudan gerçek deney çıktı dosyalarından derlenmesi, hem hiçbir sayısal tutarsızlık riskinin kalmamasını hem de öğrencinin yalnızca `--author` ve `--student-id` argümanlarını girerek saniyeler içinde teslimata hazır PDF üretebilmesini sağladı. Sıradaki son adım: **Adım 4.4 (`presentation/slides_outline.md` ve sunum / canlı demo hazırlığı)**.

### Hafta 4 — Adım 4.4 ve Milestone 4 Tamamlandı
- **Yapılan İşlemler:**
  - `presentation/slides_outline.md` oluşturularak 4 dk 30 sn (3–5 dk penceresine uygun) 10 slaytlık akış planı, zamanlama çizelgesi, slayt bazlı konuşma notları, PowerShell canlı demo komut senaryosu (`python scripts/demo.py`, `--image images/cell0527.png`, `--subset-index 1576 --show`) ve 7 maddelik jüri/hoca Q&A savunma rehberi (`401` belirsiz etiket dışlaması, `has_module_id=false` açıklaması, `C=0.001 balanced` düzenlileştirme seçimi, `10/10` LBP korunumu, `LinearSVC` kalibresiz hiper-düzlem marjinleri, `seed=999` vs `seed=42` parsimony ödünleşimi ve `poly` hata baskınlığı) belgelendi.
  - `scripts/generate_slides.py` (`reportlab.lib.pagesizes.landscape(A4)`, `--author` ve `--student-id` argümanlı) yazıldı; `scripts/generate_report.py`'deki `_load_verified_artifacts`, `_register_fonts`, `_scaled_rl_image` ve `_styled_table` yardımcıları ile `configs/experiment.yaml` ve `results/sample_predictions/demo_output.json` doğrudan bağlanarak tüm metrikler ve 7 şekil (`dataset_samples.png`, `feature_visualization.png`, `ga_convergence.png`, `selected_features_breakdown.png`, `confusion_matrices.png`, `error_analysis_grid.png`, `demo_showcase.png`) iki sütunlu 10 sayfalık `presentation/slides.pdf` (`5.354.791` bayt) içine derlendi (`python scripts/generate_slides.py`, `exit 0`).
  - `presentation/slides.pdf` tüm 10 sayfasıyla görsel olarak incelendi; ilk derlemede Slayt 2/4/7/8/9 tablolarında yakalanan `&times;` entity metni, `Mean ± Std` sütun çakışması ve hücre taşmaları giderilip yeniden derlenerek 10 sayfanın tamamının temiz, taşmasız ve okunabilir olduğu doğrulandı.
  - `README.md` depo ağacı ve uçtan uca çalıştırma komutları `generate_report.py`, `generate_slides.py`, `report/final_report.*` ve `presentation/slides.*` çıktılarını içerecek şekilde güncellendi.
- **Teknik Gerekçe:** 4 haftalık SENG435 proje planındaki tüm adımlar (Veri & Sızıntısız Bölümleme -> HOG+LBP Özellik Çıkarımı & Baseline E1 -> GA Wrapper Seçimi E2 & Çoklu Tohum/Ablasyon E3–E5 -> Hata Analizi & Görselleştirme -> Canlı CLI Demo -> Akademik Rapor PDF/DOCX -> 10 Slaytlık Sunum PDF + Canlı Demo/Q&A Rehberi) **%100 tamamlandı (Milestone 4 Kapanışı)**. Teslim öncesi yalnızca `--author` ve `--student-id` bayrakları ile rapor ve slayt PDF'lerini kişisel bilgilerle yeniden derlemek yeterlidir.

### Ek Özellik — İnteraktif Flask Web Arayüzü (`scripts/app.py`) Eklendi
- **Yapılan İşlemler:**
  - `flask` (`3.1.3`) kullanılarak tek komutla açılan (`python scripts/app.py` -> `http://127.0.0.1:5000`), üç sekmeli modern bir web muayene arayüzü (`scripts/app.py`) geliştirildi:
    1. **Canlı Hücre Muayene Stüdyosu:** Sunum örnek vakaları (`#470`, `#1180`, `#1576` GA düzeltmesi, `#237`, `#8`, `#850`), `n=334` kilitli test kümesi alt grup filtresi/rastgele seçim gezgini, sürükle-bırak harici görüntü yükleme (`np.frombuffer` + `cv2.imdecode` ile Unicode güvenli), E1 Baseline vs. E2 GA-SVM hiper-düzlem marjin göstergeleri (`threshold=0.0` uyarısıyla), 4 panelli görüntü işleme görselleştirmesi (Orijinal, HOG `198/324`, Viridis LBP `10/10`, etkileşimli 10-bin LBP histogramı) ve indirilebilir `results/sample_predictions/ui_last_prediction.png` tanı kartı.
    2. **Deneysel Sonuçlar & GA Kararlılık Analizi:** `results/metrics.csv` (E1–E5), `results/ga_multiseed_summary.json` (seeds `42, 123, 999`) ve `mono` vs `poly` hata kırılım tabloları.
    3. **Proje Grafikleri Galerisi:** Tüm sonuç/hata analizi grafiklerinin tam ekran büyütmeli galerisi.
  - `requirements.txt` dosyasına `flask>=3.0.0`, `reportlab>=4.0.0` ve `python-docx>=1.1.0` eklendi; `README.md` güncellendi.
  - `python scripts/app.py --smoke-test` çalıştırılarak tüm uç noktaların (`GET /`, `/api/overview`, `/api/samples`, `/api/predict` `#1576`/`#470`/dosya yükleme ve `/api/figure/...`) **7/7 PASS (exit code 0)** geçtiği doğrulandı.
- **Teknik Gerekçe:** Komut satırı (`demo.py`) yerine görsel, sürükle-bırak destekli ve jüri sunumlarında tek tıkla vakalar arası geçiş sağlayan etkileşimli bir tarayıcı arayüzü sunularak hem son kullanıcı deneyimi hem de sunum kalitesi yükseltildi.
