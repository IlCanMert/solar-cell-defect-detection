"use strict";

const byId = id => document.getElementById(id);
const number = (value, digits = 0) => Number(value).toLocaleString("tr-TR", {
  minimumFractionDigits: digits,
  maximumFractionDigits: digits,
});
const percent = (value, digits = 1) => "%" + number(value * 100, digits);
const label = value => value === 1 ? "Kusurlu" : "Sağlıklı";
const cellType = { mono: "Monokristal", poly: "Polikristal" };
const imageSource = value => "data:image/png;base64," + value;
const sampleSelect = byId("sampleSelect");
const fileInput = byId("fileInput");
const result = byId("result");
const status = byId("predictionStatus");
const navigation = document.querySelectorAll("nav a");
const pages = document.querySelectorAll("[data-page]");
let samples = [];
let busy = false;

function showPage() {
  const page = ["incele", "deney", "gorseller"].includes(location.hash.slice(1))
    ? location.hash.slice(1)
    : "incele";
  for (const section of pages) section.hidden = section.id !== page;
  for (const link of navigation) {
    if (link.hash === "#" + page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (response.status === 413) throw new Error("Görüntü 16 MB sınırını aşıyor. Daha küçük bir dosya seç.");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "İstek tamamlanamadı.");
  return data;
}

function setPredictionControls() {
  for (const control of document.querySelectorAll("[data-predict-control]")) {
    control.disabled = busy;
  }
  if (!samples.length) {
    sampleSelect.disabled = true;
    byId("sampleButton").disabled = true;
    byId("randomButton").disabled = true;
  }
}

function renderHistogram(bins) {
  const chart = byId("histogram");
  chart.replaceChildren();
  chart.setAttribute("aria-label", "LBP örüntüleri ve oranları. " +
    bins.map((value, index) => "Örüntü " + index + " " + percent(value)).join(", "));
  for (const [index, value] of bins.entries()) {
    const bar = document.createElement("span");
    const fill = document.createElement("i");
    fill.style.height = (value * 100) + "%";
    bar.title = "Örüntü " + index + " · " + percent(value);
    bar.append(fill, String(index));
    chart.append(bar);
  }
}

function renderModel(prefix, predicted, reference, count) {
  const prediction = byId(prefix + "Label");
  const match = byId(prefix + "Match");
  prediction.textContent = label(predicted);
  prediction.dataset.label = predicted;
  match.hidden = reference == null;
  if (reference != null) {
    match.dataset.match = predicted === reference;
    match.textContent = predicted === reference ? "✓ Etiketle eşleşti" : "× Etiketle eşleşmedi";
  }
  byId(prefix + "Features").textContent = count;
}

function renderPrediction(data) {
  const { meta } = data;
  const reference = meta.true_label;
  const details = [meta.sample_tag || "Yüklenen görüntü", cellType[meta.cell_type]].filter(Boolean);
  byId("resultSource").textContent = details.join(" · ");
  byId("referenceValue").textContent = reference == null ? "Bu görüntünün etiketi bilinmiyor" : label(reference);
  renderModel("baseline", data.baseline_label, reference, number(data.total_features) + " görüntü özelliği");
  renderModel("ga", data.ga_label, reference,
    number(data.selected_features) + " görüntü özelliği · %" + number(data.feature_reduction_pct, 2) + " daha az");

  let comparison = data.models_agree ? "İki model aynı sınıfı tahmin etti." : "İki model farklı sınıflar tahmin etti.";
  if (reference != null && !data.models_agree) {
    comparison += data.ga_label === reference
      ? " Bu görüntüde GA modelinin tahmini etiketle eşleşti."
      : " Bu görüntüde tüm özellikli modelin tahmini etiketle eşleşti.";
  }
  byId("comparison").textContent = comparison;
  byId("rawImage").src = imageSource(data.images.raw_b64);
  byId("imageSize").textContent = data.image_shape[1] + " × " + data.image_shape[0] + " piksel";
  byId("hogImage").src = imageSource(data.images.hog_b64);
  byId("lbpImage").src = imageSource(data.images.lbp_b64);
  for (const [id, score] of [["baselineScore", data.baseline_margin], ["gaScore", data.ga_margin]]) {
    byId(id).textContent = (score > 0 ? "+" : "") + number(score, 6);
  }
  byId("gaTime").textContent = number(data.timing_ms.ga_total_ms, 2) + " ms";
  renderHistogram(data.lbp_hist);
  byId("downloadCard").href = data.card_url;
  if (meta.subset_index != null && samples.some(sample => sample.subset_index === meta.subset_index)) {
    sampleSelect.value = String(meta.subset_index);
  }
  result.hidden = false;
  if (!byId("incele").hidden) {
    byId("resultTitle").focus({ preventScroll: true });
    result.scrollIntoView({ block: "start" });
  }
}

async function predict(request) {
  if (busy) return;
  busy = true;
  setPredictionControls();
  status.removeAttribute("data-error");
  status.textContent = "Görüntü inceleniyor…";
  try {
    renderPrediction(await fetchJSON("/api/predict", { method: "POST", ...request }));
    status.textContent = "İnceleme tamamlandı.";
  } catch (error) {
    status.dataset.error = "";
    status.textContent = "İnceleme tamamlanamadı. " + error.message +
      (result.hidden ? "" : " Son başarılı inceleme aşağıda duruyor.");
    status.scrollIntoView({ block: "nearest" });
  } finally {
    busy = false;
    setPredictionControls();
  }
}

function predictSample(subsetIndex) {
  return predict({
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subset_index: Number(subsetIndex) }),
  });
}

function populateSamples(presets, samplePayload) {
  samples = samplePayload.samples;
  sampleSelect.replaceChildren(new Option("Bir hücre seç", ""));
  for (const sample of samples) {
    const description = "#" + sample.subset_index + " · " + cellType[sample.cell_type] + " · " + label(sample.label);
    sampleSelect.add(new Option(description, sample.subset_index));
  }
  for (const preset of presets) {
    if (!samples.some(sample => sample.subset_index === preset.subset_index)) continue;
    const button = document.createElement("button");
    const title = document.createElement("strong");
    const description = document.createElement("span");
    button.type = "button";
    button.className = "preset";
    button.dataset.predictControl = "";
    button.dataset.example = preset.subset_index;
    title.textContent = preset.tag + " · #" + preset.subset_index;
    description.textContent = preset.desc;
    button.append(title, description);
    byId("presets").append(button);
  }
  setPredictionControls();
}

const experimentNames = {
  E1: "Tüm özelliklerle SVM",
  E2_seed42: "GA + SVM (seed 42)",
  E2_3seed_mean: "GA + SVM (3 koşu ortalaması)",
  E3: "Yalnızca HOG + SVM",
  E4: "Yalnızca LBP + SVM",
  E5_seed42: "Rastgele özellik seçimi (seed 42)",
  E5_5seed_mean: "Rastgele özellik seçimi (5 koşu ortalaması)",
};

function addTableRow(table, name, values) {
  const row = table.insertRow();
  const header = document.createElement("th");
  header.scope = "row";
  header.textContent = name;
  row.append(header);
  for (const value of values) row.insertCell().textContent = value;
}

function populateExperiments(overview) {
  const { E1: baseline, E2_seed42: ga, E2_3seed_mean: average } = overview.metrics;
  byId("reductionValue").textContent = "%" + number(average.feature_reduction_pct, 2) + " daha az özellik";
  byId("experimentFinding").textContent = "Üç GA koşusunun ortalama Macro F1 puanı " +
    number(average.test_macro_f1, 4) + ". Tüm özellikli model " + number(baseline.test_macro_f1, 4) +
    " puan aldı. GA’nın kullandığı özellik sayısı azalırken bu testte puanlar birbirine yakın kaldı.";
  const test = overview.dataset.splits.test;
  const outcomes = overview.error_summary.e2_outcome_counts;
  byId("testContext").textContent = "Test kümesinde " + test.total + " hücre var. " +
    "GA (seed 42), " + (outcomes.TP + outcomes.FN) + " kusurlu hücrenin " + outcomes.TP +
    " tanesini yakaladı. ELPV etiketlerinde panel kimliği bulunmadığından aynı panele ait hücreler farklı veri bölmelerinde yer almış olabilir.";

  const metrics = row => [
    number(row.selected_features, Number.isInteger(Number(row.selected_features)) ? 0 : 1),
    number(row.test_macro_f1, 4),
    percent(row.test_defective_recall),
    percent(row.test_accuracy),
  ];
  for (const row of [baseline, ga, average]) {
    addTableRow(byId("keyMetrics"), experimentNames[row.experiment_id], metrics(row));
  }
  for (const row of overview.metrics_list) {
    addTableRow(byId("allMetrics"), experimentNames[row.experiment_id] || row.model_name,
      [...metrics(row), number(row.total_infer_ms, 2) + " ms"]);
  }
  for (const row of overview.ga_multiseed.runs) {
    addTableRow(byId("seedMetrics"), String(row.seed), [
      number(row.selected_features), number(row.test_macro_f1, 4), number(row.search_time_s, 2) + " sn",
    ]);
  }
  for (const [type, row] of Object.entries(overview.error_summary.cell_type_breakdown)) {
    addTableRow(byId("subgroupMetrics"), cellType[type], [
      row.total,
      row.e1_errors + " hata · " + percent(row.e1_error_rate),
      row.e2_errors + " hata · " + percent(row.e2_error_rate),
    ]);
  }
  byId("experimentStatus").hidden = true;
  byId("experimentContent").hidden = false;
}

function populateGallery(figures) {
  const dialog = byId("figureDialog");
  for (const figure of figures) {
    const button = document.createElement("button");
    const image = document.createElement("img");
    const title = document.createElement("strong");
    const description = document.createElement("span");
    const source = "/api/figure/" + figure.rel_path;
    button.className = "gallery-item";
    button.type = "button";
    image.src = source;
    image.alt = "";
    image.loading = "lazy";
    title.textContent = figure.title;
    description.textContent = figure.caption;
    button.append(image, title, description);
    button.addEventListener("click", () => {
      byId("figureTitle").textContent = figure.title;
      byId("figureImage").src = source;
      byId("figureImage").alt = figure.title;
      byId("figureCaption").textContent = figure.caption;
      dialog.showModal();
    });
    byId("gallery").append(button);
  }
  byId("galleryStatus").hidden = true;
}

document.addEventListener("click", event => {
  const button = event.target.closest("[data-example]");
  if (button) predictSample(button.dataset.example);
});
byId("uploadButton").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  fileInput.value = "";
  predict({ body: form });
});
byId("sampleButton").addEventListener("click", () => {
  if (sampleSelect.value) predictSample(sampleSelect.value);
  else sampleSelect.focus();
});
byId("randomButton").addEventListener("click", () => {
  const sample = samples[Math.floor(Math.random() * samples.length)];
  if (sample) predictSample(sample.subset_index);
});
window.addEventListener("hashchange", () => {
  showPage();
  window.scrollTo(0, 0);
});
showPage();

const overviewRequest = fetchJSON("/api/overview");
overviewRequest.then(overview => {
  populateExperiments(overview);
  populateGallery(overview.gallery_figures);
}).catch(() => {
  byId("experimentStatus").textContent = "Deney sonuçları yüklenemedi. Sayfayı yeniden yükle.";
  byId("galleryStatus").textContent = "Görseller yüklenemedi. Sayfayı yeniden yükle.";
});
Promise.all([overviewRequest, fetchJSON("/api/samples")])
  .then(([overview, data]) => populateSamples(overview.presets, data))
  .catch(() => {
    sampleSelect.replaceChildren(new Option("Test hücreleri yüklenemedi", ""));
  });
