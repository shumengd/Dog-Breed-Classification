"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, sample: null, image: null, result: null, filename: "", busy: false, request: 0, benchmarks: null };
const modelNames = { cnn_from_scratch: "CNN from scratch", transfer_freeze_0: "Pretrained CNN", transfer_freeze_1: "Pretrained CNN", transfer_freeze_2: "Pretrained CNN", transfer_freeze_3: "Pretrained CNN", vit_from_scratch: "Vision Transformer" };

function icons() { if (window.lucide) window.lucide.createIcons(); }
function percent(value) { return `${(value * 100).toFixed(1)}%`; }
function showError(message) { $("error").textContent = message; $("error").hidden = !message; }
function updateControls() {
  const available = state.status?.models.find((model) => model.id === $("model-select").value)?.available;
  $("classify").disabled = state.busy || !available || (!state.sample && !state.image);
  $("classify").querySelector("span").textContent = state.busy ? "Classifying..." : "Classify image";
  $("prediction").setAttribute("aria-busy", String(state.busy));
  $("model-select").disabled = state.busy;
  $("download").disabled = !state.result || state.busy;
}

function clearPrediction() {
  state.result = null;
  $("breed").textContent = "Awaiting prediction";
  $("confidence").textContent = "--";
  $("golden-value").textContent = $("collie-value").textContent = "--";
  $("golden-bar").style.width = $("collie-bar").style.width = "0%";
  $("ground-truth").hidden = true;
  $("elapsed").textContent = "-- ms";
  $("epoch").textContent = "Epoch --";
  updateControls();
}

function setPreview(source, filename, sourceTag) {
  const image = $("preview");
  image.onload = () => { $("image-dimensions").textContent = `${image.naturalWidth} x ${image.naturalHeight} px`; };
  image.src = source;
  image.hidden = false;
  $("empty-image").hidden = true;
  $("filename").textContent = filename;
  $("source-tag").textContent = sourceTag;
  state.filename = filename;
  clearPrediction();
  showError("");
}

async function classify() {
  if (state.busy || (!state.sample && !state.image)) return;
  state.busy = true;
  const request = ++state.request;
  updateControls();
  showError("");
  try {
    const response = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: $("model-select").value, sample: state.sample, image: state.image, filename: state.filename }),
    });
    const result = await response.json();
    if (request !== state.request) return;
    if (!response.ok) throw new Error(result.error || "Prediction unavailable");
    state.result = result;
    $("breed").textContent = result.predicted_label;
    $("confidence").textContent = percent(result.confidence);
    result.probabilities.forEach((item, index) => {
      const prefix = index === 0 ? "golden" : "collie";
      $(`${prefix}-value`).textContent = percent(item.probability);
      $(`${prefix}-bar`).style.width = percent(item.probability);
    });
    $("elapsed").textContent = `${result.inference_ms.toFixed(1)} ms`;
    $("epoch").textContent = `Epoch ${result.epoch}`;
    $("model-detail").textContent = `64 x 64 RGB / epoch ${result.epoch}`;
    $("ground-truth").hidden = !result.actual_label;
    if (result.actual_label) $("actual-label").textContent = result.actual_label;
  } catch (error) {
    if (request === state.request) showError(error.message);
  } finally {
    if (request === state.request) { state.busy = false; updateControls(); }
  }
}

async function selectSample(sample) {
  state.request++;
  state.busy = false;
  state.sample = sample.id;
  state.image = null;
  setPreview(sample.url, sample.label, "Test sample");
  document.querySelectorAll(".sample").forEach((button) => {
    const active = button.dataset.id === sample.id;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (state.status.ready) await classify();
}

async function selectFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { showError("Select a JPEG, PNG, or WebP image."); return; }
  if (file.size > 8 * 1024 * 1024) { showError("Select an image smaller than 8 MB."); return; }
  const request = ++state.request;
  state.busy = false;
  const reader = new FileReader();
  const source = await new Promise((resolve, reject) => {
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Image could not be read"));
    reader.readAsDataURL(file);
  });
  if (request !== state.request) return;
  state.sample = null;
  state.image = source.split(",")[1];
  setPreview(source, file.name, "Uploaded image");
  document.querySelectorAll(".sample").forEach((button) => { button.classList.remove("active"); button.setAttribute("aria-pressed", "false"); });
  if (state.status?.ready) await classify();
}

function showBenchmark() {
  const dataset = $("benchmark-select").value === "reproduced" ? state.benchmarks.reproduced : state.benchmarks.historical;
  const tbody = $("benchmark-rows");
  tbody.replaceChildren();
  const baseline = dataset.results.find((result) => result.model === "cnn_from_scratch");
  const transfer = dataset.results.find((result) => result.model === "transfer_freeze_3");
  if (baseline && transfer) {
    const gain = transfer.test_auroc - baseline.test_auroc;
    $("gain-value").textContent = `${gain >= 0 ? "+" : ""}${gain.toFixed(4)}`;
  } else $("gain-value").textContent = "--";
  dataset.results.forEach((result) => {
    const row = document.createElement("tr");
    if (result.model === "transfer_freeze_3") row.className = "highlight";
    const frozen = result.model.startsWith("transfer_freeze_") ? result.model.split("_").at(-1) : "--";
    [modelNames[result.model] || result.model, frozen, result.epoch ?? "--", result.validation_auroc.toFixed(4), result.test_auroc.toFixed(4)].forEach((value) => {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    });
    tbody.append(row);
  });
  $("benchmark-note").textContent = dataset.note || dataset.description;
}

function switchTab(selected) {
  ["predict", "experiments"].forEach((name) => {
    const active = name === selected;
    $(`${name}-tab`).classList.toggle("active", active);
    $(`${name}-tab`).setAttribute("aria-selected", String(active));
    $(`${name}-tab`).tabIndex = active ? 0 : -1;
    $(`${name}-view`).hidden = !active;
  });
}

async function initialize() {
  icons();
  $("upload").onclick = () => $("file-input").click();
  $("file-input").onchange = (event) => { selectFile(event.target.files[0]).catch((error) => showError(error.message)); event.target.value = ""; };
  $("classify").onclick = classify;
  $("model-select").onchange = () => { clearPrediction(); showError(""); if (state.sample || state.image) classify(); };
  $("reset").onclick = () => {
    state.request++; state.busy = false; state.sample = null; state.image = null;
    $("preview").hidden = true; $("preview").removeAttribute("src"); $("empty-image").hidden = false;
    $("source-tag").textContent = "No image selected"; $("filename").textContent = "Image"; $("image-dimensions").textContent = "JPEG, PNG, WebP / 8 MB";
    document.querySelectorAll(".sample").forEach((button) => { button.classList.remove("active"); button.setAttribute("aria-pressed", "false"); });
    clearPrediction(); $("breed").textContent = "Awaiting image"; showError("");
  };
  $("download").onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "dog-breed-prediction.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const dropZone = $("drop-zone");
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); $("drag-label").hidden = false; }));
  dropZone.addEventListener("dragleave", (event) => { if (!dropZone.contains(event.relatedTarget)) { dropZone.classList.remove("dragging"); $("drag-label").hidden = true; } });
  dropZone.addEventListener("drop", (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); $("drag-label").hidden = true; selectFile(event.dataTransfer.files[0]).catch((error) => showError(error.message)); });
  ["predict", "experiments"].forEach((name) => {
    $(`${name}-tab`).onclick = () => switchTab(name);
    $(`${name}-tab`).onkeydown = (event) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault(); const target = event.key === "Home" ? "predict" : event.key === "End" ? "experiments" : name === "predict" ? "experiments" : "predict";
        switchTab(target); $(`${target}-tab`).focus();
      }
    };
  });
  $("benchmark-select").onchange = showBenchmark;
  $("figure-select").onchange = () => {
    $("training-figure").src = `/figures/${$("figure-select").value}`;
    $("training-figure").alt = `${$("figure-select").selectedOptions[0].textContent}: original experiment`;
  };
  try {
    const [statusResponse, benchmarkResponse] = await Promise.all([fetch("/api/status"), fetch("/api/benchmarks")]);
    if (!statusResponse.ok || !benchmarkResponse.ok) throw new Error("Local model connection unavailable");
    state.status = await statusResponse.json();
    state.benchmarks = await benchmarkResponse.json();
    $("connection").textContent = state.status.ready ? "Local model ready" : "Model unavailable";
    $("connection").prepend(document.createElement("span"));
    $("connection").classList.toggle("ready", state.status.ready);
    $("model-select").replaceChildren();
    state.status.models.forEach((model) => {
      const option = document.createElement("option"); option.value = model.id; option.textContent = model.name + (model.available ? "" : " (unavailable)"); option.disabled = !model.available; $("model-select").append(option);
    });
    if (state.status.default_model) $("model-select").value = state.status.default_model;
    $("model-detail").textContent = state.status.ready ? "64 x 64 RGB" : "Checkpoint unavailable";
    $("sample-count").textContent = `${state.status.samples.length} images`;
    state.status.samples.forEach((sample) => {
      const button = document.createElement("button"); button.className = "sample"; button.dataset.id = sample.id; button.setAttribute("aria-pressed", "false"); button.setAttribute("aria-label", `${sample.label} test sample ${sample.id.split("-")[1]}`);
      const image = document.createElement("img"); image.src = sample.url; image.alt = sample.label;
      const label = document.createElement("span"); label.textContent = sample.label;
      button.append(image, label); button.onclick = () => selectSample(sample); $("samples").append(button);
    });
    if (state.benchmarks.reproduced) { const option = document.createElement("option"); option.value = "reproduced"; option.textContent = "Reproduced experiments"; $("benchmark-select").append(option); }
    showBenchmark();
    if (!state.status.ready) showError("Local model files are unavailable.");
    if (state.status.samples.length) await selectSample(state.status.samples[0]);
    updateControls();
  } catch (error) { $("connection").textContent = "Disconnected"; showError(error.message); }
}

initialize();
