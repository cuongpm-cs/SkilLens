const state = {
  data: null,
  samples: [],
  reviews: {},
  selectedId: null,
  classFilter: "all",
  statusFilter: "all",
  search: "",
  showOtherBoxes: true,
  zoom: 100,
  saveTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const svgNamespace = "http://www.w3.org/2000/svg";

function reviewFor(sampleId) {
  return state.reviews[sampleId] || { status: "unreviewed", note: "", reviewer: "" };
}

function statusLabel(status) {
  return { pass: "Pass", needs_review: "Needs review", unreviewed: "Not reviewed" }[status] || "Not reviewed";
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.style.borderColor = error ? "#d99a95" : "";
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 1900);
}

function selectedSample() {
  return state.samples.find((sample) => sample.sample_id === state.selectedId) || state.samples[0];
}

function filteredSamples() {
  const needle = state.search.trim().toLowerCase();
  return state.samples.filter((sample) => {
    const review = reviewFor(sample.sample_id);
    if (state.classFilter !== "all" && sample.target_class_name !== state.classFilter) return false;
    if (state.statusFilter !== "all" && review.status !== state.statusFilter) return false;
    if (!needle) return true;
    return `${sample.sample_id} ${sample.target_display_name} ${sample.video_id} ${sample.file_name}`.toLowerCase().includes(needle);
  });
}

function renderProgress() {
  const decisions = state.samples.map((sample) => reviewFor(sample.sample_id).status);
  const pass = decisions.filter((status) => status === "pass").length;
  const needsReview = decisions.filter((status) => status === "needs_review").length;
  const reviewed = pass + needsReview;
  const total = state.samples.length || 1;
  const percent = Math.round((reviewed / total) * 100);
  $("#progressText").textContent = `${reviewed} of ${total} reviewed`;
  $("#progressPercent").textContent = `${percent}%`;
  $("#progressFill").style.width = `${percent}%`;
  $("#passCount").textContent = pass;
  $("#reviewCount").textContent = needsReview;
  $("#pendingCount").textContent = total - reviewed;

  $("#classSummary").innerHTML = state.data.classes.map((item) => {
    const rows = state.samples.filter((sample) => sample.target_class_id === item.class_id);
    const done = rows.filter((sample) => ["pass", "needs_review"].includes(reviewFor(sample.sample_id).status)).length;
    const width = rows.length ? (done / rows.length) * 100 : 0;
    return `<span class="class-meter" title="${item.display_name}: ${done}/${rows.length} reviewed"><i style="width:${width}%"></i></span>`;
  }).join("");
}

function renderSampleList() {
  const rows = filteredSamples();
  $("#visibleCount").textContent = rows.length;
  const list = $("#sampleList");
  if (!rows.length) {
    list.innerHTML = '<div class="empty-list">No images match these filters.</div>';
    return;
  }
  list.innerHTML = rows.map((sample) => {
    const review = reviewFor(sample.sample_id);
    const marker = review.status === "pass" ? "✓" : review.status === "needs_review" ? "!" : "";
    return `<button type="button" class="sample-row ${sample.sample_id === state.selectedId ? "active" : ""}"
      data-sample="${sample.sample_id}" data-status="${review.status}">
      <span class="row-state">${marker}</span>
      <span><strong>${sample.target_display_name}</strong><small>${sample.split} · ${sample.video_id}</small></span>
      <span class="row-index">${sample.sample_id.slice(3)}</span>
    </button>`;
  }).join("");
  list.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => selectSample(button.dataset.sample));
  });
}

function addSvgBox(svg, box, sample, imageWidth, imageHeight) {
  const isTarget = box.class_id === sample.target_class_id;
  if (!isTarget && !state.showOtherBoxes) return;
  const x = (box.x_center - box.width / 2) * imageWidth;
  const y = (box.y_center - box.height / 2) * imageHeight;
  const width = box.width * imageWidth;
  const height = box.height * imageHeight;
  const stroke = isTarget ? "#00a578" : "#3178c6";
  const fontSize = Math.max(15, imageWidth / 65);
  const label = isTarget ? `TARGET · ${sample.target_display_name}` : box.class_name.replaceAll("_", " ");

  const group = document.createElementNS(svgNamespace, "g");
  const rectangle = document.createElementNS(svgNamespace, "rect");
  rectangle.setAttribute("x", x);
  rectangle.setAttribute("y", y);
  rectangle.setAttribute("width", width);
  rectangle.setAttribute("height", height);
  rectangle.setAttribute("fill", isTarget ? "rgba(0,165,120,.08)" : "rgba(49,120,198,.035)");
  rectangle.setAttribute("stroke", stroke);
  rectangle.setAttribute("stroke-width", Math.max(3, imageWidth / 420));
  rectangle.setAttribute("vector-effect", "non-scaling-stroke");
  if (!isTarget) rectangle.setAttribute("stroke-dasharray", "10 7");
  group.appendChild(rectangle);

  const textWidth = Math.min(imageWidth - 4, label.length * fontSize * .62 + 15);
  const labelHeight = fontSize + 11;
  const labelY = y > labelHeight + 3 ? y - labelHeight : y + 2;
  const background = document.createElementNS(svgNamespace, "rect");
  background.setAttribute("x", Math.max(0, x));
  background.setAttribute("y", Math.max(0, labelY));
  background.setAttribute("width", textWidth);
  background.setAttribute("height", labelHeight);
  background.setAttribute("rx", "4");
  background.setAttribute("fill", stroke);
  group.appendChild(background);

  const text = document.createElementNS(svgNamespace, "text");
  text.setAttribute("x", Math.max(0, x) + 7);
  text.setAttribute("y", Math.max(0, labelY) + fontSize + 1);
  text.setAttribute("fill", "white");
  text.setAttribute("font-size", fontSize);
  text.setAttribute("font-family", "Inter, system-ui, sans-serif");
  text.setAttribute("font-weight", isTarget ? "800" : "650");
  text.textContent = label;
  group.appendChild(text);
  svg.appendChild(group);
}

function renderBoxes() {
  const image = $("#reviewImage");
  const svg = $("#boxOverlay");
  const sample = selectedSample();
  if (!sample || !image.naturalWidth) return;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  sample.boxes
    .slice()
    .sort((first, second) => Number(first.class_id === sample.target_class_id) - Number(second.class_id === sample.target_class_id))
    .forEach((box) => addSvgBox(svg, box, sample, image.naturalWidth, image.naturalHeight));
}

function renderDecision() {
  const sample = selectedSample();
  const review = reviewFor(sample.sample_id);
  $("#passButton").classList.toggle("active", review.status === "pass");
  $("#needsReviewButton").classList.toggle("active", review.status === "needs_review");
  $("#reviewNote").value = review.note || "";
  const saved = $("#savedState");
  saved.dataset.status = review.status;
  saved.querySelector("strong").textContent = statusLabel(review.status);
  saved.querySelector("small").textContent = review.updated_at
    ? `Saved ${new Date(review.updated_at).toLocaleString()}`
    : "Choose an answer above";
}

function renderSelectedSample() {
  const sample = selectedSample();
  if (!sample) return;
  $("#sampleNumber").textContent = sample.sample_id;
  $("#targetName").textContent = sample.target_display_name;
  $("#splitBadge").textContent = sample.split.toUpperCase();
  $("#videoName").textContent = sample.video_id;
  $("#fileName").textContent = sample.file_name;
  const image = $("#reviewImage");
  image.alt = `${sample.target_display_name} label review, ${sample.sample_id}`;
  const cacheSeparator = sample.image_url.includes("?") ? "&" : "?";
  image.src = `${sample.image_url}${cacheSeparator}v=1`;
  renderDecision();
  renderSampleList();
  history.replaceState(null, "", `#sample=${sample.sample_id}`);
  requestAnimationFrame(() => {
    const active = document.querySelector(`.sample-row[data-sample="${sample.sample_id}"]`);
    active?.scrollIntoView({ block: "nearest" });
  });
}

function selectSample(sampleId) {
  if (!state.samples.some((sample) => sample.sample_id === sampleId)) return;
  clearTimeout(state.saveTimer);
  state.selectedId = sampleId;
  state.zoom = 100;
  $("#zoomRange").value = "100";
  updateZoom();
  renderSelectedSample();
}

function moveSample(direction, preferUnreviewed = false) {
  const rows = filteredSamples();
  if (!rows.length) return;
  let index = rows.findIndex((sample) => sample.sample_id === state.selectedId);
  if (index < 0) index = 0;
  for (let step = 1; step <= rows.length; step += 1) {
    const next = rows[(index + direction * step + rows.length) % rows.length];
    if (!preferUnreviewed || reviewFor(next.sample_id).status === "unreviewed") {
      selectSample(next.sample_id);
      return;
    }
  }
  selectSample(rows[(index + direction + rows.length) % rows.length].sample_id);
}

async function saveDecision(status, options = {}) {
  clearTimeout(state.saveTimer);
  state.saveTimer = null;
  const sample = selectedSample();
  if (!sample) return;
  const payload = {
    sample_id: sample.sample_id,
    status,
    note: $("#reviewNote").value.trim(),
    reviewer: $("#reviewerName").value.trim(),
  };
  $("#noteStatus").textContent = "Saving…";
  try {
    const response = await fetch("/api/expert-review/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Save failed (${response.status})`);
    if (result.review) state.reviews[sample.sample_id] = result.review;
    else delete state.reviews[sample.sample_id];
    $("#noteStatus").textContent = "Saved.";
    renderDecision();
    renderProgress();
    renderSampleList();
    if (!options.quiet) showToast(status === "pass" ? "Marked as Pass" : status === "needs_review" ? "Marked as Needs review" : "Answer cleared");
    if (options.advance && $("#autoAdvance").checked) {
      setTimeout(() => moveSample(1, true), 220);
    }
  } catch (error) {
    $("#noteStatus").textContent = "Could not save. Please try again.";
    showToast(error.message, true);
  }
}

function scheduleNoteSave() {
  clearTimeout(state.saveTimer);
  $("#noteStatus").textContent = "Notes not saved yet…";
  state.saveTimer = setTimeout(() => {
    const currentStatus = reviewFor(selectedSample().sample_id).status;
    saveDecision(currentStatus, { quiet: true });
  }, 700);
}

function updateZoom() {
  state.zoom = Number($("#zoomRange").value);
  $("#zoomValue").textContent = `${state.zoom}%`;
  $("#imageStage").style.width = `${state.zoom}%`;
}

function csvCell(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function exportCsv() {
  const headers = ["sample_id", "status", "note", "reviewer", "target_object", "split", "video_id", "file_name", "updated_at"];
  const rows = state.samples.map((sample) => {
    const review = reviewFor(sample.sample_id);
    return [sample.sample_id, review.status, review.note, review.reviewer, sample.target_class_name, sample.split, sample.video_id, sample.file_name, review.updated_at || ""];
  });
  const csv = [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\r\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  link.download = `catheterization_label_review_${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
  showToast("Review CSV exported");
}

function bindEvents() {
  $("#reviewImage").addEventListener("load", renderBoxes);
  $("#showOtherBoxes").addEventListener("change", (event) => {
    state.showOtherBoxes = event.target.checked;
    renderBoxes();
  });
  $("#zoomRange").addEventListener("input", updateZoom);
  $("#resetZoom").addEventListener("click", () => {
    $("#zoomRange").value = "100";
    updateZoom();
    $("#imageViewport").scrollTo({ top: 0, left: 0, behavior: "smooth" });
  });
  $("#passButton").addEventListener("click", () => saveDecision("pass", { advance: true }));
  $("#needsReviewButton").addEventListener("click", () => saveDecision("needs_review", { advance: true }));
  $("#clearDecision").addEventListener("click", () => {
    $("#reviewNote").value = "";
    saveDecision("unreviewed");
  });
  $("#reviewNote").addEventListener("input", scheduleNoteSave);
  $("#previousSample").addEventListener("click", () => moveSample(-1));
  $("#nextSample").addEventListener("click", () => moveSample(1));
  $("#exportCsv").addEventListener("click", exportCsv);
  $("#reviewerName").addEventListener("input", (event) => localStorage.setItem("skillensExpertReviewer", event.target.value));
  $("#classFilter").addEventListener("change", (event) => { state.classFilter = event.target.value; renderSampleList(); });
  $("#statusFilter").addEventListener("change", (event) => { state.statusFilter = event.target.value; renderSampleList(); });
  $("#sampleSearch").addEventListener("input", (event) => { state.search = event.target.value; renderSampleList(); });
  window.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.key.toLowerCase() === "p") saveDecision("pass", { advance: true });
    else if (event.key.toLowerCase() === "r") saveDecision("needs_review", { advance: true });
    else if (event.key === "ArrowLeft") moveSample(-1);
    else if (event.key === "ArrowRight") moveSample(1);
  });
}

async function init() {
  try {
    const response = await fetch("/api/expert-review");
    if (!response.ok) throw new Error(`Review API returned ${response.status}`);
    state.data = await response.json();
    state.samples = state.data.samples;
    state.reviews = state.data.reviews || {};
    const classFilter = $("#classFilter");
    state.data.classes.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.class_name;
      option.textContent = `${item.display_name} (${item.sample_count})`;
      classFilter.appendChild(option);
    });
    $("#reviewerName").value = localStorage.getItem("skillensExpertReviewer") || "";
    const requested = new URLSearchParams(location.hash.slice(1)).get("sample");
    state.selectedId = state.samples.some((sample) => sample.sample_id === requested) ? requested : state.samples[0].sample_id;
    bindEvents();
    renderProgress();
    renderSelectedSample();
    updateZoom();
    $("#loading").classList.add("hidden");
  } catch (error) {
    $("#loading").innerHTML = `<div class="loading-mark">!</div><p>${error.message}</p>`;
    console.error(error);
  }
}

init();
