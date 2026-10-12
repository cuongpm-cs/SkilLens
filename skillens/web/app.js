const state = {
  data: null,
  selectedStep: null,
  filter: "all",
  search: "",
  duration: 0,
  timelineZoom: 1.5,
  activeEventId: null,
  activeEvidenceId: null,
};

const $ = (selector) => document.querySelector(selector);
const video = $("#video");
const timeline = $("#timeline");
const timelineViewport = $("#timelineViewport");
const overviewTrack = $("#overviewTrack");
const evidenceTrack = $("#evidenceTrack");
const audioTrack = $("#audioTrack");

function formatTime(seconds, includeHours = false) {
  const value = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = Math.floor(value % 60);
  if (includeHours || hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function timeRange(start, end) {
  return `${formatTime(start)}–${formatTime(end)}`;
}

function statusLabel(status) {
  return {
    complete: "Complete",
    partial: "Partial",
    unverifiable: "Unverifiable",
    unclear: "Unclear",
    supporting: "Supporting",
  }[status] || status || "Unknown";
}

function scoreText(step) {
  return step.score === null || step.score === undefined ? `—/${step.max_score}` : `${step.score}/${step.max_score}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 1800);
}

function renderHeader() {
  const data = state.data;
  $("#examTitle").textContent = data.title;
  $("#candidateName").textContent = data.candidate;
  $("#backendBadge").textContent = data.backend;
  const summary = data.summary;
  const confirmed = Number(summary.confirmed_score || 0);
  const maximum = Number(summary.maximum_score || 40);
  const percent = maximum ? Math.round((confirmed / maximum) * 100) : 0;
  $("#confirmedScore").textContent = confirmed;
  $("#maximumScore").textContent = `/ ${maximum}`;
  $("#scorePercent").textContent = `${percent}%`;
  $("#scoreRing").style.setProperty("--score-angle", `${percent * 3.6}deg`);
  const range = summary.score_range || [confirmed, confirmed];
  $("#scoreRange").textContent = `Range ${range[0]}–${range[1]}`;
  $("#reviewCount").textContent = `${summary.review_steps || 0} need review`;
  $("#criteriaCount").textContent = data.criteria.length;
}

function matchesFilter(step) {
  if (state.filter === "complete" && step.verdict !== "complete") return false;
  if (state.filter === "partial" && step.verdict !== "partial") return false;
  if (state.filter === "review" && !step.requires_review && step.verdict !== "unverifiable") return false;
  const needle = state.search.trim().toLowerCase();
  return !needle || `${step.step_id} ${step.description} ${step.reason}`.toLowerCase().includes(needle);
}

function renderCriteria() {
  const container = $("#criteriaList");
  const rows = state.data.criteria.filter(matchesFilter);
  container.innerHTML = rows.map((step) => `
    <button class="criterion-card ${state.selectedStep?.step_id === step.step_id ? "selected" : ""}"
            data-step="${step.step_id}" type="button">
      <span class="criterion-index">${String(step.step_id).padStart(2, "0")}</span>
      <span>
        <span class="criterion-title">${escapeHtml(step.description)}</span>
        <span class="criterion-sub">
          <i class="mini-dot ${escapeHtml(step.verdict)}"></i>
          ${statusLabel(step.verdict)} · ${step.evidence.length} evidence
        </span>
      </span>
      <span class="criterion-score"><strong>${step.score ?? "—"}</strong>/${step.max_score}</span>
    </button>
  `).join("");
  container.querySelectorAll(".criterion-card").forEach((button) => {
    button.addEventListener("click", () => selectCriterion(Number(button.dataset.step)));
  });
}

function renderRuler() {
  const ruler = $("#ruler");
  const duration = state.duration || 1;
  const majorInterval = duration > 1200 ? 120 : duration > 420 ? 60 : 30;
  const minorInterval = state.timelineZoom >= 4 ? 5 : state.timelineZoom >= 2 ? 10 : 15;
  const ticks = [];
  for (let second = 0; second <= duration; second += minorInterval) {
    const major = second % majorInterval === 0;
    ticks.push(`<i class="ruler-tick ${major ? "major" : "minor"}" style="left:${(second / duration) * 100}%">
      ${major ? `<span>${formatTime(second)}</span>` : ""}
    </i>`);
  }
  ruler.innerHTML = ticks.join("");
}

function centerTimelineOnPlayhead() {
  if (!timelineViewport || state.timelineZoom <= 1) return;
  const ratio = Number(video.currentTime || 0) / Math.max(state.duration, 1);
  timelineViewport.scrollLeft = Math.max(0, ratio * timeline.scrollWidth - timelineViewport.clientWidth / 2);
}

function updateTimelineZoom(center = true) {
  if (!timelineViewport) return;
  const baseWidth = Math.max(timelineViewport.clientWidth, 720);
  timeline.style.width = `${Math.round(baseWidth * state.timelineZoom)}px`;
  $("#zoomFit").classList.toggle("active", state.timelineZoom === 1);
  renderRuler();
  updatePlaybackState();
  if (center) requestAnimationFrame(centerTimelineOnPlayhead);
}

function segmentStyle(start, end) {
  const duration = state.duration || 1;
  const left = Math.max(0, Math.min(100, (start / duration) * 100));
  const width = Math.max(0.25, Math.min(100 - left, ((end - start) / duration) * 100));
  return `left:${left}%;width:${width}%`;
}

function renderOverviewTrack() {
  overviewTrack.innerHTML = state.data.timeline_events.map((event, index) => `
    <button type="button" class="event-segment ${escapeHtml(event.status)}"
      data-event="${escapeHtml(event.event_id)}"
      style="${segmentStyle(event.start_seconds, event.end_seconds)}"
      title="${escapeHtml(`${timeRange(event.start_seconds, event.end_seconds)} · ${event.headline}`)}">
      <span class="segment-code">E${String(index + 1).padStart(2, "0")}</span>
      <small>${formatTime(event.start_seconds)}</small>
    </button>
  `).join("");
  overviewTrack.querySelectorAll(".event-segment").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const row = state.data.timeline_events.find((item) => item.event_id === button.dataset.event);
      if (row?.step_ids.length) selectCriterion(row.step_ids[0]);
      seekTo(row.start_seconds, true);
    });
  });
}

function renderEvidenceTrack() {
  const evidence = state.selectedStep?.evidence || [];
  evidenceTrack.innerHTML = evidence.map((row, index) => `
    <button type="button" class="evidence-segment ${row.source === "audio" ? "audio" : ""}"
      data-evidence="${escapeHtml(row.id)}"
      style="${segmentStyle(row.start_seconds, Math.max(row.end_seconds, row.start_seconds + 1))}"
      title="${escapeHtml(`${timeRange(row.start_seconds, row.end_seconds)} · ${row.description}`)}">
      <span>${row.source === "audio" ? "A" : "V"}${index + 1}</span>
    </button>
  `).join("");
  evidenceTrack.querySelectorAll(".evidence-segment").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const row = evidence.find((item) => item.id === button.dataset.evidence);
      seekTo(row.start_seconds, true);
    });
  });
}

function renderAudioTrack() {
  const chunks = state.data.audio_chunks || [];
  audioTrack.innerHTML = chunks.map((chunk, index) => `
    <button type="button" class="audio-segment"
      data-audio="${index}"
      style="${segmentStyle(chunk.start_seconds, Math.max(chunk.end_seconds, chunk.start_seconds + 1))}"
      title="${escapeHtml(`${timeRange(chunk.start_seconds, chunk.end_seconds)} · ${chunk.text || "Transcript"}`)}">
    </button>
  `).join("");
  audioTrack.querySelectorAll(".audio-segment").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const chunk = chunks[Number(button.dataset.audio)];
      if (chunk) seekTo(chunk.start_seconds, true);
    });
  });
}

function renderDetail() {
  const step = state.selectedStep;
  if (!step) return;
  const evidenceMarkup = step.evidence.length ? step.evidence.map((row) => `
    <button type="button" class="evidence-card" data-evidence-card="${escapeHtml(row.id)}">
      <span class="evidence-card-top">
        <span class="timestamp">▶ ${timeRange(row.start_seconds, row.end_seconds)}</span>
        <span class="evidence-source">${escapeHtml(row.source)} · ${Math.round(Number(row.confidence || 0) * 100)}%</span>
      </span>
      <p>${escapeHtml(row.description)}</p>
    </button>
  `).join("") : `<div class="evidence-card"><p>No timestamped evidence was sufficient for this criterion.</p></div>`;

  $("#criterionDetail").innerHTML = `
    <div class="detail-header">
      <span class="detail-number">${String(step.step_id).padStart(2, "0")}</span>
      <div>
        <h2>${escapeHtml(step.description)}</h2>
        <div class="detail-meta">
          <span class="status-badge ${escapeHtml(step.verdict)}">${statusLabel(step.verdict)}</span>
          <span>${Math.round(Number(step.confidence || 0) * 100)}% confidence</span>
          <span>·</span><span>${escapeHtml(step.score_basis || "no score basis")}</span>
        </div>
      </div>
      <div class="detail-score">${step.score ?? "—"}<span> / ${step.max_score}</span></div>
    </div>
    <div class="judgement">
      <span class="judgement-label">LLM judgement</span>
      <p>${escapeHtml(step.reason || "No judgement reason was recorded.")}</p>
    </div>
    <div class="evidence-heading">
      <span>Timestamped evidence</span>
      <span>${step.evidence.length} event${step.evidence.length === 1 ? "" : "s"}${step.evidence.length > 4 ? " · scroll to view all" : ""}</span>
    </div>
    <div class="evidence-list ${step.evidence.length > 4 ? "is-scrollable" : ""}">${evidenceMarkup}</div>
  `;
  document.querySelectorAll("[data-evidence-card]").forEach((button) => {
    button.addEventListener("click", () => {
      const row = step.evidence.find((item) => item.id === button.dataset.evidenceCard);
      seekTo(row.start_seconds, true);
    });
  });
}

function selectCriterion(stepId) {
  const step = state.data.criteria.find((item) => item.step_id === Number(stepId));
  if (!step) return;
  state.selectedStep = step;
  renderCriteria();
  renderEvidenceTrack();
  renderDetail();
  $("#overlayStep").textContent = `Criterion ${String(step.step_id).padStart(2, "0")}`;
  $("#captionVerdict").textContent = statusLabel(step.verdict);
  $("#videoOverlay").dataset.status = step.verdict;
  updateHash();
}

function seekTo(seconds, autoplay = false) {
  video.currentTime = Math.max(0, Math.min(Number(seconds) || 0, state.duration));
  updatePlaybackState();
  updateHash();
  if (autoplay) video.play().catch(() => {});
}

function eventAtTime(seconds) {
  return state.data.timeline_events.find((event) => seconds >= event.start_seconds && seconds < event.end_seconds)
    || state.data.timeline_events.at(-1);
}

function evidenceAtTime(seconds) {
  if (!state.selectedStep) return null;
  return state.selectedStep.evidence.find((row) => seconds >= row.start_seconds - 0.5 && seconds <= row.end_seconds + 0.5) || null;
}

function updatePlaybackState() {
  const duration = state.duration || video.duration || 0;
  const current = Number(video.currentTime || 0);
  $("#timecode").textContent = `${formatTime(current)} / ${formatTime(duration)}`;
  $("#timelineCursor").textContent = formatTime(current);
  $("#playheadTime").textContent = formatTime(current);
  const trackLeft = overviewTrack.offsetLeft;
  const trackWidth = overviewTrack.clientWidth;
  $("#playhead").style.left = `${trackLeft + (current / Math.max(duration, 1)) * trackWidth}px`;

  const activeEvent = eventAtTime(current);
  if (activeEvent && activeEvent.event_id !== state.activeEventId) {
    state.activeEventId = activeEvent.event_id;
    const activeIndex = state.data.timeline_events.findIndex((event) => event.event_id === activeEvent.event_id);
    $("#activeEventLabel").textContent = `${timeRange(activeEvent.start_seconds, activeEvent.end_seconds)} · ${activeEvent.headline}`;
    $("#overlayEvent").textContent = activeEvent.headline;
    $("#overlayEvent").title = activeEvent.headline;
    $("#captionEventId").textContent = `E${String(activeIndex + 1).padStart(2, "0")}`;
    $("#captionEventTime").textContent = timeRange(activeEvent.start_seconds, activeEvent.end_seconds);
    document.querySelectorAll(".event-segment").forEach((segment) => {
      segment.classList.toggle("active", segment.dataset.event === activeEvent.event_id);
    });
  }
  const activeEvidence = evidenceAtTime(current);
  const evidenceId = activeEvidence?.id || null;
  if (evidenceId !== state.activeEvidenceId) {
    state.activeEvidenceId = evidenceId;
    document.querySelectorAll("[data-evidence], [data-evidence-card]").forEach((element) => {
      const id = element.dataset.evidence || element.dataset.evidenceCard;
      element.classList.toggle("active", id === evidenceId);
    });
  }
}

function seekFromTrack(event) {
  if (event.target !== event.currentTarget) return;
  const bounds = event.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
  seekTo(ratio * state.duration);
}

function updateHash() {
  if (!state.selectedStep) return;
  const params = new URLSearchParams({
    step: state.selectedStep.step_id,
    t: Number(video.currentTime || 0).toFixed(2),
  });
  history.replaceState(null, "", `#${params}`);
}

function applyInitialHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const step = Number(params.get("step")) || state.data.criteria[0]?.step_id;
  const timestamp = Number(params.get("t"));
  selectCriterion(step);
  if (Number.isFinite(timestamp)) seekTo(timestamp);
}

function bindControls() {
  $("#playPause").addEventListener("click", () => video.paused ? video.play() : video.pause());
  $("#backFive").addEventListener("click", () => seekTo(video.currentTime - 5));
  $("#forwardFive").addEventListener("click", () => seekTo(video.currentTime + 5));
  $("#replayEvent").addEventListener("click", () => {
    const activeEvent = state.data.timeline_events.find((event) => event.event_id === state.activeEventId);
    if (activeEvent) seekTo(activeEvent.start_seconds, true);
  });
  $("#playbackRate").addEventListener("change", (event) => { video.playbackRate = Number(event.target.value); });
  video.addEventListener("play", () => { $("#playPause").textContent = "❚❚"; });
  video.addEventListener("pause", () => { $("#playPause").textContent = "▶"; });
  video.addEventListener("timeupdate", updatePlaybackState);
  video.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(video.duration)) state.duration = video.duration;
    renderRuler();
    renderOverviewTrack();
    renderEvidenceTrack();
    renderAudioTrack();
    updateTimelineZoom(false);
    updatePlaybackState();
  });
  overviewTrack.addEventListener("click", seekFromTrack);
  evidenceTrack.addEventListener("click", seekFromTrack);
  audioTrack.addEventListener("click", seekFromTrack);

  $("#zoomOut").addEventListener("click", () => {
    state.timelineZoom = Math.max(1, Number((state.timelineZoom / 1.5).toFixed(2)));
    updateTimelineZoom();
  });
  $("#zoomFit").addEventListener("click", () => {
    state.timelineZoom = 1;
    updateTimelineZoom();
  });
  $("#zoomIn").addEventListener("click", () => {
    state.timelineZoom = Math.min(6, Number((state.timelineZoom * 1.5).toFixed(2)));
    updateTimelineZoom();
  });

  $("#criterionSearch").addEventListener("input", (event) => {
    state.search = event.target.value;
    renderCriteria();
  });
  $("#filters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]");
    if (!button) return;
    state.filter = button.dataset.filter;
    document.querySelectorAll(".filter-chip").forEach((chip) => chip.classList.toggle("active", chip === button));
    renderCriteria();
  });
  $("#copyLink").addEventListener("click", async () => {
    updateHash();
    await navigator.clipboard.writeText(location.href);
    showToast("Review link copied");
  });
  window.addEventListener("resize", () => updateTimelineZoom(false));
  window.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") {
      event.preventDefault();
      video.paused ? video.play() : video.pause();
    } else if (event.code === "ArrowLeft") {
      event.preventDefault();
      seekTo(video.currentTime - 5);
    } else if (event.code === "ArrowRight") {
      event.preventDefault();
      seekTo(video.currentTime + 5);
    }
  });
}

async function init() {
  try {
    const response = await fetch("/api/dashboard");
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    state.data = await response.json();
    state.duration = Number(state.data.duration_seconds || 0);
    video.src = state.data.video_url;
    video.controls = true;
    renderHeader();
    renderRuler();
    renderOverviewTrack();
    renderAudioTrack();
    renderCriteria();
    bindControls();
    applyInitialHash();
    updateTimelineZoom(false);
    $("#loading").classList.add("hidden");
  } catch (error) {
    $("#loading").innerHTML = `<div class="loading-mark">!</div><p>${escapeHtml(error.message)}</p>`;
    console.error(error);
  }
}

init();
