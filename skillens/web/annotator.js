"use strict";

const state = {
  project: null,
  activeVideoIndex: 0,
  activeFrameIndex: -1,
  activeClassId: null,
  boxes: [],
  selectedBoxId: null,
  status: "unreviewed",
  note: "",
  revision: 0,
  suggestions: [],
  history: [],
  future: [],
  pointer: null,
  dirty: false,
  saveTimer: null,
  savePromise: null,
  changeVersion: 0,
  zoom: 100,
};

const el = {};
const $ = (id) => document.getElementById(id);

function cacheElements() {
  [
    "progressText", "progressFill", "splitPolicy", "saveState", "saveStateText",
    "videoCount", "videoList", "projectHeading", "frameSearch", "frameFilter", "frameList",
    "previousFrame", "nextFrame", "currentFrameTitle", "currentFrameMeta",
    "undoButton", "redoButton", "copyPrevious", "loadSuggestions", "fitButton",
    "zoomRange", "zoomValue", "canvasViewport", "emptyCanvas", "imageShell",
    "frameImage", "boxOverlay", "classList", "addClassButton", "classForm",
    "className", "classColor", "cancelClass", "classEmpty", "objectCount", "objectList",
    "frameStatus", "frameNote", "confirmFrame", "markEmpty", "markReview",
    "markDraft", "toastRegion",
  ].forEach((id) => { el[id] = $(id); });
}

function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

function frameKey(videoId, filename) {
  return `${videoId}/${filename}`;
}

function currentVideo() {
  return state.project?.videos[state.activeVideoIndex] ?? null;
}

function currentFrame() {
  const video = currentVideo();
  return video?.frames[state.activeFrameIndex] ?? null;
}

function formatTime(seconds) {
  const whole = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(whole / 60);
  const remainder = whole % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function classInfo(classId) {
  return state.project.classes.find((item) => item.id === classId);
}

function statusLabel(status) {
  return {
    unreviewed: "To do",
    annotated: "Confirmed",
    empty: "No objects",
    needs_review: "Needs review",
  }[status] || "To do";
}

function setSaveState(kind, text) {
  el.saveState.dataset.state = kind;
  el.saveStateText.textContent = text;
}

function toast(message, kind = "normal") {
  const node = document.createElement("div");
  node.className = `toast ${kind === "error" ? "error" : ""}`;
  node.textContent = message;
  el.toastRegion.append(node);
  window.setTimeout(() => node.remove(), 3200);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function summaryFromFrames() {
  const { total_frames: total, complete_frames: complete } = state.project.summary;
  return { total, complete, progress: total ? complete / total : 0 };
}

function updateProgress() {
  const summary = summaryFromFrames();
  const percent = Math.round(summary.progress * 100);
  el.progressText.textContent = `${summary.complete} / ${summary.total} frames complete · ${percent}%`;
  el.progressFill.style.width = `${percent}%`;
  renderVideos();
}

function renderVideos() {
  if (!state.project) return;
  el.videoList.replaceChildren();
  state.project.videos.forEach((video, index) => {
    const complete = video.complete_count || 0;
    const percent = video.frame_count ? Math.round((complete / video.frame_count) * 100) : 0;
    const button = document.createElement("button");
    button.className = `video-card ${index === state.activeVideoIndex ? "active" : ""}`;
    button.innerHTML = `
      <span class="video-card-top">
        <strong>${video.video_id.replace("video_", "Video ")}</strong>
        <span class="split-badge ${video.split}">${video.split}</span>
      </span>
      <small>${complete} / ${video.frame_count} complete</small>
      <span class="mini-progress"><span style="width:${percent}%"></span></span>`;
    button.addEventListener("click", async () => {
      if (index === state.activeVideoIndex) return;
      await flushSave();
      try {
        await ensureVideoFrames(index);
      } catch (error) {
        toast(`Could not load this video: ${error.message}`, "error");
        return;
      }
      state.activeVideoIndex = index;
      state.activeFrameIndex = -1;
      renderVideos();
      renderFrameList();
      const next = video.frames.findIndex((frame) => !["annotated", "empty"].includes(frame.status));
      await loadFrame(next >= 0 ? next : 0);
    });
    el.videoList.append(button);
  });
}

async function ensureVideoFrames(index) {
  const video = state.project.videos[index];
  if (!video || video.frames_loaded) return;
  const response = await api(`/api/annotation-video?video=${encodeURIComponent(video.video_id)}`);
  video.frames = response.frames;
  video.complete_count = response.complete_count;
  video.frames_loaded = true;
}

function frameMatchesFilter(frame) {
  const filter = el.frameFilter.value;
  const query = el.frameSearch.value.trim().toLowerCase();
  const indexText = String(frame.index);
  const nameText = frame.filename.toLowerCase();
  if (query && !indexText.includes(query) && !nameText.includes(query)) return false;
  if (filter === "todo") return frame.status === "unreviewed";
  if (filter === "complete") return ["annotated", "empty"].includes(frame.status);
  if (filter === "needs_review") return frame.status === "needs_review";
  return true;
}

function renderFrameList() {
  const video = currentVideo();
  el.frameList.replaceChildren();
  if (!video) return;
  video.frames.forEach((frame, index) => {
    if (!frameMatchesFilter(frame)) return;
    const button = document.createElement("button");
    button.className = `frame-row ${index === state.activeFrameIndex ? "active" : ""}`;
    button.dataset.index = String(index);
    button.innerHTML = `
      <span class="status-dot ${frame.status}"></span>
      <span class="frame-copy">
        <strong>Frame ${String(frame.index).padStart(4, "0")}</strong>
        <small>${formatTime(frame.timestamp_seconds)} · ${statusLabel(frame.status)}</small>
      </span>
      <span class="box-count">${frame.box_count || 0} box${frame.box_count === 1 ? "" : "es"}</span>`;
    button.addEventListener("click", () => loadFrame(index));
    el.frameList.append(button);
  });
  requestAnimationFrame(() => {
    el.frameList.querySelector(".frame-row.active")?.scrollIntoView({ block: "nearest" });
  });
}

function renderClasses() {
  el.classList.replaceChildren();
  el.classEmpty.hidden = state.project.classes.length > 0;
  state.project.classes.forEach((item) => {
    const count = state.boxes.filter((box) => box.class_id === item.id).length;
    const button = document.createElement("button");
    button.className = `class-row ${item.id === state.activeClassId ? "active" : ""}`;
    button.style.setProperty("--class-color", item.color);
    button.innerHTML = `
      <span class="class-swatch"></span>
      <span class="class-number">${item.id + 1}</span>
      <strong>${item.label}</strong>
      <small>${count}</small>`;
    button.addEventListener("click", () => selectClass(item.id));
    el.classList.append(button);
  });
}

function selectClass(classId) {
  if (!classInfo(classId)) return;
  state.activeClassId = classId;
  if (state.selectedBoxId) {
    const box = state.boxes.find((candidate) => candidate.id === state.selectedBoxId);
    if (box && box.class_id !== classId) {
      snapshot();
      box.class_id = classId;
      markDirty();
      renderBoxes();
      renderObjects();
    }
  }
  renderClasses();
}

function renderObjects() {
  el.objectCount.textContent = String(state.boxes.length);
  el.objectList.replaceChildren();
  if (!state.boxes.length) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = "No boxes on this frame.";
    el.objectList.append(message);
    return;
  }
  state.boxes.forEach((box, index) => {
    const info = classInfo(box.class_id);
    const row = document.createElement("div");
    row.className = `object-row ${box.id === state.selectedBoxId ? "active" : ""}`;
    row.style.setProperty("--object-color", info.color);
    row.innerHTML = `
      <span class="object-dot"></span>
      <button class="select-object">${index + 1}. ${info.label}</button>
      <button class="delete-object" title="Delete box">×</button>`;
    row.querySelector(".select-object").addEventListener("click", () => {
      state.selectedBoxId = box.id;
      state.activeClassId = box.class_id;
      renderBoxes(); renderObjects(); renderClasses();
    });
    row.querySelector(".delete-object").addEventListener("click", () => deleteBox(box.id));
    el.objectList.append(row);
  });
}

function styleBox(node, box) {
  node.style.left = `${box.x * 100}%`;
  node.style.top = `${box.y * 100}%`;
  node.style.width = `${box.width * 100}%`;
  node.style.height = `${box.height * 100}%`;
}

function renderBoxes() {
  el.boxOverlay.replaceChildren();
  state.boxes.forEach((box) => {
    const info = classInfo(box.class_id);
    const node = document.createElement("div");
    node.className = `bbox ${box.id === state.selectedBoxId ? "selected" : ""}`;
    node.dataset.boxId = box.id;
    node.style.setProperty("--box-color", info.color);
    styleBox(node, box);
    const label = document.createElement("span");
    label.className = "bbox-label";
    label.textContent = info.label;
    node.append(label);
    if (box.id === state.selectedBoxId) {
      ["nw", "n", "ne", "e", "se", "s", "sw", "w"].forEach((handle) => {
        const grip = document.createElement("span");
        grip.className = "resize-handle";
        grip.dataset.handle = handle;
        node.append(grip);
      });
    }
    el.boxOverlay.append(node);
  });
}

function updateStatus() {
  el.frameStatus.className = `status-badge ${state.status}`;
  el.frameStatus.textContent = statusLabel(state.status);
}

function snapshot() {
  state.history.push({ boxes: deepCopy(state.boxes), selectedBoxId: state.selectedBoxId });
  if (state.history.length > 50) state.history.shift();
  state.future = [];
  updateHistoryButtons();
}

function restore(snapshotValue) {
  state.boxes = deepCopy(snapshotValue.boxes);
  state.selectedBoxId = snapshotValue.selectedBoxId;
  markDirty();
  renderAllFrameDetails();
}

function undo() {
  if (!state.history.length) return;
  state.future.push({ boxes: deepCopy(state.boxes), selectedBoxId: state.selectedBoxId });
  restore(state.history.pop());
  updateHistoryButtons();
}

function redo() {
  if (!state.future.length) return;
  state.history.push({ boxes: deepCopy(state.boxes), selectedBoxId: state.selectedBoxId });
  const value = state.future.pop();
  state.boxes = deepCopy(value.boxes);
  state.selectedBoxId = value.selectedBoxId;
  markDirty();
  renderAllFrameDetails();
  updateHistoryButtons();
}

function updateHistoryButtons() {
  el.undoButton.disabled = !state.history.length;
  el.redoButton.disabled = !state.future.length;
}

function markDirty() {
  if (!currentFrame()) return;
  state.dirty = true;
  state.changeVersion += 1;
  if (["annotated", "empty"].includes(state.status)) {
    state.status = "unreviewed";
    updateStatus();
  }
  setSaveState("saving", "Unsaved changes");
  window.clearTimeout(state.saveTimer);
  state.saveTimer = window.setTimeout(() => saveFrame(), 650);
}

function renderAllFrameDetails() {
  renderBoxes();
  renderObjects();
  renderClasses();
  updateStatus();
}

function imageUrl(video, frame) {
  return `/media/annotation-frame?video=${encodeURIComponent(video.video_id)}&frame=${encodeURIComponent(frame.filename)}`;
}

async function loadFrame(index) {
  const video = currentVideo();
  if (!video || index < 0 || index >= video.frames.length) return;
  if (index === state.activeFrameIndex && currentFrame()) return;
  await flushSave();
  state.activeFrameIndex = index;
  const frame = currentFrame();
  const response = await api(`/api/annotation-frame?video=${encodeURIComponent(video.video_id)}&frame=${encodeURIComponent(frame.filename)}`);
  const record = response.record || {};
  if (response.record) {
    state.project.annotations[frameKey(video.video_id, frame.filename)] = deepCopy(response.record);
  }
  state.boxes = deepCopy(record.boxes || []);
  state.selectedBoxId = null;
  state.status = record.status || "unreviewed";
  state.note = record.note || "";
  state.revision = Number(record.revision || 0);
  state.suggestions = deepCopy(response.suggestions || []);
  state.history = [];
  state.future = [];
  state.dirty = false;
  state.changeVersion = 0;
  el.frameNote.value = state.note;
  el.currentFrameTitle.textContent = `Frame ${String(frame.index).padStart(4, "0")}`;
  el.currentFrameMeta.textContent = `${formatTime(frame.timestamp_seconds)} · ${video.split.toUpperCase()} · ${video.video_id.replace("video_", "")}`;
  el.emptyCanvas.hidden = true;
  el.imageShell.hidden = false;
  el.frameImage.src = response.image_url || imageUrl(video, frame);
  el.loadSuggestions.disabled = !state.suggestions.length;
  el.loadSuggestions.textContent = state.suggestions.length ? `Load suggestions (${state.suggestions.length})` : "No suggestions";
  renderFrameList();
  renderAllFrameDetails();
  updateHistoryButtons();
  setSaveState("saved", "Saved");
}

function updateLocalRecord(record, preserveEditor = false) {
  const video = currentVideo();
  const frame = currentFrame();
  if (!video || !frame) return;
  const wasComplete = ["annotated", "empty"].includes(frame.status);
  const isComplete = ["annotated", "empty"].includes(record.status);
  if (wasComplete !== isComplete) {
    const delta = isComplete ? 1 : -1;
    video.complete_count += delta;
    state.project.summary.complete_frames += delta;
  }
  state.project.annotations[frameKey(video.video_id, frame.filename)] = deepCopy(record);
  frame.status = record.status;
  frame.box_count = record.boxes.length;
  frame.revision = record.revision;
  state.revision = record.revision;
  if (!preserveEditor) {
    state.status = record.status;
    state.dirty = false;
  }
  renderFrameList();
  updateProgress();
  updateStatus();
}

async function saveFrame(options = {}) {
  const video = currentVideo();
  const frame = currentFrame();
  if (!video || !frame) return null;
  window.clearTimeout(state.saveTimer);
  if (state.savePromise) {
    try { await state.savePromise; } catch (_) { /* the first request reported its error */ }
  }
  if (!state.dirty && !options.force) return null;
  const saveVersion = state.changeVersion;
  const payload = {
    video_id: video.video_id,
    filename: frame.filename,
    boxes: state.boxes,
    status: state.status,
    note: el.frameNote.value.trim(),
    expected_revision: state.revision,
  };
  setSaveState("saving", "Saving…");
  state.savePromise = api("/api/annotation-save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  try {
    const response = await state.savePromise;
    const changedWhileSaving = saveVersion !== state.changeVersion;
    updateLocalRecord(response.record, changedWhileSaving);
    if (changedWhileSaving) {
      setSaveState("saving", "Unsaved changes");
      state.saveTimer = window.setTimeout(() => saveFrame(), 150);
    } else {
      setSaveState("saved", "Saved");
    }
    return response.record;
  } catch (error) {
    if (error.status === 409 && error.payload?.record) {
      toast("This frame changed in another tab. Reloading the saved version.", "error");
      const record = error.payload.record;
      state.boxes = deepCopy(record.boxes || []);
      state.note = record.note || "";
      el.frameNote.value = state.note;
      updateLocalRecord(record);
      renderAllFrameDetails();
    } else {
      setSaveState("error", "Save failed");
      toast(error.message, "error");
    }
    throw error;
  } finally {
    state.savePromise = null;
  }
}

async function flushSave() {
  window.clearTimeout(state.saveTimer);
  if (state.savePromise) {
    try { await state.savePromise; } catch (_) { /* surfaced by saveFrame */ }
  }
  if (state.dirty) {
    try { await saveFrame(); } catch (_) { return; }
  }
}

async function setFrameStatus(status, moveNext = false) {
  if (!currentFrame()) return;
  if (status === "annotated" && !state.boxes.length) {
    toast("Draw at least one box, or choose No objects.", "error");
    return;
  }
  if (status === "empty") {
    if (state.boxes.length && !window.confirm("Remove all boxes and mark this frame as No objects?")) return;
    if (state.boxes.length) snapshot();
    state.boxes = [];
    state.selectedBoxId = null;
  }
  state.status = status;
  state.dirty = true;
  state.changeVersion += 1;
  renderAllFrameDetails();
  try {
    await saveFrame({ force: true });
    if (moveNext) await navigate(1, true);
  } catch (_) { /* saveFrame already explained the error */ }
}

async function navigate(delta, preferTodo = false) {
  const video = currentVideo();
  if (!video) return;
  await flushSave();
  let index = state.activeFrameIndex + delta;
  if (preferTodo && delta > 0) {
    const nextTodo = video.frames.findIndex((frame, candidate) => candidate > state.activeFrameIndex && !["annotated", "empty"].includes(frame.status));
    if (nextTodo >= 0) index = nextTodo;
  }
  if (index >= 0 && index < video.frames.length) await loadFrame(index);
}

function boxId() {
  return `box-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function deleteBox(id = state.selectedBoxId) {
  if (!id) return;
  const index = state.boxes.findIndex((box) => box.id === id);
  if (index < 0) return;
  snapshot();
  state.boxes.splice(index, 1);
  state.selectedBoxId = null;
  markDirty();
  renderAllFrameDetails();
}

function pointFromEvent(event) {
  const rect = el.boxOverlay.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function onPointerDown(event) {
  if (!currentFrame() || event.button !== 0) return;
  if (!state.project.classes.length || state.activeClassId === null) {
    toast("Add and choose an object class before drawing a box.", "error");
    return;
  }
  const point = pointFromEvent(event);
  const boxNode = event.target.closest(".bbox");
  const handle = event.target.dataset.handle;
  if (boxNode) {
    const id = boxNode.dataset.boxId;
    state.selectedBoxId = id;
    const box = state.boxes.find((candidate) => candidate.id === id);
    state.activeClassId = box.class_id;
    snapshot();
    state.pointer = {
      type: handle ? "resize" : "move",
      handle,
      start: point,
      original: deepCopy(box),
      boxId: id,
      changed: false,
    };
    renderAllFrameDetails();
  } else {
    state.selectedBoxId = null;
    state.pointer = { type: "draw", start: point, current: point };
    renderBoxes(); renderObjects();
  }
  el.boxOverlay.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function moveBox(pointer, point) {
  const box = state.boxes.find((candidate) => candidate.id === pointer.boxId);
  if (!box) return;
  const dx = point.x - pointer.start.x;
  const dy = point.y - pointer.start.y;
  box.x = Math.min(1 - box.width, Math.max(0, pointer.original.x + dx));
  box.y = Math.min(1 - box.height, Math.max(0, pointer.original.y + dy));
  pointer.changed = Math.abs(dx) > .0005 || Math.abs(dy) > .0005;
}

function resizeBox(pointer, point) {
  const box = state.boxes.find((candidate) => candidate.id === pointer.boxId);
  if (!box) return;
  const original = pointer.original;
  const minSize = .003;
  let left = original.x;
  let top = original.y;
  let right = original.x + original.width;
  let bottom = original.y + original.height;
  if (pointer.handle.includes("w")) left = Math.min(point.x, right - minSize);
  if (pointer.handle.includes("e")) right = Math.max(point.x, left + minSize);
  if (pointer.handle.includes("n")) top = Math.min(point.y, bottom - minSize);
  if (pointer.handle.includes("s")) bottom = Math.max(point.y, top + minSize);
  box.x = Math.max(0, left);
  box.y = Math.max(0, top);
  box.width = Math.min(1, right) - box.x;
  box.height = Math.min(1, bottom) - box.y;
  pointer.changed = true;
}

function renderDrawingBox(pointer) {
  renderBoxes();
  const left = Math.min(pointer.start.x, pointer.current.x);
  const top = Math.min(pointer.start.y, pointer.current.y);
  const width = Math.abs(pointer.current.x - pointer.start.x);
  const height = Math.abs(pointer.current.y - pointer.start.y);
  const node = document.createElement("div");
  node.className = "drawing-box";
  styleBox(node, { x: left, y: top, width, height });
  el.boxOverlay.append(node);
}

function onPointerMove(event) {
  if (!state.pointer) return;
  const point = pointFromEvent(event);
  if (state.pointer.type === "draw") {
    state.pointer.current = point;
    renderDrawingBox(state.pointer);
  } else if (state.pointer.type === "move") {
    moveBox(state.pointer, point);
    renderBoxes();
  } else {
    resizeBox(state.pointer, point);
    renderBoxes();
  }
  event.preventDefault();
}

function onPointerUp(event) {
  const pointer = state.pointer;
  if (!pointer) return;
  if (pointer.type === "draw") {
    const left = Math.min(pointer.start.x, pointer.current.x);
    const top = Math.min(pointer.start.y, pointer.current.y);
    const width = Math.abs(pointer.current.x - pointer.start.x);
    const height = Math.abs(pointer.current.y - pointer.start.y);
    if (width >= .003 && height >= .003) {
      snapshot();
      const id = boxId();
      state.boxes.push({ id, class_id: state.activeClassId, x: left, y: top, width, height });
      state.selectedBoxId = id;
      markDirty();
    }
  } else if (pointer.changed) {
    markDirty();
  } else {
    // The snapshot was only provisional for a click without movement.
    state.history.pop();
    updateHistoryButtons();
  }
  state.pointer = null;
  try { el.boxOverlay.releasePointerCapture(event.pointerId); } catch (_) { /* no-op */ }
  renderAllFrameDetails();
}

function importSuggestions() {
  if (!state.suggestions.length) return;
  if (state.boxes.length && !window.confirm("Replace the current boxes with silver suggestions?")) return;
  snapshot();
  state.boxes = state.suggestions.map((box) => ({ ...deepCopy(box), id: boxId() }));
  state.selectedBoxId = state.boxes[0]?.id || null;
  if (state.boxes[0]) state.activeClassId = state.boxes[0].class_id;
  markDirty();
  renderAllFrameDetails();
  toast(`${state.boxes.length} suggestions loaded. Check every box before confirming.`);
}

async function copyPreviousBoxes() {
  const video = currentVideo();
  if (!video || state.activeFrameIndex <= 0) {
    toast("There is no previous frame.", "error");
    return;
  }
  const videoIndex = state.activeVideoIndex;
  const frameIndex = state.activeFrameIndex;
  const previous = video.frames[frameIndex - 1];
  const key = frameKey(video.video_id, previous.filename);
  let record = state.project.annotations[key];
  if (!record) {
    try {
      const response = await api(`/api/annotation-frame?video=${encodeURIComponent(video.video_id)}&frame=${encodeURIComponent(previous.filename)}`);
      record = response.record;
      if (record) state.project.annotations[key] = deepCopy(record);
    } catch (error) {
      toast(`Could not load the previous frame: ${error.message}`, "error");
      return;
    }
  }
  if (videoIndex !== state.activeVideoIndex || frameIndex !== state.activeFrameIndex) return;
  if (!record?.boxes?.length) {
    toast("The previous frame has no saved boxes.", "error");
    return;
  }
  if (state.boxes.length && !window.confirm("Replace the current boxes with boxes from the previous frame?")) return;
  snapshot();
  state.boxes = record.boxes.map((box) => ({ ...deepCopy(box), id: boxId() }));
  state.selectedBoxId = state.boxes[0]?.id || null;
  markDirty();
  renderAllFrameDetails();
}

function fittedImageWidth() {
  const image = el.frameImage;
  if (!image.naturalWidth || !image.naturalHeight) return 0;
  const viewport = el.canvasViewport;
  const style = window.getComputedStyle(viewport);
  const horizontalPadding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const verticalPadding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  const availableWidth = Math.max(1, viewport.clientWidth - horizontalPadding);
  const availableHeight = Math.max(1, viewport.clientHeight - verticalPadding);
  return Math.min(availableWidth, availableHeight * image.naturalWidth / image.naturalHeight);
}

function setZoom(value) {
  state.zoom = Math.max(60, Math.min(300, Number(value) || 100));
  const fittedWidth = fittedImageWidth();
  if (fittedWidth) {
    el.imageShell.style.width = `${Math.max(1, Math.floor(fittedWidth * state.zoom / 100))}px`;
  }
  el.zoomRange.value = String(state.zoom);
  el.zoomValue.textContent = `${state.zoom}%`;
}

function fitImage() {
  setZoom(100);
  el.canvasViewport.scrollTo({ top: 0, left: 0, behavior: "auto" });
}

const classColors = [
  "#007c91", "#e57a44", "#6c5ce7", "#2d8bcb", "#00a878",
  "#c44e8b", "#946b2d", "#607d8b", "#d1495b", "#3a7d44",
];

function showClassForm(show = true) {
  el.classForm.hidden = !show;
  if (show) {
    el.className.value = "";
    el.classColor.value = classColors[state.project.classes.length % classColors.length];
    requestAnimationFrame(() => el.className.focus());
  }
}

async function addClass(event) {
  event.preventDefault();
  const label = el.className.value.trim();
  if (!label) {
    toast("Enter a class name.", "error");
    el.className.focus();
    return;
  }
  try {
    const response = await api("/api/annotation-class", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, color: el.classColor.value }),
    });
    state.project.classes = response.classes;
    state.activeClassId = response.class.id;
    state.suggestions = [];
    showClassForm(false);
    renderClasses();
    renderObjects();
    el.loadSuggestions.disabled = true;
    el.loadSuggestions.textContent = "Suggestions unavailable";
    toast(`${response.class.label} added as class ${response.class.id}.`);
  } catch (error) {
    toast(error.message, "error");
  }
}

function bindEvents() {
  el.frameFilter.addEventListener("change", renderFrameList);
  el.frameSearch.addEventListener("input", renderFrameList);
  el.previousFrame.addEventListener("click", () => navigate(-1));
  el.nextFrame.addEventListener("click", () => navigate(1));
  el.undoButton.addEventListener("click", undo);
  el.redoButton.addEventListener("click", redo);
  el.copyPrevious.addEventListener("click", copyPreviousBoxes);
  el.loadSuggestions.addEventListener("click", importSuggestions);
  el.addClassButton.addEventListener("click", () => showClassForm(el.classForm.hidden));
  el.cancelClass.addEventListener("click", () => showClassForm(false));
  el.classForm.addEventListener("submit", addClass);
  el.fitButton.addEventListener("click", fitImage);
  el.zoomRange.addEventListener("input", (event) => setZoom(event.target.value));
  el.frameImage.addEventListener("load", fitImage);
  window.addEventListener("resize", () => window.requestAnimationFrame(() => setZoom(state.zoom)));
  el.confirmFrame.addEventListener("click", () => setFrameStatus("annotated", true));
  el.markEmpty.addEventListener("click", () => setFrameStatus("empty", true));
  el.markReview.addEventListener("click", () => setFrameStatus("needs_review", true));
  el.markDraft.addEventListener("click", () => setFrameStatus("unreviewed", false));
  el.frameNote.addEventListener("input", () => {
    state.note = el.frameNote.value;
    markDirty();
  });
  el.boxOverlay.addEventListener("pointerdown", onPointerDown);
  el.boxOverlay.addEventListener("pointermove", onPointerMove);
  el.boxOverlay.addEventListener("pointerup", onPointerUp);
  el.boxOverlay.addEventListener("pointercancel", onPointerUp);
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty || state.savePromise) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  document.addEventListener("keydown", (event) => {
    const editingText = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault(); event.shiftKey ? redo() : undo(); return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault(); redo(); return;
    }
    if (editingText) return;
    if (/^[1-9]$/.test(event.key)) {
      const classId = Number(event.key) - 1;
      if (classInfo(classId)) selectClass(classId);
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); deleteBox(); return; }
    if (event.key === "ArrowLeft") { event.preventDefault(); navigate(-1); return; }
    if (event.key === "ArrowRight") { event.preventDefault(); navigate(1); return; }
    if (event.key === " ") { event.preventDefault(); setFrameStatus("annotated", true); return; }
    if (event.key.toLowerCase() === "s") { importSuggestions(); return; }
    if (event.key.toLowerCase() === "c") { copyPreviousBoxes(); return; }
    if (event.key.toLowerCase() === "f") fitImage();
  });
}

async function initialize() {
  cacheElements();
  bindEvents();
  try {
    state.project = await api("/api/annotation-project");
    state.project.annotations = {};
    await ensureVideoFrames(0);
    state.activeClassId = state.project.classes[0]?.id ?? null;
    el.videoCount.textContent = String(state.project.videos.length);
    el.projectHeading.textContent = `${state.project.videos.length} OSCE videos`;
    el.splitPolicy.textContent = "Leakage-safe video split · train / dev / test locked";
    renderClasses();
    renderVideos();
    renderFrameList();
    updateProgress();
    const firstVideo = currentVideo();
    const firstTodo = firstVideo.frames.findIndex((frame) => !["annotated", "empty"].includes(frame.status));
    await loadFrame(firstTodo >= 0 ? firstTodo : 0);
  } catch (error) {
    setSaveState("error", "Could not load project");
    toast(error.message, "error");
  }
}

document.addEventListener("DOMContentLoaded", initialize);
