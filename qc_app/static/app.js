// CompoSET QC Interface - Main Application

// ── State ───────────────────────────────────────────────────────────────────

const state = {
  scenes: [],
  sceneIndex: 0,
  batchIndex: 0,
  mode: "base",  // "base", "variations", or "l1"
  hideReviewed: false,
  // Snapshots of on-disk data for diff coloring
  onDiskCorrected: null,   // UC1: corrected caption as loaded from server
  onDiskVariations: {},     // UC2: {var_id: {positive, negative}} as loaded
  currentBatch: null,       // UC2: current batch of variation objects
  actedOn: new Set(),       // UC2: var IDs explicitly submitted/discarded this batch
};

// ── Diff Algorithm ──────────────────────────────────────────────────────────
// Port of bold_diff() from pipeline/gen_report.py

function lcsOpcodes(a, b) {
  // Compute opcodes similar to Python's SequenceMatcher.get_opcodes()
  const m = a.length, n = b.length;
  // Build LCS table
  const dp = Array.from({length: m + 1}, () => new Uint16Array(n + 1));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i-1] === b[j-1]
        ? dp[i-1][j-1] + 1
        : Math.max(dp[i-1][j], dp[i][j-1]);
    }
  }

  // Backtrack to get matching pairs
  const matches = [];
  let i = m, j = n;
  while (i > 0 && j > 0) {
    if (a[i-1] === b[j-1]) {
      matches.push([i-1, j-1]);
      i--; j--;
    } else if (dp[i-1][j] >= dp[i][j-1]) {
      i--;
    } else {
      j--;
    }
  }
  matches.reverse();

  // Convert matches to opcodes
  const opcodes = [];
  let ai = 0, bj = 0;
  for (const [mi, mj] of matches) {
    if (mi > ai || mj > bj) {
      opcodes.push(["replace", ai, mi, bj, mj]);
    }
    opcodes.push(["equal", mi, mi + 1, mj, mj + 1]);
    ai = mi + 1;
    bj = mj + 1;
  }
  if (ai < m || bj < n) {
    opcodes.push(["replace", ai, m, bj, n]);
  }
  return opcodes;
}

function boldDiff(original, modified) {
  // Returns array of {word, weight} where weight is "normal" or "bold"
  const origWords = original.split(/\s+/).filter(w => w);
  const modWords = modified.split(/\s+/).filter(w => w);
  if (!modWords.length) return [];

  const opcodes = lcsOpcodes(origWords, modWords);
  const result = [];

  for (const [tag, i1, i2, j1, j2] of opcodes) {
    if (tag === "equal") {
      for (let k = j1; k < j2; k++) {
        result.push({word: modWords[k], weight: "normal"});
      }
    } else {
      for (let k = j1; k < j2; k++) {
        result.push({word: modWords[k], weight: "bold"});
      }
    }
  }
  return result;
}

// ── Diff Rendering ──────────────────────────────────────────────────────────

function renderDiffHTML(original, currentText, onDiskText) {
  // Three-way diff rendering:
  //   orange bold: words in onDiskText that differ from original (auto-corrections)
  //   teal bold: words in currentText that differ from onDiskText (user edits)
  // We render currentText with appropriate coloring.

  const origWords = original.split(/\s+/).filter(w => w);
  const diskWords = onDiskText.split(/\s+/).filter(w => w);
  const curWords = currentText.split(/\s+/).filter(w => w);

  // First pass: which words in onDiskText differ from original
  const diskDiff = boldDiff(original, onDiskText);
  const diskBoldSet = new Set();
  diskDiff.forEach((d, i) => { if (d.weight === "bold") diskBoldSet.add(i); });

  // Second pass: diff between onDiskText and currentText
  const userDiff = boldDiff(onDiskText, currentText);

  // Now render currentText. For each word in currentText:
  // If it was changed by the user (differs from onDisk) -> teal
  // Else map back to onDisk position and check if that was an auto-correction -> orange
  // Else -> normal

  // Map current words back to disk words via opcodes
  const opcodes = lcsOpcodes(diskWords, curWords);
  const html = [];

  for (const [tag, i1, i2, j1, j2] of opcodes) {
    if (tag === "equal") {
      for (let k = j1; k < j2; k++) {
        const diskIdx = i1 + (k - j1);
        if (diskBoldSet.has(diskIdx)) {
          html.push(`<span class="diff-auto">${escapeHTML(curWords[k])}</span>`);
        } else {
          html.push(escapeHTML(curWords[k]));
        }
      }
    } else {
      // These words were changed by the user
      for (let k = j1; k < j2; k++) {
        html.push(`<span class="diff-user">${escapeHTML(curWords[k])}</span>`);
      }
    }
  }

  return html.join(" ");
}

function renderSimpleDiffHTML(original, modified) {
  // Two-way diff: orange bold for differences
  const diff = boldDiff(original, modified);
  return diff.map(d => {
    if (d.weight === "bold") {
      return `<span class="diff-auto">${escapeHTML(d.word)}</span>`;
    }
    return escapeHTML(d.word);
  }).join(" ");
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── API Helpers ─────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  });
  return res.json();
}

// ── Scene Navigation ────────────────────────────────────────────────────────

async function loadScenes() {
  state.scenes = await fetchJSON("/api/scenes");
  renderSceneSelector();
}

function isSceneReviewed(s) {
  if (state.mode === "base") {
    return s.base_reviewed;
  }
  if (state.mode === "l1") {
    return s.l1_reviewed;
  }
  // Variations: reviewed if all variations have been reviewed
  return s.n_variations > 0 && s.n_reviewed >= s.n_variations;
}

function renderSceneSelector() {
  const sel = document.getElementById("scene-select");
  sel.innerHTML = "";
  state.scenes.forEach((s, i) => {
    if (state.hideReviewed && isSceneReviewed(s)) return;

    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${s.id}`;
    if (state.mode === "variations") {
      opt.textContent += ` (${s.n_reviewed}/${s.n_variations} reviewed)`;
    } else if (state.mode === "l1") {
      opt.textContent += s.l1_reviewed ? " [reviewed]" : "";
    } else {
      opt.textContent += s.base_reviewed ? " [reviewed]" : "";
    }
    sel.appendChild(opt);
  });
  if (!sel.options.length) {
    const opt = document.createElement("option");
    opt.value = -1;
    opt.textContent = "(all reviewed)";
    opt.disabled = true;
    sel.appendChild(opt);
    sel.value = -1;
    return;
  }
  // If current scene is hidden, jump to first visible
  if (sel.querySelector(`option[value="${state.sceneIndex}"]`)) {
    sel.value = state.sceneIndex;
  } else {
    state.sceneIndex = parseInt(sel.options[0].value);
    sel.value = state.sceneIndex;
  }
}

function currentSceneId() {
  return state.scenes[state.sceneIndex]?.id;
}

async function autoApproveBatch() {
  // Auto-approve any variations in current batch that weren't explicitly acted on
  if (state.mode !== "variations" || !state.currentBatch) return;
  const sid = currentSceneId();
  const promises = [];
  for (const v of state.currentBatch) {
    if (!v.qc_status && !state.actedOn.has(v.id)) {
      promises.push(postJSON(`/api/variation/${sid}/${v.id}/save`, {
        positive: v.positive,
        negative: v.negative,
      }));
    }
  }
  if (promises.length) await Promise.all(promises);
}

async function jumpToLastBatch() {
  if (state.mode !== "variations") return;
  await autoApproveBatch();
  const totalVars = state.scenes[state.sceneIndex]?.n_variations || 0;
  const totalBatches = Math.max(1, Math.ceil(totalVars / 4));
  state.batchIndex = totalBatches - 1;
  await loadScenes();
  loadVariations();
  saveSession();
}

async function navigate(delta) {
  if (state.mode === "variations") {
    await autoApproveBatch();

    // Try next batch first
    const totalVars = state.scenes[state.sceneIndex]?.n_variations || 0;
    const totalBatches = Math.ceil(totalVars / 4);
    const newBatch = state.batchIndex + delta;

    if (newBatch >= 0 && newBatch < totalBatches) {
      state.batchIndex = newBatch;
      await loadScenes();
      loadVariations();
      saveSession();
      return;
    }
  }

  // Move to next/prev scene, skipping hidden ones
  let newIdx = state.sceneIndex + delta;
  while (newIdx >= 0 && newIdx < state.scenes.length) {
    if (!state.hideReviewed || !isSceneReviewed(state.scenes[newIdx])) break;
    newIdx += delta;
  }
  if (newIdx >= 0 && newIdx < state.scenes.length) {
    state.sceneIndex = newIdx;
    state.batchIndex = delta < 0 ? 999 : 0;  // 999 = will clamp to last batch
    document.getElementById("scene-select").value = newIdx;
    if (state.mode === "base") loadBaseCaption();
    else if (state.mode === "l1") loadL1Pairs();
    else { await loadScenes(); loadVariations(); }
    saveSession();
  }
}

// ── UC1: Base Caption Correction ────────────────────────────────────────────

async function loadBaseCaption() {
  const sid = currentSceneId();
  if (!sid) return;

  const data = await fetchJSON(`/api/scene/${sid}/base`);

  // Store on-disk corrected for diff coloring
  state.onDiskCorrected = data.corrected_caption;

  // Base image
  const imgEl = document.getElementById("base-image");
  imgEl.src = data.base_image || "";
  imgEl.style.display = data.base_image ? "block" : "none";

  // Base caption (read-only)
  const baseDiv = document.getElementById("base-caption-text");
  baseDiv.innerHTML = formatCaptionReadonly(data.base_caption);

  // Corrected caption (editable with diff)
  renderCorrectedCaption(data.base_caption, data.corrected_caption);

  // Reset submit button
  const btn = document.getElementById("btn-submit-base");
  btn.textContent = "Submit";
  btn.className = "btn btn-submit";
  btn.disabled = false;

  updateNavInfo();
}

function formatCaptionReadonly(caption) {
  let html = `<div class="caption-line"><strong>base_caption:</strong> "${escapeHTML(caption.base_caption)}"</div>`;
  for (const [key, val] of Object.entries(caption.detail_sentences)) {
    html += `<div class="caption-line"><strong>${escapeHTML(key)}:</strong> "${escapeHTML(val)}"</div>`;
  }
  return html;
}

function renderCorrectedCaption(baseCaption, correctedCaption) {
  const container = document.getElementById("corrected-caption-editor");
  container.innerHTML = "";

  // Create editable blocks for each caption line
  const lines = [
    {key: "base_caption", base: baseCaption.base_caption, corrected: correctedCaption.base_caption},
    ...Object.keys(correctedCaption.detail_sentences).map(key => ({
      key,
      base: baseCaption.detail_sentences[key] || "",
      corrected: correctedCaption.detail_sentences[key],
    })),
  ];

  for (const line of lines) {
    const row = document.createElement("div");
    row.className = "caption-edit-row";

    const label = document.createElement("strong");
    label.textContent = line.key + ": ";
    row.appendChild(label);

    const editDiv = document.createElement("div");
    editDiv.className = "caption-editable";
    editDiv.contentEditable = true;
    editDiv.dataset.key = line.key;
    editDiv.dataset.base = line.base;
    editDiv.dataset.ondisk = line.corrected;

    // Render with diff highlighting
    editDiv.innerHTML = renderDiffHTML(line.base, line.corrected, line.corrected);

    // On input: recompute diff
    editDiv.addEventListener("input", () => {
      const plainText = editDiv.textContent;
      const cursorOffset = saveCursorOffset(editDiv);
      editDiv.innerHTML = renderDiffHTML(line.base, plainText, line.corrected);
      restoreCursorOffset(editDiv, cursorOffset);
    });

    row.appendChild(editDiv);
    container.appendChild(row);
  }
}

function getEditedCaption() {
  const rows = document.querySelectorAll("#corrected-caption-editor .caption-edit-row");
  const result = {base_caption: "", detail_sentences: {}};

  rows.forEach(row => {
    const editDiv = row.querySelector(".caption-editable");
    const key = editDiv.dataset.key;
    const text = editDiv.textContent.trim();
    if (key === "base_caption") {
      result.base_caption = text;
    } else {
      result.detail_sentences[key] = text;
    }
  });
  return result;
}

async function submitBaseCaption() {
  const sid = currentSceneId();
  const data = getEditedCaption();
  await postJSON(`/api/scene/${sid}/base`, data);

  const btn = document.getElementById("btn-submit-base");
  btn.textContent = "Submitted";
  btn.className = "btn btn-submitted";
  btn.disabled = true;

  // Refresh scene list
  await loadScenes();
}

// ── UC2: Variation Review ───────────────────────────────────────────────────

async function loadVariations() {
  const sid = currentSceneId();
  if (!sid) return;

  const data = await fetchJSON(`/api/scene/${sid}/variations`);

  // Clamp batch index
  const totalBatches = Math.ceil(data.variations.length / 4);
  if (state.batchIndex >= totalBatches) state.batchIndex = totalBatches - 1;
  if (state.batchIndex < 0) state.batchIndex = 0;

  // Base image
  const imgEl = document.getElementById("var-base-image");
  imgEl.src = data.base_image || "";

  // Get current batch
  const start = state.batchIndex * 4;
  const batch = data.variations.slice(start, start + 4);

  // Track current batch for auto-approve on navigate
  state.currentBatch = batch;
  state.actedOn = new Set();

  // Store on-disk values
  state.onDiskVariations = {};
  for (const v of data.variations) {
    state.onDiskVariations[v.id] = {positive: v.positive, negative: v.negative};
  }

  // Render grid
  const grid = document.getElementById("var-grid");
  grid.innerHTML = "";

  for (let i = 0; i < 4; i++) {
    const cell = document.createElement("div");
    cell.className = "var-cell";

    if (i >= batch.length) {
      cell.classList.add("var-cell-empty");
      grid.appendChild(cell);
      continue;
    }

    const v = batch[i];
    const isDiscarded = v.qc_status === "discarded";
    const isReviewed = v.qc_status === "approved" || v.qc_status === "edited";

    if (isDiscarded) cell.classList.add("var-cell-discarded");

    // Variation image
    const img = document.createElement("img");
    img.className = "var-image";
    if (v.image) {
      img.src = v.image;
    } else {
      img.style.display = "none";
      const placeholder = document.createElement("div");
      placeholder.className = "var-image-placeholder";
      placeholder.textContent = `${v.id} (no image)`;
      cell.appendChild(placeholder);
    }
    cell.appendChild(img);

    // Edit type badge
    const badge = document.createElement("div");
    badge.className = "var-badge";
    badge.textContent = v.edit_types.join(", ");
    cell.appendChild(badge);

    // Edit details
    if (v.edits.length) {
      const details = document.createElement("div");
      details.className = "var-edit-details";
      details.textContent = v.edits.map(e =>
        `${e.attribute}: ${e.from} \u2192 ${e.to} (${e.target_object})`
      ).join("; ");
      cell.appendChild(details);
    }

    // Positive caption (label + editable in one row)
    const posRow = document.createElement("div");
    posRow.className = "var-caption-row";

    const posLabel = document.createElement("div");
    posLabel.className = "var-caption-label";
    posLabel.textContent = "positive:";
    posRow.appendChild(posLabel);

    const posEdit = document.createElement("div");
    posEdit.className = "caption-editable var-caption-pos";
    posEdit.contentEditable = !isDiscarded;
    posEdit.dataset.varId = v.id;
    posEdit.textContent = v.positive;
    posRow.appendChild(posEdit);
    cell.appendChild(posRow);

    // Negative caption (label + editable in one row, with diff)
    const negRow = document.createElement("div");
    negRow.className = "var-caption-row";

    const negLabel = document.createElement("div");
    negLabel.className = "var-caption-label";
    negLabel.textContent = "negative:";
    negRow.appendChild(negLabel);

    const negEdit = document.createElement("div");
    negEdit.className = "caption-editable var-caption-neg";
    negEdit.contentEditable = !isDiscarded;
    negEdit.dataset.varId = v.id;
    negEdit.dataset.ondiskNeg = v.negative;
    negEdit.innerHTML = renderSimpleDiffHTML(v.positive, v.negative);
    negRow.appendChild(negEdit);
    cell.appendChild(negRow);

    // Update neg diff when pos or neg changes
    const updateNegDiff = () => {
      const posText = posEdit.textContent;
      const negText = negEdit.textContent;
      const cursorOffset = saveCursorOffset(negEdit);
      negEdit.innerHTML = renderDiffHTML(posText, negText, v.negative);
      restoreCursorOffset(negEdit, cursorOffset);
    };
    posEdit.addEventListener("input", () => {
      // When positive changes, re-render negative diff
      const posText = posEdit.textContent;
      const negText = negEdit.textContent;
      negEdit.innerHTML = renderSimpleDiffHTML(posText, negText);
    });
    negEdit.addEventListener("input", updateNegDiff);

    // Buttons
    const btnRow = document.createElement("div");
    btnRow.className = "var-btn-row";

    const submitBtn = document.createElement("button");
    submitBtn.className = isReviewed ? "btn btn-submitted" : "btn btn-submit";
    submitBtn.textContent = isReviewed ? "Submitted" : "Submit";
    submitBtn.disabled = isDiscarded;
    submitBtn.onclick = async () => {
      if (submitBtn.textContent === "Submitted") {
        // Undo: re-enable editing
        submitBtn.textContent = "Submit";
        submitBtn.className = "btn btn-submit";
        posEdit.contentEditable = true;
        negEdit.contentEditable = true;
        return;
      }
      const pos = posEdit.textContent.trim();
      const neg = negEdit.textContent.trim();
      await postJSON(`/api/variation/${sid}/${v.id}/save`, {positive: pos, negative: neg});
      state.actedOn.add(v.id);
      submitBtn.textContent = "Submitted";
      submitBtn.className = "btn btn-submitted";
      await loadScenes();
    };
    btnRow.appendChild(submitBtn);

    const discardBtn = document.createElement("button");
    discardBtn.className = isDiscarded ? "btn btn-discarded" : "btn btn-discard";
    discardBtn.textContent = isDiscarded ? "Discarded" : "Discard";
    discardBtn.disabled = isDiscarded;
    discardBtn.onclick = async () => {
      await postJSON(`/api/variation/${sid}/${v.id}/discard`, {});
      state.actedOn.add(v.id);
      cell.classList.add("var-cell-discarded");
      posEdit.contentEditable = false;
      negEdit.contentEditable = false;
      submitBtn.disabled = true;
      discardBtn.textContent = "Discarded";
      discardBtn.className = "btn btn-discarded";
      discardBtn.disabled = true;
      await loadScenes();
    };
    btnRow.appendChild(discardBtn);

    cell.appendChild(btnRow);
    grid.appendChild(cell);
  }

  updateNavInfo();
}

// ── UC3: Caption Verbosities (short/medium/long) — read-only review ──────────

async function loadL1Pairs() {
  const sid = currentSceneId();
  if (!sid) return;

  const data = await fetchJSON(`/api/scene/${sid}/verbosities`);
  if (data.error) {
    document.getElementById("l1-list").innerHTML =
      `<div class="l1-empty">${escapeHTML(data.error)}</div>`;
    updateNavInfo();
    return;
  }

  const list = document.getElementById("l1-list");
  list.innerHTML = "";

  for (const v of data.variations) {
    const row = document.createElement("div");
    row.className = "verbosity-row";

    // Left: var image + id
    const imgCol = document.createElement("div");
    imgCol.className = "verbosity-image";
    if (v.image) {
      const img = document.createElement("img");
      img.src = v.image;
      img.alt = v.id;
      imgCol.appendChild(img);
    }
    const idLabel = document.createElement("div");
    idLabel.className = "verbosity-id";
    idLabel.textContent = `${v.id}` + (v.edit_type ? ` — ${v.edit_type}` : "");
    imgCol.appendChild(idLabel);
    row.appendChild(imgCol);

    // Right: L0 + short + medium + long, each pos (plain) + neg (diff)
    const capCol = document.createElement("div");
    capCol.className = "verbosity-captions";

    const tiers = [
      { label: "L0",     pos: v.l0.positive,   neg: v.l0.negative   },
      { label: "short",  pos: v.short.base,    neg: v.short.var     },
      { label: "medium", pos: v.medium.base,   neg: v.medium.var    },
      { label: "long",   pos: v.long.base,     neg: v.long.var      },
    ];
    for (const t of tiers) {
      const block = document.createElement("div");
      block.className = "verbosity-block";
      const posRow = document.createElement("div");
      posRow.innerHTML = `<span class="verbosity-label">${t.label} pos:</span> ${escapeHTML(t.pos)}`;
      block.appendChild(posRow);
      const negRow = document.createElement("div");
      negRow.innerHTML = `<span class="verbosity-label">${t.label} neg:</span> ${renderSimpleDiffHTML(t.pos, t.neg)}`;
      block.appendChild(negRow);
      capCol.appendChild(block);
    }

    row.appendChild(capCol);
    list.appendChild(row);
  }


  updateNavInfo();
}

function updateNavInfo() {
  const sid = currentSceneId();
  const infoEl = document.getElementById("nav-info");

  if (state.mode === "variations") {
    const totalVars = state.scenes[state.sceneIndex]?.n_variations || 0;
    const totalBatches = Math.ceil(totalVars / 4);
    const start = state.batchIndex * 4 + 1;
    const end = Math.min(start + 3, totalVars);
    infoEl.textContent = `Batch ${state.batchIndex + 1}/${totalBatches} (v${String(start).padStart(2,"0")}-v${String(end).padStart(2,"0")})`;
  } else {
    const sceneNum = state.sceneIndex + 1;
    infoEl.textContent = `Scene ${sceneNum}/${state.scenes.length}`;
    if (state.mode === "l1") {
      const count = document.querySelectorAll("#l1-list .verbosity-row").length;
      infoEl.textContent += ` (${count} variations)`;
    }
  }
}

// ── Cursor Management ───────────────────────────────────────────────────────
// Save/restore cursor position in contenteditable after innerHTML replacement

function saveCursorOffset(el) {
  const sel = window.getSelection();
  if (!sel.rangeCount || !el.contains(sel.anchorNode)) return 0;

  const range = document.createRange();
  range.setStart(el, 0);
  range.setEnd(sel.anchorNode, sel.anchorOffset);
  return range.toString().length;
}

function restoreCursorOffset(el, offset) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
  let pos = 0;
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const len = node.textContent.length;
    if (pos + len >= offset) {
      const sel = window.getSelection();
      const range = document.createRange();
      range.setStart(node, offset - pos);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
      return;
    }
    pos += len;
  }
}

// ── Session Persistence ────────────────────────────────────────────────────

function saveSession() {
  sessionStorage.setItem("qc_state", JSON.stringify({
    mode: state.mode,
    sceneIndex: state.sceneIndex,
    batchIndex: state.batchIndex,
    batchIndexSceneId: state.scenes?.[state.sceneIndex]?.id ?? null,
    hideReviewed: state.hideReviewed,
  }));
}

function restoreSession() {
  const saved = sessionStorage.getItem("qc_state");
  if (!saved) return false;
  try {
    const s = JSON.parse(saved);
    state.mode = s.mode || "base";
    state.sceneIndex = s.sceneIndex || 0;
    state.batchIndex = s.batchIndex || 0;
    state._restoredBatchSceneId = s.batchIndexSceneId || null;
    state.hideReviewed = s.hideReviewed || false;
    return true;
  } catch { return false; }
}

// Called after loadScenes() so state.scenes is populated.
// If the restored batchIndex was saved against a different scene than the one
// we're now on, reset it to 0 — stale batch indices from other scenes should
// not bleed through.
function validateRestoredBatch() {
  const currentSid = state.scenes?.[state.sceneIndex]?.id ?? null;
  if (state._restoredBatchSceneId && state._restoredBatchSceneId !== currentSid) {
    state.batchIndex = 0;
  }
  delete state._restoredBatchSceneId;
}

// ── Tab Switching ───────────────────────────────────────────────────────────

function switchMode(mode, resetBatch = false) {
  state.mode = mode;
  if (resetBatch) state.batchIndex = 0;

  document.getElementById("tab-base").classList.toggle("tab-active", mode === "base");
  document.getElementById("tab-vars").classList.toggle("tab-active", mode === "variations");
  document.getElementById("tab-l1").classList.toggle("tab-active", mode === "l1");
  document.getElementById("view-base").style.display = mode === "base" ? "flex" : "none";
  document.getElementById("view-vars").style.display = mode === "variations" ? "flex" : "none";
  document.getElementById("view-l1").style.display = mode === "l1" ? "block" : "none";

  renderSceneSelector();
  if (mode === "base") loadBaseCaption();
  else if (mode === "variations") loadVariations();
  else loadL1Pairs();
  saveSession();
}

// ── Init ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  // Restore session state BEFORE loading scenes (so hideReviewed is set)
  const restored = restoreSession();

  const toggleBtn = document.getElementById("btn-hide-reviewed");
  if (restored) {
    toggleBtn.classList.toggle("btn-toggle-active", state.hideReviewed);
  }

  await loadScenes();
  validateRestoredBatch();

  // Tab buttons
  document.getElementById("tab-base").onclick = () => switchMode("base");
  document.getElementById("tab-vars").onclick = () => switchMode("variations");
  document.getElementById("tab-l1").onclick = () => switchMode("l1");

  // Hide reviewed toggle
  toggleBtn.onclick = () => {
    state.hideReviewed = !state.hideReviewed;
    toggleBtn.classList.toggle("btn-toggle-active", state.hideReviewed);
    renderSceneSelector();
    if (state.mode === "base") loadBaseCaption();
    else if (state.mode === "variations") loadVariations();
    else loadL1Pairs();
    saveSession();
  };

  // Scene selector
  document.getElementById("scene-select").onchange = (e) => {
    state.sceneIndex = parseInt(e.target.value);
    state.batchIndex = 0;
    if (state.mode === "base") loadBaseCaption();
    else if (state.mode === "variations") loadVariations();
    else loadL1Pairs();
    saveSession();
  };

  // Navigation
  document.getElementById("btn-prev").onclick = () => navigate(-1);
  document.getElementById("btn-next").onclick = () => navigate(1);
  document.getElementById("btn-last").onclick = jumpToLastBatch;

  // Submit base caption
  document.getElementById("btn-submit-base").onclick = submitBaseCaption;

  // Keyboard navigation
  document.addEventListener("keydown", (e) => {
    // Don't navigate when editing
    if (e.target.isContentEditable) return;
    if (e.key === "ArrowRight" || e.key === "n") navigate(1);
    if (e.key === "ArrowLeft" || e.key === "p") navigate(-1);
  });
  // Load initial view (restored or default)
  switchMode(state.mode, !restored);
});
