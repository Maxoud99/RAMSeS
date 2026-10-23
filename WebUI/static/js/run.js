/* The live run view.
 *
 * Progress is measured in completed stages, never as a synthetic percentage:
 * the sub-stages take wildly different times (0.3s to 65s on SKAB), so a
 * time-based bar would be a lie.
 *
 * Falls back to polling if EventSource fails twice, so a proxy that buffers
 * SSE does not leave the page frozen.
 */

import { $, $$, el, getJSON, postJSON, duration } from "./dom.js";

const root = $("#run-root");
const jobId = root.dataset.job;

const PHASES = [
  [1, "Loading training data"], [2, "Loading test data"],
  [3, "Training / loading models"], [4, "Injecting synthetic anomalies"],
  [5, "Preparing data"], [6, "Model selection"], [7, "Writing results"],
];
// The online phase follows "Writing results" when the run asked for it
// (WebUI/markers.py ONLINE_PHASE_NUMBER); its scenarios are its sub-rows.
const ONLINE_PHASE = [8, "Online phase"];
const ONLINE_ORDER = ["fixed", "periodic", "threshold", "adaptive_threshold"];
const ONLINE_NAMES = {
  fixed: "Fixed", periodic: "Periodic", threshold: "Threshold",
  adaptive_threshold: "Adaptive threshold", fixed_random: "Fixed random",
};
const SUBSTAGES = [
  ["ga", "Genetic algorithm"], ["thompson", "Thompson Sampling"],
  ["gan", "GAN perturbations"], ["offby", "Off-by-threshold"],
  ["montecarlo", "Monte Carlo"], ["aggregation", "Rank aggregation"],
];

let stickToBottom = true;
let allLines = [];
let startedAt = Date.now() / 1000;
let finished = false;
let sseFailures = 0;

/* ── Rail ─────────────────────────────────────────────────────────────────── */

function buildRail(requestedStages, params = {}) {
  const rail = $("#rail");
  const rows = [];
  if (params.enable_online && !PHASES.some(([n]) => n === ONLINE_PHASE[0])) PHASES.push(ONLINE_PHASE);
  for (const [number, title] of PHASES) {
    rows.push(el("li", { id: `phase-${number}`, "data-status": "pending" },
      el("span", { class: "dot" }),
      el("span", { class: "label", text: `${number}. ${title}` }),
      el("span", { class: "meta" })));
    if (number === 6) {
      for (const [key, label] of SUBSTAGES) {
        // Pre-populate from the request so the plan is visible before the run
        // reaches it; stages that were not asked for start as skipped.
        const asked = key === "aggregation"
          ? !requestedStages || requestedStages.length === 5
          : !requestedStages || requestedStages.includes(key);
        rows.push(el("li", { id: `stage-${key}`, class: "sub",
                             "data-status": asked ? "pending" : "skipped" },
          el("span", { class: "dot" }),
          el("span", { class: "label", text: label }),
          el("span", { class: "meta", text: asked ? "" : "not selected" })));
      }
    }
    if (number === ONLINE_PHASE[0]) {
      // The scenarios the request named, so the plan shows before the phase
      // starts; the log's own order replaces it when it does.
      const strategy = String(params.online_strategy || params.strategy || "").toLowerCase();
      const keys = strategy === "all" ? ONLINE_ORDER : (ONLINE_NAMES[strategy] ? [strategy] : []);
      keys.forEach((key) => rows.push(onlineRow(key)));
    }
  }
  rail.replaceChildren(...rows);
}

/* One scenario of the online phase: its name, its window count at the right,
 * and under it what it is doing. */
function onlineRow(key, label) {
  return el("li", { id: `stage-online_${key}`, class: "sub online-row", "data-status": "pending",
                    title: label || null },
    el("span", { class: "dot" }),
    el("span", { class: "label", text: ONLINE_NAMES[key] || key }),
    el("span", { class: "meta" }),
    el("span", { class: "row-note" }));
}

function ensureOnlinePhase() {
  if (!PHASES.some(([n]) => n === ONLINE_PHASE[0])) {
    PHASES.push(ONLINE_PHASE);
    $("#rail").append(el("li", { id: `phase-${ONLINE_PHASE[0]}`, "data-status": "pending" },
      el("span", { class: "dot" }),
      el("span", { class: "label", text: `${ONLINE_PHASE[0]}. ${ONLINE_PHASE[1]}` }),
      el("span", { class: "meta" })));
  }
}

/* A scenario's progress, from its `online_<key>` stage entry (WebUI/jobs.py
 * Job._record_online): running with window n/total, waiting for a
 * re-optimization, done, or failed with its error. */
function setOnlineStage(key, entry) {
  const scenario = key.replace(/^online_/, "");
  ensureOnlinePhase();
  let row = $(`#stage-online_${scenario}`);
  if (!row) {
    row = onlineRow(scenario, entry.label);
    // In the phase's own order: after the last online row, or after the phase.
    const rows = $$("#rail li.online-row");
    const anchor = rows.length ? rows[rows.length - 1] : $(`#phase-${ONLINE_PHASE[0]}`);
    anchor.after(row);
  }
  if (entry.label) row.title = entry.label;
  const waiting = entry.waiting;
  row.dataset.status = entry.status === "pending" ? "pending" : entry.status;
  row.classList.toggle("is-waiting", !!waiting && entry.status === "running");
  const total = Number(entry.total) || 0;
  const done = Number(entry.window) || 0;
  $(".meta", row).textContent = waiting && entry.status === "running" ? "waiting"
    : total && entry.status !== "pending" ? `${done}/${total}` : "";
  const applied = /(\d+) re-optimization\(s\) applied/.exec(entry.text || "");
  let note = "";
  if (entry.status === "failed") note = entry.text || "";
  else if (waiting && entry.status === "running") {
    note = `Waiting for the re-optimization started at window ${waiting.since_window} (due at window ${waiting.until_window})`;
  } else if (applied) {
    note = `${applied[1]} re-optimization${applied[1] === "1" ? "" : "s"} applied`;
  }
  $(".row-note", row).textContent = note;
  if (entry.status !== "pending") revealOnline(onlinePageUrl());
  updateProgress();
}

function onlinePageUrl() {
  const params = currentParams || {};
  return params.dataset && params.entity
    ? `/online/${encodeURIComponent(params.dataset)}/${encodeURIComponent(params.entity)}` : null;
}

function revealOnline(url) {
  const link = $("#view-online");
  if (!link || !url) return;
  link.href = url;
  link.hidden = false;
}

/* The phase as a whole: skipped (too short) or failed, from the snapshot. */
function applyOnline(state) {
  if (!state) return;
  ensureOnlinePhase();
  const row = $(`#phase-${ONLINE_PHASE[0]}`);
  if (state.status === "skipped") {
    row.dataset.status = "skipped";
    $(".meta", row).textContent = "skipped";
    if (state.error) row.title = state.error;
  } else if (state.status === "failed") {
    row.dataset.status = "failed";
    if (state.error) row.title = state.error;
  } else if (state.status === "complete") {
    row.dataset.status = "done";
  }
}

function setPhase(number) {
  for (const [n] of PHASES) {
    const row = $(`#phase-${n}`);
    if (!row) continue;
    if (n < number) row.dataset.status = "done";
    else if (n === number) row.dataset.status = "running";
  }
  // The same phase, named in the facts under the title, so it is readable
  // without scanning the rail. The stylesheet words it for a finished run.
  const phase = PHASES.find(([n]) => n === number);
  if (phase) $("#run-phase").textContent = `${phase[0]}. ${phase[1]}`;
}

function setStage(key, entry) {
  if (entry.group === "online" || key.startsWith("online_")) {
    setOnlineStage(key, entry);
    return;
  }
  const row = $(`#stage-${key}`);
  if (!row) return;
  row.dataset.status = entry.status;
  const meta = $(".meta", row);
  // No per-stage duration here: the pipeline times its own modules and the
  // comprehensive report is the binding record of those numbers.
  if (entry.status === "done") {
    meta.textContent = "";
  } else if (entry.status === "skipped") {
    meta.textContent = "skipped";
  } else if (entry.status === "running") {
    meta.textContent = "running…";
  }
  if (entry.text) row.title = entry.text;
  updateProgress();
}

function updateProgress() {
  const isSettled = (r) => ["done", "skipped"].includes(r.dataset.status);
  // The bar counts every sub-row, the online scenarios too; the fact under
  // the title counts the selection stages it names.
  const rows = $$("#rail li.sub");
  const settled = rows.filter(isSettled).length;
  const pct = rows.length ? Math.round((settled / rows.length) * 100) : 0;
  $("#progress").style.width = `${pct}%`;
  const stages = $$("#rail li.sub:not(.online-row)");
  $("#run-settled").textContent = stages.length
    ? `${stages.filter(isSettled).length} / ${stages.length}` : "—";
}

/* ── Console ──────────────────────────────────────────────────────────────── */

function lineClass(line) {
  if (/ERROR|❌|Traceback/.test(line)) return "line-error";
  if (/WARNING|⚠/.test(line)) return "line-warn";
  if (/STAGE|Sub-stage|✓ \[|EXECUTION COMPLETE|Partial run complete|ONLINE (PHASE|SCENARIO)/.test(line)) return "line-stage";
  return "";
}

function isImportant(line) {
  return /STAGE|Sub-stage|✓ \[|EXECUTION COMPLETE|Partial run complete|WARNING|ERROR|Generation|Final Decision|Best ensemble|ONLINE (PHASE|SCENARIO|REOPT)/.test(line);
}

function renderConsole() {
  const box = $("#console");
  const filter = ($("#log-filter").value || "").toLowerCase();
  const onlyImportant = $("#important-only").checked;
  const visible = allLines.filter((line) =>
    (!onlyImportant || isImportant(line)) &&
    (!filter || line.toLowerCase().includes(filter)));
  box.replaceChildren(...visible.slice(-4000).map((line) =>
    el("div", { class: lineClass(line), text: line })));
  if (stickToBottom) box.scrollTop = box.scrollHeight;
}

function appendLines(lines) {
  if (!lines || !lines.length) return;
  allLines = allLines.concat(lines);
  renderConsole();
}

/* ── Status ───────────────────────────────────────────────────────────────── */

function banner(kind, text) {
  $("#banners").append(el("p", { class: `banner banner-${kind}`, text }));
}

let currentParams = null;

function applySnapshot(snapshot) {
  startedAt = snapshot.started_at || startedAt;
  currentParams = snapshot.params || currentParams;
  $("#argv").textContent = (snapshot.argv || []).join(" ");
  $("#download-log").href = `/api/runs/${jobId}/log?download=1`;
  const params = snapshot.params || {};
  $("#run-subtitle").textContent =
    `${params.dataset || "?"} · entity ${params.entity || "?"}`;
  if (snapshot.phase) setPhase(snapshot.phase.number);
  Object.entries(snapshot.stages || {}).forEach(([key, entry]) => setStage(key, entry));
  applyOnline(snapshot.online);
  (snapshot.warnings || []).forEach((w) => banner("warn", w.text));
  if (snapshot.status && snapshot.status !== "running") finish(snapshot);
}

function finish(snapshot) {
  if (finished) return;
  finished = true;
  const status = snapshot.status;
  const titles = {
    succeeded: "Run complete",
    succeeded_with_warnings: "Run complete, with warnings",
    failed: "Run failed", cancelled: "Run cancelled", timeout: "Run timed out",
  };
  $("#run-title").textContent = titles[status] || status;
  // Presentation only: the stylesheet keys the status pill and the progress
  // bar's colour off this, so the outcome is a glyph and a word, not a hue.
  root.dataset.status = status;
  // The ticker below stops at finish, so a run opened after it ended would
  // show no time at all; the snapshot carries the measured total.
  if (typeof snapshot.elapsed === "number") $("#elapsed").textContent = duration(snapshot.elapsed);
  $("#cancel").hidden = true;
  $$("#rail li[data-status='running']").forEach((row) => {
    row.dataset.status = status === "succeeded" ? "done" : "failed";
  });
  if (snapshot.failure_reason) {
    banner(status.startsWith("succeeded") ? "warn" : "error", snapshot.failure_reason);
  }
  if (snapshot.result_url) {
    const link = $("#view-result");
    link.href = snapshot.result_url;
    link.hidden = false;                 // never auto-navigate; the log matters
  }
  if (snapshot.report_url) {
    const link = $("#view-report");
    link.href = snapshot.report_url;
    link.hidden = false;
  }
  // Only results this run wrote: an older run's online files for the same
  // entity are not offered as this one's (WebUI/jobs.py sets online_url).
  if (snapshot.online_url) {
    revealOnline(snapshot.online_url);
    // A run without --explain has no explanation to lead to: its online
    // results are then the run's way forward.
    if (!snapshot.result_url) $("#view-online").classList.add("primary", "icon-arrow");
  } else {
    $("#view-online").hidden = true;
  }
  applyOnline(snapshot.online);
  // Nothing streams into the log any more, so "auto-scrolling" would be a
  // promise about output that is not coming.
  $("#stick-hint").textContent = ENDED_HINT;
  updateProgress();
}

const ENDED_HINT = "Run ended — the log above is complete.";

/* ── Transport ────────────────────────────────────────────────────────────── */

function connectSSE(cursor) {
  const source = new EventSource(`/api/runs/${jobId}/events?cursor=${cursor}`);

  source.addEventListener("hello", (event) => applySnapshot(JSON.parse(event.data)));
  source.addEventListener("log", (event) => appendLines(JSON.parse(event.data).lines));
  source.addEventListener("phase", (event) => setPhase(JSON.parse(event.data).number));
  source.addEventListener("stage", (event) => {
    const data = JSON.parse(event.data);
    setStage(data.key, data);
  });
  source.addEventListener("warning", (event) => banner("warn", JSON.parse(event.data).text));
  source.addEventListener("status", (event) => {
    finish(JSON.parse(event.data));
    source.close();
  });
  source.onerror = () => {
    if (finished) { source.close(); return; }
    sseFailures += 1;
    if (sseFailures >= 2) {
      source.close();
      banner("info", "Live streaming unavailable — falling back to polling.");
      poll();
    }
  };
  return source;
}

async function poll() {
  let cursor = allLines.length;
  while (!finished) {
    try {
      const snapshot = await getJSON(`/api/runs/${jobId}`);
      applySnapshot(snapshot);
      const chunk = await getJSON(`/api/runs/${jobId}/log?offset=${cursor}`);
      appendLines(chunk.lines);
      cursor = chunk.cursor;
      if (snapshot.status !== "running" && snapshot.status !== "starting") {
        finish(snapshot);
        return;
      }
    } catch (e) { /* transient; try again */ }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

/* ── Wiring ───────────────────────────────────────────────────────────────── */

async function init() {
  let snapshot;
  try {
    snapshot = await getJSON(`/api/runs/${jobId}`);
  } catch (e) {
    root.replaceChildren(el("p", { class: "banner banner-error", text: "Unknown run." }));
    return;
  }
  buildRail((snapshot.params || {}).stages, snapshot.params || {});
  applySnapshot(snapshot);

  const chunk = await getJSON(`/api/runs/${jobId}/log`);
  appendLines(chunk.lines);

  if (!finished) connectSSE(chunk.cursor);

  const box = $("#console");
  box.addEventListener("scroll", () => {
    // Disengage the moment the reader scrolls up; re-engage at the bottom.
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
    stickToBottom = atBottom;
    $("#stick-hint").textContent = finished ? ENDED_HINT : atBottom
      ? "Auto-scrolling. Scroll up to pause."
      : "Paused — scroll to the bottom to resume.";
  });

  $("#log-filter").addEventListener("input", renderConsole);
  $("#important-only").addEventListener("change", renderConsole);
  $("#cancel").addEventListener("click", async () => {
    $("#cancel").disabled = true;
    try { await postJSON(`/api/runs/${jobId}/cancel`, {}); } catch (e) {}
  });

  setInterval(() => {
    if (finished) return;
    $("#elapsed").textContent = duration(Date.now() / 1000 - startedAt);
  }, 1000);
}

init();
