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

function buildRail(requestedStages) {
  const rail = $("#rail");
  const rows = [];
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
  }
  rail.replaceChildren(...rows);
}

function setPhase(number) {
  for (const [n] of PHASES) {
    const row = $(`#phase-${n}`);
    if (!row) continue;
    if (n < number) row.dataset.status = "done";
    else if (n === number) row.dataset.status = "running";
  }
}

function setStage(key, entry) {
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
  const rows = $$("#rail li.sub");
  const settled = rows.filter((r) => ["done", "skipped"].includes(r.dataset.status)).length;
  const pct = rows.length ? Math.round((settled / rows.length) * 100) : 0;
  $("#progress").style.width = `${pct}%`;
}

/* ── Console ──────────────────────────────────────────────────────────────── */

function lineClass(line) {
  if (/ERROR|❌|Traceback/.test(line)) return "line-error";
  if (/WARNING|⚠/.test(line)) return "line-warn";
  if (/STAGE|Sub-stage|✓ \[|EXECUTION COMPLETE|Partial run complete/.test(line)) return "line-stage";
  return "";
}

function isImportant(line) {
  return /STAGE|Sub-stage|✓ \[|EXECUTION COMPLETE|Partial run complete|WARNING|ERROR|Generation|Final Decision|Best ensemble/.test(line);
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

function applySnapshot(snapshot) {
  startedAt = snapshot.started_at || startedAt;
  $("#argv").textContent = (snapshot.argv || []).join(" ");
  $("#download-log").href = `/api/runs/${jobId}/log?download=1`;
  const params = snapshot.params || {};
  $("#run-subtitle").textContent =
    `${params.dataset || "?"} · entity ${params.entity || "?"}`;
  if (snapshot.phase) setPhase(snapshot.phase.number);
  Object.entries(snapshot.stages || {}).forEach(([key, entry]) => setStage(key, entry));
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
  updateProgress();
}

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
  buildRail((snapshot.params || {}).stages);
  applySnapshot(snapshot);

  const chunk = await getJSON(`/api/runs/${jobId}/log`);
  appendLines(chunk.lines);

  if (!finished) connectSSE(chunk.cursor);

  const box = $("#console");
  box.addEventListener("scroll", () => {
    // Disengage the moment the reader scrolls up; re-engage at the bottom.
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
    stickToBottom = atBottom;
    $("#stick-hint").textContent = atBottom
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
