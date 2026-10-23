/* The online phase page.
 *
 * Reads /api/online/<dataset>/<entity> — the pipeline's own summary.json and
 * one <scenario>.json per replayed scenario (schema: the docstring of
 * Model_Selection/online_phase.py) — and draws it: how the scenarios compare,
 * each one's scores window by window, and what each re-optimisation did.
 * Nothing here computes a score; every number on the page is one the
 * pipeline wrote.
 *
 * The charts are inline SVG built here, no library: lines for the scores, a
 * crosshair and tooltip on pointer and keyboard, and a table twin behind
 * "Show data" so no value is reachable only by hovering. A run still writing
 * its results is polled every 5 seconds until it is complete, interrupted or
 * skipped.
 */

import { $, el, getJSON, duration, timeAgo, familyClass } from "./dom.js";
import { createMonitor } from "./monitor.js";

const root = $("#online-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;
const API = `/api/online/${encodeURIComponent(dataset)}/${encodeURIComponent(entity)}`;
const POLL_MS = 5000;
// Built once and moved into every re-render, so its playback state survives
// the poll (WebUI/static/js/monitor.js).
const monitor = createMonitor({ dataset, entity });

/* One colour per scenario (ramses.css --viz-*), the same on every chart and
 * in every legend, so a scenario is recognised across the page. `name` is the
 * short form used where the full label does not fit. */
const SCENARIOS = {
  fixed: { name: "Fixed", cls: "sc-fixed" },
  periodic: { name: "Periodic", cls: "sc-periodic" },
  threshold: { name: "Threshold", cls: "sc-threshold" },
  adaptive_threshold: { name: "Adaptive threshold", cls: "sc-adaptive" },
  fixed_random: { name: "Fixed random", cls: "sc-other" },
};
const LIVE = new Set(["running", "pending", "waiting_for_reopt"]);
const STATE_LABELS = {
  running: "Running", pending: "Pending", waiting_for_reopt: "Waiting for re-optimization",
  complete: "Complete", interrupted: "Interrupted", skipped: "Skipped", failed: "Failed",
};
const REASONS = {
  periodic: "Periodic", relative_drop: "Relative drop", adaptive_drop: "Adaptive drop",
};
const SIGNALS = {
  ensemble: { title: "Ensemble fitness", short: "Ensemble", noun: "the ensemble's fitness" },
  single: { title: "Single-model score", short: "Single model", noun: "the single model's score" },
};

/* What survives a refresh while the run is polled: the open scenario tab, the
 * tables opened with "Show data", and which control had focus. */
const ui = { tab: null, expanded: new Set(), lastStamp: null, timer: null, catalog: null };
let charts = [];
let resizeObserver = null;

/* ── Small helpers ────────────────────────────────────────────────────────── */

function scenarioMeta(key) {
  return SCENARIOS[key] || { name: key, cls: "sc-other" };
}

function num(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function plural(n, word, many = `${word}s`) {
  return `${n} ${n === 1 ? word : many}`;
}

function stamp(epochSeconds) {
  if (!epochSeconds) return "—";
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function waitText(seconds) {
  return isNum(seconds) && seconds > 0 ? duration(seconds) : "—";
}

/* A Carbon tag for a state: an icon and a word, never the colour alone. */
function stateTag(status, extra = "") {
  return el("span", { class: `state-tag${extra ? ` ${extra}` : ""}`, "data-state": status || "unknown" },
    STATE_LABELS[status] || status || "Unknown");
}

/* A short stroke of the scenario's colour: the key beside its name in the
 * legend, the tables and the tooltip. */
function lineKey(key) {
  return el("span", { class: `key-line ${scenarioMeta(key).cls}`, "aria-hidden": "true" });
}

function chips(names) {
  return (names || []).map((name) => el("span", { class: familyClass(name), text: name }));
}

/* Windows whose labels are one class have no score (scorable === false):
 * runs of them, as [first, last] window indices, so a chart can mark the
 * stretch as a gap instead of leaving the reader to wonder why the line
 * breaks. A summary from before `scorable` existed has none. */
function noScoreRuns(windows) {
  const runs = [];
  let run = null;
  (windows || []).forEach((w) => {
    if (w.scorable === false) {
      if (run && run[1] === w.i - 1) run[1] = w.i;
      else { run = [w.i, w.i]; runs.push(run); }
    } else {
      run = null;
    }
  });
  return runs;
}

function noScoreKey() {
  return el("li", {}, el("span", { class: "key-noscore", "aria-hidden": "true" }),
            el("span", { text: "no score: the window's labels are one class" }));
}

/* Drawn under the lines, the full height of each panel. */
function drawNoScore(node, runs, xs, top, height) {
  runs.forEach(([a, b]) => {
    const x0 = xs.x(a - 0.5), x1 = xs.x(b + 0.5);
    node.append(svg("rect", { class: "no-score", x: x0, y: top, width: Math.max(1, x1 - x0), height }));
  });
}

function noScoreNote(w) {
  if (w.scorable !== false) return "";
  const n = isNum(w.n_anomalous) && isNum(w.n_samples)
    ? ` (${w.n_anomalous} of ${w.n_samples} samples labelled anomalous)` : "";
  return `No score: the window's labels are one class${n}, so F1 and PR-AUC are undefined; it cannot trigger.`;
}

function samplesText(summary, w) {
  if (!isNum(w.start) || !isNum(w.end)) return "";
  const where = summary.window_coordinates === "online_stream" ? "stream samples" : "samples";
  return `${where} ${w.start}–${w.end}`;
}

/* The scenario's rule in words, from the parameters it actually ran with. */
function ruleText(key, params) {
  const p = params || {};
  const delay = isNum(p.reopt_delay) ? ` · switch-in after L = ${p.reopt_delay}` : "";
  if (key === "fixed") return "Never re-optimizes: the offline selection scores every window.";
  if (key === "fixed_random") return "Keeps one random selection for the whole stream (command-line baseline).";
  if (key === "periodic") return `Re-optimizes every ${p.update_interval} windows${delay}.`;
  if (key === "threshold") {
    return `Re-optimizes when the ensemble's fitness or the single model's score falls more `
      + `than ${p.drop_pct}% below its baseline${delay}.`;
  }
  if (key === "adaptive_threshold") {
    return `Re-optimizes below mean − ${p.adaptive_k}·std of the last ${p.adaptive_window} scores `
      + `since the last switch-in; the ${p.drop_pct}% rule applies until ${p.adaptive_window} `
      + `scores exist${delay}.`;
  }
  return "";
}

function windowsOf(doc) {
  return ((doc && doc.windows) || []).slice().sort((a, b) => a.i - b.i);
}

/* ── SVG ──────────────────────────────────────────────────────────────────── */

const SVGNS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

/* The event marks, one shape per kind so they read without colour: trigger ▲,
 * suppressed ×, applied ●, not applied ○, failed ◆. Drawn the same in the
 * chart, its legend and the events table. */
function glyph(kind, x, y, size = 5) {
  const r = size;
  if (kind === "trigger") {
    return svg("path", { class: "glyph glyph-trigger", d: `M${x} ${y - r}L${x + r} ${y + r * 0.8}H${x - r}Z` });
  }
  if (kind === "suppressed") {
    return svg("path", { class: "glyph glyph-suppressed",
      d: `M${x - r * 0.8} ${y - r * 0.8}L${x + r * 0.8} ${y + r * 0.8}M${x + r * 0.8} ${y - r * 0.8}L${x - r * 0.8} ${y + r * 0.8}` });
  }
  if (kind === "applied") return svg("circle", { class: "glyph glyph-applied", cx: x, cy: y, r: r * 0.9 });
  if (kind === "not_applied" || kind === "waiting") {
    return svg("circle", { class: `glyph glyph-${kind}`, cx: x, cy: y, r: r * 0.8 });
  }
  return svg("path", { class: "glyph glyph-failed", d: `M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z` });
}

function glyphIcon(kind) {
  return svg("svg", { class: "glyph-icon", viewBox: "0 0 14 14", width: 14, height: 14,
                      "aria-hidden": "true", focusable: "false" }, glyph(kind, 7, 7, 5));
}

const GLYPH_LABELS = {
  trigger: "Trigger", suppressed: "Trigger suppressed", applied: "Re-optimization applied",
  not_applied: "Not applied (stream ended)", failed: "Failed, timed out or aborted",
  waiting: "Waiting for a re-optimization",
};

/* Clean tick values: steps of 1, 2 or 5 times a power of ten. */
function niceStep(range, count) {
  const raw = range / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * mag;
}

/* The y range of a score chart: the data with a margin, on clean ticks, never
 * outside [0, 1]. Not from zero: these are lines, and on a 0–1 axis the
 * differences between scenarios — a few hundredths — would be flat. */
function yScaleOf(values, height, top) {
  const vs = values.filter(isNum);
  let lo = vs.length ? Math.min(...vs) : 0;
  let hi = vs.length ? Math.max(...vs) : 1;
  const span = Math.max(hi - lo, 0.04);
  lo = Math.max(0, lo - span * 0.2);
  hi = Math.min(1, hi + span * 0.2);
  const step = niceStep(hi - lo, 4);
  lo = Math.max(0, Math.floor(lo / step + 1e-9) * step);
  hi = Math.min(1, Math.ceil(hi / step - 1e-9) * step);
  if (hi <= lo) hi = Math.min(1, lo + step);
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(+v.toFixed(6));
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const y = (v) => top + (1 - (v - lo) / (hi - lo)) * height;
  return { y, ticks, digits, lo, hi };
}

/* Windows are cells: window i is centred in [i − ½, i + ½], so a threshold in
 * effect for a window spans its cell and the steps line up with the points. */
function xScaleOf(first, last, left, width) {
  const lo = first - 0.5, hi = last + 0.5;
  const x = (i) => left + ((i - lo) / (hi - lo)) * width;
  const cell = width / (hi - lo);
  const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];
  const room = Math.max(1, Math.floor(width / 40));
  const step = steps.find((s) => (last - first) / s + 1 <= room) || steps[steps.length - 1];
  const ticks = [];
  for (let i = Math.ceil(first / step) * step; i <= last; i += step) ticks.push(i);
  return { x, ticks, cell, lo, hi };
}

function linePath(points, x, y) {
  let d = "";
  let pen = false;
  for (const p of points) {
    if (!isNum(p.v)) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(p.i).toFixed(1)} ${y(p.v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

function yAxis(g, scale, left, right) {
  for (const t of scale.ticks) {
    const yy = scale.y(t);
    g.append(svg("line", { class: "grid", x1: left, x2: right, y1: yy, y2: yy }),
             svg("text", { class: "tick", x: left - 8, y: yy, dy: "0.32em", "text-anchor": "end",
                           text: t.toFixed(scale.digits) }));
  }
}

function xAxis(g, xs, baseline, left, right, label) {
  g.append(svg("line", { class: "axis", x1: left, x2: right, y1: baseline, y2: baseline }));
  for (const i of xs.ticks) {
    const xx = xs.x(i);
    g.append(svg("line", { class: "axis", x1: xx, x2: xx, y1: baseline, y2: baseline + 4 }),
             svg("text", { class: "tick", x: xx, y: baseline + 16, "text-anchor": "middle", text: String(i) }));
  }
  if (label) {
    g.append(svg("text", { class: "axis-title", x: right, y: baseline + 32, "text-anchor": "end", text: label }));
  }
}

/* ── Interactive plot: crosshair, tooltip, keyboard ───────────────────────── */

/* Wires one plot. `indices` are the window positions the crosshair snaps to;
 * `readout(i)` returns {title, rows, text} for the tooltip and the live
 * region. The plot is one tab stop: arrow keys step through windows, Home and
 * End jump to the ends, Escape hides the tooltip. */
function interactive(plot, tip, live, layout, indices, readout) {
  let current = null;
  let pinned = false;

  const show = (i, announce) => {
    current = i;
    const { x } = layout();
    const cross = plot.querySelector(".crosshair");
    const xx = x(i);
    if (cross) {
      cross.setAttribute("x1", xx); cross.setAttribute("x2", xx);
      cross.removeAttribute("visibility");
    }
    const r = readout(i);
    tip.replaceChildren(
      el("p", { class: "viz-tip-title", text: r.title }),
      ...r.rows);
    tip.hidden = false;
    const width = plot.clientWidth;
    const tipWidth = tip.offsetWidth;
    let left = xx + 12;
    if (left + tipWidth > width) left = xx - 12 - tipWidth;
    // Beside the crosshair where the plot leaves room; on a narrow screen it
    // would cover the plot it describes, so it opens under the plot instead.
    const narrow = width < 560;
    if (narrow) left = xx - tipWidth / 2;
    tip.style.left = `${Math.max(0, Math.min(left, width - tipWidth))}px`;
    tip.style.top = narrow ? `${plot.clientHeight + 4}px` : "8px";
    if (announce && live) live.textContent = r.text;
  };
  const hide = () => {
    tip.hidden = true;
    const cross = plot.querySelector(".crosshair");
    if (cross) cross.setAttribute("visibility", "hidden");
  };
  const nearest = (clientX) => {
    const box = plot.getBoundingClientRect();
    const { x } = layout();
    let best = indices[0], dist = Infinity;
    for (const i of indices) {
      const d = Math.abs(x(i) - (clientX - box.left));
      if (d < dist) { dist = d; best = i; }
    }
    return best;
  };

  plot.addEventListener("pointermove", (event) => {
    if (!indices.length) return;
    pinned = false;
    show(nearest(event.clientX), false);
  });
  plot.addEventListener("pointerleave", () => {
    if (document.activeElement !== plot) hide();
  });
  plot.addEventListener("focus", () => {
    if (!indices.length) return;
    show(current === null ? indices[0] : current, true);
  });
  plot.addEventListener("blur", () => { if (!pinned) hide(); pinned = false; });
  plot.addEventListener("keydown", (event) => {
    if (!indices.length) return;
    const at = Math.max(0, indices.indexOf(current === null ? indices[0] : current));
    const moves = {
      ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: indices.length - 1,
      PageUp: at + 5, PageDown: at - 5,
    };
    if (event.key === "Escape") { hide(); return; }
    if (!(event.key in moves)) return;
    event.preventDefault();
    const next = Math.max(0, Math.min(indices.length - 1, moves[event.key]));
    show(indices[next], true);
  });
  return { refresh: () => { if (!tip.hidden && current !== null) show(current, false); } };
}

/* A chart's frame: a heading, a legend, the plot (one tab stop), a hidden live
 * region for the keyboard readout, and the "Show data" table twin. `draw(plot,
 * width)` (re)builds the SVG at the plot's width; it is called once inserted
 * and again whenever the width changes. */
function chartFrame({ id, title, sub, legend, height, label, draw, table }) {
  const tip = el("div", { class: "viz-tip", hidden: true });
  const live = el("p", { class: "visually-hidden", "aria-live": "polite" });
  const plot = el("div", {
    class: "viz-plot", tabindex: "0", "data-focus": `plot-${id}`,
    role: "group", "aria-label": `${label} Use the arrow keys to step through the windows.`,
    style: `height: ${height}px`,
  }, tip);
  const dataId = `${id}-data`;
  const open = ui.expanded.has(dataId);
  const dataBox = el("div", { class: "viz-data table-scroll", id: dataId, hidden: !open }, table);
  const toggle = el("button", {
    type: "button", class: "link-button viz-toggle", "aria-expanded": open ? "true" : "false",
    "aria-controls": dataId, "data-focus": `toggle-${id}`,
    onclick: () => {
      const show = dataBox.hidden;
      dataBox.hidden = !show;
      toggle.setAttribute("aria-expanded", show ? "true" : "false");
      toggle.textContent = show ? "Hide data" : "Show data";
      if (show) ui.expanded.add(dataId); else ui.expanded.delete(dataId);
    },
  }, open ? "Hide data" : "Show data");
  // Heading, legend and plot are one block, so paper never parts them.
  const figure = el("figure", { class: "viz", id, "aria-labelledby": `${id}-title` },
    el("div", { class: "viz-frame" },
      el("div", { class: "viz-head" },
        el("h3", { class: "viz-title", id: `${id}-title`, text: title }),
        sub ? el("p", { class: "viz-sub", text: sub }) : null),
      legend,
      plot),
    live, toggle, dataBox);
  charts.push({ plot, tip, live, draw, width: 0 });
  return figure;
}

function drawChart(chart, width) {
  chart.width = width;
  const old = chart.plot.querySelector("svg");
  if (old) old.remove();
  const result = chart.draw(width);
  chart.plot.prepend(result.svg);
  chart.layout = result.layout;
  if (!chart.interaction) {
    chart.interaction = interactive(chart.plot, chart.tip, chart.live, () => chart.layout(),
                                    result.indices, result.readout);
  } else {
    chart.interaction.refresh();
  }
}

/* Draws every chart at its plot's width. One in a hidden tab has no width
 * yet: it is drawn at the width of a drawn chart in the same kind of place
 * (every tab panel is as wide as the others), so a printout — which shows
 * every panel — has it, and it is redrawn if its own width differs once its
 * tab opens. */
function drawCharts() {
  if (resizeObserver) resizeObserver.disconnect();
  const redraw = (chart) => {
    const width = Math.floor(chart.plot.clientWidth);
    if (width && width !== chart.width) drawChart(chart, width);
  };
  charts.forEach(redraw);
  const panelWidth = Math.max(0, ...charts.filter((c) => c.plot.clientWidth && c.plot.closest(".tab-panel"))
    .map((c) => c.width || 0));
  if (panelWidth) charts.forEach((chart) => { if (!chart.width) drawChart(chart, panelWidth); });
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(() => charts.forEach(redraw));
    charts.forEach((chart) => resizeObserver.observe(chart.plot));
  }
}

/* Paper shows every chart, the ones in closed tabs too, at the width of an
 * A4 or Letter page's text column, so their labels print at the size they
 * read on screen; the screen widths come back after printing. */
const PRINT_WIDTH = 680;
window.addEventListener("beforeprint", () => {
  charts.forEach((chart) => drawChart(chart, PRINT_WIDTH));
});
window.addEventListener("afterprint", () => {
  charts.forEach((chart) => { chart.width = 0; });
  drawCharts();
});

/* ── Comparison charts ────────────────────────────────────────────────────── */

/* Which scenarios scored exactly the same as an earlier one on every window
 * they have — they deployed the same selection throughout — so their lines
 * lie under it. Said in the legend, because otherwise a hidden line reads as
 * a missing one. */
function coincidences(series) {
  const same = new Map();
  series.forEach((s, k) => {
    if (!s.points.length) return;
    for (let j = 0; j < k; j += 1) {
      const t = series[j];
      if (same.has(t.key)) continue;
      const theirs = new Map(t.points.map((p) => [p.i, p.v]));
      const equal = s.points.every((p) => theirs.has(p.i)
        && (theirs.get(p.i) === p.v || (!isNum(p.v) && !isNum(theirs.get(p.i)))));
      if (equal) {
        same.set(s.key, s.points.length === t.points.length
          ? `same as ${t.meta.name} on every window`
          : `same as ${t.meta.name} on the ${plural(s.points.length, "window")} it scored`);
        break;
      }
    }
  });
  return same;
}

function comparisonChart(payload, signal) {
  const summary = payload.summary;
  const order = scenarioOrder(payload);
  const series = order.map((key) => {
    const ws = windowsOf(payload.scenarios[key]);
    return {
      key, meta: scenarioMeta(key), windows: ws,
      points: ws.map((w) => ({ i: w.i, v: w[`${signal}_score`] })),
    };
  }).filter((s) => s.points.length);
  if (!series.length) return null;
  const same = coincidences(series);
  const indices = [...new Set(series.flatMap((s) => s.points.map((p) => p.i)))].sort((a, b) => a - b);
  const byWindow = new Map(series.map((s) => [s.key, new Map(s.windows.map((w) => [w.i, w]))]));
  const first = indices[0], last = indices[indices.length - 1];
  const info = SIGNALS[signal];
  const id = `cmp-${signal}`;
  const height = 260;

  const legend = el("ul", { class: "viz-legend", "aria-label": "Scenarios" },
    series.map((s) => el("li", {},
      lineKey(s.key), el("span", { text: s.meta.name }),
      same.has(s.key) ? el("span", { class: "viz-legend-note", text: same.get(s.key) }) : null)),
    series.some((s) => s.windows.some((w) => w.reopt_applied))
      ? el("li", { class: "viz-legend-mark" },
          el("span", { class: "key-dot", "aria-hidden": "true" }), el("span", { text: "re-optimization applied" }))
      : null,
    // Every scenario replays the same windows, so one scenario's gaps are all of them.
    noScoreRuns(series[0].windows).length ? noScoreKey() : null);
  const gaps = noScoreRuns(series[0].windows);

  const deployedOf = (w) => (signal === "ensemble"
    ? (w.deployed_ensemble || []).join(" + ") : (w.deployed_single || ""));

  const readout = (i) => {
    const any = series.map((s) => byWindow.get(s.key).get(i)).find(Boolean);
    const where = any ? samplesText(summary, any) : "";
    const title = `Window ${i}${where ? ` · ${where}` : ""}`;
    const rows = [];
    const words = [title];
    series.forEach((s) => {
      const w = byWindow.get(s.key).get(i);
      if (!w) return;
      const value = w[`${signal}_score`];
      const applied = w.reopt_applied ? "re-optimization applied" : "";
      rows.push(el("div", { class: "viz-tip-row" },
        lineKey(s.key),
        el("strong", { text: num(value, 4) }),
        el("span", { class: "viz-tip-name", text: s.meta.name }),
        applied ? el("span", { class: "viz-tip-flag", text: "● applied" }) : null,
        el("span", { class: "viz-tip-detail mono", text: deployedOf(w) })));
      words.push(`${s.meta.name} ${num(value, 4)}${applied ? `, ${applied}` : ""}, deployed ${deployedOf(w) || "nothing"}`);
    });
    const gap = any ? noScoreNote(any) : "";
    if (gap) { rows.push(el("p", { class: "viz-tip-note", text: gap })); words.push(gap); }
    return { title, rows, text: words.join("; ") };
  };

  const draw = (width) => {
    const m = { top: 12, right: 16, bottom: 40, left: 44 };
    const plotW = Math.max(40, width - m.left - m.right);
    const plotH = height - m.top - m.bottom;
    const xs = xScaleOf(first, last, m.left, plotW);
    const ys = yScaleOf(series.flatMap((s) => s.points.map((p) => p.v)), plotH, m.top);
    const node = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-labelledby": `${id}-svg-title ${id}-svg-desc` },
      svg("title", { id: `${id}-svg-title`, text: `${info.title} per window` }),
      svg("desc", { id: `${id}-svg-desc`, text: series.map((s) => {
        const vs = s.points.map((p) => p.v).filter(isNum);
        return `${s.meta.name}: ${vs.length} windows, from ${num(Math.min(...vs))} to ${num(Math.max(...vs))}`;
      }).join(". ") }));
    drawNoScore(node, gaps, xs, m.top, plotH);
    const grid = svg("g", { class: "viz-grid" });
    yAxis(grid, ys, m.left, m.left + plotW);
    xAxis(grid, xs, m.top + plotH, m.left, m.left + plotW, "Window");
    node.append(grid);
    // Drawn last-first, so where lines coincide the first scenario (the one
    // the legend's "same as" points to) is the line on top.
    const lines = svg("g", { class: "viz-lines" });
    series.slice().reverse().forEach((s) => {
      lines.append(svg("path", { class: `line ${s.meta.cls}`, d: linePath(s.points, xs.x, ys.y) }));
    });
    node.append(lines);
    const marks = svg("g", { class: "viz-marks" });
    series.slice().reverse().forEach((s) => {
      s.windows.filter((w) => w.reopt_applied && isNum(w[`${signal}_score`])).forEach((w) => {
        marks.append(svg("circle", { class: `dot ${s.meta.cls}`, cx: xs.x(w.i), cy: ys.y(w[`${signal}_score`]), r: 5 }));
      });
      // A lone window has no line to draw, so it is a dot.
      if (s.points.length === 1 && isNum(s.points[0].v)) {
        marks.append(svg("circle", { class: `dot ${s.meta.cls}`, cx: xs.x(s.points[0].i), cy: ys.y(s.points[0].v), r: 4 }));
      }
    });
    node.append(marks,
      svg("line", { class: "crosshair", x1: 0, x2: 0, y1: m.top, y2: m.top + plotH, visibility: "hidden" }));
    return { svg: node, layout: () => ({ x: xs.x }), indices, readout };
  };

  const table = el("table", { class: "summary-table viz-table" },
    el("caption", { class: "visually-hidden", text: `${info.title} per window, by scenario` }),
    el("thead", {}, el("tr", {},
      el("th", { scope: "col", text: "Window" }),
      el("th", { scope: "col", text: "Samples" }),
      series.map((s) => el("th", { scope: "col", class: "num", text: s.meta.name })))),
    el("tbody", {}, indices.map((i) => {
      const any = series.map((s) => byWindow.get(s.key).get(i)).find(Boolean);
      return el("tr", {},
        el("td", { class: "num", text: String(i) }),
        el("td", { class: "mono", text: any && isNum(any.start) ? `${any.start}–${any.end}` : "—" }),
        series.map((s) => {
          const w = byWindow.get(s.key).get(i);
          const value = w && w.scorable === false ? "no score" : (w ? num(w[`${signal}_score`], 4) : "—");
          return el("td", { class: "num",
            text: w ? `${value}${w.reopt_applied ? " ●" : ""}` : "—" });
        }));
    })));

  return chartFrame({
    id, title: `${info.title} per window`,
    sub: signal === "ensemble"
      ? "The ensemble each scenario had deployed, scored on every window."
      : "The single model each scenario had deployed, scored on every window.",
    legend, height, draw, table,
    label: `${info.title} per window for ${plural(series.length, "scenario")}.`,
  });
}

/* ── Scenario timeline ────────────────────────────────────────────────────── */

/* Every re-optimisation as one record: its trigger window, where it was due,
 * and how it ended — read from the events, which close every `started`. */
function reoptsOf(doc) {
  const out = [];
  const open = new Map();
  for (const e of (doc && doc.events) || []) {
    if (e.type === "started") {
      const r = { start: e.window, apply: e.apply_window, end: null, outcome: null, event: null };
      out.push(r);
      open.set(e.window, r);
    } else if (["applied", "failed", "not_applied", "aborted"].includes(e.type)) {
      const r = open.get(e.started_at_window);
      if (r) { r.outcome = e.type; r.event = e; open.delete(e.started_at_window); }
    }
  }
  return out;
}

function timelineChart(payload, key) {
  const summary = payload.summary;
  const doc = payload.scenarios[key];
  const row = rowOf(summary, key) || {};
  const ws = windowsOf(doc);
  if (!ws.length) return null;
  const meta = scenarioMeta(key);
  const n = ws[ws.length - 1].i + 1;
  const byWindow = new Map(ws.map((w) => [w.i, w]));
  const events = (doc && doc.events) || [];
  const reopts = reoptsOf(doc);
  const waiting = (doc && doc.waiting) || row.waiting || null;
  const dueLast = Math.max(n - 1, ...reopts.map((r) => (isNum(r.apply) ? r.apply : 0)),
                           waiting && isNum(waiting.until_window) ? waiting.until_window : 0);
  const first = ws[0].i;
  const indices = ws.map((w) => w.i);
  const id = `tl-${key}`;
  const panelH = 132, laneH = 64, gap = 28;
  const height = 12 + 20 + panelH + gap + 20 + panelH + 16 + laneH + 40;
  const hasThreshold = ws.some((w) => w.thresholds && (isNum(w.thresholds.ensemble) || isNum(w.thresholds.single)));

  const legendItems = [
    el("li", {}, lineKey(key), el("span", { text: `${meta.name} (scores)` })),
    hasThreshold ? el("li", {}, el("span", { class: "key-dash", "aria-hidden": "true" }),
                      el("span", { text: "threshold in effect" })) : null,
    reopts.length ? el("li", {}, el("span", { class: "key-band", "aria-hidden": "true" }),
                       el("span", { text: "re-optimization pending" })) : null,
    dueLast > n - 1 ? el("li", {}, el("span", { class: "key-after", "aria-hidden": "true" }),
                         el("span", { text: "after the stream" })) : null,
    noScoreRuns(ws).length ? noScoreKey() : null,
  ];
  const gaps = noScoreRuns(ws);
  const kinds = new Set();
  events.forEach((e) => {
    if (e.type === "trigger") kinds.add(e.suppressed_because ? "suppressed" : "trigger");
    if (e.type === "applied") kinds.add("applied");
    if (e.type === "not_applied") kinds.add("not_applied");
    if (e.type === "failed" || e.type === "aborted") kinds.add("failed");
  });
  if (waiting) kinds.add("waiting");
  ["trigger", "suppressed", "applied", "not_applied", "failed", "waiting"].filter((k) => kinds.has(k))
    .forEach((k) => legendItems.push(el("li", {}, glyphIcon(k), el("span", { text: GLYPH_LABELS[k].toLowerCase() }))));
  const legend = el("ul", { class: "viz-legend", "aria-label": "Legend" }, legendItems);

  const triggerAt = new Map(events.filter((e) => e.type === "trigger").map((e) => [e.window, e]));

  const readout = (i) => {
    const w = byWindow.get(i);
    if (!w) {
      return { title: `Window ${i}`, rows: [el("p", { class: "viz-tip-note", text: "After the last window: the stream had ended." })],
               text: `Window ${i}: after the stream ended.` };
    }
    const where = samplesText(summary, w);
    const title = `Window ${i}${where ? ` · ${where}` : ""}`;
    const rows = [];
    const words = [title];
    for (const signal of ["ensemble", "single"]) {
      const value = w[`${signal}_score`];
      const t = w.thresholds ? w.thresholds[signal] : null;
      const mode = w.threshold_modes ? w.threshold_modes[signal] : null;
      const below = (w.below || []).includes(signal);
      const thresholdText = isNum(t) ? `threshold ${num(t, 4)}${mode ? ` (${mode})` : ""}` : "";
      rows.push(el("div", { class: "viz-tip-row" },
        lineKey(key), el("strong", { text: num(value, 4) }),
        el("span", { class: "viz-tip-name", text: SIGNALS[signal].short }),
        el("span", { class: "viz-tip-detail", text: [thresholdText, below ? "below it" : ""].filter(Boolean).join(", ") })));
      words.push(`${SIGNALS[signal].short} ${num(value, 4)}${thresholdText ? `, ${thresholdText}` : ""}${below ? ", below it" : ""}`);
      if (w.errors && w.errors[signal]) {
        rows.push(el("p", { class: "viz-tip-note", text: `${SIGNALS[signal].short}: ${w.errors[signal]}` }));
      }
    }
    const notes = [];
    if (w.trigger) {
      const tr = w.trigger;
      notes.push(`${tr.suppressed_because ? "× Trigger suppressed" : "▲ Trigger"} (${REASONS[tr.reason] || tr.reason})`
        + `${tr.text ? `: ${tr.text}` : ""}${tr.started && isNum(tr.apply_window) ? ` — due at window ${tr.apply_window}` : ""}`);
    }
    if (w.reopt_applied) {
      const a = w.reopt_applied;
      notes.push(`● Re-optimization from window ${a.started_at_window} applied`
        + ` (ran ${duration(a.duration_s)}${a.waited_s > 0 ? `, waited ${duration(a.waited_s)}` : ""})`);
    }
    if (w.reopt_failed) {
      notes.push(`◆ Re-optimization from window ${w.reopt_failed.started_at_window} `
        + `${w.reopt_failed.timed_out ? "timed out" : "failed"}`);
    }
    if (w.reopt_pending && !w.reopt_applied) {
      notes.push(`Re-optimization from window ${w.reopt_pending.started_at_window} pending, due at window ${w.reopt_pending.apply_window}`);
    }
    if (w.rescored) notes.push("Re-scored after the switch-in (L = 0).");
    if (noScoreNote(w)) notes.push(noScoreNote(w));
    notes.forEach((t) => { rows.push(el("p", { class: "viz-tip-note", text: t })); words.push(t); });
    const deployed = `Deployed: ensemble ${(w.deployed_ensemble || []).join(" + ") || "—"} · single ${w.deployed_single || "—"}`;
    rows.push(el("p", { class: "viz-tip-note mono", text: deployed }));
    words.push(deployed);
    return { title, rows, text: words.join("; ") };
  };

  const draw = (width) => {
    const m = { top: 12, right: 16, left: 44 };
    const plotW = Math.max(40, width - m.left - m.right);
    const xs = xScaleOf(first, dueLast, m.left, plotW);
    const panels = [
      { signal: "ensemble", top: m.top + 20 },
      { signal: "single", top: m.top + 20 + panelH + gap + 20 },
    ];
    const laneTop = panels[1].top + panelH + 16;
    const laneMid = laneTop + 26;
    const axisY = laneTop + laneH;
    const node = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-labelledby": `${id}-svg-title ${id}-svg-desc` },
      svg("title", { id: `${id}-svg-title`, text: `${meta.name}: scores, thresholds and re-optimizations per window` }),
      svg("desc", { id: `${id}-svg-desc`, text: `${plural(ws.length, "window")}; `
        + `${plural(row.n_triggers || 0, "trigger")}, ${plural(row.n_reopts_applied || 0, "re-optimization")} applied, `
        + `${row.n_reopts_not_applied || 0} not applied, ${row.n_reopts_failed || 0} failed, `
        + `${row.n_suppressed || 0} suppressed.` }));

    // The stretch past the last window, where a late re-optimisation was due.
    if (dueLast > n - 1) {
      const x0 = xs.x(n - 0.5), x1 = m.left + plotW;
      node.append(svg("rect", { class: "after-stream", x: x0, y: panels[0].top, width: x1 - x0,
                                height: axisY - panels[0].top }));
      // Named in place where it is wide enough; the legend names it always.
      if (x1 - x0 > 110) {
        node.append(svg("text", { class: "lane-label", x: x1 - 6, y: panels[0].top + 14, "text-anchor": "end",
                                  text: "after the stream" }));
      }
    }
    // Pending re-optimisations, from trigger to switch-in, across both panels.
    reopts.forEach((r) => {
      if (!isNum(r.start)) return;
      const end = isNum(r.apply) ? r.apply : r.start;
      const x0 = xs.x(r.start), x1 = xs.x(Math.min(end, dueLast));
      panels.forEach((p) => node.append(svg("rect", { class: "pending", x: x0, y: p.top, width: Math.max(0, x1 - x0), height: panelH })));
    });

    panels.forEach((p) => drawNoScore(node, gaps, xs, p.top, panelH));
    panels.forEach((p) => {
      const values = ws.map((w) => w[`${p.signal}_score`])
        .concat(ws.map((w) => (w.thresholds ? w.thresholds[p.signal] : null)));
      const ys = yScaleOf(values, panelH, p.top);
      p.ys = ys;
      const g = svg("g", { class: "viz-grid" });
      yAxis(g, ys, m.left, m.left + plotW);
      g.append(svg("line", { class: "axis", x1: m.left, x2: m.left + plotW, y1: p.top + panelH, y2: p.top + panelH }),
               svg("text", { class: "panel-title", x: m.left, y: p.top - 8, text: SIGNALS[p.signal].title }));
      node.append(g);

      // The threshold in effect, a step per window; its mode named where it
      // starts and wherever it changes (relative → adaptive).
      let d = "", prev = null, prevMode = null;
      const labels = [];
      ws.forEach((w) => {
        const t = w.thresholds ? w.thresholds[p.signal] : null;
        const mode = w.threshold_modes ? w.threshold_modes[p.signal] : null;
        if (!isNum(t)) { prev = null; return; }
        const x0 = Math.max(m.left, xs.x(w.i - 0.5)), x1 = Math.min(m.left + plotW, xs.x(w.i + 0.5));
        const yy = ys.y(t);
        d += prev !== null && prev.i === w.i - 1 ? `V${yy.toFixed(1)}H${x1.toFixed(1)}` : `M${x0.toFixed(1)} ${yy.toFixed(1)}H${x1.toFixed(1)}`;
        if (mode && mode !== prevMode) labels.push({ x: x0, y: yy, mode });
        prev = { i: w.i, t };
        prevMode = mode;
      });
      if (d) {
        node.append(svg("path", { class: "threshold", d }));
        labels.forEach((l) => node.append(svg("text", { class: "threshold-label", x: l.x + 2, y: l.y + 14, text: l.mode })));
      }
      // Switch-ins: a hairline through the panel at each applied window.
      ws.filter((w) => w.reopt_applied).forEach((w) => {
        node.append(svg("line", { class: "switch-line", x1: xs.x(w.i), x2: xs.x(w.i), y1: p.top, y2: p.top + panelH }));
      });
      const points = ws.map((w) => ({ i: w.i, v: w[`${p.signal}_score`] }));
      node.append(svg("path", { class: `line ${meta.cls}`, d: linePath(points, xs.x, ys.y) }));
      const dots = svg("g", { class: "viz-marks" });
      const every = ws.length <= 48;
      ws.forEach((w) => {
        const v = w[`${p.signal}_score`];
        if (!isNum(v)) return;
        if (w.reopt_applied) dots.append(svg("circle", { class: `dot ${meta.cls}`, cx: xs.x(w.i), cy: ys.y(v), r: 5 }));
        else if (every) dots.append(svg("circle", { class: `point ${meta.cls}`, cx: xs.x(w.i), cy: ys.y(v), r: 3 }));
      });
      node.append(dots);
    });

    // The events lane: what the policy did, on the same window axis.
    const lane = svg("g", { class: "viz-lane" });
    lane.append(svg("text", { class: "panel-title", x: m.left, y: laneTop + 8, text: "Re-optimizations" }));
    if (events.length || waiting) {
      lane.append(svg("line", { class: "grid", x1: m.left, x2: m.left + plotW, y1: laneMid, y2: laneMid }));
    }
    reopts.forEach((r) => {
      if (!isNum(r.start) || !isNum(r.apply)) return;
      lane.append(svg("line", { class: "lane-span", x1: xs.x(r.start), x2: xs.x(Math.min(r.apply, dueLast)), y1: laneMid, y2: laneMid }));
      const waited = r.event && r.event.waited_s > 0 ? r.event.waited_s : 0;
      if (waited) {
        // The stream stood still just before the switch-in window.
        const x1 = xs.x(r.apply), x0 = Math.max(xs.x(r.start), x1 - xs.cell / 2);
        lane.append(svg("rect", { class: "wait-span", x: x0, y: laneMid - 3, width: Math.max(2, x1 - x0), height: 6 }),
                    svg("text", { class: "lane-label", x: x1, y: laneMid + 22, "text-anchor": "middle",
                                  text: `waited ${duration(waited)}` }));
      }
    });
    if (waiting && isNum(waiting.until_window)) {
      const x1 = xs.x(waiting.until_window);
      lane.append(glyph("waiting", x1, laneMid, 6),
                  svg("text", { class: "lane-label", x: x1, y: laneMid + 22, "text-anchor": "middle",
                                text: `waiting${waiting.since ? ` ${duration(Date.now() / 1000 - waiting.since)}` : ""}` }));
    }
    events.forEach((e) => {
      if (e.type === "trigger" && isNum(e.window)) {
        lane.append(glyph(e.suppressed_because ? "suppressed" : "trigger", xs.x(e.window), laneMid, 6));
      } else if (e.type === "applied" && isNum(e.window)) {
        lane.append(glyph("applied", xs.x(e.window), laneMid, 6));
      } else if (e.type === "not_applied" && isNum(e.apply_window)) {
        lane.append(glyph("not_applied", xs.x(Math.min(e.apply_window, dueLast)), laneMid, 6));
      } else if ((e.type === "failed" || e.type === "aborted") && isNum(e.window ?? e.apply_window)) {
        lane.append(glyph("failed", xs.x(e.window ?? e.apply_window), laneMid, 6));
      }
    });
    if (!events.length && !waiting) {
      lane.append(svg("text", { class: "lane-label", x: m.left + plotW / 2, y: laneMid + 4, "text-anchor": "middle",
        text: key === "fixed" ? "Never re-optimizes" : "No trigger fired" }));
    }
    node.append(lane);
    const axis = svg("g", { class: "viz-grid" });
    xAxis(axis, xs, axisY, m.left, m.left + plotW, "Window");
    node.append(axis,
      svg("line", { class: "crosshair", x1: 0, x2: 0, y1: panels[0].top, y2: axisY, visibility: "hidden" }));
    return { svg: node, layout: () => ({ x: xs.x }), indices, readout };
  };

  const table = el("table", { class: "summary-table viz-table" },
    el("caption", { class: "visually-hidden", text: `${meta.name}: every window` }),
    el("thead", {}, el("tr", {},
      ["Window", "Ensemble", "Threshold", "Single model", "Threshold", "Event", "Deployed ensemble", "Deployed single"]
        .map((h, k) => el("th", { scope: "col", class: [0, 1, 2, 3, 4].includes(k) ? "num" : null, text: h })))),
    el("tbody", {}, ws.map((w) => {
      const tr = triggerAt.get(w.i);
      const event = [
        tr ? (tr.suppressed_because ? "× suppressed" : "▲ trigger") : "",
        w.reopt_applied ? "● applied" : "",
        w.reopt_failed ? "◆ failed" : "",
      ].filter(Boolean).join(", ");
      const t = w.thresholds || {};
      const gap = w.scorable === false;
      return el("tr", {},
        el("td", { class: "num", text: String(w.i) }),
        el("td", { class: "num", text: gap ? "no score" : num(w.ensemble_score, 4) }),
        el("td", { class: "num", text: num(t.ensemble, 4) }),
        el("td", { class: "num", text: gap ? "no score" : num(w.single_score, 4) }),
        el("td", { class: "num", text: num(t.single, 4) }),
        el("td", { text: event || "—" }),
        el("td", { class: "detector", text: (w.deployed_ensemble || []).join(" + ") || "—" }),
        el("td", { class: "detector", text: w.deployed_single || "—" }));
    })));

  return chartFrame({
    id, title: "Scores, thresholds and re-optimizations",
    sub: null, legend, height, draw, table,
    label: `${meta.name}: ensemble fitness and single-model score per window, with the thresholds in effect and the re-optimizations.`,
  });
}

/* ── Tables ───────────────────────────────────────────────────────────────── */

function rowOf(summary, key) {
  return (summary.scenarios || []).find((s) => s && s.key === key);
}

function scenarioOrder(payload) {
  const summary = payload.summary;
  const order = Array.isArray(summary.scenario_order) && summary.scenario_order.length
    ? summary.scenario_order : (summary.scenarios || []).map((s) => s.key);
  return order.filter((key) => key in SCENARIOS || rowOf(summary, key));
}

/* The best mean in a column, or nothing when every scenario has the same one
 * (no scenario did better) or fewer than two have one. Ties share the mark.
 * Only finished scenarios that scored the same windows compete: a mean over
 * the first seven windows is not comparable with one over all twelve. */
function comparable(rows) {
  const done = rows.filter((r) => r.status === "complete");
  const most = Math.max(0, ...done.map((r) => r.n_windows || 0));
  return done.filter((r) => (r.n_windows || 0) === most);
}

function bestOf(rows, field) {
  const vals = comparable(rows).map((r) => r[field]).filter(isNum);
  if (vals.length < 2) return null;
  const best = Math.max(...vals);
  if (vals.every((v) => Math.abs(v - best) < 1e-12)) return null;
  return best;
}

function comparisonTable(payload) {
  const summary = payload.summary;
  const rows = scenarioOrder(payload).map((key) => rowOf(summary, key)).filter(Boolean);
  const bestEns = bestOf(rows, "mean_ensemble_score");
  const partial = comparable(rows).length < rows.length;
  const bestSingle = bestOf(rows, "mean_single_score");
  const scoreCell = (value, best) => {
    const top = best !== null && isNum(value) && Math.abs(value - best) < 1e-12;
    return el("td", { class: `num${top ? " is-best" : ""}` },
      num(value),
      top ? el("span", { class: "best-mark", "aria-hidden": "true", text: " ★" }) : null,
      top ? el("span", { class: "visually-hidden", text: " (best mean)" }) : null);
  };
  const count = (value, title) => el("td", { class: "num", title: title || null,
                                             text: isNum(value) ? String(value) : "—" });
  // Two header rows: what each group of columns measures, then the columns.
  const group = (label, span) => el("th", { scope: "colgroup", colspan: String(span), class: "group-head", text: label });
  const col = (label, cls) => el("th", { scope: "col", class: cls || "num", text: label });
  const table = el("table", { class: "summary-table compare-table" },
    el("caption", { class: "visually-hidden", text: "The scenarios compared" }),
    el("thead", {},
      el("tr", {},
        el("th", { scope: "col", rowspan: "2", class: "scenario-head", text: "Scenario" }),
        group("Ensemble fitness", 2), group("Single-model score", 2),
        group("Re-optimizations", 6), group("Time", 2)),
      el("tr", {},
        col("Mean"), col("Min"), col("Mean"), col("Min"),
        col("Triggers"), col("Started"), col("Applied"), col("Not applied"), col("Failed"), col("Suppressed"),
        col("Waited"), col("Wall"))),
    el("tbody", {}, rows.map((r) => {
      const failedNote = [r.n_reopts_timed_out ? `${r.n_reopts_timed_out} timed out` : "",
                          r.n_reopts_aborted ? `${r.n_reopts_aborted} aborted` : ""].filter(Boolean).join(", ");
      const failed = (r.n_reopts_failed || 0) + (r.n_reopts_aborted || 0);
      return el("tr", {},
        el("th", { scope: "row", class: "scenario-cell" },
          lineKey(r.key),
          el("span", { class: "scenario-name", text: scenarioMeta(r.key).name }),
          r.status === "complete" ? null : stateTag(r.status, "is-small")),
        scoreCell(r.mean_ensemble_score, bestEns),
        el("td", { class: "num", text: num(r.min_ensemble_score) }),
        scoreCell(r.mean_single_score, bestSingle),
        el("td", { class: "num", text: num(r.min_single_score) }),
        count(r.n_triggers),
        count(r.n_reopts_started, r.n_reopts_reused ? `${r.n_reopts_reused} reused from an earlier scenario` : null),
        count(r.n_reopts_applied),
        count(r.n_reopts_not_applied, "Due after the last window, so never switched in"),
        el("td", { class: "num" }, String(failed),
          failedNote ? el("span", { class: "cell-note", text: ` (${failedNote})` }) : null),
        count(r.n_suppressed, "Triggers while a re-optimization was pending"),
        el("td", { class: "num", title: r.n_waits ? plural(r.n_waits, "wait") : null, text: waitText(r.total_wait_s) }),
        el("td", { class: "num", text: isNum(r.wall_time_s) ? duration(r.wall_time_s) : "—" }));
    })));
  const notes = [];
  if (bestEns !== null || bestSingle !== null) notes.push("★ highest mean in its column; ties share it.");
  else if (comparable(rows).length > 1) notes.push("No scenario had a higher mean than another: they scored the same on every window.");
  if (partial) notes.push("Only scenarios that finished every window compete for ★: a mean over fewer windows is not comparable.");
  notes.push("Waited: time the stream stood still for a re-optimization to finish.");
  return el("div", { class: "stack-sm" },
    el("div", { class: "table-scroll compare-scroll" }, table),
    el("p", { class: "table-note", text: notes.join(" ") }));
}

function eventsTable(payload, key) {
  const summary = payload.summary;
  const doc = payload.scenarios[key];
  const events = (doc && doc.events) || [];
  const ws = windowsOf(doc);
  const lastWindow = ws.length ? ws[ws.length - 1].i : null;
  const byWindow = new Map(ws.map((w) => [w.i, w]));
  if (!events.length) {
    return el("p", { class: "empty-note", text: key === "fixed"
      ? "No events: the fixed scenario never re-optimizes."
      : "No events: no trigger fired on any window." });
  }
  // The wait is reported on the event that ended it.
  const waitedFor = new Map(events.filter((e) => e.type === "applied" || e.type === "failed")
    .map((e) => [e.started_at_window, e.waited_s]));
  const describe = (e) => {
    const w = isNum(e.window) ? byWindow.get(e.window) : null;
    switch (e.type) {
      case "trigger": {
        const text = w && w.trigger && w.trigger.text ? w.trigger.text : "";
        const base = `${REASONS[e.reason] || e.reason || "Rule"}${text ? `: ${text}` : ""}`;
        return { glyph: e.suppressed_because ? "suppressed" : "trigger",
                 name: e.suppressed_because ? "Trigger suppressed" : "Trigger",
                 detail: e.suppressed_because
                   ? `${base}. A re-optimization was already pending, so none started.` : base,
                 dur: "" };
      }
      case "started":
        return { glyph: null, name: "Re-optimization started",
                 detail: `Runs in the background; switch-in due at window ${e.apply_window}.`
                   + `${e.reused_from ? ` Result reused from ${scenarioMeta(e.reused_from).name}.` : ""}`,
                 dur: "" };
      case "waiting":
        return { glyph: "waiting", name: "Stream waiting",
                 detail: `For the re-optimization started at window ${e.started_at_window} to finish.`,
                 dur: waitText(waitedFor.get(e.started_at_window)) };
      case "applied":
        return { glyph: "applied", name: "Applied",
                 detail: el("span", {}, "Switched in: ensemble ", chips(e.new_ensemble), " · single ",
                            chips(e.new_single ? [e.new_single] : []),
                            e.reused_from ? ` (reused from ${scenarioMeta(e.reused_from).name})` : ""),
                 dur: `ran ${duration(e.duration_s)}${e.waited_s > 0 ? ` · waited ${duration(e.waited_s)}` : ""}` };
      case "failed":
        return { glyph: "failed", name: e.timed_out ? "Timed out" : "Failed",
                 detail: `Re-optimization from window ${e.started_at_window}: ${e.error || "no reason recorded"}. The current selection stays.`,
                 dur: isNum(e.duration_s) ? `ran ${duration(e.duration_s)}` : "" };
      case "not_applied":
        return { glyph: "not_applied", name: "Not applied (stream ended)",
                 detail: `Re-optimization from window ${e.started_at_window} was due at window ${e.apply_window}, `
                   + `after the last window${isNum(lastWindow) ? ` (${lastWindow})` : ""}; it was stopped.`,
                 dur: isNum(e.duration_s) ? `ran ${duration(e.duration_s)}` : "" };
      case "aborted":
        return { glyph: "failed", name: "Aborted",
                 detail: `Re-optimization from window ${e.started_at_window} was stopped because the scenario failed`
                   + `${e.error ? `: ${e.error}` : ""}.`,
                 dur: "" };
      default:
        return { glyph: null, name: e.type, detail: "", dur: "" };
    }
  };
  const table = el("table", { class: "summary-table events-table" },
    el("caption", { class: "visually-hidden", text: `${scenarioMeta(key).name}: events` }),
    el("thead", {}, el("tr", {},
      el("th", { scope: "col", class: "num", text: "Window" }),
      el("th", { scope: "col", text: "Event" }),
      el("th", { scope: "col", text: "Details" }),
      el("th", { scope: "col", text: "Duration" }))),
    el("tbody", {}, events.map((e) => {
      const d = describe(e);
      return el("tr", {},
        el("td", { class: "num", text: isNum(e.window) ? String(e.window) : "—" }),
        el("td", { class: "event-name" }, d.glyph ? glyphIcon(d.glyph) : el("span", { class: "glyph-gap" }), d.name),
        el("td", { class: "event-detail" }, d.detail),
        el("td", { class: "mono", text: d.dur || "—" }));
    })));
  return el("div", { class: "table-scroll" }, table);
}

/* The deployed selection as contiguous spans of windows. */
function selectionSpans(payload, key) {
  const doc = payload.scenarios[key];
  const ws = windowsOf(doc);
  if (!ws.length) return el("p", { class: "empty-note", text: "No window scored yet." });
  const spans = [];
  ws.forEach((w) => {
    const sig = `${(w.deployed_ensemble || []).join("+")}|${w.deployed_single || ""}`;
    const last = spans[spans.length - 1];
    if (last && last.sig === sig && last.to === w.i - 1) { last.to = w.i; return; }
    spans.push({ sig, from: w.i, to: w.i, ensemble: w.deployed_ensemble || [], single: w.deployed_single,
                 applied: w.reopt_applied });
  });
  const table = el("table", { class: "summary-table spans-table" },
    el("caption", { class: "visually-hidden", text: `${scenarioMeta(key).name}: deployed selection over time` }),
    el("thead", {}, el("tr", {},
      el("th", { scope: "col", text: "Windows" }),
      el("th", { scope: "col", text: "Ensemble" }),
      el("th", { scope: "col", text: "Single model" }),
      el("th", { scope: "col", text: "From" }))),
    el("tbody", {}, spans.map((s, k) => el("tr", {},
      el("td", { class: "mono", text: s.from === s.to ? `${s.from}` : `${s.from}–${s.to}` }),
      el("td", { class: "chips-cell" }, chips(s.ensemble)),
      el("td", { class: "chips-cell" }, chips(s.single ? [s.single] : [])),
      el("td", { text: s.applied
        ? `Re-optimization started at window ${s.applied.started_at_window}`
        : k === 0 ? "Offline selection" : "—" })))));
  return el("div", { class: "table-scroll" }, table);
}

/* ── Header, notices, sections ────────────────────────────────────────────── */

function injectionText(summary) {
  const copies = (summary.injection && summary.injection.copies) || [];
  if (!copies.length) return { value: summary.anomaly_type || "—", sub: "" };
  const counts = new Map();
  copies.forEach((c) => counts.set(c.anomaly_type, (counts.get(c.anomaly_type) || 0) + 1));
  const parts = [...counts.entries()].map(([type, k]) => (k > 1 ? `${type} ×${k}` : type));
  return {
    value: summary.inject_online_regime ? "Regime shift" : parts.join(", "),
    sub: summary.inject_online_regime
      ? `${parts.join(", ")} · ${plural(copies.length, "copy", "copies")}`
      : `${plural(copies.length, "copy", "copies")}${isNum(summary.seed) ? ` · seed ${summary.seed}` : ""}`,
  };
}

function factsRow(summary) {
  const fact = (label, value, sub) => el("div", {},
    el("dt", { text: label }),
    el("dd", {}, el("span", { class: "fact-value", text: value }),
      sub ? el("span", { class: "fact-sub", text: sub }) : null));
  const available = summary.n_windows_available;
  const injection = injectionText(summary);
  // What the windows were scored on. A summary from before label sources
  // existed was always scored on injected anomalies.
  const real = summary.label_source === "real";
  const scorable = summary.n_windows_scorable;
  const scorableText = isNum(scorable) && isNum(summary.n_windows)
    ? `${scorable} of ${plural(summary.n_windows, "window")} scorable` : "";
  const metric = summary.decision_metric || {};
  const timeout = summary.reopt_timeout;
  return el("dl", { class: "facts-row" },
    fact("Windows", isNum(summary.n_windows) ? String(summary.n_windows) : "—",
         isNum(available) && available !== summary.n_windows ? `of ${available} available` : "every online window"),
    fact("Window", isNum(summary.window_size) ? `${summary.window_size} samples` : "—",
         isNum(summary.step_size) ? `step ${summary.step_size}` : ""),
    fact("Switch-in delay", isNum(summary.reopt_delay) ? `L = ${summary.reopt_delay}` : "—",
         isNum(timeout) ? `timeout ${duration(timeout)}` : "no timeout"),
    fact("Decision metric", metric.label || metric.spec || "—",
         [metric.formula, isNum(summary.vus_window) ? `VUS window ${summary.vus_window}` : ""].filter(Boolean).join(" · ")),
    fact("Labels", real ? "Real labels" : "Injected anomalies",
         [real ? "no injection online" : "", scorableText].filter(Boolean).join(" · ")),
    real ? null : fact("Injected", injection.value, injection.sub),
    fact("Updated", summary.updated_at ? timeAgo(summary.updated_at) : "—",
         summary.generated_at ? `started ${stamp(summary.generated_at)}` : ""));
}

function offlineBlock(summary) {
  const off = summary.offline_selection;
  if (!off) return null;
  const ens = off.ensemble || {};
  const single = off.single || {};
  const score = (s) => (s && s.scores && isNum(s.scores.score) ? num(s.scores.score) : "—");
  const choice = off.framework_choice ? off.framework_choice.replace(/_/g, " ") : "—";
  return el("section", { class: "offline-block", "aria-labelledby": "offline-title" },
    el("h2", { id: "offline-title", class: "section-title", text: "Starting selection" }),
    el("p", { class: "section-desc", text:
      "Every scenario starts from what the offline phase selected; a re-optimization replaces it." }),
    el("dl", { class: "facts offline-facts" },
      el("div", {}, el("dt", { text: "Ensemble" }),
        el("dd", {}, chips(ens.members),
          ens.meta_model_type ? el("span", { class: "dim small", text: ` ${ens.meta_model_type} meta-model` }) : null)),
      el("div", {}, el("dt", { text: "Single model" }), el("dd", {}, chips(single.model ? [single.model] : []))),
      el("div", {}, el("dt", { text: "Offline decision" }), el("dd", { class: "is-decision", text: choice })),
      el("div", {}, el("dt", { text: "Offline scores" }),
        el("dd", {},
          el("span", { class: "mono", text: `ensemble ${score(ens)} · single ${score(single)}` }),
          off.comparable_with_windows === false
            ? el("span", { class: "state-tag is-small", "data-state": "note" }, "Not comparable with window scores")
            : null,
          off.comparability_note ? el("span", { class: "fact-sub", text: off.comparability_note }) : null))));
}

function notices(payload) {
  const summary = payload.summary;
  const out = [];
  const status = summary.status;
  if (status === "interrupted") {
    out.push(el("div", { class: "banner banner-warn", role: "status" },
      el("strong", { text: "Interrupted." }),
      ` ${summary.error || "The pipeline stopped before the online phase finished."}`
      + " The results below stop at the last window it wrote."));
  } else if (status === "failed") {
    out.push(el("div", { class: "banner banner-error", role: "alert" },
      el("strong", { text: "The online phase failed." }), ` ${summary.error || "See the run's log."}`));
  } else if (status === "skipped") {
    const f = summary.feasibility || {};
    out.push(el("div", { class: "banner banner-info" },
      el("strong", { text: "Skipped." }),
      ` ${summary.error || f.reason || "This entity is too short for the online phase."}`,
      el("p", { class: "banner-detail", text:
        "Only a longer entity makes room: the online part is always the last 20% of the test series, "
        + `and a window is never shorter than ${f.min_window_size || 100} samples.` })));
  }
  (summary.scenarios || []).forEach((r) => {
    const name = scenarioMeta(r.key).name;
    if (r.status === "failed") {
      out.push(el("div", { class: "banner banner-error", role: "alert" },
        el("strong", { text: `${name} failed.` }), ` ${r.error || "No reason recorded."}`));
    } else if (r.status === "waiting_for_reopt" && r.waiting) {
      out.push(el("div", { class: "banner banner-info" },
        el("strong", { text: `${name} is waiting.` }),
        ` The stream stands at window ${r.waiting.until_window} until the re-optimization started at window `
        + `${r.waiting.since_window} finishes${r.waiting.since ? ` (${duration(Date.now() / 1000 - r.waiting.since)} so far)` : ""}.`));
    }
  });
  return out;
}

function header(payload) {
  const summary = payload.summary;
  const label = payload.dataset_label || summary.dataset_label || dataset;
  const order = scenarioOrder(payload);
  const live = isLive(summary);
  const desc = summary.status === "skipped"
    ? "The online phase was skipped for this entity: there was no window to replay."
    : order.length > 1
    ? `${plural(order.length, "scenario")} replayed on the same ${plural(summary.n_windows || 0, "window")} of the online stream, the last 20% of the test series, each starting from the offline selection.`
    : order.length === 1
      ? `The ${scenarioMeta(order[0]).name.toLowerCase()} scenario replayed on ${plural(summary.n_windows || 0, "window")} of the online stream, the last 20% of the test series.`
      : "The online phase replays the last 20% of the test series window by window.";
  return el("header", { class: "page-head online-head" },
    el("div", { class: "page-head-main" },
      el("div", { class: "title-row" },
        el("h1", { text: `${label} · entity ${summary.entity || entity}` }),
        stateTag(summary.status)),
      el("p", { class: "page-desc", text: desc }),
      live ? el("p", { class: "live-note", text: `Updating every ${POLL_MS / 1000} seconds while the run writes these results.` }) : null));
}

/* ── Scenario tabs (Carbon line tabs) ─────────────────────────────────────── */

function scenarioTabs(payload) {
  const order = scenarioOrder(payload).filter((key) => payload.scenarios[key] || rowOf(payload.summary, key));
  if (!order.length) return null;
  const hashKey = decodeURIComponent((location.hash || "").replace(/^#scenario-/, ""));
  if (!ui.tab || !order.includes(ui.tab)) ui.tab = order.includes(hashKey) ? hashKey : order[0];
  const tabs = [];
  const panels = [];
  const select = (key, focus) => {
    ui.tab = key;
    tabs.forEach((t) => {
      const on = t.dataset.key === key;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    panels.forEach((p) => { p.hidden = p.dataset.key !== key; });
    try { history.replaceState(null, "", `#scenario-${key}`); } catch (e) { /* not essential */ }
    // A panel shown for the first time has had no width to draw at.
    drawCharts();
    revealSelectedTab();
  };
  order.forEach((key, k) => {
    const row = rowOf(payload.summary, key) || {};
    const on = key === ui.tab;
    const tab = el("button", {
      type: "button", role: "tab", class: "tab", id: `tab-${key}`, "data-key": key,
      "data-focus": `tab-${key}`, "aria-controls": `panel-${key}`,
      "aria-selected": on ? "true" : "false", tabindex: on ? "0" : "-1",
      onclick: () => select(key, false),
      onkeydown: (event) => {
        const moves = { ArrowRight: k + 1, ArrowLeft: k - 1, Home: 0, End: order.length - 1 };
        if (!(event.key in moves)) return;
        event.preventDefault();
        select(order[(moves[event.key] + order.length) % order.length], true);
      },
    }, lineKey(key), el("span", { text: scenarioMeta(key).name }),
       row.status && row.status !== "complete" ? stateTag(row.status, "is-small") : null);
    tabs.push(tab);
    panels.push(scenarioPanel(payload, key, !on));
  });
  return el("section", { class: "scenario-section", "aria-labelledby": "scenarios-title" },
    el("h2", { id: "scenarios-title", class: "section-title", text: order.length > 1 ? "Each scenario" : "Window by window" }),
    el("p", { class: "section-desc", text: order.length > 1
      ? "What each policy did, window by window: its scores against the threshold in effect, when it re-optimized, and what it deployed."
      : "What the policy did, window by window: its scores against the threshold in effect, when it re-optimized, and what it deployed." }),
    el("div", { class: "tabs-scroll" },
      el("div", { class: "tabs", role: "tablist", "aria-label": "Scenarios" }, tabs)),
    panels);
}

/* The tab strip scrolls sideways on a narrow screen; the selected tab is
 * brought into it, without moving the page. */
function revealSelectedTab() {
  const tab = root.querySelector('.tab[aria-selected="true"]');
  const strip = tab && tab.closest(".tabs-scroll");
  if (!strip) return;
  const left = tab.offsetLeft, right = left + tab.offsetWidth;
  if (left < strip.scrollLeft) strip.scrollLeft = left;
  else if (right > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = right - strip.clientWidth;
}

function scenarioPanel(payload, key, hidden) {
  const doc = payload.scenarios[key];
  const row = rowOf(payload.summary, key) || {};
  const meta = scenarioMeta(key);
  const params = (doc && doc.params) || row.params;
  const ws = windowsOf(doc);
  const planned = row.n_windows_planned || (doc && doc.n_windows_planned);
  const body = [];
  body.push(el("div", { class: "panel-head-row" },
    el("h3", { class: "panel-heading" }, lineKey(key), el("span", { text: row.label || meta.name })),
    stateTag(row.status || (doc && doc.status))));
  body.push(el("p", { class: "rule-text", text: ruleText(key, params) }));
  const progress = isNum(planned) && ws.length < planned ? `${ws.length} of ${planned} windows scored so far.` : "";
  if (progress) body.push(el("p", { class: "live-note", text: progress }));
  if (row.error && row.status !== "failed") body.push(el("p", { class: "banner banner-warn", text: row.error }));
  const chart = timelineChart(payload, key);
  body.push(chart || el("p", { class: "empty-note", text: "No window scored yet." }));
  body.push(el("h4", { class: "sub-heading", text: "Events" }), eventsTable(payload, key));
  body.push(el("h4", { class: "sub-heading", text: "Deployed selection over time" }), selectionSpans(payload, key));
  return el("div", { class: "tab-panel", role: "tabpanel", id: `panel-${key}`, "data-key": key,
                     "aria-labelledby": `tab-${key}`, tabindex: "0", hidden }, body);
}

/* ── Page ─────────────────────────────────────────────────────────────────── */

function isLive(summary) {
  return LIVE.has(summary.status)
    || (summary.status !== "interrupted" && summary.status !== "failed"
        && (summary.scenarios || []).some((r) => LIVE.has(r.status)));
}

function render(payload) {
  const summary = payload.summary;
  const focusKey = document.activeElement && root.contains(document.activeElement)
    ? document.activeElement.dataset.focus : null;
  charts = [];
  const hasWindows = Object.values(payload.scenarios || {}).some((d) => windowsOf(d).length);
  const parts = [header(payload)];
  if (summary.status !== "skipped") parts.push(factsRow(summary));
  parts.push(...notices(payload));
  if (summary.status !== "skipped") parts.push(monitor.element);
  const offline = offlineBlock(summary);
  if (hasWindows) {
    parts.push(el("section", { class: "compare-section", "aria-labelledby": "compare-title" },
      el("h2", { id: "compare-title", class: "section-title",
                 text: scenarioOrder(payload).length > 1 ? "Scenarios compared" : "Summary" }),
      el("p", { class: "section-desc", text: scenarioOrder(payload).length > 1
        ? "The same windows and the same starting selection for every scenario: only the re-optimization policy differs. Scores are per window, on the decision metric."
        : "Scores are per window, on the decision metric. Run the online phase with “Compare all four” to set the policies side by side." }),
      comparisonTable(payload)));
    const ensembleChart = comparisonChart(payload, "ensemble");
    const singleChart = comparisonChart(payload, "single");
    parts.push(el("section", { class: "charts-section", "aria-labelledby": "charts-title" },
      el("h2", { id: "charts-title", class: "section-title", text: "Scores per window" }),
      el("p", { class: "section-desc", text:
        "Two charts, not one: the ensemble's fitness and the single model's score use different F1 definitions, so their values are not on one scale." }),
      el("div", { class: "viz-pair" }, ensembleChart, singleChart)));
    if (offline) parts.push(offline);
    parts.push(scenarioTabs(payload));
  } else {
    if (offline) parts.push(offline);
    if (summary.status !== "skipped") {
      parts.push(el("p", { class: "empty-note", text: isLive(summary)
        ? "No window has been scored yet. The first results appear here as soon as the first scenario scores window 0."
        : "No window was scored." }));
    }
  }
  root.replaceChildren(el("div", { class: "online-page" }, parts));
  drawCharts();
  revealSelectedTab();
  if (focusKey) {
    const again = root.querySelector(`[data-focus="${focusKey}"]`);
    if (again) again.focus({ preventScroll: true });
  }
}

async function renderEmpty() {
  const label = (ui.catalog && (ui.catalog.datasets || []).find((d) => d.name === dataset)) || {};
  const card = el("section", { class: "card stack empty-card" },
    el("h1", { text: "No online results yet" }),
    el("p", { class: "muted", text:
      `Nothing has been written for ${label.label || dataset} · entity ${entity}. The online phase runs after `
      + "the offline selection when a run has “Online phase” turned on, and writes its results here window by window." }),
    el("p", {}, el("a", { class: "button primary icon-arrow", href: "/#sec-online", text: "Configure a run" })));
  root.replaceChildren(card);
  // Whether it could run at all: the configure page would refuse a too-short
  // entity, so say so here rather than send the reader there to find out.
  try {
    const f = await getJSON(`/api/online/feasibility/${encodeURIComponent(dataset)}/${encodeURIComponent(entity)}`);
    card.insertBefore(f.ok
      ? el("p", { class: "banner banner-info", text:
          `This entity is long enough: ${plural(f.n_windows, "window")} of ${f.window_size} samples with the default window sizing.` })
      : el("p", { class: "banner banner-warn" }, el("strong", { text: "This entity is too short for the online phase. " }), f.reason || ""),
    card.lastElementChild);
  } catch (e) { /* the entity's series could not be read; the run would say why */ }
}

async function load() {
  clearTimeout(ui.timer);
  let payload;
  try {
    payload = await getJSON(API);
  } catch (error) {
    if (error.status === 404) {
      if (ui.lastStamp === null) renderEmpty();
    } else if (ui.lastStamp === null) {
      root.replaceChildren(el("p", { class: "banner banner-error", text:
        `The online results could not be read: ${error.message}` }));
    }
    return;
  }
  const summary = payload.summary || {};
  const stampNow = `${summary.updated_at}|${summary.status}`;
  if (stampNow !== ui.lastStamp) {
    ui.lastStamp = stampNow;
    render(payload);
    monitor.update(payload);
  }
  if (isLive(summary)) ui.timer = setTimeout(load, POLL_MS);
}

/* The catalogue supplies the dataset's display label. The header's view tabs
 * are rendered by the server (templates/_view_tabs.html), not probed here. */
async function loadCatalog() {
  try {
    ui.catalog = await getJSON("/api/catalog");
  } catch (e) { /* the raw dataset name is shown instead */ }
}

loadCatalog().finally(() => load());
