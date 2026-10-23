/* The online page's live monitor: the series the online phase replays, its
 * real and injected anomalies, and the window being scored, moving.
 *
 * Reads /api/online/<dataset>/<entity>/stream (WebUI/stream.py), which serves
 * decimated slices of the stream files the pipeline writes as it goes
 * (Model_Selection/online_stream.py): min/max buckets, so a one-sample spike
 * survives any zoom, and anomalies as exact spans, so none is decimated away.
 *
 * While the run is live the window follows the scenario being replayed: the
 * page's own poll (online.js, every 5 s) hands each new payload to `update`.
 * A finished run is played back: play / pause / step / scrub and a speed,
 * moving the window through the stream. Under the series, the selected
 * scenario's score track for that window: the deployed detector's anomaly
 * scores, the threshold they were cut at, the flagged samples, and a mark
 * wherever a re-optimization was switched in.
 *
 * The element is built once and moved into each re-render of the page, so
 * its state (scenario, position, playback) survives the poll.
 */

import { el, getJSON } from "./dom.js";

const SCENARIO_NAMES = {
  fixed: "Fixed", periodic: "Periodic", threshold: "Threshold",
  adaptive_threshold: "Adaptive threshold", fixed_random: "Fixed random",
};
const SCENARIO_CLASSES = {
  fixed: "sc-fixed", periodic: "sc-periodic", threshold: "sc-threshold",
  adaptive_threshold: "sc-adaptive", fixed_random: "sc-other",
};
const SIGNAL_NAMES = { single: "Single model", ensemble: "Ensemble" };
const SPEEDS = [1, 2, 4, 8];
const LIVE_STATES = new Set(["running", "pending", "waiting_for_reopt"]);
const NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  }
  for (const c of children) if (c) node.append(c);
  return node;
}

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 3) => (isNum(v) ? v.toFixed(d) : "—");

function icon(d) {
  return svg("svg", { viewBox: "0 0 16 16", width: 16, height: 16, "aria-hidden": "true", class: "mon-icon" },
    svg("path", { d }));
}
const ICONS = {
  play: "M4 2.5v11l9-5.5z",
  pause: "M4 2.5h3v11H4zM9 2.5h3v11H9z",
  back: "M3 2.5h2v11H3zM13 2.5v11L6 8z",
  forward: "M11 2.5h2v11h-2zM3 2.5v11L10 8z",
};

function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return { ticks, digits: Math.max(0, -Math.floor(Math.log10(step))) };
}

export function createMonitor({ dataset, entity }) {
  const API = `/api/online/${encodeURIComponent(dataset)}/${encodeURIComponent(entity)}/stream`;
  const state = {
    meta: null, payload: null, missing: false,
    scenario: null, userScenario: false, signal: "single", channel: 0, copy: 0, view: "follow",
    pos: 0, follow: false, playing: false, speed: 2, timer: null,
    hover: null, width: 0, ticket: 0,
  };
  const seriesCache = new Map();
  const windowCache = new Map();

  /* ── Elements ───────────────────────────────────────────────────────────── */
  const ids = { title: "mon-title", scrub: "mon-scrub", table: "mon-table" };
  const scenarioSel = el("select", { id: "mon-scenario", "aria-label": "Scenario" });
  const signalSel = el("select", { id: "mon-signal", "aria-label": "Score track" },
    ...Object.entries(SIGNAL_NAMES).map(([k, v]) => el("option", { value: k }, `${v} scores`)));
  const channelSel = el("select", { id: "mon-channel", "aria-label": "Channel" });
  const copySel = el("select", { id: "mon-copy", "aria-label": "Injected copy" });
  const viewSel = el("select", { id: "mon-view", "aria-label": "View" },
    el("option", { value: "follow" }, "Follow the window"),
    el("option", { value: "whole" }, "Whole stream"));
  const field = (label, control, id) => el("div", { class: "mon-field", "data-field": id },
    el("label", { for: control.id, text: label }), control);
  const channelField = field("Channel", channelSel, "channel");
  const copyField = field("Injected copy", copySel, "copy");

  const playBtn = el("button", { type: "button", class: "small mon-play", "aria-label": "Play" }, icon(ICONS.play), el("span", { text: "Play" }));
  const backBtn = el("button", { type: "button", class: "small ghost mon-step", "aria-label": "Previous window" }, icon(ICONS.back));
  const fwdBtn = el("button", { type: "button", class: "small ghost mon-step", "aria-label": "Next window" }, icon(ICONS.forward));
  const scrub = el("input", { type: "range", id: ids.scrub, class: "mon-scrub", min: "0", max: "0", step: "1", value: "0",
                              "aria-label": "Window position" });
  const speedSel = el("select", { id: "mon-speed", "aria-label": "Playback speed" },
    ...SPEEDS.map((s) => el("option", { value: String(s), selected: s === state.speed }, `${s} windows/s`)));
  const liveBtn = el("button", { type: "button", class: "small tertiary mon-live", hidden: true }, "Back to live");
  const position = el("span", { class: "mon-position", "aria-hidden": "true" });

  const readout = el("p", { class: "mon-readout" });
  const announce = el("p", { class: "visually-hidden", "aria-live": "polite" });
  const legend = el("ul", { class: "viz-legend mon-legend", "aria-label": "Legend" });
  const tip = el("div", { class: "viz-tip", hidden: true });
  const plot = el("div", { class: "viz-plot mon-plot", tabindex: "0", role: "group",
    "aria-label": "The replayed series with its anomalies and the current window, and under it the "
      + "score track. Arrow keys move a cursor sample by sample; the controls above move the window." }, tip);
  const note = el("p", { class: "mon-note" });
  const tableBox = el("div", { class: "viz-data table-scroll", id: ids.table, hidden: true });
  const tableToggle = el("button", { type: "button", class: "link-button viz-toggle", "aria-expanded": "false",
                                     "aria-controls": ids.table }, "Show data");

  const element = el("section", { class: "monitor-section", "aria-labelledby": ids.title },
    el("h2", { id: ids.title, class: "section-title", text: "Live monitor" }),
    el("p", { class: "section-desc", text:
      "The test series as the online phase replays it, with its anomalies, and the window being scored. "
      + "While the run is live the window follows the scenario being replayed; afterwards, play it back." }),
    el("div", { class: "mon-controls" },
      field("Scenario", scenarioSel, "scenario"), field("Score track", signalSel, "signal"),
      channelField, copyField, field("View", viewSel, "view")),
    el("div", { class: "mon-playback", role: "group", "aria-label": "Playback" },
      backBtn, playBtn, fwdBtn, scrub, position, field("Speed", speedSel, "speed"), liveBtn),
    readout, announce,
    el("figure", { class: "viz mon-figure", "aria-labelledby": ids.title }, legend, plot),
    note, tableToggle, tableBox);

  /* ── Data ───────────────────────────────────────────────────────────────── */
  const scenarioDoc = () => (state.payload && state.payload.scenarios || {})[state.scenario];
  const windows = () => ((scenarioDoc() && scenarioDoc().windows) || []);
  const current = () => windows()[Math.min(state.pos, Math.max(0, windows().length - 1))];
  const scenarioLive = () => {
    const doc = scenarioDoc();
    const row = ((state.payload && state.payload.summary.scenarios) || []).find((r) => r.key === state.scenario);
    return LIVE_STATES.has((doc && doc.status) || (row && row.status));
  };
  const runLive = () => {
    const s = state.payload && state.payload.summary;
    return !!s && (LIVE_STATES.has(s.status)
      || (s.scenarios || []).some((r) => LIVE_STATES.has(r.status)));
  };

  function viewport() {
    const meta = state.meta;
    const total = meta.total_length;
    const w = current();
    if (state.view === "whole" || !w) return [0, total];
    // Six windows' width around the window, so its neighbourhood and the
    // anomalies coming up are in view; clamped at the ends of the stream.
    const span = Math.min(total, Math.max(meta.window_size * 6, 200));
    let lo = Math.round((w.start + w.end) / 2 - span / 2);
    lo = Math.max(0, Math.min(lo, total - span));
    return [lo, lo + span];
  }

  async function seriesFor(lo, hi) {
    // A chunk three viewports wide at the plot's resolution, reused while the
    // viewport stays inside it, so following the window costs a request
    // every few steps rather than one per step.
    const points = Math.max(200, Math.round((state.width || 800) * 2));
    for (const [key, data] of seriesCache) {
      const [ch, cp, a, b, pts] = key.split("|").map(Number);
      if (ch === state.channel && cp === state.copy && a <= lo && b >= hi
          && (b - a) / pts <= (hi - lo) / points * 3.5) return data;
    }
    const span = hi - lo;
    const total = state.meta.total_length;
    const a = Math.max(0, lo - (state.view === "whole" ? 0 : span));
    const b = Math.min(total, hi + (state.view === "whole" ? 0 : span));
    const n = state.view === "whole" ? points : points * 3;
    const key = `${state.channel}|${state.copy}|${a}|${b}|${n}`;
    const data = await getJSON(`${API}?channels=${state.channel}&copy=${state.copy}&from=${a}&to=${b}&max_points=${n}`);
    if (seriesCache.size > 24) seriesCache.delete(seriesCache.keys().next().value);
    seriesCache.set(key, data);
    return data;
  }

  async function windowFor(i, live) {
    const key = `${state.scenario}|${i}`;
    if (windowCache.has(key)) return windowCache.get(key);
    const data = await getJSON(`${API}?series=0&scenario=${encodeURIComponent(state.scenario)}&window=${i}&max_points=2000`);
    const w = data.window || null;
    // A live window's file can still be missing; only a written one is kept.
    if (w) {
      if (windowCache.size > 200) windowCache.delete(windowCache.keys().next().value);
      windowCache.set(key, w);
    }
    return w;
  }

  /* ── Drawing ────────────────────────────────────────────────────────────── */
  function legendItems(series) {
    const items = [el("li", {}, el("span", { class: "key-line mon-key-series", "aria-hidden": "true" }),
                      el("span", { text: `Series${state.meta.n_channels > 1 ? ` (channel ${state.channel})` : ""}` }))];
    if (state.meta.has_real_labels) {
      items.push(el("li", {}, el("span", { class: "mon-key-real", "aria-hidden": "true" }),
                    el("span", { text: "Real anomaly (labelled)" })));
    }
    if (state.meta.has_injected_labels) {
      const kind = (state.meta.copies[state.copy] || {}).anomaly_type || "anomaly";
      items.push(el("li", {}, el("span", { class: "mon-key-injected", "aria-hidden": "true" }),
                    el("span", { text: `Injected: ${kind}` })));
    }
    items.push(el("li", {}, el("span", { class: "mon-key-window", "aria-hidden": "true" }),
                  el("span", { text: "Current window" })));
    items.push(el("li", {}, el("span", { class: `key-line ${SCENARIO_CLASSES[state.scenario] || "sc-other"}`, "aria-hidden": "true" }),
                  el("span", { text: `${SIGNAL_NAMES[state.signal]} score (${SCENARIO_NAMES[state.scenario] || state.scenario})` })));
    items.push(el("li", {}, el("span", { class: "key-dash", "aria-hidden": "true" }), el("span", { text: "threshold" })));
    items.push(el("li", {}, el("span", { class: `mon-key-flag ${SCENARIO_CLASSES[state.scenario] || "sc-other"}`, "aria-hidden": "true" }),
                  el("span", { text: "flagged" })));
    if (windows().some((w) => w.reopt_applied)) {
      items.push(el("li", {}, el("span", { class: "mon-key-switch", "aria-hidden": "true" }),
                    el("span", { text: "re-optimization applied" })));
    }
    legend.replaceChildren(...items);
  }

  function rangePath(xs, mins, maxs, xOf, yOf, bucket) {
    // One sample per point: a line. Buckets: the band between each bucket's
    // minimum and maximum, closed, so a spike inside a bucket is still drawn.
    let d = "";
    if (bucket === 1) {
      let pen = false;
      xs.forEach((x, k) => {
        const v = maxs[k];
        if (!isNum(v)) { pen = false; return; }
        d += `${pen ? "L" : "M"}${xOf(x + 0.5).toFixed(1)} ${yOf(v).toFixed(1)}`;
        pen = true;
      });
      return { d, area: false };
    }
    const top = [], bottom = [];
    xs.forEach((x, k) => {
      if (!isNum(maxs[k]) || !isNum(mins[k])) return;
      const cx = xOf(x + bucket / 2);
      top.push(`${cx.toFixed(1)} ${yOf(maxs[k]).toFixed(1)}`);
      bottom.push(`${cx.toFixed(1)} ${yOf(mins[k]).toFixed(1)}`);
    });
    if (!top.length) return { d: "", area: true };
    d = `M${top.join("L")}L${bottom.reverse().join("L")}Z`;
    return { d, area: true };
  }

  function draw(series, win) {
    const meta = state.meta;
    const width = state.width || plot.clientWidth || 800;
    const [lo, hi] = viewport();
    const m = { left: 48, right: 12 };
    const plotW = Math.max(40, width - m.left - m.right);
    const top = { y: 16, h: 190 };
    // Room between the panels for the window's name under the series and
    // the score track's title above its own panel.
    const score = { y: top.y + top.h + 52, h: 110 };
    const lane = { y: score.y + score.h + 8, h: 10 };
    const axisY = lane.y + lane.h + 10;
    const height = axisY + 34;
    const xOf = (p) => m.left + ((p - lo) / Math.max(1, hi - lo)) * plotW;
    const clipX = (p) => Math.max(m.left, Math.min(m.left + plotW, xOf(p)));

    const s = series.series[0];
    const vals = s.min.concat(s.max).filter(isNum);
    let yLo = Math.min(...vals), yHi = Math.max(...vals);
    if (!vals.length) { yLo = 0; yHi = 1; }
    const pad = (yHi - yLo) * 0.08 || 0.5;
    yLo -= pad; yHi += pad;
    const yOf = (v) => top.y + top.h - ((v - yLo) / (yHi - yLo)) * top.h;

    const node = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-labelledby": "mon-svg-title mon-svg-desc" },
      svg("title", { id: "mon-svg-title", text: "Live monitor" }),
      svg("desc", { id: "mon-svg-desc", text: readout.textContent }),
      svg("defs", {}, svg("pattern", { id: "mon-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse",
                                       patternTransform: "rotate(45)" },
        svg("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "mon-hatch-line" })),
        // The fetched chunk runs past the viewport on both sides; lines are
        // clipped to the plot so they never draw over the axis labels.
        svg("clipPath", { id: "mon-clip" },
          svg("rect", { x: m.left, y: 0, width: plotW, height }))));
    const clipped = () => svg("g", { "clip-path": "url(#mon-clip)" });

    // Bands: real anomalies solid, injected hatched; both across the series
    // panel, with a strip along its top so a narrow one is still visible.
    for (const [a, b] of series.real_spans || []) {
      const x0 = clipX(a), x1 = Math.max(clipX(b), x0 + 1.5);
      node.append(svg("rect", { class: "mon-real", x: x0, y: top.y, width: x1 - x0, height: top.h }),
                  svg("rect", { class: "mon-real-strip", x: x0, y: top.y - 6, width: x1 - x0, height: 4 }));
    }
    for (const [a, b] of series.injected_spans || []) {
      const x0 = clipX(a), x1 = Math.max(clipX(b), x0 + 1.5);
      node.append(svg("rect", { class: "mon-injected", x: x0, y: top.y, width: x1 - x0, height: top.h }),
                  svg("rect", { class: "mon-injected-strip", x: x0, y: top.y - 6, width: x1 - x0, height: 4 }));
    }

    // Grid and axes.
    const g = svg("g", { class: "viz-grid" });
    const yt = niceTicks(yLo, yHi);
    yt.ticks.forEach((t) => g.append(
      svg("line", { class: "grid", x1: m.left, x2: m.left + plotW, y1: yOf(t), y2: yOf(t) }),
      svg("text", { class: "tick", x: m.left - 6, y: yOf(t), dy: "0.32em", "text-anchor": "end", text: t.toFixed(yt.digits) })));
    g.append(svg("line", { class: "axis", x1: m.left, x2: m.left + plotW, y1: axisY, y2: axisY }));
    const xt = niceTicks(lo, hi, Math.max(2, Math.floor(plotW / 90)));
    xt.ticks.filter((t) => t >= lo && t <= hi).forEach((t) => g.append(
      svg("line", { class: "axis", x1: xOf(t), x2: xOf(t), y1: axisY, y2: axisY + 4 }),
      svg("text", { class: "tick", x: xOf(t), y: axisY + 16, "text-anchor": "middle", text: String(Math.round(t)) })));
    g.append(svg("text", { class: "axis-title", x: m.left + plotW, y: axisY + 30, "text-anchor": "end", text: "Sample" }));
    node.append(g);

    // The offline/online boundary.
    if (meta.offline_size >= lo && meta.offline_size <= hi) {
      const bx = xOf(meta.offline_size);
      node.append(svg("line", { class: "mon-boundary", x1: bx, x2: bx, y1: top.y - 8, y2: axisY }),
                  svg("text", { class: "lane-label", x: bx - 4, y: top.y + 12, "text-anchor": "end", text: "offline" }),
                  svg("text", { class: "lane-label", x: bx + 4, y: top.y + 12, text: "online" }));
    }

    // Re-optimizations switched in: a hairline at the first window each new
    // selection scored.
    windows().filter((w) => w.reopt_applied && w.start >= lo && w.start <= hi).forEach((w) => {
      const x = xOf(w.start);
      node.append(svg("line", { class: "mon-switch", x1: x, x2: x, y1: top.y, y2: lane.y + lane.h }),
                  svg("path", { class: "mon-switch-mark", d: `M${x - 5} ${score.y - 7}h10l-5 7z` }));
    });

    // The current window: its tint under the lines, its outline over them.
    const cur = current();
    const winBox = cur ? { x: clipX(cur.start), y: top.y - 8,
                           w: Math.max(2, clipX(cur.end) - clipX(cur.start)),
                           h: lane.y + lane.h - top.y + 8 } : null;
    if (winBox) node.append(svg("rect", { class: "mon-window-fill", x: winBox.x, y: winBox.y,
                                          width: winBox.w, height: winBox.h }));

    // The series.
    const path = rangePath(s.x, s.min, s.max, xOf, yOf, s.bucket);
    const seriesG = clipped();
    seriesG.append(svg("path", { class: path.area ? "mon-series-area" : "mon-series", d: path.d }));
    node.append(seriesG);

    if (winBox) {
      node.append(svg("rect", { class: "mon-window", x: winBox.x, y: winBox.y, width: winBox.w, height: winBox.h }));
      node.append(svg("text", { class: "mon-window-label", x: winBox.x + 4, y: top.y + top.h + 14,
                                text: `window ${cur.i}` }));
    }

    // Score track.
    const cls = SCENARIO_CLASSES[state.scenario] || "sc-other";
    node.append(svg("text", { class: "panel-title", x: m.left, y: score.y - 12,
                              text: `${SIGNAL_NAMES[state.signal]} score — ${SCENARIO_NAMES[state.scenario] || state.scenario}` }));
    node.append(svg("line", { class: "axis", x1: m.left, x2: m.left + plotW, y1: score.y + score.h, y2: score.y + score.h }));
    const track = win && win[state.signal];
    if (track && track.x.length) {
      const sv = track.min.concat(track.max).filter(isNum);
      if (isNum(track.threshold)) sv.push(track.threshold);
      let sLo = Math.min(...sv), sHi = Math.max(...sv);
      const sp = (sHi - sLo) * 0.08 || 0.5;
      sLo -= sp; sHi += sp;
      const sy = (v) => score.y + score.h - ((v - sLo) / (sHi - sLo)) * score.h;
      const st = niceTicks(sLo, sHi, 2);
      const sg = svg("g", { class: "viz-grid" });
      st.ticks.forEach((t) => sg.append(
        svg("line", { class: "grid", x1: m.left, x2: m.left + plotW, y1: sy(t), y2: sy(t) }),
        svg("text", { class: "tick", x: m.left - 6, y: sy(t), dy: "0.32em", "text-anchor": "end", text: t.toFixed(st.digits) })));
      node.append(sg);
      const sPath = rangePath(track.x, track.min, track.max, xOf, sy, track.bucket);
      const scoreG = clipped();
      scoreG.append(svg("path", { class: `${sPath.area ? "mon-score-area" : "line"} ${cls}`, d: sPath.d }));
      node.append(scoreG);
      if (isNum(track.threshold)) {
        node.append(svg("line", { class: "threshold", x1: clipX(win.start), x2: clipX(win.end),
                                  y1: sy(track.threshold), y2: sy(track.threshold) }));
      }
      for (const [a, b] of track.flagged || []) {
        const x0 = clipX(a), x1 = Math.max(clipX(b), x0 + 2);
        node.append(svg("rect", { class: `mon-flag ${cls}`, x: x0, y: lane.y, width: x1 - x0, height: lane.h }));
      }
      if (!isNum(track.threshold)) {
        node.append(svg("text", { class: "lane-label", x: m.left + plotW, y: score.y - 12, "text-anchor": "end",
                                  text: plotW > 560 ? "no threshold: the window's labels are one class"
                                                    : "no threshold" }));
      }
    } else {
      node.append(svg("text", { class: "lane-label", x: m.left + plotW / 2, y: score.y + score.h / 2, "text-anchor": "middle",
        text: cur ? "No scores for this window yet" : "No window scored yet" }));
    }
    node.append(svg("text", { class: "lane-label", x: m.left - 6, y: lane.y + lane.h - 1, "text-anchor": "end", text: "flags" }));
    node.append(svg("line", { class: "crosshair", x1: 0, x2: 0, y1: top.y, y2: lane.y + lane.h, visibility: "hidden" }));

    const old = plot.querySelector("svg");
    if (old) old.remove();
    plot.style.height = `${height}px`;
    plot.prepend(node);
    state.layout = { xOf, lo, hi, left: m.left, plotW, series, win };
    if (state.hover !== null) showTip(state.hover);
  }

  /* ── Tooltip ────────────────────────────────────────────────────────────── */
  function sampleAt(p) {
    const L = state.layout;
    if (!L) return null;
    const s = L.series.series[0];
    const k = Math.max(0, Math.min(s.x.length - 1, Math.floor((p - s.x[0]) / s.bucket)));
    const inSpan = (spans) => (spans || []).find(([a, b]) => p >= a && p < b);
    const track = L.win && L.win[state.signal];
    let scoreText = null;
    if (track && p >= L.win.start && p < L.win.end) {
      const j = Math.max(0, Math.min(track.x.length - 1, Math.floor((p - track.x[0]) / track.bucket)));
      const flagged = (track.flagged || []).some(([a, b]) => p >= a && p < b);
      scoreText = `${track.bucket === 1 ? fmt(track.max[j], 4) : `${fmt(track.min[j], 4)}–${fmt(track.max[j], 4)}`}${flagged ? " · flagged" : ""}`;
    }
    const inj = inSpan(L.series.injected_spans);
    return {
      p, value: s.bucket === 1 ? fmt(s.max[k], 4) : `${fmt(s.min[k], 4)} to ${fmt(s.max[k], 4)} (${s.bucket} samples)`,
      real: state.meta.has_real_labels ? (inSpan(L.series.real_spans) ? "anomalous" : "normal") : null,
      injected: state.meta.has_injected_labels ? (inj ? `injected ${inj[2] || "anomaly"}` : "no injected anomaly") : null,
      score: scoreText,
      part: p < state.meta.offline_size ? "offline" : "online",
    };
  }

  function showTip(p) {
    const L = state.layout;
    if (!L || p < L.lo || p >= L.hi) { tip.hidden = true; return; }
    state.hover = p;
    const info = sampleAt(p);
    const x = L.xOf(p + 0.5);
    const cross = plot.querySelector(".crosshair");
    if (cross) { cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.removeAttribute("visibility"); }
    const rows = [
      el("p", { class: "viz-tip-note mon-tip-value" }, el("strong", { text: info.value }),
         state.meta.n_channels > 1 ? ` (channel ${state.channel})` : ""),
      info.real ? el("p", { class: "viz-tip-note", text: `Real label: ${info.real}` }) : null,
      info.injected ? el("p", { class: "viz-tip-note", text: info.injected.replace(/^injected /, "Injected: ") }) : null,
      info.score ? el("p", { class: "viz-tip-note", text: `${SIGNAL_NAMES[state.signal]} score: ${info.score}` }) : null,
    ];
    tip.replaceChildren(el("p", { class: "viz-tip-title", text: `Sample ${Math.floor(p)} · ${info.part}` }),
      ...rows.filter(Boolean));
    tip.hidden = false;
    const width = plot.clientWidth, tw = tip.offsetWidth;
    let left = x + 12;
    if (left + tw > width) left = x - 12 - tw;
    tip.style.left = `${Math.max(0, Math.min(left, width - tw))}px`;
    tip.style.top = "8px";
    return info;
  }

  function hideTip() {
    state.hover = null;
    tip.hidden = true;
    const cross = plot.querySelector(".crosshair");
    if (cross) cross.setAttribute("visibility", "hidden");
  }

  plot.addEventListener("pointermove", (e) => {
    const L = state.layout;
    if (!L) return;
    const box = plot.getBoundingClientRect();
    const p = L.lo + ((e.clientX - box.left - L.left) / L.plotW) * (L.hi - L.lo);
    showTip(Math.floor(p));
  });
  plot.addEventListener("pointerleave", () => { if (document.activeElement !== plot) hideTip(); });
  plot.addEventListener("blur", hideTip);
  plot.addEventListener("keydown", (e) => {
    const L = state.layout;
    if (!L) return;
    const stepBy = Math.max(1, L.series.series[0].bucket);
    const at = state.hover === null ? (current() ? current().start : L.lo) : state.hover;
    const moves = { ArrowRight: at + stepBy, ArrowLeft: at - stepBy, Home: L.lo, End: L.hi - 1 };
    if (e.key === "Escape") { hideTip(); return; }
    if (!(e.key in moves)) return;
    e.preventDefault();
    const info = showTip(Math.max(L.lo, Math.min(L.hi - 1, moves[e.key])));
    if (info) announce.textContent = `Sample ${Math.floor(info.p)}, value ${info.value}`
      + `${info.real ? `, real label ${info.real}` : ""}${info.injected ? `, ${info.injected}` : ""}`
      + `${info.score ? `, score ${info.score}` : ""}.`;
  });

  /* ── Rendering pipeline ─────────────────────────────────────────────────── */
  function syncControls() {
    const ws = windows();
    scrub.max = String(Math.max(0, ws.length - 1));
    scrub.value = String(Math.min(state.pos, Math.max(0, ws.length - 1)));
    scrub.disabled = ws.length < 2;
    backBtn.disabled = state.pos <= 0;
    fwdBtn.disabled = state.pos >= ws.length - 1;
    playBtn.disabled = ws.length < 2;
    playBtn.setAttribute("aria-label", state.playing ? "Pause" : "Play");
    playBtn.replaceChildren(icon(state.playing ? ICONS.pause : ICONS.play), el("span", { text: state.playing ? "Pause" : "Play" }));
    const live = runLive() && scenarioLive();
    liveBtn.hidden = !(live && !state.follow);
    const w = current();
    const planned = (scenarioDoc() && scenarioDoc().n_windows_planned) || ws.length;
    position.textContent = w ? `Window ${w.i} of ${planned}` : "";
    scrub.setAttribute("aria-valuetext", w ? `window ${w.i}, samples ${w.start} to ${w.end}` : "no window");
  }

  function describe(win) {
    const w = current();
    if (!w) return "No window has been scored yet.";
    const track = win && win[state.signal];
    const parts = [`Window ${w.i}: samples ${w.start}–${w.end}`];
    // n_anomalous counts the labels the window was scored on.
    const kind = state.meta.label_source === "real" ? "labelled" : "injected";
    if (w.scorable === false) {
      parts.push(`${w.n_anomalous ? "anomalous throughout" : `no ${kind} anomaly in it`} (no score)`);
    } else if (isNum(w.n_anomalous)) {
      parts.push(`${w.n_anomalous} of ${w.n_samples} samples ${kind} anomalous`);
    }
    const value = state.signal === "single" ? w.single_score : w.ensemble_score;
    parts.push(`${SIGNAL_NAMES[state.signal].toLowerCase()} score ${fmt(value, 4)}`);
    if (track) {
      parts.push(isNum(track.threshold) ? `threshold ${fmt(track.threshold, 4)}, ${track.n_flagged} flagged` : "no threshold");
    }
    parts.push(state.signal === "single" ? `deployed ${w.deployed_single || "—"}`
                                         : `deployed ${(w.deployed_ensemble || []).join(" + ") || "—"}`);
    if (w.reopt_applied) parts.push(`re-optimization from window ${w.reopt_applied.started_at_window} applied here`);
    if (state.follow && runLive() && scenarioLive()) parts.push("live");
    return `${parts.join(" · ")}.`;
  }

  async function render() {
    if (!state.meta || !state.scenario) return;
    const ticket = ++state.ticket;
    syncControls();
    const [lo, hi] = viewport();
    let series, win = null;
    try {
      series = await seriesFor(lo, hi);
      const w = current();
      if (w) win = await windowFor(w.i);
    } catch (error) {
      if (ticket === state.ticket) note.textContent = `The stream could not be read: ${error.message}`;
      return;
    }
    if (ticket !== state.ticket) return;
    readout.textContent = describe(win);
    if (!state.playing) announce.textContent = readout.textContent;
    legendItems(series);
    draw(series, win);
    if (!tableBox.hidden) fillTable(win);
  }

  /* ── The data table: the current window, sample by sample ───────────────── */
  async function fillTable(win) {
    const w = current();
    if (!w) { tableBox.replaceChildren(el("p", { class: "empty-note", text: "No window scored yet." })); return; }
    const n = w.end - w.start;
    const cap = Math.min(n, 1000);
    let rows;
    try {
      const data = await getJSON(`${API}?channels=${state.channel}&copy=${state.copy}&from=${w.start}&to=${w.start + cap}&max_points=${2 * cap + 2}`
        + `&scenario=${encodeURIComponent(state.scenario)}&window=${w.i}`);
      const s = data.series[0];
      const track = data.window && data.window[state.signal];
      const inSpan = (spans, p) => (spans || []).some(([a, b]) => p >= a && p < b);
      rows = s.x.map((p, k) => {
        const j = track ? p - track.x[0] : -1;
        return el("tr", {},
          el("td", { class: "num", text: String(p) }),
          el("td", { class: "num", text: fmt(s.max[k], 4) }),
          state.meta.has_real_labels ? el("td", { text: inSpan(data.real_spans, p) ? "anomalous" : "—" }) : null,
          state.meta.has_injected_labels ? el("td", { text: inSpan(data.injected_spans, p) ? "injected" : "—" }) : null,
          el("td", { class: "num", text: track && j >= 0 && j < track.x.length ? fmt(track.max[j], 4) : "—" }),
          el("td", { text: track && inSpan(track.flagged, p) ? "flagged" : "—" }));
      });
    } catch (error) {
      tableBox.replaceChildren(el("p", { class: "banner banner-warn", text: error.message }));
      return;
    }
    const more = n > cap ? [el("p", { class: "viz-sub", text: `The first ${cap} of the window's ${n} samples.` })] : [];
    tableBox.replaceChildren(el("table", { class: "summary-table viz-table" },
      el("caption", { class: "visually-hidden", text: `Window ${w.i}, sample by sample` }),
      el("thead", {}, el("tr", {},
        el("th", { scope: "col", class: "num", text: "Sample" }),
        el("th", { scope: "col", class: "num", text: "Value" }),
        state.meta.has_real_labels ? el("th", { scope: "col", text: "Real label" }) : null,
        state.meta.has_injected_labels ? el("th", { scope: "col", text: "Injected" }) : null,
        el("th", { scope: "col", class: "num", text: `${SIGNAL_NAMES[state.signal]} score` }),
        el("th", { scope: "col", text: "Prediction" }))),
      el("tbody", {}, rows)), ...more);
  }

  tableToggle.addEventListener("click", () => {
    const open = tableBox.hidden;
    tableBox.hidden = !open;
    tableToggle.setAttribute("aria-expanded", open ? "true" : "false");
    tableToggle.textContent = open ? "Hide data" : "Show data";
    if (open) fillTable(null);
  });

  /* ── Playback ───────────────────────────────────────────────────────────── */
  function moveTo(pos, { detach = true } = {}) {
    const n = windows().length;
    state.pos = Math.max(0, Math.min(n - 1, pos));
    if (detach) state.follow = false;
    render();
  }
  function stop() {
    state.playing = false;
    clearInterval(state.timer);
    state.timer = null;
    syncControls();
  }
  function play() {
    if (windows().length < 2) return;
    if (state.pos >= windows().length - 1) state.pos = 0;
    state.playing = true;
    state.follow = false;
    clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.pos >= windows().length - 1) { stop(); render(); return; }
      moveTo(state.pos + 1);
    }, 1000 / state.speed);
    syncControls();
    render();
  }
  playBtn.addEventListener("click", () => (state.playing ? stop() : play()));
  backBtn.addEventListener("click", () => { stop(); moveTo(state.pos - 1); });
  fwdBtn.addEventListener("click", () => { stop(); moveTo(state.pos + 1); });
  // Read before stop(), which redraws the controls from the old position.
  scrub.addEventListener("input", () => { const to = Number(scrub.value); stop(); moveTo(to); });
  speedSel.addEventListener("change", () => {
    state.speed = Number(speedSel.value) || 2;
    if (state.playing) play();
  });
  liveBtn.addEventListener("click", () => {
    stop();
    state.follow = true;
    moveTo(windows().length - 1, { detach: false });
  });
  element.addEventListener("keydown", (e) => {
    // Space plays and pauses from anywhere in the panel but a text control.
    if (e.key === " " && !["SELECT", "INPUT", "BUTTON"].includes(e.target.tagName)) {
      e.preventDefault();
      if (state.playing) stop(); else play();
    }
  });

  scenarioSel.addEventListener("change", () => {
    stop();
    state.scenario = scenarioSel.value;
    state.userScenario = true;
    state.follow = runLive() && scenarioLive();
    state.pos = state.follow ? windows().length - 1 : Math.min(state.pos, Math.max(0, windows().length - 1));
    render();
  });
  signalSel.addEventListener("change", () => { state.signal = signalSel.value; render(); });
  channelSel.addEventListener("change", () => { state.channel = Number(channelSel.value) || 0; render(); });
  copySel.addEventListener("change", () => { state.copy = Number(copySel.value) || 0; render(); });
  viewSel.addEventListener("change", () => { state.view = viewSel.value; render(); });

  if ("ResizeObserver" in window) {
    new ResizeObserver(() => {
      const w = Math.floor(plot.clientWidth);
      if (w && w !== state.width) { state.width = w; render(); }
    }).observe(plot);
  }

  /* ── Feeding it ─────────────────────────────────────────────────────────── */
  async function loadMeta() {
    try {
      const data = await getJSON(`${API}?series=0`);
      state.meta = data.meta;
      state.missing = false;
    } catch (error) {
      state.missing = error.status === 404;
      state.meta = null;
    }
  }

  async function update(payload) {
    state.payload = payload;
    const order = (payload.summary.scenario_order || []).filter((k) => payload.scenarios && payload.scenarios[k]);
    if (!state.meta) await loadMeta();
    if (!state.meta) {
      // No stream files: a run from before the monitor, or one whose online
      // phase has not built its stream yet. Asked again on the next poll.
      element.hidden = true;
      return;
    }
    element.hidden = false;
    // The scenario: the one being replayed while the run is live (so the
    // window moves on to the next scenario when one finishes), unless the
    // reader picked one; otherwise the first.
    const liveRow = (payload.summary.scenarios || []).find((r) => LIVE_STATES.has(r.status) && order.includes(r.key));
    const first = !state.scenario;
    if (liveRow && !state.userScenario && state.scenario !== liveRow.key) {
      state.scenario = liveRow.key;
      if (first || state.follow) state.follow = true;
    } else if (!state.scenario || !order.includes(state.scenario)) {
      state.scenario = order[0];
      state.pos = 0;
    }
    if (first && liveRow) state.follow = true;
    scenarioSel.replaceChildren(...order.map((k) => el("option", { value: k, selected: k === state.scenario },
      SCENARIO_NAMES[k] || k)));
    const meta = state.meta;
    channelField.hidden = meta.n_channels < 2;
    if (channelSel.options.length !== meta.n_channels) {
      channelSel.replaceChildren(...meta.channel_names.map((name, c) =>
        el("option", { value: String(c), selected: c === state.channel }, `Channel ${name}`)));
    }
    copyField.hidden = !(meta.has_injected_labels && meta.copies.length > 1);
    if (copySel.options.length !== meta.copies.length) {
      copySel.replaceChildren(...meta.copies.map((c) =>
        el("option", { value: String(c.copy), selected: c.copy === state.copy }, `${c.copy}: ${c.anomaly_type}`)));
    }
    note.textContent = meta.label_source === "real"
      ? "Real labels: nothing is injected online; the window is scored on the labelled anomalies (solid bands)."
      : `Injected anomalies (${(meta.copies[state.copy] || {}).anomaly_type || "—"}, hatched) are scored; `
        + `${meta.has_real_labels ? "the series' real anomalies (solid) are shown for comparison." : "this dataset has no real labels."}`
        + (meta.copies.length > 1 ? ` The window holds the same time slice of all ${meta.copies.length} copies; one copy is drawn at a time.` : "");
    // Live: the window follows the newest one the scenario has scored. Once
    // the run is over the panel is a player.
    if (state.follow && !state.playing) state.pos = Math.max(0, windows().length - 1);
    if (!runLive()) state.follow = false;
    render();
  }

  element.hidden = true;
  return { element, update };
}
