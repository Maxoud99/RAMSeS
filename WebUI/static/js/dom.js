/* Shared helpers: element creation, formatting, and the theme toggle.
 * No framework, no build step — plain ES modules. */

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = {};
    try { detail = await response.json(); } catch (e) {}
    const error = new Error(detail.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return response.json();
}

export async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `${response.status}`);
    error.status = response.status;
    error.detail = data;
    throw error;
  }
  return data;
}

/* Rates are small fractions; a plain percentage reads better than 0.0164. */
export function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function duration(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}

export function timeAgo(epochSeconds) {
  if (!epochSeconds) return "";
  const delta = Date.now() / 1000 - epochSeconds;
  if (delta < 90) return "just now";
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  return new Date(epochSeconds * 1000).toLocaleDateString();
}

export function familyClass(name) {
  const family = String(name).split("_")[0].toLowerCase();
  return `chip chip-${family}`;
}

/* Paragraph-aware rendering: the narratives contain blank-line paragraph
 * breaks, and textContent alone would collapse them into one block. */
export function proseNode(text, className = "prose") {
  const wrapper = el("div", { class: className });
  String(text || "").split(/\n{2,}/).forEach((para) => {
    if (para.trim()) wrapper.append(el("p", { text: para.trim() }));
  });
  return wrapper;
}

/* ── Sortable tables ─────────────────────────────────────────────────────────
 *
 * Click a header to sort by it; click again to reverse; a third click restores
 * the order the table was built in. That third state matters here because the
 * default order is itself a finding — context features by contribution, stages in
 * pipeline order — so a sort must be undoable without a reload.
 *
 * Headers are the <th> elements themselves, made focusable, rather than buttons
 * inside them: the print stylesheet hides every <button>, which would blank the
 * header row on paper.
 */

const BLANK_CELLS = new Set(["", "—", "–", "-", "n/a", "not available", "null"]);
/* Strict enough that "context feature 7" is not mistaken for the number 7. */
const STRICT_NUMBER = /^[+-]?[\d,]*\.?\d+(?:[eE][+-]?\d+)?\s*%?$/;

function cellText(row, index) {
  const cell = row.cells[index];
  return cell ? cell.textContent.trim() : "";
}

function toNumber(text) {
  const match = text.replace(/,/g, "").match(/[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?/);
  return match ? parseFloat(match[0]) : NaN;
}

/* A column is numeric if its header says so (the payload's `align` already
 * knows) or if every value present looks like a number. The header hint is what
 * lets "3 (tie)" sort as 3. */
function columnIsNumeric(head, rows, index) {
  if (head && head.cells[index] && head.cells[index].classList.contains("num")) return true;
  let seen = 0;
  for (const row of rows) {
    const text = cellText(row, index);
    if (BLANK_CELLS.has(text.toLowerCase())) continue;
    if (!STRICT_NUMBER.test(text)) return false;
    seen += 1;
  }
  return seen > 0;
}

/* Rows beyond a fold are hidden by class, so the fold has to follow the sort —
 * otherwise sorting would reorder the table while still hiding whichever rows
 * happened to start at the bottom. */
export function applyFold(table) {
  const fold = parseInt(table.dataset.collapseAfter || "0", 10);
  if (!fold) return;
  Array.from(table.tBodies[0].rows).forEach((row, i) => {
    row.classList.toggle("extra", i >= fold);
  });
}

export function makeSortable(table) {
  const head = table.tHead && table.tHead.rows[0];
  const body = table.tBodies[0];
  if (!head || !body || body.rows.length < 2) return table;

  const original = Array.from(body.rows);
  const numeric = original.length
    ? Array.from(head.cells).map((_c, i) => columnIsNumeric(head, original, i))
    : [];
  let active = -1;
  let direction = 0; // 0 none, 1 first click, -1 second

  const render = (rows) => {
    rows.forEach((row) => body.append(row));
    applyFold(table);
  };

  const sortBy = (index) => {
    if (index === active) direction = direction === 1 ? -1 : 0;
    // Numbers open on the largest — for a contribution or a claim count that is
    // the interesting end; text opens A→Z.
    else { active = index; direction = 1; }
    if (direction === 0) {
      active = -1;
      render(original.slice());
    } else {
      const sign = numeric[index] ? -direction : direction;
      const decorated = Array.from(body.rows).map((row, i) => ({ row, i }));
      decorated.sort((a, b) => {
        const ta = cellText(a.row, index), tb = cellText(b.row, index);
        const ba = BLANK_CELLS.has(ta.toLowerCase());
        const bb = BLANK_CELLS.has(tb.toLowerCase());
        // Missing values sink in both directions: "no value recorded" is not
        // smaller than every value, it is absent.
        if (ba || bb) return ba && bb ? a.i - b.i : (ba ? 1 : -1);
        let cmp;
        if (numeric[index]) {
          const na = toNumber(ta), nb = toNumber(tb);
          cmp = (Number.isNaN(na) ? 0 : na) - (Number.isNaN(nb) ? 0 : nb);
        } else {
          // Natural order, so context feature 2 precedes context feature 10.
          cmp = ta.localeCompare(tb, undefined, { numeric: true, sensitivity: "base" });
        }
        return cmp * sign || a.i - b.i; // stable
      });
      render(decorated.map((d) => d.row));
    }
    Array.from(head.cells).forEach((cell, i) => {
      const state = i !== active ? "none"
        : (numeric[i] ? (direction === 1 ? "descending" : "ascending")
                      : (direction === 1 ? "ascending" : "descending"));
      cell.setAttribute("aria-sort", state);
    });
  };

  Array.from(head.cells).forEach((cell, i) => {
    cell.setAttribute("aria-sort", "none");
    cell.tabIndex = 0;
    cell.title = "Sort by this column";
    cell.addEventListener("click", () => sortBy(i));
    cell.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        sortBy(i);
      }
    });
  });
  return table;
}

/* Theme: system by default, with a manual override that persists. */
function initTheme() {
  const button = $("#theme-toggle");
  if (!button) return;
  const order = ["", "light", "dark"];
  const label = { "": "Theme: system", light: "Theme: light", dark: "Theme: dark" };
  const glyph = { "": "◐", light: "☀", dark: "☾" };
  const apply = (value) => {
    document.documentElement.setAttribute("data-theme", value);
    button.textContent = glyph[value];
    button.title = label[value];
    try {
      if (value) localStorage.setItem("ramses-theme", value);
      else localStorage.removeItem("ramses-theme");
    } catch (e) {}
  };
  let current = document.documentElement.getAttribute("data-theme") || "";
  apply(current);
  button.addEventListener("click", () => {
    current = order[(order.indexOf(current) + 1) % order.length];
    apply(current);
  });
}

initTheme();
