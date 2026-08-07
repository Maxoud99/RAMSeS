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
