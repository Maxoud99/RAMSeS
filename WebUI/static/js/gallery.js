/* Lightbox for the plot galleries.
 *
 * The per-window sets run to 173 frames, so nothing is rendered eagerly: the
 * lightbox holds one <img>, and pages of items are fetched as the user moves
 * past what has been loaded.
 */

import { $, el } from "./dom.js";

let items = [];
let index = 0;
let options = {};
let attached = false;

function refs() {
  return {
    box: $("#lightbox"),
    img: $("#lightbox-img"),
    title: $("#lightbox-title"),
    count: $("#lightbox-count"),
    prev: $("#lightbox-prev"),
    next: $("#lightbox-next"),
    close: $("#lightbox-close"),
  };
}

function show() {
  const { img, title, count, prev, next } = refs();
  const item = items[index];
  if (!item) return;
  img.src = item.src;
  img.alt = item.title || "";
  title.textContent = item.title || item.name || "";
  const total = options.total || items.length;
  count.textContent = `${index + 1} / ${total}`;
  prev.disabled = index === 0;
  next.disabled = index >= total - 1;
}

async function move(delta) {
  const target = index + delta;
  const total = options.total || items.length;
  if (target < 0 || target >= total) return;
  // Fetch the next page only when the user actually reaches its edge.
  if (target >= items.length && options.loadMore) {
    const more = await options.loadMore(items.length);
    if (more && more.length) items = items.concat(more);
  }
  if (target < items.length) {
    index = target;
    show();
  }
}

function close() {
  const { box, img } = refs();
  box.hidden = true;
  img.src = "";
  document.body.style.removeProperty("overflow");
}

function onKey(event) {
  const { box } = refs();
  if (box.hidden) return;
  if (event.key === "Escape") close();
  else if (event.key === "ArrowRight") move(1);
  else if (event.key === "ArrowLeft") move(-1);
}

export function attachLightbox() {
  if (attached) return;
  const { prev, next, close: closeButton, box } = refs();
  if (!box) return;
  prev.addEventListener("click", () => move(-1));
  next.addEventListener("click", () => move(1));
  closeButton.addEventListener("click", close);
  box.addEventListener("click", (event) => { if (event.target === box) close(); });
  document.addEventListener("keydown", onKey);
  attached = true;
}

export async function openLightbox(initialItems, startIndex = 0, opts = {}) {
  options = opts || {};
  items = initialItems ? initialItems.slice() : [];
  if (!items.length && options.lazyAll) {
    items = (await options.lazyAll()) || [];
  }
  if (!items.length) return;
  index = Math.min(Math.max(0, startIndex), items.length - 1);
  const { box } = refs();
  box.hidden = false;
  document.body.style.overflow = "hidden";
  show();
}
