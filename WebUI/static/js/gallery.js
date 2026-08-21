/* Lightbox for the plot galleries.
 *
 * The per-window sets run to 173 frames, so nothing is rendered eagerly: the
 * lightbox holds one <img>, and pages of items are fetched as the user reaches
 * past what has been loaded — either by stepping to the edge of a page or by
 * typing a figure number into the counter, which may cross several at once.
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
    input: $("#lightbox-index"),
    total: $("#lightbox-total"),
    prev: $("#lightbox-prev"),
    next: $("#lightbox-next"),
    close: $("#lightbox-close"),
  };
}

function totalCount() {
  return options.total || items.length;
}

function show() {
  const { img, title, input, total, prev, next } = refs();
  const item = items[index];
  if (!item) return;
  img.src = item.src;
  img.alt = item.title || "";
  title.textContent = item.title || item.name || "";
  const count = totalCount();
  // Not written while the field has focus: overwriting what someone is halfway
  // through typing is how a jump to 173 becomes a jump to 1.
  if (document.activeElement !== input) input.value = String(index + 1);
  input.max = String(count);
  // `size` does nothing on a number input, so the field is widened to the digits
  // it has to hold. 2ch of slack leaves room for the spinner.
  input.style.width = `${String(count).length + 2}ch`;
  total.textContent = ` / ${count}`;
  prev.disabled = index === 0;
  next.disabled = index >= count - 1;
}

/* Load pages until `target` is in `items`, or until the source runs dry.
 *
 * Sets are paged 60 at a time, so reaching frame 95 from a fresh open needs one
 * more page and frame 173 needs two — a jump has to be able to cross more than
 * one page boundary, unlike move(), which only ever steps over the edge of the
 * last one. The empty-page guard is what stops a short or lying `total` from
 * spinning here forever. */
async function ensureLoaded(target) {
  while (target >= items.length && options.loadMore) {
    const more = await options.loadMore(items.length);
    if (!more || !more.length) break;
    items = items.concat(more);
  }
  return target < items.length;
}

async function goTo(target) {
  if (target < 0 || target >= totalCount()) return;
  if (await ensureLoaded(target)) {
    index = target;
    show();
  }
}

async function move(delta) {
  await goTo(index + delta);
}

/* Read the typed figure number and jump to it.
 *
 * Clamped rather than rejected: someone who types past the end means the last
 * frame, and silently refusing to move looks like the control is broken. */
async function jumpFromInput() {
  const { box, input } = refs();
  // blur fires as the lightbox closes; without this the field would re-request
  // the image it had just been told to drop.
  if (box.hidden) return;
  const typed = parseInt(input.value, 10);
  if (!Number.isFinite(typed)) {
    input.value = String(index + 1);
    return;
  }
  const target = Math.min(Math.max(1, typed), totalCount()) - 1;
  await goTo(target);
  input.value = String(index + 1);
}

function close() {
  const { box, img } = refs();
  box.hidden = true;
  img.src = "";
  document.body.style.removeProperty("overflow");
}

function onKey(event) {
  const { box, input } = refs();
  if (box.hidden) return;
  // While the number field has focus the arrow keys belong to it — they move
  // the caret and step the value — so only Escape is global there.
  if (event.target === input) {
    if (event.key === "Escape") { input.blur(); close(); }
    return;
  }
  if (event.key === "Escape") close();
  else if (event.key === "ArrowRight") move(1);
  else if (event.key === "ArrowLeft") move(-1);
}

export function attachLightbox() {
  if (attached) return;
  const { prev, next, input, close: closeButton, box } = refs();
  if (!box) return;
  prev.addEventListener("click", () => move(-1));
  next.addEventListener("click", () => move(1));
  closeButton.addEventListener("click", close);
  // Enter jumps; blur commits whatever is left in the field, so a click
  // elsewhere is not a silently discarded number. `change` covers the spinner.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); jumpFromInput(); }
  });
  input.addEventListener("change", () => jumpFromInput());
  input.addEventListener("blur", () => jumpFromInput());
  input.addEventListener("focus", () => input.select());
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
