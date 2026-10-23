/* Which section of a long page the reader is in.
 *
 * The contents lists that travel with the reader — the explanation's rail and
 * the sidebar the comprehensive report and the documentation share — all ask
 * this, and they ask it here, so one scroll position never gets two answers.
 * Presentation only: the caller turns the answer into a class on a link.
 *
 * From the scroll position rather than an IntersectionObserver. A section
 * taller than the viewport reports nothing in view while the reader is in the
 * middle of it; a short one (the 106px agreement strip) never crossed the thin
 * band the explanation used to watch; and the sections at the foot of a page
 * cannot scroll far enough up to reach any band set part-way down the screen,
 * so the rail went on naming the section above them. */

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

/* What the reader does to move the page themselves. Any of these ends a hold
 * (below) at once: the jump they asked for is over the moment they take over. */
const INPUTS = ["wheel", "touchstart", "keydown"];

/* How long the page has to be still before a followed link's jump counts as
 * finished. Long enough to span the frames of a smooth scroll, short enough
 * that the rail never looks stuck. */
const SETTLE_MS = 160;

/* Tracks `ids` (fragment targets, in page order) and calls `mark(id)` whenever
 * the current one changes. `sectionOf(node)` maps a target to the box whose
 * top counts — the stage card around a stage's heading — and defaults to the
 * target itself. Returns `follow(id)`, for the caller to invoke when a link to
 * one of the ids is followed.
 *
 * The rule:
 *   the current section is the last whose top has reached the line, a little
 *   below the sticky header; above the first section, the first;
 *   at the foot of the page, where the last sections can never rise to that
 *   line, it is the one the reader followed a link to when its top is in
 *   view, and otherwise the last section whose top is in view;
 *   a followed link is marked at once, and the mark stands until the jump has
 *   settled, so a smooth scroll does not flicker through every section it
 *   passes on the way. */
export function trackSections(ids, { mark, sectionOf = (node) => node }) {
  let line = null;
  let current;
  let held = null;
  let settle = 0;
  let frame = 0;
  let tries = 0;

  /* The line: below the sticky header (and the strip of chips on a narrow
     screen) plus a line or two, so a heading counts as reached once it is
     near the top, not only once it has gone under the header. The clearance
     is the stylesheet's own, read back as the resolved scroll-margin-top of a
     target — the custom property would come back as an unevaluated calc() —
     so the two cannot disagree and the phone layout's smaller value applies
     there. Read lazily, from a target that is in the document: read once
     when the list was built, the sections were not attached yet, the margin
     came back as 0 and every heading lit its link a section late. */
  const readLine = () => {
    const node = ids.map((id) => document.getElementById(id)).find(Boolean);
    if (!node) return null;
    const clearance = parseFloat(getComputedStyle(node).scrollMarginTop);
    if (!Number.isFinite(clearance)) return null;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    return clearance + 3 * rem;
  };

  const compute = () => {
    const boxes = [];
    ids.forEach((id) => {
      const node = document.getElementById(id);
      if (node) boxes.push({ id, rect: (sectionOf(node) || node).getBoundingClientRect() });
    });
    if (!boxes.length) return undefined;
    if (line === null) line = readLine();
    const at = line === null ? 0 : line;

    let pick = boxes[0].id;
    boxes.forEach((box) => { if (box.rect.top <= at) pick = box.id; });

    const doc = document.documentElement;
    const atEnd = window.scrollY > 0
      && window.scrollY + window.innerHeight >= doc.scrollHeight - 2;
    if (atEnd) {
      const inView = boxes.filter((box) => box.rect.top >= 0 && box.rect.top < window.innerHeight);
      const asked = decodeURIComponent((location.hash || "").replace(/^#/, ""));
      const followed = inView.find((box) => box.id === asked);
      if (followed) pick = followed.id;
      else if (inView.length) pick = inView[inView.length - 1].id;
    }
    return pick;
  };

  const apply = (id) => {
    if (id === current) return;
    current = id;
    mark(id);
  };

  const update = () => {
    frame = 0;
    if (held) return;
    const id = compute();
    // Not built yet (the caller inserts the sections after this returns):
    // look again on the next frames rather than waiting for a scroll.
    if (id === undefined) {
      if (tries++ < 120) schedule();
      return;
    }
    apply(id);
  };
  function schedule() {
    if (!frame) frame = requestAnimationFrame(update);
  }
  const release = () => {
    if (!held) return;
    held = null;
    clearTimeout(settle);
    schedule();
  };

  window.addEventListener("scroll", () => {
    if (held) {
      clearTimeout(settle);
      settle = setTimeout(release, SETTLE_MS);
      return;
    }
    schedule();
  }, { passive: true });
  window.addEventListener("resize", () => { line = null; schedule(); }, { passive: true });
  INPUTS.forEach((type) =>
    window.addEventListener(type, release, { capture: true, passive: true }));
  schedule();

  return function follow(id) {
    if (!ids.includes(id)) return;
    held = id;
    apply(id);
    clearTimeout(settle);
    settle = setTimeout(release, SETTLE_MS);
  };
}

/* Keep the marked link in view inside the contents: a column that scrolls on
 * a short screen, a strip that scrolls sideways on a narrow one. The contents
 * scroll, never the page — scrollIntoView would move both. */
export function revealInRail(link, list, box) {
  if (!link || !list) return;
  const behavior = REDUCED_MOTION.matches ? "auto" : "smooth";
  const l = link.getBoundingClientRect();
  if (!l.width && !l.height) return;           // not displayed (collapsed, a hidden sublist)
  if (list.scrollWidth > list.clientWidth + 1) {
    const b = list.getBoundingClientRect();
    if (l.left < b.left || l.right > b.right) {
      list.scrollTo({ left: list.scrollLeft + l.left - b.left - (b.width - l.width) / 2, behavior });
    }
  } else if (box && box.scrollHeight > box.clientHeight + 1) {
    const b = box.getBoundingClientRect();
    if (l.top < b.top || l.bottom > b.bottom) {
      box.scrollTo({ top: box.scrollTop + l.top - b.top - b.height / 3, behavior });
    }
  }
}
