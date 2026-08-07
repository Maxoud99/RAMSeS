/* The run form.
 *
 * The command preview comes from the server's dry-run endpoint rather than
 * being rebuilt here, so what the user copies is exactly what gets spawned.
 */

import { $, $$, el, getJSON, postJSON, pct, timeAgo, familyClass } from "./dom.js";

const STAGE_LABELS = {
  ga: "Genetic algorithm", thompson: "Thompson Sampling", gan: "GAN perturbations",
  offby: "Off-by-threshold", montecarlo: "Monte Carlo",
};
const ALL_STAGES = Object.keys(STAGE_LABELS);

let catalog = null;
let selectedDetectors = new Set();

function datasetOf() { return $("#dataset").value; }
function entityOf() { return $("#entity").value; }

function selectedStages() {
  return $$("#stages input:checked").map((i) => i.value);
}

function currentBody(extra = {}) {
  return {
    dataset: datasetOf(),
    entity: entityOf(),
    stages: selectedStages(),
    detectors: Array.from(selectedDetectors),
    explain: $("#explain").checked,
    parallel: $("#parallel").checked,
    enable_online: $("#enable_online").checked,
    overwrite: $("#overwrite").checked,
    iteration: Number($("#iteration").value) || 5,
    timeout: Number($("#timeout").value) || 7200,
    llm_model: $("#llm_model").value.trim() || null,
    llm_base_url: $("#llm_base_url").value.trim() || null,
    ...extra,
  };
}

/* ── Rendering ─────────────────────────────────────────────────────────────── */

function renderWarnings() {
  const box = $("#warnings");
  box.replaceChildren(...(catalog.warnings || []).map((w) =>
    el("p", { class: "banner banner-warn", text: w.text })));
}

function renderDatasets() {
  const select = $("#dataset");
  // `value` stays the canonical key the CLI expects; only the label is pretty.
  select.replaceChildren(...catalog.datasets.map((d) =>
    el("option", { value: d.name, disabled: !d.runnable },
      `${d.label || d.name} — ${d.n_entities} entities` +
      (d.runnable ? "" : " (not supported)"))));
  select.addEventListener("change", () => { renderEntities(); refreshDetectors(); });
  renderEntities();
}

function renderEntities() {
  const dataset = catalog.datasets.find((d) => d.name === datasetOf());
  const filter = ($("#entity-filter").value || "").toLowerCase();
  const trained = new Set((dataset && dataset.trained) || []);
  const entities = (dataset ? dataset.entities : [])
    .filter((e) => !filter || e.toLowerCase().includes(filter))
    // Entities with checkpoints first: those run immediately, the rest have to
    // train from scratch. Junk files in the data directory sink to the bottom.
    .sort((a, b) => (trained.has(b) - trained.has(a)) || a.localeCompare(b, undefined, { numeric: true }));

  const select = $("#entity");
  const previous = select.value;
  select.replaceChildren(...entities.map((e) => el("option", { value: e },
    trained.has(e) ? e : `${e} — needs training`)));
  if (entities.includes(previous)) select.value = previous;
  // size must stay 1: any larger turns the <select> into an always-open
  // listbox that never collapses after a choice.
  select.size = 1;
}

async function refreshDetectors() {
  const dataset = datasetOf(), entity = entityOf();
  const box = $("#detectors");
  if (!dataset || !entity) { box.replaceChildren(); return; }

  // Availability is per entity (it depends on which .pth files exist), so it
  // needs its own request rather than coming from the cached catalog.
  let detectors;
  try {
    detectors = (await getJSON(
      `/api/detectors/${encodeURIComponent(dataset)}/${encodeURIComponent(entity)}`)).detectors;
  } catch (e) {
    detectors = (catalog.all_detectors || []).map(
      (name) => ({ name, available: true, params: null }));
  }

  selectedDetectors = new Set(detectors.filter((d) => d.available).map((d) => d.name));

  const byFamily = {};
  detectors.forEach((d) => {
    const family = d.family || d.name.split("_")[0];
    (byFamily[family] = byFamily[family] || []).push(d);
  });

  box.replaceChildren(...Object.entries(byFamily).map(([family, members]) => {
    const inputs = [];
    const selectable = members.filter((d) => d.available);

    // Clicking the family name toggles the whole family: if any member is
    // unchecked it selects them all, otherwise it clears them.
    const familyButton = el("button", {
      type: "button", class: "small",
      style: "width: 6em; text-align: left; font-family: var(--font-detector);",
      title: `Select or clear every ${family} detector`,
      onclick: () => {
        const turnOn = selectable.some((d) => !selectedDetectors.has(d.name));
        inputs.forEach((input) => {
          if (input.disabled) return;
          input.checked = turnOn;
          if (turnOn) selectedDetectors.add(input.value);
          else selectedDetectors.delete(input.value);
        });
        onChange();
      },
    }, family);

    const chips = members.map((d) => {
      const input = el("input", {
        type: "checkbox", value: d.name,
        checked: d.available, disabled: !d.available,
        onchange: () => {
          if (input.checked) selectedDetectors.add(d.name);
          else selectedDetectors.delete(d.name);
          onChange();
        },
      });
      inputs.push(input);
      return el("label", {
        class: "toggle",
        title: d.available ? "" : "No trained model for this entity",
      }, input, el("span", { class: "detector", text: d.name }),
         d.params && d.params.label
           ? el("span", { class: "hint", text: d.params.label }) : null);
    });

    return el("div", { class: "row" }, familyButton, ...chips);
  }));
  onChange();
}

function renderStages() {
  $("#stages").replaceChildren(...ALL_STAGES.map((token) =>
    el("label", { class: "toggle" },
      el("input", { type: "checkbox", value: token, checked: true, onchange: onChange }),
      el("span", { text: STAGE_LABELS[token] }))));
}

/* The banner encodes what a partial run actually does, up front, rather than
 * letting the user discover it four minutes later. */
function renderPartialBanner() {
  const stages = selectedStages();
  const banner = $("#partial-banner");
  const partial = stages.length > 0 && stages.length < ALL_STAGES.length;
  banner.hidden = !partial;
  if (!partial) return;
  const children = [
    el("strong", { text: "Partial run." }),
    el("span", { text: " Rank aggregation, the final ensemble-vs-single decision and the " +
      "comprehensive report are skipped, and the run goes sequential (parallel is ignored)." }),
  ];
  if ($("#explain").checked) {
    children.push(el("p", { class: "small", style: "margin: var(--sp-2) 0 0;", text:
      "The pipeline only writes explanations on a full run, so they will be generated " +
      "in a second pass once the stages finish." }));
  }
  banner.replaceChildren(...children);
}

function renderResults() {
  const box = $("#results");
  const results = catalog.results || [];
  if (!results.length) {
    box.replaceChildren(el("p", { class: "muted" },
      "No explanations yet — run the pipeline with explanations enabled."));
    return;
  }
  box.replaceChildren(...results.map((r) => el("a", {
    class: "card card-tight row-between", href: `/result/${r.dataset}/${r.entity}`,
    style: "text-decoration: none; color: inherit;",
  },
    el("div", {},
      el("strong", { text: `${r.dataset} · ${r.entity}` }),
      el("div", { class: "small muted", text:
        `${r.framework_choice ? r.framework_choice.replace(/_/g, " ") : "no decision"} · ${r.n_stages} stages explained` })),
    el("div", { class: "small muted mono", text:
      [r.hallucination_rate !== null && r.hallucination_rate !== undefined
        ? `${pct(r.hallucination_rate)} unsupported` : null,
       timeAgo(r.generated_at)].filter(Boolean).join(" · ") }))));
}

async function renderHealth() {
  try {
    const health = await getJSON("/api/health");
    const llm = health.llm || {};
    $("#llm-health").replaceChildren(
      el("span", { class: llm.reachable ? "badge badge-ok" : "badge badge-warn" },
         llm.reachable ? "✓ LLM server reachable" : "! LLM server unreachable"),
      el("span", { text: ` ${llm.model} at ${llm.base_url}` }),
      llm.reachable ? null : el("span", { class: "muted", text:
        " — the run will finish but write no narratives. Start it with `ollama serve`." }));
  } catch (e) { /* health is advisory */ }
}

/* ── Interaction ──────────────────────────────────────────────────────────── */

let previewTimer = null;

function onChange() {
  const count = selectedDetectors.size;
  $("#detector-count").textContent = `${count} of 11 selected`;
  const tooFew = count < 2;
  $("#detector-error").hidden = !tooFew;
  if (tooFew) $("#detector-error").textContent = "Select at least two detectors.";
  $("#start").disabled = tooFew || !selectedStages().length || !entityOf();

  renderPartialBanner();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updatePreview, 150);
}

async function updatePreview() {
  if (!entityOf()) return;
  try {
    const data = await postJSON("/api/runs", currentBody({ dry_run: true }));
    $("#command").textContent = data.command;
  } catch (error) {
    $("#command").textContent = `# ${error.message}`;
  }
}

async function submit(event) {
  event.preventDefault();
  $("#start-error").textContent = "";
  $("#start").disabled = true;
  try {
    const data = await postJSON("/api/runs", currentBody());
    location.href = data.url;
  } catch (error) {
    $("#start").disabled = false;
    $("#start-error").textContent = error.status === 409
      ? `A run is already in progress. View it at /run/${error.detail.active_job_id}`
      : error.message;
  }
}

async function init() {
  catalog = await getJSON("/api/catalog");
  renderWarnings();
  renderDatasets();
  renderStages();
  renderResults();
  renderHealth();
  await refreshDetectors();

  $("#entity").addEventListener("change", refreshDetectors);
  $("#entity-filter").addEventListener("input", renderEntities);
  $("#run-form").addEventListener("submit", submit);
  $$("#run-form input, #run-form select").forEach((node) =>
    node.addEventListener("change", onChange));

  $$("[data-select]").forEach((button) => button.addEventListener("click", () => {
    const all = button.dataset.select === "all";
    $$("#detectors input").forEach((input) => {
      if (input.disabled) return;
      input.checked = all;
      if (all) selectedDetectors.add(input.value); else selectedDetectors.delete(input.value);
    });
    onChange();
  }));

  $$("[data-stages]").forEach((button) => button.addEventListener("click", () => {
    const wanted = button.dataset.stages === "robustness"
      ? ["gan", "offby", "montecarlo"] : ALL_STAGES;
    $$("#stages input").forEach((input) => { input.checked = wanted.includes(input.value); });
    onChange();
  }));

  $("#copy-command").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#command").textContent);
      $("#copy-command").textContent = "Copied";
      setTimeout(() => { $("#copy-command").textContent = "Copy"; }, 1200);
    } catch (e) { /* clipboard may be blocked; the text is selectable anyway */ }
  });
}

init();
