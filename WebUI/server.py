"""
Flask app: pages, JSON API, SSE progress stream, and image serving.

Thin by design — routes validate input and delegate. Bound to localhost only:
there is no authentication and the app spawns processes with user-supplied
arguments, so it must never be reachable from another machine.
"""

import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_file, url_for)

from Utils.pipeline_spec import (ALL_ANOMALY_TYPES, ALL_DETECTORS, ALL_STAGES,
                                 DECISION_METRICS, DEFAULT_LLM_BASE_URL,
                                 DEFAULT_LLM_MODEL, ONLINE_BASELINE_SCENARIOS,
                                 ONLINE_SCENARIOS, parse_online_labels,
                                 parse_online_strategy)
from WebUI import artifacts, catalog, jobs, ondemand, online, paths, plots, stream

SSE_KEEPALIVE_SECONDS = 15
SSE_LOG_BATCH = 40           # thousands of lines: one frame per line floods the tab
SSE_MAX_SUBSCRIBERS = 8
# Installed model ids the health check passes back: enough for the configure
# page to name a replacement, few enough that a server with dozens of models
# does not turn one status line into a catalogue.
HEALTH_MODEL_SAMPLE = 8


def _ollama_name(name: str) -> str:
    """`name` as Ollama resolves it: case-insensitive, with the
    `registry.ollama.ai/` host and `library/` namespace implied, and `:latest`
    when no tag is given. Only the last path segment can carry the tag."""
    name = name.strip().lower()
    for implied in ("registry.ollama.ai/", "library/"):
        if name.startswith(implied):
            name = name[len(implied):]
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def _model_installed(model: str, installed: List[str]) -> bool:
    """Would the server answer a chat request for `model`?

    The only server ever checked is the pipeline default, Ollama's own port,
    so this follows Ollama's name resolution rather than plain equality. It
    lists "phi4:latest" but serves "phi4", "PHI4" and "library/phi4" as that
    same model (checked against Ollama 0.12.11), and a warning telling someone
    to pull a model they already have is as wrong as a missing one. A
    different tag ("phi4:14b") is a different model.
    """
    wanted = _ollama_name(model)
    return any(_ollama_name(m) == wanted for m in installed)


def _health(model: Optional[str] = None,
            base_url: Optional[str] = None) -> Dict[str, Any]:
    """Is the LLM server up, and does it have the model a run would ask for?

    Checked before a run, not after a 4-minute wait. Reachability alone was not
    enough: Ollama answers /models whether or not the configured model is
    pulled, so a missing model showed as "reachable" and the run then wrote no
    narratives — every chat request 404s, and narrate_entity records each one
    as a per-stage error rather than failing the run.

    `model` and `base_url` are the run form's Advanced overrides, resolved the
    way the run resolves them (app.py: the flag, else the pipeline default), so
    the verdict is about the run about to start. That is also why no
    environment variable is consulted: this check used to read
    RAMSES_LLM_MODEL / RAMSES_LLM_BASE_URL, which the run never does, and could
    call a model installed that the run would not ask for.

    Never raises and keeps the 1.5 s budget: health is advisory, and the
    configure page renders whatever comes back.
    """
    model = (model or "").strip() or DEFAULT_LLM_MODEL
    base_url = (base_url or "").strip() or DEFAULT_LLM_BASE_URL
    health: Dict[str, Any] = {
        "checked": True, "reachable": False, "base_url": base_url,
        "model": model, "model_available": None,
        "available_models": [], "n_available_models": 0,
    }

    # A custom server is reported as unchecked rather than probed. This is a
    # GET that any page open in the browser can make to localhost, so fetching
    # whatever URL the query string names would turn it into a request relay.
    # Checking the default server instead, as this did, answered a question
    # about a server the run will not use: "not installed" for a model the
    # custom server has, "unreachable" when only the default one is down.
    # The trailing slash is ignored because LLMClient strips it too.
    if base_url.rstrip("/") != DEFAULT_LLM_BASE_URL.rstrip("/"):
        health.update(checked=False, reachable=None)
        return health

    try:
        import requests
        resp = requests.get(DEFAULT_LLM_BASE_URL.rstrip("/") + "/models", timeout=1.5)
    except Exception:
        resp = None
    health["reachable"] = resp is not None and resp.status_code < 500

    # None means "could not tell", which is not the same claim as "missing":
    # a server that is up but has no /models route, or answers it with
    # something other than the OpenAI list shape, must not be reported as
    # lacking a model it may well serve.
    if health["reachable"] and resp.status_code == 200:
        try:
            listing = resp.json().get("data")
        except Exception:
            listing = None
        if isinstance(listing, list):
            installed = [str(m["id"]) for m in listing
                         if isinstance(m, dict) and m.get("id")]
            health.update(model_available=_model_installed(model, installed),
                          available_models=installed[:HEALTH_MODEL_SAMPLE],
                          n_available_models=len(installed))
    return health


def _json_body() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}


def _validate_run(body: Dict[str, Any]) -> Optional[str]:
    if not str(body.get("dataset") or "").strip():
        return "dataset is required"
    if not str(body.get("entity") or "").strip():
        return "entity is required"
    stages = body.get("stages")
    if stages is not None:
        unknown = {str(s).lower() for s in stages} - set(ALL_STAGES)
        if unknown:
            return f"unknown stage(s): {', '.join(sorted(unknown))}"
        if not stages:
            return "select at least one stage"
    detectors = body.get("detectors")
    if detectors is not None:
        unknown = {str(d) for d in detectors} - set(ALL_DETECTORS)
        if unknown:
            return f"unknown detector(s): {', '.join(sorted(unknown))}"
        if len(set(detectors)) < 2:
            return "select at least two detectors"
    # `anomaly_types` (list) from the multi-select, or the older single
    # `anomaly_type` (a name, "a,b", or a list) — the run injects every one.
    raw_types = body.get("anomaly_types", body.get("anomaly_type"))
    if raw_types is not None:
        parts = raw_types if isinstance(raw_types, (list, tuple)) else str(raw_types).split(",")
        tokens = [str(p).strip().lower() for p in parts if str(p).strip()]
        if not tokens:
            return "select at least one anomaly type"
        unknown = [t for t in tokens if t not in ALL_ANOMALY_TYPES]
        if unknown:
            return f"unknown anomaly type: {', '.join(unknown)}"
    rate = body.get("anomaly_rate")
    if rate is not None:
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            return f"anomaly rate must be a number, got {rate!r}"
        if not 0.0 < rate <= 1.0:
            return f"anomaly rate must be greater than 0 and at most 1, got {rate}"
    metrics = body.get("decision_metrics")
    if metrics is None:
        metrics = body.get("decision_metric")
    if metrics is not None:
        if isinstance(metrics, str):
            metrics = [metrics]
        unknown = [m for m in metrics
                   if str(m).strip().lower().replace("-", "_") not in DECISION_METRICS]
        if unknown:
            return f"unknown decision metric(s): {', '.join(map(str, unknown))}"
        if not metrics:
            return "select at least one decision metric"
        if isinstance(metrics, dict):
            try:
                weights = {m: float(w) for m, w in metrics.items()}
            except (TypeError, ValueError):
                return "decision metric weights must be numbers"
            if any(w < 0 for w in weights.values()):
                return "decision metric weights must not be negative"
            if not any(w > 0 for w in weights.values()):
                return "select at least one decision metric"
    return _validate_online(body)


def _validate_online(body: Dict[str, Any]) -> Optional[str]:
    """The online-phase parameters, checked whether or not the phase is on.

    Values are parsed by the CLI's own parsers, so the UI cannot accept what the
    pipeline would refuse after a minutes-long offline phase. Messages name the
    request field rather than the CLI flag behind it.
    """
    for flag in ("enable_online", "inject_online_regime"):
        value = body.get(flag)
        if value is not None and not isinstance(value, bool):
            return f"{flag} must be true or false, got {value!r}"
    strategy = body.get("online_strategy")
    field = "online_strategy"
    if strategy is None or (isinstance(strategy, str) and not strategy.strip()):
        strategy, field = body.get("strategy"), "strategy"
    if strategy is not None and not (isinstance(strategy, str) and not strategy.strip()):
        if not isinstance(strategy, str):
            return f"{field} must be one of {', '.join(ONLINE_SCENARIOS)}, all"
        try:
            # The random-selection baseline is a command-line tool, not one of
            # the compared scenarios.
            parse_online_strategy(strategy, allow_baselines=False)
        except ValueError as e:
            return f"{field}: {str(e).partition(': ')[2]}"
    for key, _flag, parse in jobs.ONLINE_PARAMS:
        value = body.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            parse(value)
        except ValueError as e:
            return f"{key}: {str(e).partition(': ')[2]}"
    try:
        labels = parse_online_labels(body.get("online_labels"))
    except ValueError as e:
        return f"online_labels: {str(e).partition(': ')[2]}"
    if body.get("enable_online") is True:
        iteration = _iteration(body.get("iteration"))
        if iteration is None:
            return f"iteration must be a whole number, got {body.get('iteration')!r}"
        # An entity too short for one online window — or, with real labels,
        # one whose online part holds no labelled anomaly — would run the
        # whole offline phase only to skip the online one. Refused here with
        # the pipeline's own verdict; an entity whose data cannot be read is
        # left to the run, which reports why.
        verdict = online.feasibility(str(body.get("dataset")).strip(),
                                     str(body.get("entity")).strip(), iteration,
                                     online_labels=labels,
                                     inject_online_regime=body.get("inject_online_regime") is True)
        if verdict is not None and verdict.get("label_error"):
            return f"online_labels: {verdict['label_error']}"
        if verdict is not None and not verdict["ok"]:
            return f"enable_online: {verdict['reason']}"
    return None


def _iteration(value) -> Optional[int]:
    """--iteration as the run would take it (default 5), or None if invalid."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return 5
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    return int(number)


def create_app(**overrides) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(JSON_SORT_KEYS=False, **overrides)
    manager = jobs.manager()

    @app.context_processor
    def _asset_stamp():
        """Cache-bust CSS/JS by their own mtime.

        The server runs with use_reloader=False (the reloader would orphan a
        running pipeline), so an edit to result.js or ramses.css needs a restart
        already — and without a stamp the browser then keeps serving the old
        file anyway, which reads as "the feature I just added is missing".
        """
        static_dir = Path(app.static_folder or "")

        def asset(filename: str) -> str:
            url = url_for("static", filename=filename)
            try:
                return f"{url}?v={int((static_dir / filename).stat().st_mtime)}"
            except OSError:
                return url

        return {"asset": asset}

    # ── Pages ───────────────────────────────────────────────────────────────

    @app.get("/")
    def page_index():
        return render_template("index.html")

    @app.get("/run/<job_id>")
    def page_run(job_id):
        if manager.get(job_id) is None:
            abort(404)
        return render_template("run.html", job_id=job_id)

    @app.get("/result/<dataset>/<entity>")
    def page_result(dataset, entity):
        return render_template("result.html", dataset=dataset, entity=entity,
                               views=result_views(dataset, entity), current="explanation")

    @app.get("/report/<dataset>/<entity>")
    def page_report(dataset, entity):
        """The comprehensive report on its own page.

        Separate from /result on purpose: it is the pipeline's numeric record,
        not narration, and it exists for runs made without --explain, which
        have no explanation page at all.
        """
        return render_template("report.html", dataset=dataset, entity=entity,
                               views=result_views(dataset, entity), current="report")

    @app.get("/online/<dataset>/<entity>")
    def page_online(dataset, entity):
        """The online phase: each scenario's timeline and their comparison.

        Its own page for the same reason as the report: a run without
        --explain has online results but no explanation page. The page reads
        /api/online/<dataset>/<entity> and shows its own empty state, so an
        entity with no online data yet is not a 404 — a run may be writing it.
        """
        return render_template("online.html", dataset=dataset, entity=entity,
                               views=result_views(dataset, entity), current="online")

    @app.get("/docs/<dataset>/<entity>")
    def page_docs(dataset, entity):
        """Stage documentation, one section per pipeline stage.

        Its own page because it is reference material: the glossaries were long
        enough that every card opened with a wall of definitions before its
        finding. The stage cards link here with a fragment, so a reader arrives
        at the section for the stage they were reading.
        """
        return render_template("docs.html", dataset=dataset, entity=entity,
                               views=result_views(dataset, entity), current="docs")

    # ── Catalog and health ──────────────────────────────────────────────────

    def owns(dataset: str, entity: str) -> bool:
        """Does a live job of this web UI write these online results now?"""
        job = manager.active()
        if job is None or job.is_done():
            return False
        params = job.params or {}
        return (str(params.get("dataset") or "").lower() == str(dataset).lower()
                and str(params.get("entity") or "").lower() == str(entity).lower())

    @app.get("/api/catalog")
    def api_catalog():
        data = catalog.catalog(refresh=request.args.get("refresh") == "1")
        data = dict(data)
        results = []
        for summary in (artifacts.entity_summary(ds, ent)
                        for ds, ent in artifacts.known_entities()):
            if not summary:
                continue
            # Each explained result says whether online-phase data exists for
            # it, and where; `online_results` below also lists entities that
            # have online data but no explanations (a run without --explain).
            info = online.info(summary["dataset"], summary["entity"], is_live=owns)
            results.append({**summary, "has_online": info is not None, "online": info})
        data["results"] = results
        data["online_results"] = [
            i for i in (online.info(ds, ent, is_live=owns)
                        for ds, ent in online.known_entities()) if i]
        return jsonify(data)

    @app.get("/api/views/<dataset>/<entity>")
    def api_views(dataset, entity):
        return jsonify(result_views(dataset, entity))

    @app.get("/api/detectors/<dataset>/<entity>")
    def api_detectors(dataset, entity):
        """Detector availability is per entity, so it needs its own lookup —
        the catalog cannot precompute it for every entity in every dataset."""
        return jsonify({"dataset": dataset, "entity": entity,
                        "detectors": catalog.detectors_for(dataset, entity)})

    @app.get("/api/health")
    def api_health():
        active = manager.active()
        # `?model=` and `?base_url=` are the run form's Advanced overrides.
        # The model is only compared against the default server's list and the
        # URL only against the default; neither is ever fetched or sent
        # anywhere, so they need no validation beyond what _health does.
        return jsonify({"llm": _health(request.args.get("model"),
                                       request.args.get("base_url")),
                        "active_job": active.id if active else None})

    # ── Runs ────────────────────────────────────────────────────────────────

    @app.post("/api/runs")
    def api_start_run():
        body = _json_body()
        error = _validate_run(body)
        if error:
            return jsonify({"error": error}), 400
        try:
            argv = jobs.build_argv(body)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # The command preview and the spawned command come from one function,
        # so what the user copies is exactly what runs.
        if body.get("dry_run"):
            return jsonify({"argv": argv, "command": " ".join(argv)})

        try:
            job = manager.start(body, argv=argv,
                                timeout=int(body.get("timeout") or jobs.DEFAULT_TIMEOUT))
        except RuntimeError as active_id:
            return jsonify({"error": "A run is already in progress.",
                            "active_job_id": str(active_id)}), 409
        return jsonify({"job_id": job.id, "argv": job.argv,
                        "url": f"/run/{job.id}"}), 201

    @app.get("/api/runs")
    def api_list_runs():
        return jsonify({"jobs": manager.recent()})

    @app.get("/api/runs/<job_id>")
    def api_get_run(job_id):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job.snapshot())

    @app.get("/api/runs/<job_id>/log")
    def api_run_log(job_id):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        if request.args.get("download") == "1" and job.log_path and job.log_path.exists():
            return send_file(job.log_path, mimetype="text/plain",
                             as_attachment=True, download_name=f"{job_id}.log")
        offset = max(0, int(request.args.get("offset") or 0))
        return jsonify(job.tail(offset))

    @app.post("/api/runs/<job_id>/cancel")
    def api_cancel_run(job_id):
        if manager.get(job_id) is None:
            abort(404)
        return jsonify({"cancelled": manager.cancel(job_id)}), 202

    @app.get("/api/runs/<job_id>/events")
    def api_run_events(job_id):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        # EventSource reconnects on its own and replays from Last-Event-ID, so
        # a dropped connection resumes rather than losing the run's history.
        start = request.headers.get("Last-Event-ID") or request.args.get("cursor") or 0
        try:
            start = max(0, int(start))
        except (TypeError, ValueError):
            start = 0
        return Response(_event_stream(job, start), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no",
                                 "Connection": "keep-alive"})

    # ── Explanations ────────────────────────────────────────────────────────

    @app.get("/api/explanations/<dataset>/<entity>")
    def api_explanations(dataset, entity):
        payload = artifacts.build_payload(dataset, entity)
        if payload is None:
            return jsonify({"error": "no_artifacts",
                            "hint": "Run this dataset/entity with explanations enabled."}), 404
        # Titles and captions only: `src` is a filename and the pair picker's
        # detector list is a query parameter, both of which must stay canonical.
        payload["plots"] = plots.manifest(dataset, entity)
        # Each regime-bearing stage names its own plot subdirectory in STAGES,
        # so both Thompson stages get their regimes paired without either one
        # being named here.
        for stage in payload["stages"]:
            stems = (artifacts.STAGE_BY_KEY.get(stage["key"]) or {}).get("regimes")
            if not stems:
                continue
            # Several stems mean the card offers a toggle; the first is default.
            variants = plots.regime_plot_variants(dataset, entity, list(stems))
            for regime in stage.get("regimes", []):
                figures = variants.get(regime["index"])
                if figures:
                    # `plots` carries every set for the toggle; `plot` and
                    # `plot_caption` stay as the default one so a client that
                    # ignores the toggle still renders correctly. The captions
                    # travel with the images: these sets show different
                    # quantities over the same window range, and only plots.py
                    # knows which is which.
                    regime["plots"] = figures
                    regime["plot"] = figures[0]["src"]
                    regime["plot_caption"] = figures[0].get("caption")
        return jsonify(payload)

    @app.get("/api/docs/<dataset>/<entity>")
    def api_docs(dataset, entity):
        payload = artifacts.documentation(dataset, entity)
        if payload is None:
            return jsonify({"error": "no_artifacts",
                            "hint": "Run this dataset/entity with explanations enabled."}), 404
        return jsonify(payload)

    @app.get("/api/explanations/<dataset>/<entity>/download")
    def api_download(dataset, entity):
        nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
        if nl_dir is None:
            abort(404)
        stage = request.args.get("stage") or "global"
        if stage == "global":
            target = artifacts.global_text_path(dataset, entity)
        else:
            meta = artifacts.STAGE_BY_KEY.get(stage)
            if meta is None:
                abort(404)
            exact = nl_dir / f"{meta['nl']}.txt"
            target = exact if exact.exists() else artifacts._newest(nl_dir, f"{meta['nl']}*.txt")
        if target is None or not Path(target).is_file():
            abort(404)
        return send_file(target, mimetype="text/plain",
                         as_attachment=True, download_name=Path(target).name)

    @app.get("/api/comprehensive/<dataset>/<entity>")
    def api_comprehensive(dataset, entity):
        if request.args.get("download") == "1":
            target = artifacts.comprehensive_path(dataset, entity)
            if target is None:
                abort(404)
            return send_file(target, mimetype="text/plain",
                             as_attachment=True, download_name=target.name)
        report = artifacts.comprehensive_report(dataset, entity)
        if report is None:
            return jsonify({
                "error": "no_report",
                "hint": "The comprehensive report is written only by a full run; "
                        "partial runs skip it.",
            }), 404
        return jsonify(report)

    # ── Online phase ────────────────────────────────────────────────────────

    @app.get("/api/online/<dataset>/<entity>")
    def api_online(dataset, entity):
        """The online phase's summary and per-scenario window series.

        Read straight from results/online/<dataset>/<entity>/ (schema in
        Model_Selection/online_phase.py), which the pipeline rewrites after
        every window, so polling this during a run shows it progressing.
        `?scenario=<key>` narrows the series to one scenario and
        `?summary=1` leaves them out.
        """
        payload = online.payload(
            dataset, entity, scenario=(request.args.get("scenario") or None),
            summary_only=request.args.get("summary") == "1", is_live=owns)
        if payload is None:
            return jsonify({
                "error": "no_online_results",
                "hint": "Run this dataset/entity with the online phase enabled.",
            }), 404
        # No-store: the files change after every window while a run is live.
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/online/<dataset>/<entity>/stream")
    def api_online_stream(dataset, entity):
        """The live monitor's data: a decimated slice of the replayed series,
        its real and injected anomalies as exact spans, and one scenario's
        scores, threshold and flagged samples for one window.

        Query: channels=0,1 | all, from, to, max_points (default 2000),
        copy, series=0 (leave the series out), scenario, window (default the
        scenario's latest written window). Read from the stream files the
        pipeline writes beside summary.json (Model_Selection/online_stream.py);
        404 until there are any, never cached since they grow while a run is
        live.
        """
        scenario = request.args.get("scenario") or None
        if scenario is not None and scenario not in ONLINE_SCENARIOS + ONLINE_BASELINE_SCENARIOS:
            return jsonify({"error": "unknown_scenario",
                            "hint": f"scenario must be one of {', '.join(ONLINE_SCENARIOS)}"}), 400
        try:
            body = stream.view(online.online_dir(dataset, entity), request.args)
        except FileNotFoundError:
            return jsonify({"error": "no_stream",
                            "hint": "This run wrote no live-monitor stream (it predates the "
                                    "monitor, or its online phase has not started)."}), 404
        except ValueError as e:
            return jsonify({"error": "bad_request", "hint": str(e)}), 400
        response = jsonify(body)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/online/feasibility/<dataset>/<entity>")
    def api_online_feasibility(dataset, entity):
        """Can this entity run the online phase, and if not, why not?

        `?iteration=I` is the run's --iteration (default 5), which sizes the
        windows; `?online_labels=real|injected` (default: real where the data
        has labels) and `?inject_online_regime=1` are the run's label options.
        In real-label mode the answer counts the online windows that hold a
        labelled anomaly, and is not ok when there is nothing to score. The answer comes from the same functions run_app uses to cut
        the windows and to skip the phase, on the series length it would load,
        so the run form can disable the online option with the pipeline's own
        reason before anything starts.
        """
        iteration = _iteration(request.args.get("iteration"))
        if iteration is None:
            return jsonify({"error": "bad_iteration",
                            "hint": "iteration must be a whole number."}), 400
        try:
            labels = parse_online_labels(request.args.get("online_labels"))
        except ValueError:
            return jsonify({"error": "bad_online_labels",
                            "hint": "online_labels must be real, injected or auto."}), 400
        regime = str(request.args.get("inject_online_regime") or "").lower() in ("1", "true")
        verdict = online.feasibility(dataset, entity, iteration, online_labels=labels,
                                     inject_online_regime=regime)
        if verdict is None:
            return jsonify({"error": "unknown_entity",
                            "hint": "This entity's test series could not be read "
                                    "with the configured dataset path."}), 404
        return jsonify(verdict)

    @app.post("/api/explanations/<dataset>/<entity>/narrate")
    def api_narrate(dataset, entity):
        """Regenerate narratives from existing IR.

        Two uses: the LLM server was down during the run, and partial runs —
        which never reach the pipeline's narration step at all.
        """
        body = _json_body()
        argv = jobs.build_narrate_argv(dataset, entity, body.get("llm_model"),
                                       body.get("llm_base_url"))
        try:
            job = manager.start({"dataset": dataset, "entity": entity,
                                 "explain": True, "narrate_only": True},
                                argv=argv, timeout=1800)
        except RuntimeError as active_id:
            return jsonify({"error": "A run is already in progress.",
                            "active_job_id": str(active_id)}), 409
        return jsonify({"job_id": job.id, "url": f"/run/{job.id}"}), 201

    # ── Plots and media ─────────────────────────────────────────────────────

    @app.get("/api/plots/<dataset>/<entity>")
    def api_plots(dataset, entity):
        return jsonify(plots.manifest(dataset, entity))

    @app.get("/api/plots/<dataset>/<entity>/gallery/<path:gallery_id>")
    def api_gallery(dataset, entity, gallery_id):
        offset = max(0, int(request.args.get("offset") or 0))
        limit = max(1, min(int(request.args.get("limit") or 60), 200))
        return jsonify(plots.gallery_page(dataset, entity, gallery_id, offset, limit))

    @app.get("/api/plots/<dataset>/<entity>/ranking-gap")
    def api_ranking_gap(dataset, entity):
        """The ranking gap for any detector pair, drawn per request.

        Eleven detectors is 55 unordered pairs per entity and a reader looks at
        one or two, so these are rendered rather than written by the pipeline.
        Everything it needs is the IR's `context_feature_shares` block; nothing is
        written to results/.
        """
        a = (request.args.get("a") or "").strip()
        b = (request.args.get("b") or "").strip()
        if not a or not b:
            abort(400)
        png = ondemand.render_ranking_gap(dataset, entity, a, b)
        if png is None:
            abort(404)
        # No-store: the pair is a query parameter, and a stale cached body would
        # outlive the run that produced the numbers in it.
        return Response(png, mimetype="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/plots/<dataset>/<entity>/per-window")
    def api_per_window(dataset, entity):
        """One per-window frame of a Thompson set, drawn per request.

        These were nine folders of PNGs — over a thousand frames per entity, of
        which a reader opens a handful — so the pipeline persists the per-context-feature
        numbers and the frame is rendered here. `scope` and the gallery's stride
        are arguments rather than separate sets.
        """
        kind = (request.args.get("kind") or "").strip()
        scope = (request.args.get("scope") or "top").strip()
        try:
            t = int(request.args.get("t") or 0)
        except ValueError:
            abort(400)
        if not kind or scope not in ("top", "all") or t < 0:
            abort(400)
        png = ondemand.render_per_window(dataset, entity, kind, t, scope)
        if png is None:
            abort(404)
        return Response(png, mimetype="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/media/<path:relpath>")
    def media(relpath):
        target = plots.safe_media_path(relpath)
        if target is None:
            abort(404)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        # conditional=True gives ETag/Range handling, which the 173-image
        # galleries rely on for caching.
        return send_file(target, mimetype=mime, conditional=True)

    return app


def result_views(dataset: str, entity: str) -> Dict[str, bool]:
    """Which views of one result exist, so every page shows the same tabs.

    The result, report, docs and online pages each used to decide their own
    header tabs, and disagreed: a run without --explain showed "Explanation"
    and "Docs" tabs on its report page that led to empty pages, while its
    online page hid them. The pages now render the tabs from this, server-side
    (templates/_view_tabs.html), with the criteria the catalog uses, so the
    tabs and the results list cannot disagree either. The documentation
    describes the explanation's stages, so it exists exactly when the
    explanation does.
    """
    explained = artifacts.entity_summary(dataset, entity) is not None
    return {
        "explanation": explained,
        "report": artifacts.comprehensive_path(dataset, entity) is not None,
        "online": online.load_summary(dataset, entity) is not None,
        "docs": explained,
    }


def _sse(event: str, data: Any, event_id: Optional[int] = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data)}\n\n"


def _event_stream(job: "jobs.Job", start_cursor: int) -> Iterator[str]:
    """Log lines, stage transitions and the final status, as SSE frames."""
    yield _sse("hello", {**job.snapshot(), "cursor_from": start_cursor})

    cursor = start_cursor
    last_stages: Dict[str, Any] = {}
    last_phase = None
    last_warnings = 0
    last_beat = time.time()

    while True:
        with job.condition:
            job.condition.wait(timeout=1.0)
            snapshot = {
                "cursor": job.cursor,
                "stages": {k: dict(v) for k, v in job.stages.items()},
                "phase": dict(job.phase) if job.phase else None,
                "warnings": list(job.warnings),
                "status": job.status,
                "done": job.is_done(),
            }

        if snapshot["cursor"] > cursor:
            batch = job.tail(cursor)
            lines = batch["lines"]
            for i in range(0, len(lines), SSE_LOG_BATCH):
                chunk = lines[i:i + SSE_LOG_BATCH]
                cursor = batch["from"] + i + len(chunk)
                yield _sse("log", {"from": batch["from"] + i, "lines": chunk}, cursor)
            cursor = batch["cursor"]
            last_beat = time.time()

        if snapshot["phase"] != last_phase:
            last_phase = snapshot["phase"]
            if last_phase:
                yield _sse("phase", last_phase)
                last_beat = time.time()

        for key, entry in snapshot["stages"].items():
            if last_stages.get(key) != entry:
                last_stages[key] = dict(entry)
                yield _sse("stage", {"key": key, **entry})
                last_beat = time.time()

        if len(snapshot["warnings"]) > last_warnings:
            for warning in snapshot["warnings"][last_warnings:]:
                yield _sse("warning", warning)
            last_warnings = len(snapshot["warnings"])
            last_beat = time.time()

        if snapshot["done"]:
            yield _sse("status", job.snapshot())
            return

        if time.time() - last_beat > SSE_KEEPALIVE_SECONDS:
            yield ": keepalive\n\n"
            last_beat = time.time()


def serve(host: str = "127.0.0.1", port: int = 5000) -> None:
    from werkzeug.serving import run_simple
    app = create_app()
    # Templates recompile when the file changes, so markup edits need only a
    # browser reload. Unlike the reloader below, this restarts nothing.
    app.jinja_env.auto_reload = True
    print(f"RAMSeS Web UI → http://{host}:{port}")
    # threaded: each SSE connection holds a thread. No reloader: it would
    # restart the process and orphan a running pipeline. No debug: the Werkzeug
    # debugger is remote code execution for anything that reaches the port.
    run_simple(host, port, app, threaded=True, use_reloader=False, use_debugger=False)
