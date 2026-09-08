"""
Flask app: pages, JSON API, SSE progress stream, and image serving.

Thin by design — routes validate input and delegate. Bound to localhost only:
there is no authentication and the app spawns processes with user-supplied
arguments, so it must never be reachable from another machine.
"""

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_file, url_for)

from Utils.pipeline_spec import (ALL_ANOMALY_TYPES, ALL_DETECTORS, ALL_STAGES,
                                 DECISION_METRICS, DEFAULT_LLM_BASE_URL,
                                 DEFAULT_LLM_MODEL)
from WebUI import artifacts, catalog, jobs, ondemand, paths, plots

SSE_KEEPALIVE_SECONDS = 15
SSE_LOG_BATCH = 40           # thousands of lines: one frame per line floods the tab
SSE_MAX_SUBSCRIBERS = 8


def _health() -> Dict[str, Any]:
    """Is the LLM server up? Checked before a run, not after a 4-minute wait."""
    base_url = os.environ.get("RAMSES_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    model = os.environ.get("RAMSES_LLM_MODEL", DEFAULT_LLM_MODEL)
    reachable = False
    try:
        import requests
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=1.5)
        reachable = resp.status_code < 500
    except Exception:
        reachable = False
    return {"reachable": reachable, "base_url": base_url, "model": model}


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
    anomaly_type = body.get("anomaly_type")
    if anomaly_type is not None and str(anomaly_type).lower() not in ALL_ANOMALY_TYPES:
        return f"unknown anomaly type: {anomaly_type}"
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
    return None


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
        return render_template("result.html", dataset=dataset, entity=entity)

    @app.get("/report/<dataset>/<entity>")
    def page_report(dataset, entity):
        """The comprehensive report on its own page.

        Separate from /result on purpose: it is the pipeline's numeric record,
        not narration, and it exists for runs made without --explain, which
        have no explanation page at all.
        """
        return render_template("report.html", dataset=dataset, entity=entity)

    @app.get("/docs/<dataset>/<entity>")
    def page_docs(dataset, entity):
        """Stage documentation, one section per pipeline stage.

        Its own page because it is reference material: the glossaries were long
        enough that every card opened with a wall of definitions before its
        finding. The stage cards link here with a fragment, so a reader arrives
        at the section for the stage they were reading.
        """
        return render_template("docs.html", dataset=dataset, entity=entity)

    # ── Catalog and health ──────────────────────────────────────────────────

    @app.get("/api/catalog")
    def api_catalog():
        data = catalog.catalog(refresh=request.args.get("refresh") == "1")
        data = dict(data)
        data["results"] = [
            s for s in (artifacts.entity_summary(ds, ent)
                        for ds, ent in artifacts.known_entities()) if s]
        return jsonify(data)

    @app.get("/api/detectors/<dataset>/<entity>")
    def api_detectors(dataset, entity):
        """Detector availability is per entity, so it needs its own lookup —
        the catalog cannot precompute it for every entity in every dataset."""
        return jsonify({"dataset": dataset, "entity": entity,
                        "detectors": catalog.detectors_for(dataset, entity)})

    @app.get("/api/health")
    def api_health():
        active = manager.active()
        return jsonify({"llm": _health(),
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
        written to myresults/.
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
