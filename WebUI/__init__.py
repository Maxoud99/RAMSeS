"""
Local web UI for RAMSeS: configure a run, watch it live, read the explanation.

Deliberately dependency-light — Flask and the standard library only. It never
imports `app.py`, `Explainability.*`, torch or matplotlib: the pipeline is
driven as a subprocess (see WebUI.jobs) and its artifacts are read off disk
(see WebUI.artifacts), so the server starts instantly and never holds the
pipeline's multi-gigabyte working set.

Run with:  python -m WebUI
"""

__all__ = ["create_app", "serve"]


def create_app(**overrides):
    """Lazy re-export so `import WebUI` stays cheap."""
    from WebUI.server import create_app as _create_app
    return _create_app(**overrides)


def serve(**kwargs):
    from WebUI.server import serve as _serve
    return _serve(**kwargs)
