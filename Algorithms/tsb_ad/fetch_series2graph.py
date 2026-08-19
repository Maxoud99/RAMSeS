"""Fetch `Series2Graph.py` from the TSB-AD wheel into the vendored tree.

    python -m Algorithms.tsb_ad.fetch_series2graph

Everything else under `Algorithms/tsb_ad` is committed. This one file is not,
because it is not Apache-2.0 like the rest of that wheel: it is patent-pending
and licensed for research use only, so redistributing it inside an Apache-2.0
repository would grant rights we do not hold. `Algorithms/series2graph_detector`
carries the full reasoning.

Fetching leaves the header byte-identical, which matters: it is the authors'
statement of their own terms, and stripping it would turn a licensing question
into a misrepresentation. This script never rewrites the file it downloads.

Nothing else in the pipeline calls this. It is a setup step, run once, and the
Series2Graph family fails with a message naming this command until it has
been.
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

PACKAGE = "TSB-AD"
VERSION = "1.5"                       # the release Algorithms/tsb_ad vendors
MEMBER = "TSB_AD/models/Series2Graph.py"
TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "models", "Series2Graph.py")


def _wheel_url(version: str) -> str:
    """The wheel's URL from the PyPI JSON API, rather than a hardcoded hash."""
    api = f"https://pypi.org/pypi/{PACKAGE}/{version}/json"
    with urllib.request.urlopen(api, timeout=60) as response:
        meta = json.load(response)
    for entry in meta["urls"]:
        if entry["packagetype"] == "bdist_wheel":
            return entry["url"]
    raise RuntimeError(f"no wheel published for {PACKAGE} {version}")


def fetch(version: str = VERSION, target: str = TARGET, force: bool = False) -> str:
    if os.path.exists(target) and not force:
        print(f"already present: {target}\n(pass --force to re-download)")
        return target
    url = _wheel_url(version)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as response:
        blob = response.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as wheel:
        source = wheel.read(MEMBER)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(source)

    head = source.decode("utf8", "replace").splitlines()[:12]
    print(f"\nwrote {target} ({len(source)} bytes)\n")
    print("The authors' own terms, verbatim from the file you just fetched:\n")
    for line in head:
        if line.startswith("#"):
            print(f"    {line}")
    print("\nThis file is gitignored. It is yours to use under those terms; do "
          "not commit it.")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=VERSION,
                        help=f"TSB-AD release to take it from (default {VERSION})")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already there")
    args = parser.parse_args(argv)
    try:
        fetch(args.version, force=args.force)
    except Exception as exc:
        print(f"fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
