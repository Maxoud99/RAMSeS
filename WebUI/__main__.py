"""`python -m WebUI` — start the local web UI."""

import argparse

from WebUI.server import serve


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m WebUI",
        description="Local web UI for configuring RAMSeS runs and reading their explanations.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Localhost by default. There is no authentication and the app "
                             "spawns processes with user-supplied arguments, so binding to a "
                             "public interface is not supported.")
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
