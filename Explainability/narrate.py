"""
CLI for the LLM narration layer.

Reads the Intermediate Representation JSONs an `--explain` pipeline run wrote
under myresults/explanations_ir/{dataset}/{entity}/, generates natural-language
narratives with a local open-weights LLM, verifies each narrative against its
IR (atom-matching faithfulness), and writes everything under
myresults/explanations_nl/{dataset}/{entity}/.

Usage:
    python -m Explainability.narrate --dataset SMD --entity machine-1-6 --iteration 5
    python -m Explainability.narrate ... --model llama3.1:8b --base-url http://localhost:1234/v1
    python -m Explainability.narrate ... --stages monte_carlo,off_by_threshold,global
"""

from __future__ import annotations

import argparse
import sys

try:
    from Explainability.llm import (DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient,
                                    _stage_file_map, narrate_entity,
                                    GLOBAL_MODES)
except ImportError:  # executed as a loose script from inside the directory
    from llm import (DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient,  # type: ignore
                     _stage_file_map, narrate_entity,  # type: ignore
                     GLOBAL_MODES)

_STAGE_TOKENS = sorted(set(_stage_file_map(0)) | {"global"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m Explainability.narrate",
        description="Generate + verify natural-language explanations from the IR files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base-dir", default="myresults/explanations_ir")
    parser.add_argument("--out-dir", default="myresults/explanations_nl")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated subset of: " + ", ".join(_STAGE_TOKENS))
    parser.add_argument("--global-mode", default="concat", choices=list(GLOBAL_MODES),
                        help="How to build the global document: 'concat' merges the "
                             "per-stage narratives deterministically (default); 'llm' "
                             "narrates the global IR's own atoms and verifies it.")
    args = parser.parse_args(argv)

    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in stages if s not in _STAGE_TOKENS]
        if unknown:
            parser.error(f"unknown stage token(s) {unknown}; "
                         f"valid: {', '.join(_STAGE_TOKENS)}")

    client = LLMClient(base_url=args.base_url, model=args.model,
                       timeout=args.timeout)
    try:
        report = narrate_entity(args.dataset, args.entity, args.iteration, client,
                                base_dir=args.base_dir, out_dir=args.out_dir,
                                stages=stages, global_mode=args.global_mode)
    except ConnectionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"{'stage':<26} {'status':>8} {'words':>6} {'halluc.':>8} "
          f"{'omiss.':>7} {'warn':>5} {'rep':>4}")
    print("-" * 71)
    for stage_key in sorted(report["stages"]):
        info = report["stages"][stage_key]
        v = info.get("verify") or {}
        halluc = f"{v['hallucination_rate']:.3f}" if v else "-"
        omiss = f"{v['omission_rate']:.3f}" if v else "-"
        warn = str(len(v.get("attribution_warnings", []))) if v else "-"
        rep = "yes" if info.get("repaired") else "-"
        print(f"{stage_key:<26} {info['status']:>8} "
              f"{str(info.get('words', '-')):>6} {halluc:>8} {omiss:>7} "
              f"{warn:>5} {rep:>4}")
    ov = report["overall"]
    print("-" * 71)
    print(f"overall: hallucination {ov['hallucination_rate']:.3f} "
          f"({ov['n_claims']} claims) | omission {ov['omission_rate']:.3f} "
          f"({ov['n_required']} required)")
    print(f"report: {report['faithfulness_txt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
