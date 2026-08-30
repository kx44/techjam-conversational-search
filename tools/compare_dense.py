"""A/B dense retrieval against any harness, including one you wrote yourself.

Runs the same agent twice - dense retrieval on, then off - and reports both.
Everything else is held constant, so the difference is attributable.

    python3 tools/compare_dense.py                      # our two harnesses
    python3 tools/compare_dense.py --harness mysim      # yours as well
    python3 tools/compare_dense.py --weights 0,.25,.5,1 # sweep its fusion weight

Your harness needs one callable. Put a module on the path exposing:

    run(agent, samples, products, catalog_ids) -> dict with a "score" key
        agent        an Agent; call agent.reset(session_id, profile) per session
                     then agent.respond(session_id, message, turn, 10)
        samples      list of dicts with "sample_id", "scenario_type",
                     "ground_truth": {"parent_asin": ...}, "user_profile"
        products     {parent_asin: product dict}
        catalog_ids  set of valid parent_asin
    Optionally also return "hit_rate_at_10" and "mrr" and they will be shown.

`tools/realistic_sim.py` is a worked example of exactly that contract.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def fresh(base, dense: bool):
    """A clean agent sharing the built index - avoids rebuilding for each run."""
    import starter.agent as A

    A.USE_DENSE = dense
    agent = A.Agent.__new__(A.Agent)
    agent.__dict__.update(base.__dict__)
    agent._cache, agent._sessions = {}, {}
    agent._vectors = dict(base._vectors)
    agent._intents = dict(base._intents)
    if not dense:
        agent._index = None
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B dense retrieval")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--harness", action="append", default=[],
                        help="importable module exposing run(agent, samples, products, ids)")
    parser.add_argument("--weights", default="",
                        help="comma-separated DENSE_WEIGHT values to sweep, e.g. 0,.25,.5,1")
    args = parser.parse_args()

    import starter.agent as A
    from evaluator import local_evaluator as E

    samples = E.load_jsonl(args.dataset)
    catalog_ids, categories, products = E.catalog_index(args.catalog)

    A.USE_DENSE = True
    started = time.time()
    base = A.Agent(args.catalog)
    print(f"index built in {time.time() - started:.0f}s  |  "
          f"dense attached: {base._index is not None}  |  "
          f"intent detector: {base._intent is not None}")
    if base._index is None:
        print("\n  dense retrieval is NOT attached. Build the matrix first:\n"
              "    python3 tools/build_embeddings.py --catalog data/catalog.jsonl\n"
              "  Without it both columns below would be the same agent.\n")

    harnesses = [("reference evaluator",
                  lambda a: E.evaluate(a, samples, catalog_ids, categories, products))]
    for name in args.harness:
        module = importlib.import_module(name)
        harnesses.append((name, lambda a, m=module: m.run(a, samples, products, catalog_ids)))

    def score(result):
        return result.get("score", result.get("recommended_technical_score"))

    if args.weights:
        values = [float(w) for w in args.weights.split(",")]
        print(f"\n{'dense weight':<16}" + "".join(f"{n[:20]:>22}" for n, _ in harnesses))
        for weight in values:
            A.DENSE_WEIGHT = weight
            cells = "".join(f"{score(fn(fresh(base, weight > 0))):>22.4f}" for _, fn in harnesses)
            print(f"{weight:<16.2f}{cells}", flush=True)
        return

    print(f"\n{'harness':<30}{'dense on':>11}{'dense off':>12}{'delta':>10}")
    for name, fn in harnesses:
        on, off = score(fn(fresh(base, True))), score(fn(fresh(base, False)))
        print(f"{name:<30}{on:>11.4f}{off:>12.4f}{on - off:>+10.4f}", flush=True)
    print(f"\ndense weight {A.DENSE_WEIGHT}, depth {A.DENSE_LIMIT}, "
          f"fused into rank fusion against two BM25 retrievers at weight 1.0 each.")


if __name__ == "__main__":
    main()
