"""
Compare two speculative-decoding evals on the same prompts (DR-0): paired bootstrap of tokens/step.

    python scripts/compare_spec.py results/Rsd_hi_k4_frozen_sd_seq/eval_indiccorp_eval.json:fixed_k \\
        results/Rsd_hi_k4_frozen_sd_seq/eval_indiccorp_eval.json:tree_static_n40

Each side is `path[:policy]` (default policy fixed_k). Both sides must hold per-prompt `tokens` / `steps` for the
same prompts (evaluate.py with the same dataset and spec_n: make_prompts is seeded). Prints, for B - A, the
difference of accepted length (sum tokens / sum steps) with a 95% CI (2,000 resamples over prompts), both
measured speed-ups, and, if the folder has a bench_verify.json, the cost-model speed-up
tokens/step / ((verify_ms(k positions) + draft_ms) / greedy_ms) for chain policies.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def split_spec(spec):
    """'path[:policy]' -> (path, policy); a Windows drive letter ('D:/...') is not a policy separator."""
    head, sep, tail = spec.rpartition(":")
    if sep and tail and not any(c in tail for c in "/\\") and not tail.endswith(".json"):
        return head, tail
    return spec, "fixed_k"


def load_side(spec):
    path, policy = split_spec(spec)
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = next((e for e in rec.get("spec_decode") or [] if e["policy"] == policy), None)
    if entry is None:
        raise SystemExit(f"{path}: no spec_decode entry for policy {policy!r}")
    pp = entry.get("per_prompt") or {}
    if "tokens" not in pp:
        raise SystemExit(f"{path}:{policy}: no per-prompt tokens/steps (re-run evaluate.py with this version)")
    return Path(path), policy, entry, pp


def accepted(tokens, steps, idx=None):
    idx = range(len(tokens)) if idx is None else idx
    return sum(tokens[i] for i in idx) / sum(steps[i] for i in idx)


def paired_bootstrap(a, b, n=2000, seed=0):
    """(diff, lo, hi) of accepted length B - A; a, b = (tokens, steps) per prompt, same prompts."""
    (ta, sa), (tb, sb) = a, b
    if len(ta) != len(tb):
        raise ValueError(f"different prompt counts: {len(ta)} vs {len(tb)}")
    rng, m = random.Random(seed), len(ta)
    d0 = accepted(tb, sb) - accepted(ta, sa)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        ds.append(accepted(tb, sb, idx) - accepted(ta, sa, idx))
    ds.sort()
    return d0, ds[int(0.025 * n)], ds[int(0.975 * n)]


def cost_model_speedup(entry, folder):
    """tokens/step / ((verify_ms at k positions + draft_ms) / greedy_ms), chain policies only; None otherwise."""
    bench_path = folder / "bench_verify.json"
    if not bench_path.exists() or entry["policy"].startswith("tree"):
        return None
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    k = len(entry["accept_rate_per_head"]) + 1
    passes = {p["positions"]: p["ms"] for p in bench["passes"]}
    verify = passes.get(k) or min(passes.items(), key=lambda kv: abs(kv[0] - k))[1]
    return entry["mean_accepted_len"] / ((verify + (bench.get("draft_ms") or 0.0)) / bench["greedy_ms"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference: path[:policy]")
    ap.add_argument("b", help="candidate: path[:policy]")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args(argv)
    (pa, pol_a, ea, ppa), (pb, pol_b, eb, ppb) = load_side(args.a), load_side(args.b)
    d, lo, hi = paired_bootstrap((ppa["tokens"], ppa["steps"]), (ppb["tokens"], ppb["steps"]), n=args.n)
    rows = [("A", pa, pol_a, ea), ("B", pb, pol_b, eb)]
    for tag, path, pol, e in rows:
        cm = cost_model_speedup(e, path.parent)
        print(f"{tag}: {path.parent.name}/{path.name}:{pol}  tokens/step {e['mean_accepted_len']:.3f}  "
              f"speed-up x{e['speedup']:.3f}" + (f"  (cost model x{cm:.3f})" if cm else ""))
    print(f"B - A tokens/step: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  (paired bootstrap, {args.n} resamples, "
          f"{len(ppa['tokens'])} prompts){'  *' if lo > 0 or hi < 0 else ''}")
    print(f"B - A speed-up: {eb['speedup'] - ea['speedup']:+.3f}")
    return d, lo, hi


if __name__ == "__main__":
    sys.exit(main() and 0)
