"""
Fit tree draft policies for a run (JN-9) and score them offline.

python scripts/fit_tree.py --run_dir runs/Rsd_hi_k4_frozen_sd [--nodes 3 7 15 25] [--n_calib 300] [--n_eval 200]
python scripts/fit_tree.py --run_dir ... --nodes 64 --sample 0.7,0.9 --compare results/<run>/tree_policy_static_n64.json

Calibration: greedy continuations of IndicCorp *train* prompts (rows >= 1000, so never eval).
Scoring: the same on IndicCorp eval prompts (make_prompts seed 1). For each node budget it fits
StaticTree and EntropyTree and reports the oracle accepted drafts/step on the eval prompts, next
to the chain (FixedK) and the (3,2,1)-style product tree. Writes
results/{run_name}/tree_policy_{policy}_n{N}.json (load with tree_policy.load_policy) and
results/{run_name}/tree_fit.json (the scores).

--sample T,top_p (sampling-aware trees): calibration and scoring text is SAMPLED from head 0 instead of
greedy, so the tree holds the rank paths that sampled decoding (evaluate.py --tree_sample) follows. Files
get the suffix _fitT{T}p{top_p} (tree_policy_static_n64_fitT0.7p0.9.json, tree_fit_fitT0.7p0.9.json).
The scores then also hold, per budget, `ceiling`: the same fit with a drafter that equals the verifier
(head 0's own ranks given the true prefix), i.e. what is lost to sampling itself and not to the heads;
and --compare trees (e.g. the greedy-fitted one) scored on the same sampled text.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtp.data.corpus import load_split
from mtp.device import autocast_ctx
from mtp.eval.spec_decode import make_prompts
from mtp.eval.tree_policy import (StaticTree, accepted_length, collect_ranks, fit_entropy, fit_static, load_policy,
                                  save_policy, widths_tree)
from mtp.model.checkpoint import load_run

PRODUCT = {3: (1, 1, 1), 7: (2, 2, 1), 15: (3, 2, 1), 25: (5, 2, 1)}


def score(policy_tree, ranks, ents=None, policy=None):
    if policy is None:
        return sum(accepted_length(policy_tree, r) for r in ranks) / len(ranks)
    tot = 0
    for r, e in zip(ranks, ents):
        cell = sum(1 << d for d, (x, m) in enumerate(zip(e, policy.medians)) if x > m)
        tot += accepted_length(policy.trees.get(cell, policy.fallback), r)
    return tot / len(ranks)


def score_policy(policy, ranks, ents):
    """Oracle accepted drafts/step of a loaded StaticTree / EntropyTree."""
    return score(policy.nodes, ranks) if isinstance(policy, StaticTree) else score(None, ranks, ents, policy)


def parse_sample(text):
    t, top_p = (float(x) for x in text.split(","))
    if t <= 0 or not 0 < top_p <= 1:
        raise argparse.ArgumentTypeError(f"need T > 0 and 0 < top_p <= 1, got {text!r}")
    return t, top_p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", default=None)
    ap.add_argument("--nodes", type=int, nargs="+", default=[3, 7, 15, 25])
    ap.add_argument("--n_calib", type=int, default=300)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--sample", type=parse_sample, default=None, metavar="T,TOP_P",
                    help="fit and score on text sampled at this temperature / nucleus instead of greedy text")
    ap.add_argument("--compare", nargs="+", default=[], metavar="TREE.json",
                    help="other tree policies to score on the same eval text (e.g. the greedy-fitted tree)")
    ap.add_argument("--ignore_eos", choices=["auto", "on", "off"], default="off",
                    help="mask EOS while generating the text, as evaluate.py does (auto: its rule); default off")
    ap.add_argument("--no_save", action="store_true", help="scores only: do not write tree_policy files")
    args = ap.parse_args(argv)

    model, tok, cfg = load_run(args.run_dir, step=int(args.step) if args.step else None)
    amp = lambda: autocast_ctx(cfg)
    if args.ignore_eos == "auto":
        from scripts.evaluate import resolve_ignore_eos
        ignore_eos = resolve_ignore_eos("auto", cfg)
    else:
        ignore_eos = args.ignore_eos == "on"
    calib = make_prompts(load_split(cfg.lang, "train", 4 * args.n_calib), tok, n=args.n_calib, seed=0)
    evalp = make_prompts(load_split(cfg.lang, "eval"), tok, n=args.n_eval, seed=1)
    tag = f"_fitT{args.sample[0]:g}p{args.sample[1]:g}" if args.sample else ""
    print(f"{cfg.run_name} step {model.loaded_step}: {len(calib)} calibration / {len(evalp)} eval prompts"
          + (f" | sampled T={args.sample[0]:g} top_p={args.sample[1]:g}" if args.sample else "")
          + (" | ignore_eos" if ignore_eos else ""), flush=True)
    kw = dict(amp=amp, ignore_eos=ignore_eos)
    if args.sample:      # seeds: calibration 100000 + i, eval 200000 + i (evaluate.py uses 1..n)
        cr, ce, co = collect_ranks(model, tok, calib, args.max_new_tokens, sample=args.sample, seed0=100_000,
                                   oracle=True, **kw)
        er, ee, eo = collect_ranks(model, tok, evalp, args.max_new_tokens, sample=args.sample, seed0=200_000,
                                   oracle=True, **kw)
    else:
        cr, ce = collect_ranks(model, tok, calib, args.max_new_tokens, **kw)
        er, ee = collect_ranks(model, tok, evalp, args.max_new_tokens, **kw)

    out = Path(args.out_dir) / cfg.run_name
    rows = []
    for n in args.nodes:
        st, et = fit_static(cr, n), fit_entropy(cr, ce, n)
        row = {"max_nodes": n, "static": score(st.nodes, er), "entropy": score(None, er, ee, et)}
        if n in PRODUCT:
            row[f"product{PRODUCT[n]}"] = score(widths_tree(PRODUCT[n]), er)
        if args.sample:
            row["ceiling"] = score(fit_static(co, n).nodes, eo)
            row["widths"] = [sum(1 for _, d, _ in st.nodes if d == dd) for dd in range(1, model.num_heads)]
        rows.append(row)
        if not args.no_save:
            save_policy(st, out / f"tree_policy_static_n{n}{tag}.json")
            save_policy(et, out / f"tree_policy_entropy_n{n}{tag}.json")
        print("  " + " | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
              flush=True)
    chain_tree = StaticTree.from_widths((1,) * (model.num_heads - 1)).nodes
    chain = score(chain_tree, er)
    print(f"  chain (FixedK) {chain:.3f}   (accepted drafts/step; +1 = tokens/step)")
    rec = {"run_name": cfg.run_name, "step": model.loaded_step, "n_calib_positions": len(cr),
           "n_eval_positions": len(er), "chain": chain, "rows": rows,
           "note": "oracle accepted drafts per step on IndicCorp eval greedy text (teacher-forced)"}
    if args.sample:
        rec.update(sample=list(args.sample), ignore_eos=ignore_eos, chain_ceiling=score(chain_tree, eo),
                   note="oracle accepted drafts per step on IndicCorp eval text SAMPLED from head 0 (teacher-forced); "
                        "ceiling = same fit with head 0's own ranks (a drafter equal to the verifier)")
        print(f"  chain ceiling {rec['chain_ceiling']:.3f}")
    compare = {}
    for p in args.compare:
        compare[Path(p).stem.replace("tree_policy_", "")] = score_policy(load_policy(p), er, ee)
    if compare:
        rec["compare"] = compare
        print("  compare: " + " | ".join(f"{k} {v:.3f}" for k, v in compare.items()))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tree_fit{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
