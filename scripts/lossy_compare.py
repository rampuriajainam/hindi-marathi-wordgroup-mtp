"""Lossy (typical acceptance) sweep: linguistic vs random run, paired over prompts.

    python scripts/lossy_compare.py --pair results_dir/R12_hi_k4_struct_x4 results_dir/R12a_hi_k4_struct_x4_random \
        [--pair ...] --out results/drafter/lossy_compare.json

Reads quality_{dataset}.json (scripts/lossy_quality.py) and eval_{dataset}.json of both runs. Metrics per policy:
ppl (judge: plain ganga-1b), ppl_norep (only tokens that do not close a repeated 4-gram: repetitive text is cheap
for the judge), tokens/step, 4-gram repeats.
  same eps       A - B at the same policy;
  matched speed  A's (ppl at its tokens/step) - B's ppl interpolated at that tokens/step along B's curve
                 (piecewise linear over B's policies, fixed_k included): a run that accepts less is closer to
                 greedy and gets a lower ppl for free, so this is the fair quality comparison.
Paired bootstrap over prompts (2,000 resamples, 95% CI; the prompts are the same for every run).
"""

import argparse
import json
import math
import random
from pathlib import Path

QUAL = ("ppl", "ppl_norep")


def load(run, ds):
    q = json.loads((Path(run) / f"quality_{ds}.json").read_text(encoding="utf-8"))
    e = json.loads((Path(run) / f"eval_{ds}.json").read_text(encoding="utf-8"))
    steps = {x["policy"]: x["per_prompt"] for x in e["spec_decode"]}
    out = {}
    for p in q["policies"]:
        pp = p["per_prompt"]
        out[p["policy"]] = {"nll": pp["nll_sum"], "ntok": pp["n_tok"], "nll_nr": pp["nll_norep_sum"],
                            "n_nr": pp["n_norep"], "rep4": pp["rep4"],
                            "tokens": steps[p["policy"]]["tokens"], "steps": steps[p["policy"]]["steps"]}
    return out


def metrics(x, idx):
    s = lambda k: sum(x[k][i] for i in idx)
    return {"ppl": math.exp(s("nll") / s("ntok")), "ppl_norep": math.exp(s("nll_nr") / s("n_nr")),
            "tps": s("tokens") / s("steps"), "rep4": s("rep4") / len(idx)}


def interp(curve, x, key):
    """curve: [metrics] of one run over its policies; value of `key` at tokens/step x (None outside the range)."""
    pts = sorted((c["tps"], c[key]) for c in curve)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def compare(A, B, idx):
    """{policy: {metric: same-eps diff, 'matched_'+q: matched-speed diff}} on prompt indices idx."""
    ma = {p: metrics(A[p], idx) for p in A}
    mb = {p: metrics(B[p], idx) for p in B}
    out = {}
    for p in A:
        d = {k: ma[p][k] - mb[p][k] for k in ("ppl", "ppl_norep", "tps", "rep4")}
        for q in QUAL:
            y = interp(list(mb.values()), ma[p]["tps"], q)
            d["matched_" + q] = None if y is None else ma[p][q] - y
        out[p] = d
    return out


def ci(vals, d0, n):
    v = sorted(x for x in vals if x is not None)
    if d0 is None or len(v) < 0.9 * n:      # matched speed outside B's range in too many resamples
        return None
    return d0, v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LINGUISTIC", "RANDOM"))
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    fmt = lambda t, w=3: "        n/a          " if t is None else \
        f"{t[0]:+.{w}f} [{t[1]:+.{w}f},{t[2]:+.{w}f}]{'*' if t[1] > 0 or t[2] < 0 else ' '}"
    res = []
    for la, ra in a.pair:
        for ds in a.datasets:
            A, B = load(la, ds), load(ra, ds)
            m = len(next(iter(A.values()))["nll"])
            d0 = compare(A, B, range(m))
            rng = random.Random(0)
            boots = [compare(A, B, [rng.randrange(m) for _ in range(m)]) for _ in range(a.n_boot)]
            print(f"\n{Path(la).name} - {Path(ra).name}  [{ds}]   (* = 95% CI excludes 0; negative ppl = A better)")
            print(f"  {'policy':12s} {'tok/step A':>10s}  {'d tok/step':>22s}  {'d rep4':>22s}  {'d ppl, same eps':>24s}"
                  f"  {'d ppl, matched speed':>24s}  {'d ppl_norep, matched':>24s}")
            for p in A:
                row = {"linguistic": Path(la).name, "random": Path(ra).name, "dataset": ds, "policy": p,
                       "A": metrics(A[p], range(m)), "B": metrics(B[p], range(m))}
                for k in d0[p]:
                    row[k] = ci([b[p][k] for b in boots], d0[p][k], a.n_boot)
                res.append(row)
                print(f"  {p:12s} {row['A']['tps']:10.3f}  {fmt(row['tps'])}  {fmt(row['rep4'])}  {fmt(row['ppl'])}"
                      f"  {fmt(row['matched_ppl'])}  {fmt(row['matched_ppl_norep'])}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
