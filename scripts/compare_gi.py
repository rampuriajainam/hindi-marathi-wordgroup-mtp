"""Linguistic vs random run, side by side: GI (any grouping), tokens/step and per-head top-1, with paired bootstrap CIs.

    python scripts/compare_gi.py --pair results/R11_hi_k4_struct_karthika results/R11a_hi_k4_struct_random_karthika \
        --groupers own karthika stanza --datasets indiccorp_eval flores_hi

`own` = the GI stored in eval_{dataset}.json (scored with the grouper evaluate.py ran with); any other name reads
gi_{name}_{dataset}.json written by scripts/rescore_gi.py. Paired bootstrap over prompts (2,000 resamples, 95% CI);
both runs must be evaluated on the same prompts (evaluate.py with the same dataset and spec_n).
"""

import argparse
import json
import random
from pathlib import Path


def boot(a, b, ratio, n=2000, seed=0):
    """a, b: per-prompt (num, den) lists; difference of pooled ratios num/den, x100 if ratio == 'pct'."""
    (na, da), (nb, db) = a, b
    k = 100 if ratio == "pct" else 1
    rng, m = random.Random(seed), len(na)
    d0 = k * (sum(na) / sum(da) - sum(nb) / sum(db))
    ds = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        ds.append(k * (sum(na[i] for i in idx) / sum(da[i] for i in idx) - sum(nb[i] for i in idx) / sum(db[i] for i in idx)))
    ds.sort()
    return d0, ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def gi(run_dir, dataset, grouper, policy="fixed_k"):
    if grouper == "own":
        rec = json.loads((Path(run_dir) / f"eval_{dataset}.json").read_text(encoding="utf-8"))
        e = [x for x in rec.get("spec_decode") or [] if x["policy"] == policy][0]["per_prompt"]
    else:
        p = Path(run_dir) / f"gi_{grouper}_{dataset}.json"
        e = [x for x in json.loads(p.read_text(encoding="utf-8"))["policies"] if x["policy"] == policy][0]
    return e["gi_hits"], e["gi_spans"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LINGUISTIC", "RANDOM"))
    ap.add_argument("--groupers", nargs="+", default=["own"])
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]{' *' if t[1] > 0 or t[2] < 0 else ''}"
    res = []
    for la, ra in a.pair:
        for ds in a.datasets:
            recs = [json.loads((Path(r) / f"eval_{ds}.json").read_text(encoding="utf-8")) for r in (la, ra)]
            row = {"linguistic": Path(la).name, "random": Path(ra).name, "dataset": ds}
            row["top1_diff_pts"] = [100 * (x["top1"] - y["top1"]) for x, y in zip(recs[0]["per_head"], recs[1]["per_head"])]
            row["in_group_top1_diff_pts"] = [100 * ((x.get("top1_in_group") or 0) - (y.get("top1_in_group") or 0))
                                             for x, y in zip(recs[0]["per_head"], recs[1]["per_head"])]
            pp = [[e for e in r.get("spec_decode") or [] if e["policy"] == "fixed_k"][0]["per_prompt"] for r in recs]
            row["tok_per_step_diff"] = boot((pp[0]["tokens"], pp[0]["steps"]), (pp[1]["tokens"], pp[1]["steps"]), "raw")
            row["gi_diff_pts"] = {}
            for g in a.groupers:
                try:
                    row["gi_diff_pts"][g] = boot(gi(la, ds, g), gi(ra, ds, g), "pct")
                except (FileNotFoundError, IndexError, KeyError) as e:
                    row["gi_diff_pts"][g] = None
            res.append(row)
            print(f"{row['linguistic']} - {row['random']}  [{ds}]")
            print("   top-1 h1-3 (pts): " + " / ".join(f"{x:+.2f}" for x in row["top1_diff_pts"][1:]) +
                  "   in-group h1-3: " + " / ".join(f"{x:+.2f}" for x in row["in_group_top1_diff_pts"][1:]))
            print(f"   tokens/step: {fmt(row['tok_per_step_diff'])}")
            for g, v in row["gi_diff_pts"].items():
                print(f"   GI ({g}): {fmt(v) if v else 'n/a'}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
