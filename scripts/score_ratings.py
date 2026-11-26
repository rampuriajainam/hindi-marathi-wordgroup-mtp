"""
Score filled rating sheets (scripts/make_rating_sheet.py) against the key.

    python scripts/score_ratings.py --key ratings/hi_key.json --sheets ratings/hi_sheet_jai.csv ratings/hi_sheet_om.csv

Per system: mean fluency, mean coherence, mean rank, with 95% bootstrap CIs over items; pairwise win rates by rank;
and, for raters who share items, Spearman agreement of fluency and coherence on the shared continuations. Sheets may
cover different items (ratings/INSTRUCTIONS.md splits Hindi 1-30 / 21-50); a continuation rated by several raters
counts once, with its scores averaged.
"""

import argparse
import csv
import json
import random
from pathlib import Path


def read_sheet(path):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            def num(col):
                v = (r.get(col) or "").strip()
                return float(v) if v else None
            out.append((r["item"], r["label"], num("fluency (1-5)"), num("coherence with prompt (1-5)"),
                        num("rank in item (1=best)")))
    return out


def ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), means[int(0.025 * n)], means[int(0.975 * n)]


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="default: <lang>_scores.json next to the key")
    args = ap.parse_args(argv)
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))["key"]
    # Sheets may cover different items, and some items may be rated by several raters (the overlap used for
    # agreement). Each (item, continuation) counts once: its scores are averaged over the raters who rated it.
    cell = {}          # (item, label) -> {"fluency": [...], "coherence": [...], "rank": [...]} over raters
    per_rater = {}     # path -> {(item, label): {"fluency": v, "coherence": v}}
    for path in args.sheets:
        for item, label, flu, coh, rank in read_sheet(path):
            d = cell.setdefault((item, label), {"fluency": [], "coherence": [], "rank": []})
            for name, v in (("fluency", flu), ("coherence", coh), ("rank", rank)):
                if v is not None:
                    d[name].append(v)
            per_rater.setdefault(path, {})[(item, label)] = {"fluency": flu, "coherence": coh}
    per_sys, ranks = {}, {}
    for (item, label), d in cell.items():
        sysname = key[item][label]
        s = per_sys.setdefault(sysname, {"fluency": [], "coherence": [], "rank": []})
        for name, vals in d.items():
            if vals:
                s[name].append(sum(vals) / len(vals))
        if d["rank"]:
            ranks.setdefault(item, {})[sysname] = sum(d["rank"]) / len(d["rank"])
    res = {"systems": {}, "pairwise_win_rate": {}, "rater_agreement_fluency": {}, "rater_agreement_coherence": {},
           "items_per_rater": {Path(p).name: len({i for i, _ in v}) for p, v in per_rater.items()}}
    for s, d in sorted(per_sys.items()):
        res["systems"][s] = {m: (dict(zip(("mean", "lo", "hi"), ci(v))) if v else None) for m, v in d.items()}
        res["systems"][s]["n"] = len(d["fluency"])
        print(s, {m: (round(x["mean"], 2) if x else None) for m, x in res["systems"][s].items() if m != "n"})
    names = sorted(per_sys)
    for a in names:
        for b in names:
            if a < b:
                both = [r for r in ranks.values() if a in r and b in r]
                if both:
                    win = sum(r[a] < r[b] for r in both) + 0.5 * sum(r[a] == r[b] for r in both)
                    res["pairwise_win_rate"][f"{a} > {b}"] = win / len(both)
    paths = list(per_rater)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            for m in ("fluency", "coherence"):
                common = sorted(c for c in set(per_rater[paths[i]]) & set(per_rater[paths[j]])
                                if per_rater[paths[i]][c][m] is not None and per_rater[paths[j]][c][m] is not None)
                if len(common) > 2:
                    res[f"rater_agreement_{m}"][f"{Path(paths[i]).name} vs {Path(paths[j]).name}"] = {
                        "spearman": spearman([per_rater[paths[i]][c][m] for c in common],
                                             [per_rater[paths[j]][c][m] for c in common]),
                        "n_continuations": len(common)}
    print(json.dumps({k: v for k, v in res.items() if k != "systems"}, indent=1))
    out = Path(args.out) if args.out else Path(args.key).with_name(Path(args.key).stem.replace("_key", "") + "_scores.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
