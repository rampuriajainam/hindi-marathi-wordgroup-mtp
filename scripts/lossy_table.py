"""
Summary table of decoding policies: speed (eval_{dataset}.json) next to quality (quality_{dataset}*.json from
scripts/lossy_quality.py).

    python scripts/lossy_table.py --results results/lossy_tree/<run> [...] [--datasets indiccorp_eval flores_hi]
        [--judges sarvam-1 bloom-1b7] [--md out.md]

One row per (run, dataset, policy): tok/step (mean_accepted_len), speed-up, tokens/sec, match rate vs the
policy's reference ("vs": greedy, or the entry's `baseline`, e.g. sample_0.7_0.9 for sampled trees), share of
lossy accepts (lossy_accepts / accepted drafts, tree + typ entries), ppl / ppl_norep / ppl_word under
the default judge (quality_{dataset}.json), ppl_word and ppl under each extra judge
(quality_{dataset}.{slug}.json; default: every slug found), rep4, identical rate, mean first divergence (over the
prompts that diverge). Policies: exact first, then lossy by eps descending, sampled, `human` last. Fields
missing from older files are shown as '-'.
"""

import argparse
import json
import math
import re
from pathlib import Path

LOSSY_SHARE_KEYS = ("lossy_accept_share", "lossy_share", "share_lossy", "lossy_accept_rate", "lossy_rate")
SAMPLE_RE = re.compile(r"samp|sample")
EPS_RE = re.compile(r"(?:typical_|typ)([0-9.]+(?:e-?\d+)?)")


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def policy_eps(name):
    m = EPS_RE.search(name)
    try:
        return float(m.group(1).rstrip(".")) if m else None
    except ValueError:
        return None


def policy_base(name):
    """Policy name without its lossy / sampling suffix: 'tree_static_n64+pld4x6+typ0.1' -> 'tree_static_n64+pld4x6'."""
    if name.startswith("typical_"):
        return "fixed_k"
    return re.sub(r"\+(typ|samp)[^+]*", "", name)


def policy_kind(name):
    if name == "human":
        return 3
    if SAMPLE_RE.search(name):
        return 2
    return 1 if policy_eps(name) is not None else 0


def order_policies(names):
    """Exact first, then lossy, then sampled, then `human`; within each, by base (fixed_k chain first, then
    first-seen order), lossy by eps descending."""
    names = list(dict.fromkeys(names))
    bases = list(dict.fromkeys(["fixed_k"] + [policy_base(n) for n in names]))   # chain before trees
    def key(item):
        i, n = item
        eps = policy_eps(n)
        return (policy_kind(n), bases.index(policy_base(n)), -(eps if eps is not None else 0.0), i)
    return [n for _, n in sorted(enumerate(names), key=key)]


def judge_files(folder, dataset):
    """{slug: path} of the extra-judge quality files of a dataset."""
    out = {}
    for p in sorted(Path(folder).glob(f"quality_{dataset}.*.json")):
        out[p.name[len(f"quality_{dataset}."):-len(".json")]] = p
    return out


def mean_div(q):
    div = ((q or {}).get("per_prompt") or {}).get("div")
    if not div:
        return None
    d = [x for x in div if x is not None]
    return sum(d) / len(d) if d else None


def lossy_share(e):
    """Share of accepted drafts that were not head 0's argmax: lossy_accepts (tree + typ entries, summed over
    prompts) / accepted drafts (sum of tokens - steps: every step also emits one verifier token)."""
    for k in LOSSY_SHARE_KEYS:
        if e.get(k) is not None:
            return e[k]
    if e.get("lossy_accepts") is None:
        return None
    pp = e.get("per_prompt") or {}
    try:
        drafts = sum(pp["tokens"]) - sum(pp["steps"])
    except (KeyError, TypeError):
        return None
    return e["lossy_accepts"] / drafts if drafts > 0 else None


def build_rows(folders, datasets=None, judges=None):
    """[row dict] for every run folder and dataset (eval_*.json present)."""
    rows = []
    for folder in map(Path, folders):
        names = datasets or sorted(p.stem[len("eval_"):] for p in folder.glob("eval_*.json"))
        for ds in names:
            ev = _load(folder / f"eval_{ds}.json") or {}
            q0 = _load(folder / f"quality_{ds}.json") or {}
            extra = judge_files(folder, ds)
            slugs = judges if judges is not None else list(extra)
            qx = {s: (_load(extra[s]) if s in extra else None) or {} for s in slugs}
            spec = {e["policy"]: e for e in (ev.get("spec_decode") or [])}
            qual = {p["policy"]: p for p in q0.get("policies") or []}
            qual_x = {s: {p["policy"]: p for p in q.get("policies") or []} for s, q in qx.items()}
            pols = list(spec) + [p for p in qual if p not in spec]
            for s in slugs:
                pols += [p for p in qual_x[s] if p not in pols]
            run = ev.get("run_name") or q0.get("run_name") or folder.name
            for pol in order_policies(pols):
                e, q = spec.get(pol, {}), qual.get(pol, {})
                row = {"run": run, "dataset": ds, "policy": pol,
                       "tok_step": e.get("mean_accepted_len", q.get("mean_accepted_len")),
                       "speedup": e.get("speedup", q.get("speedup")), "tok_s": e.get("tokens_per_sec"),
                       "match_rate": e.get("match_rate"), "reference": e.get("baseline", "greedy" if e else None),
                       "lossy_share": lossy_share(e),
                       "ppl": q.get("ppl"), "ppl_norep": q.get("ppl_norep"), "ppl_word": q.get("ppl_word"),
                       "rep4": q.get("rep4"), "identical_rate": q.get("identical_rate"), "mean_div": mean_div(q),
                       "mean_new_tokens": q.get("mean_new_tokens", e.get("mean_new_tokens")), "judges": {}}
                for s in slugs:
                    xq = qual_x[s].get(pol, {})
                    row["judges"][s] = {"ppl_word": xq.get("ppl_word"), "ppl": xq.get("ppl")}
                    for k in ("rep4", "mean_new_tokens"):     # run-token quantities: same under every judge
                        if row[k] is None:
                            row[k] = xq.get(k)
                rows.append(row)
    return rows


def _f(v, fmt=".2f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return format(v, fmt)


def to_markdown(rows):
    slugs = list(dict.fromkeys(s for r in rows for s in r["judges"]))
    head = (["policy", "tok/step", "speed-up", "tok/s", "match", "vs", "lossy acc", "ppl", "ppl_norep", "ppl_word"]
            + [f"{s} {k}" for s in slugs for k in ("ppl_word", "ppl")]
            + ["rep4", "identical", "mean div", "len"])
    out, group = [], None
    for r in rows:
        if (r["run"], r["dataset"]) != group:
            group = (r["run"], r["dataset"])
            out += ["", f"### {r['run']} | {r['dataset']}", "", "| " + " | ".join(head) + " |",
                    "|" + "---|" * len(head)]
        cells = [r["policy"], _f(r["tok_step"]), _f(r["speedup"]), _f(r["tok_s"], ".1f"), _f(r["match_rate"]),
                 r["reference"] or "-", _f(r["lossy_share"], ".3f"), _f(r["ppl"]), _f(r["ppl_norep"]), _f(r["ppl_word"])]
        for s in slugs:
            j = r["judges"].get(s, {})
            cells += [_f(j.get("ppl_word")), _f(j.get("ppl"))]
        cells += [_f(r["rep4"], ".3f"), _f(r["identical_rate"]), _f(r["mean_div"], ".1f"),
                  _f(r["mean_new_tokens"], ".1f")]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out).lstrip("\n") + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="results/{run} folders")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every eval_*.json in each folder")
    ap.add_argument("--judges", nargs="+", default=None, help="extra-judge slugs (default: every one found)")
    ap.add_argument("--md", default=None, help="write the markdown table here")
    args = ap.parse_args(argv)
    md = to_markdown(build_rows(args.results, args.datasets, args.judges))
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
