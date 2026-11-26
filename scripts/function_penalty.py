"""Down-weight function words in the drafts, everywhere or only where a new group is predicted to start.

    python scripts/function_penalty.py --runs R12=... R12a=... --pairs R12-R12a --out results/drafter/function_penalty.json

The drafter over-bets function words (draft_text_analysis.py): it wins inside groups and loses at group starts.
Inference-only fix, no training: subtract beta from the logits of the 50 most frequent tokens (counted on --n_func
train sentences) in draft heads 1..3,
  global  at every position;
  gated   only where the head's own S2 boundary probe says its target starts a new group (sigmoid > --tau).
Teacher-forced pass over the eval texts; head 0 (the verifier) is never changed. Per position t:
  agree_d   argmax of head d at t == head 0's argmax at t+d (the greedy acceptance test, given the true prefix);
  len       1 + a1 + a1*a2 + a1*a2*a3: tokens per step if a step started at t (proxy).
Split by the target token (hi_rules_v0: continues vs starts a group). Paired bootstrap over sentences of each
setting vs beta = 0 (same run), and of --pairs A-B at the same setting.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def pooled(rows, key):
    s = sum(r.get(key, (0.0, 0))[0] for r in rows)
    n = sum(r.get(key, (0.0, 0))[1] for r in rows)
    return s / n if n else float("nan")


def boot_diff(ra, rb, ka, kb=None, n=2000, seed=0):
    kb = kb or ka
    rng = random.Random(seed)
    m = len(ra)
    ds = []
    for _ in range(n):
        pick = [rng.randrange(m) for _ in range(m)]
        ds.append(pooled([ra[i] for i in pick], ka) - pooled([rb[i] for i in pick], kb))
    ds.sort()
    return pooled(ra, ka) - pooled(rb, kb), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def add(row, key, v):
    s = row.setdefault(key, [0.0, 0])
    s[0] += v
    s[1] += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"])
    ap.add_argument("--n", type=int, default=300, help="sentences per dataset")
    ap.add_argument("--n_func", type=int, default=5000, help="train sentences to count function tokens on")
    ap.add_argument("--betas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--grouper", default="hi_rules_v0")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import collections
    import contextlib

    import torch

    from evaluate import load_texts
    from mtp.data.corpus import load_split
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    settings = [("none", 0.0)] + [(m, b) for m in ("global", "gated") for b in args.betas]
    data, func_ids = {}, None
    stats = {}
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        model, tok, cfg = load_run(run_dir)
        device = next(model.parameters()).device
        amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else contextlib.nullcontext
        if func_ids is None:
            train = [t for t in load_split(cfg.lang, "train", args.n_func) if t.strip()]
            counts = collections.Counter(t for s in train for t in tok(s)["input_ids"])
            func_ids = torch.tensor([t for t, _ in counts.most_common(50)], device=device)
            print("function tokens:", " ".join(tok.convert_ids_to_tokens(func_ids.tolist()[:25])), "...")
            grouper = get_grouper(args.grouper, cfg.lang)
            for ds in args.datasets:
                texts = [t for t in load_texts(ds, cfg, None) if t.strip()][: args.n]
                rows = label_batch(texts, tok, grouper, args.max_length)
                data[ds] = [(r["input_ids"], r["group_id"]) for r in rows]
        k = model.num_heads
        stats[name] = {}
        for ds, sents in data.items():
            rows = []
            for ids, gid in sents:
                row = {}
                L = len(ids)
                if L < k + 2:
                    rows.append(row)
                    continue
                with torch.no_grad(), amp():
                    out = model(input_ids=torch.tensor([ids], device=device))
                logits = [lg[0].float() for lg in out.logits]
                bl = out.aux.get("boundary_logits")
                if bl is None:
                    raise SystemExit(f"{name}: no boundary probes (needs an S2 run)")
                v = logits[0].argmax(-1)                                  # verifier argmax at every position
                n = L - k                                                  # t with all targets t+d+1 < L
                for mode, beta in settings:
                    agree = []
                    for d in range(1, k):
                        lg = logits[d][:n].clone()
                        if mode != "none":
                            pen = torch.zeros_like(lg)
                            pen[:, func_ids] = beta
                            if mode == "gated":
                                pen *= (torch.sigmoid(bl[d][0, :n].float()) > args.tau)[:, None]
                            lg -= pen
                        agree.append((lg.argmax(-1) == v[d: d + n]).float())
                    length = 1 + agree[0] + agree[0] * agree[1] + agree[0] * agree[1] * agree[2]
                    key = f"{mode}{beta:g}"
                    for t in range(n):
                        add(row, f"{key}|len|all", float(length[t]))
                        for d in range(1, k):
                            a, b = gid[t + d], gid[t + d + 1]
                            if a < 0 or b < 0:
                                continue
                            sp = "in" if a == b else "start"
                            add(row, f"{key}|agree_h{d}|{sp}", float(agree[d - 1][t]))
                            add(row, f"{key}|agree_h{d}|all", float(agree[d - 1][t]))
                rows.append(row)
            stats[name][ds] = rows
            print(f"\n{name} {ds}:  setting: tokens/step proxy | h1 agree in / start / all (%)")
            for mode, beta in settings:
                key = f"{mode}{beta:g}"
                print(f"  {key:10s} {pooled(rows, f'{key}|len|all'):.4f} | " + " / ".join(
                    f"{100 * pooled(rows, f'{key}|agree_h1|{sp}'):.2f}" for sp in ("in", "start", "all")), flush=True)
        del model
        torch.cuda.empty_cache()

    res = {"tau": args.tau, "betas": args.betas, "runs": {}, "vs_none": {}, "pairs": {}}
    keys = sorted({kk for rows in next(iter(stats.values())).values() for r in rows for kk in r})
    for name in stats:
        res["runs"][name] = {ds: {kk: pooled(rows, kk) for kk in keys} for ds, rows in stats[name].items()}
        res["vs_none"][name] = {}
        for ds in args.datasets:
            res["vs_none"][name][ds] = {}
            print(f"\n{name} [{ds}] tokens/step proxy vs no penalty (* = CI excludes 0)")
            for mode, beta in settings[1:]:
                key = f"{mode}{beta:g}"
                d = boot_diff(stats[name][ds], stats[name][ds], f"{key}|len|all", "none0|len|all")
                res["vs_none"][name][ds][key] = d
                print(f"  {key:10s} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{' *' if d[1] > 0 or d[2] < 0 else ''}")
    for pair in args.pairs:
        a, b = pair.split("-")
        res["pairs"][pair] = {}
        for ds in args.datasets:
            res["pairs"][pair][ds] = {}
            print(f"\n{pair} [{ds}] tokens/step proxy, same setting (* = CI excludes 0)")
            for mode, beta in settings:
                key = f"{mode}{beta:g}"
                d = boot_diff(stats[a][ds], stats[b][ds], f"{key}|len|all")
                res["pairs"][pair][ds][key] = d
                print(f"  {key:10s} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{' *' if d[1] > 0 or d[2] < 0 else ''}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
