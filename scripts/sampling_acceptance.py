"""Speculative SAMPLING acceptance, estimated offline: do group-trained heads put their probability mass where the
verifier does, beyond top-1?

    python scripts/sampling_acceptance.py --runs R12=checkpoints/kaggle_out/r12/runs/R12_hi_k4_struct_x4 \
        R12a=checkpoints/kaggle_out/r12/runs/R12a_hi_k4_struct_x4_random --pairs R12-R12a \
        --temps 1.0 0.7 --out results/drafter/sampling_acceptance.json

Under speculative sampling (Leviathan et al. 2023; lossless in distribution) a draft from q is accepted with
probability sum_v min(p(v), q(v)) = 1 - TV(p, q). Greedy decoding only sees argmax q, where group and random runs tie;
this asks whether the full distributions differ. Teacher-forced pass over the eval texts: head d at position t
(q, predicts token t+d+1) is compared with the run's own head 0 at t+d (p, the verifier given the true prefix, i.e.
the case where the previous drafts were accepted). Both at temperature T. Also reported: greedy agreement
argmax q == argmax p. Split by the target token: "in" = continues the hi_rules_v0 group of the token before it,
"start" = starts a new group. Paired bootstrap over sentences (2,000 resamples, 95% CI) for each --pairs A-B.
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
    """rows: per-sentence {key: [sum, n]} -> sum / n."""
    s = sum(r.get(key, (0.0, 0))[0] for r in rows)    # short sentences can lack a key
    n = sum(r.get(key, (0.0, 0))[1] for r in rows)
    return s / n if n else float("nan")


def boot_diff(ra, rb, key, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(ra)
    ds = []
    for _ in range(n):
        pick = [rng.randrange(m) for _ in range(m)]
        ds.append(pooled([ra[i] for i in pick], key) - pooled([rb[i] for i in pick], key))
    ds.sort()
    return pooled(ra, key) - pooled(rb, key), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    ap.add_argument("--pairs", nargs="*", default=[], help="A-B pairs (group run - random run)")
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"])
    ap.add_argument("--temps", nargs="+", type=float, default=[1.0, 0.7])
    ap.add_argument("--grouper", default="hi_rules_v0")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import contextlib

    import torch

    from evaluate import load_texts
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    data = {}       # dataset -> (input ids, group ids) per sentence; same tokenizer for every run
    stats = {}      # run -> dataset -> per-sentence rows {f"{T}|h{d}|{split}|{metric}": [sum, n]}
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        model, tok, cfg = load_run(run_dir, device=args.device)
        device = next(model.parameters()).device
        amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else contextlib.nullcontext
        k = model.num_heads
        if not data:
            grouper = get_grouper(args.grouper, cfg.lang)
            for ds in args.datasets:
                texts = [t for t in load_texts(ds, cfg, args.n) if t.strip()]
                rows = label_batch(texts, tok, grouper, args.max_length)
                data[ds] = [(r["input_ids"], r["group_id"]) for r in rows]
        stats[name] = {}
        for ds, sents in data.items():
            out_rows = []
            with torch.no_grad():
                for ids, gid in sents:
                    with amp():
                        logits = model(input_ids=torch.tensor([ids], device=device)).logits
                    L = len(ids)
                    row = {}
                    for T in args.temps:
                        p_all = torch.softmax(logits[0][0].float() / T, dim=-1)              # [L, V] verifier
                        for d in range(1, k):
                            n = L - d - 1                       # positions t with a target t+d+1 < L
                            if n <= 0:
                                continue
                            q = torch.softmax(logits[d][0, :n].float() / T, dim=-1)          # head d at t
                            p = p_all[d: d + n]                                              # head 0 at t+d
                            acc = torch.minimum(p, q).sum(-1).tolist()
                            agree = (p.argmax(-1) == q.argmax(-1)).tolist()
                            for t in range(n):
                                a, b = gid[t + d], gid[t + d + 1]
                                if a < 0 or b < 0:
                                    continue
                                split = "in" if a == b else "start"
                                for sp in (split, "all"):
                                    for metric, v in (("acc", acc[t]), ("agree", float(agree[t]))):
                                        key = f"{T}|h{d}|{sp}|{metric}"
                                        s = row.setdefault(key, [0.0, 0])
                                        s[0] += v
                                        s[1] += 1
                    out_rows.append(row)
            stats[name][ds] = out_rows
            print(f"{name} {ds}: " + "  ".join(
                f"T{T} h{d} acc {pooled(out_rows, f'{T}|h{d}|all|acc'):.4f}" for T in args.temps for d in range(1, k)),
                flush=True)
        del model
        torch.cuda.empty_cache()

    keys = sorted({key for r in next(iter(stats.values()))[args.datasets[0]] for key in r})
    res = {"grouper": args.grouper, "temps": args.temps, "runs": {}, "pairs": {}}
    for name in stats:
        res["runs"][name] = {ds: {key: pooled(rows, key) for key in keys} for ds, rows in stats[name].items()}
    for pair in args.pairs:
        a, b = pair.split("-")
        res["pairs"][pair] = {ds: {key: boot_diff(stats[a][ds], stats[b][ds], key) for key in keys}
                              for ds in args.datasets}
        for ds in args.datasets:
            print(f"\n{pair} [{ds}]  (points; * = CI excludes 0)")
            for T in args.temps:
                for metric in ("acc", "agree"):
                    cells = []
                    for d in range(1, 4):
                        for sp in ("in", "start", "all"):
                            key = f"{T}|h{d}|{sp}|{metric}"
                            if key in res["pairs"][pair][ds]:
                                x, lo, hi = res["pairs"][pair][ds][key]
                                cells.append(f"h{d}/{sp} {100 * x:+.2f}{'*' if lo > 0 or hi < 0 else ''}")
                    print(f"  T{T} {metric:5s} " + "  ".join(cells))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
