"""Route between two drafters by group position: the group-trained heads where the target continues a group, other
heads where it starts one. Offline ceiling (true boundaries) and realistic version (the S2 boundary probe).

    python scripts/route_drafters.py --runs S=checkpoints/kaggle/frozen/rsd-struct \
        R=checkpoints/kaggle/frozen/rsd-struct-rand P=checkpoints/kaggle/frozen/rsd \
        --inside S --starts R P --probe S --out results/drafter/route_drafters.json

Group-trained heads gain inside groups and lose at group starts (sampling_acceptance.py). If a drafter could tell the
two apart it could keep both. Needs frozen-backbone runs: head 0 is the same base model in all of them, so the heads
are interchangeable and every drafter is checked against the same verifier.

Teacher-forced pass over the eval texts. At position t, head d drafts the token t+d+1; it counts as accepted if its
argmax equals head 0's argmax at t+d (the greedy acceptance test given the true prefix). Per position, the
tokens/step proxy = 1 + a1 + a1*a2 + a1*a2*a3. Variants per start-expert X:
  inside-only / X-only   one drafter everywhere;
  oracle                 inside-expert where the target truly continues a group (--grouper), X where it starts one;
  probe                  the same switch decided by --probe's boundary probe for head d (sigmoid > --tau).
Paired bootstrap over sentences (2,000 resamples, 95% CI) of each routed variant vs the best single drafter.
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


def boot_diff(rows, ka, kb, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(rows)
    ds = []
    for _ in range(n):
        pick = [rows[rng.randrange(m)] for _ in range(m)]
        ds.append(pooled(pick, ka) - pooled(pick, kb))
    ds.sort()
    return pooled(rows, ka) - pooled(rows, kb), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def add(row, key, v):
    s = row.setdefault(key, [0.0, 0])
    s[0] += v
    s[1] += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir (frozen-backbone runs)")
    ap.add_argument("--inside", required=True, help="drafter used where the target continues a group")
    ap.add_argument("--starts", nargs="+", required=True, help="candidate drafters for group starts")
    ap.add_argument("--probe", required=True, help="run whose S2 boundary probe decides the switch")
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"])
    ap.add_argument("--n", type=int, default=300, help="sentences per dataset")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--grouper", default="hi_rules_v0")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import contextlib

    import torch

    from evaluate import load_texts
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    data, store = {}, {}       # store[name][ds] = list of (head argmax [k, L], probe sigmoid [k, L] or None)
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        model, tok, cfg = load_run(run_dir)
        if not getattr(model, "freeze_backbone", cfg.get("freeze_backbone") if hasattr(cfg, "get") else True):
            raise SystemExit(f"{name}: not a frozen-backbone run; heads are not interchangeable")
        device = next(model.parameters()).device
        amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else contextlib.nullcontext
        if not data:
            grouper = get_grouper(args.grouper, cfg.lang)
            for ds in args.datasets:
                texts = [t for t in load_texts(ds, cfg, None) if t.strip()][: args.n]
                data[ds] = [(r["input_ids"], r["group_id"]) for r in label_batch(texts, tok, grouper, args.max_length)]
        store[name] = {}
        for ds, sents in data.items():
            out = []
            with torch.no_grad():
                for ids, _ in sents:
                    with amp():
                        o = model(input_ids=torch.tensor([ids], device=device))
                    arg = torch.stack([lg[0].argmax(-1) for lg in o.logits]).cpu()
                    bl = o.aux.get("boundary_logits")
                    prob = torch.stack([torch.sigmoid(b[0].float()) for b in bl]).cpu() if bl is not None else None
                    out.append((arg, prob))
            store[name][ds] = out
        del model
        torch.cuda.empty_cache()
        print(f"loaded {name}", flush=True)

    if store[args.probe][args.datasets[0]][0][1] is None:
        raise SystemExit(f"{args.probe} has no boundary probes")
    res = {"tau": args.tau, "inside": args.inside, "probe": args.probe, "datasets": {}}
    for ds, sents in data.items():
        rows = []
        verifier_mismatch = 0
        for i, (ids, gid) in enumerate(sents):
            L = len(ids)
            k = store[args.inside][ds][i][0].shape[0]
            n = L - k
            row = {}
            v = store[args.inside][ds][i][0][0]
            for name in store:
                verifier_mismatch += int((store[name][ds][i][0][0] != v).sum())
            if n <= 0:
                rows.append(row)
                continue
            prob = store[args.probe][ds][i][1]
            agree = {name: [(store[name][ds][i][0][d][:n] == v[d: d + n]) for d in range(1, k)] for name in store}
            for t in range(n):
                true_in = [gid[t + d] >= 0 and gid[t + d] == gid[t + d + 1] for d in range(1, k)]
                pred_start = [float(prob[d][t]) > args.tau for d in range(1, k)]
                variants = {name: [bool(agree[name][d - 1][t]) for d in range(1, k)] for name in store}
                for x in args.starts:
                    a_in, a_x = variants[args.inside], variants[x]
                    variants[f"oracle_{x}"] = [a_in[j] if true_in[j] else a_x[j] for j in range(k - 1)]
                    variants[f"probe_{x}"] = [a_x[j] if pred_start[j] else a_in[j] for j in range(k - 1)]
                for key, a in variants.items():
                    length, run = 1.0, 1.0
                    for x in a:
                        run *= x
                        length += run
                    add(row, f"{key}|len", length)
                    split = "in" if true_in[0] else "start"
                    add(row, f"{key}|h1|{split}", float(a[0]))
                # probe quality for head 1 against the true boundary
                add(row, "probe_acc_h1", float(pred_start[0] == (not true_in[0])))
            rows.append(row)
        singles = list(store)
        best = max(singles, key=lambda s: pooled(rows, f"{s}|len"))
        d = {"verifier_mismatch_tokens": verifier_mismatch, "probe_acc_h1": pooled(rows, "probe_acc_h1"),
             "best_single": best, "variants": {}}
        keys = singles + [f"{p}_{x}" for x in args.starts for p in ("oracle", "probe")]
        print(f"\n[{ds}] verifier mismatches across runs: {verifier_mismatch} tokens; probe accuracy (h1) "
              f"{100 * d['probe_acc_h1']:.1f}%; best single drafter: {best}")
        print(f"  {'variant':22s} {'tok/step':>9s} {'h1 in':>7s} {'h1 start':>9s}   vs best single [95% CI]")
        for key in keys:
            cell = {"len": pooled(rows, f"{key}|len"), "h1_in": pooled(rows, f"{key}|h1|in"),
                    "h1_start": pooled(rows, f"{key}|h1|start")}
            if key != best:
                cell["vs_best"] = boot_diff(rows, f"{key}|len", f"{best}|len")
            d["variants"][key] = cell
            vb = cell.get("vs_best")
            print(f"  {key:22s} {cell['len']:9.4f} {100 * cell['h1_in']:6.2f}% {100 * cell['h1_start']:8.2f}%   "
                  + (f"{vb[0]:+.4f} [{vb[1]:+.4f}, {vb[2]:+.4f}]{' *' if vb[1] > 0 or vb[2] < 0 else ''}" if vb else "(best)"))
        res["datasets"][ds] = d
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
