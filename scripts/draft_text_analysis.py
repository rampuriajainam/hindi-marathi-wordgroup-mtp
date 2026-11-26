"""The drafts as text: what do the draft heads propose, judged on their own rather than by "did it match"?

    python scripts/draft_text_analysis.py --runs R12=... R12a=... --pairs R12-R12a --out results/drafter/draft_text.json

Teacher-forced pass over the eval texts. At a position t the engine would draft the block [t0, d1, d2, d3]:
t0 = head 0's argmax, d = head d's argmax (all from the true prefix x<=t). Per block, on every --stride-th position:
  fluency   mean NLL of d1..d3 given x<=t + t0 under a FIXED judge (plain ganga-1b): is the draft plausible Hindi
            whatever the reference says ("fluency_wrong": only blocks where d1 differs from the reference);
  dup       adjacent repeats inside the block (e.g. "के के"): parallel heads guess independently, so this is the
            classic incoherent-draft failure;
and per drafted token that differs from the reference token (a wrong draft), its kind:
  sibling   same stem after stripping one inflectional ending (जाता -> जाते; inflection_confusions.py);
  repeat    a token already in the prefix;
  function  one of the 50 most frequent reference tokens (है, के, में, ...);
  other.
Plus "func_share": share of ALL drafts that are such function tokens (the reference share is printed alongside):
over-producing them = playing safe. Paired bootstrap over sentences (2,000 resamples, 95% CI) for --pairs A-B.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KINDS = ("sibling", "repeat", "function", "other")


def pooled(rows, key):
    s = sum(r.get(key, (0.0, 0))[0] for r in rows)
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
    ap.add_argument("--stride", type=int, default=3, help="score every stride-th position")
    ap.add_argument("--judge", default="LingoIITGN/ganga-1b")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import contextlib

    import torch
    from transformers import AutoModelForCausalLM

    from evaluate import load_texts
    from inflection_confusions import is_deva, stem
    from mtp.device import autocast_ctx, pick_dtype, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    data, func, words = {}, None, None
    judge = None
    stats = {}
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        model, tok, cfg = load_run(run_dir)
        device = next(model.parameters()).device
        amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else contextlib.nullcontext
        if judge is None:
            judge = AutoModelForCausalLM.from_pretrained(args.judge, dtype=pick_dtype()).to(device).eval()
            for ds in args.datasets:
                texts = [t for t in load_texts(ds, cfg, None) if t.strip()][: args.n]
                data[ds] = [list(tok(t, truncation=True, max_length=args.max_length)["input_ids"]) for t in texts]
            counts = collections.Counter(t for sents in data.values() for ids in sents for t in ids)
            func = {t for t, _ in counts.most_common(50)}
            pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
            words = [p[1:] if p.startswith("▁") else None for p in pieces]   # whole-word tokens -> the word

        def kind(draft, ref, prefix):
            wd, wr = words[draft], words[ref]
            if wd and wr and is_deva(wd) and is_deva(wr) and stem(wd) == stem(wr):
                return "sibling"
            if draft in prefix:
                return "repeat"
            return "function" if draft in func else "other"

        stats[name] = {}
        for ds, sents in data.items():
            rows = []
            for ids in sents:
                row = {}
                L = len(ids)
                with torch.no_grad(), amp():
                    logits = model(input_ids=torch.tensor([ids], device=device)).logits
                arg = [lg[0].argmax(-1).tolist() for lg in logits]               # [k][L]
                blocks = []
                for t in range(0, L - 4, args.stride):
                    block = [arg[0][t], arg[1][t], arg[2][t], arg[3][t]]           # predicts t+1 .. t+4
                    ref = ids[t + 1: t + 5]
                    blocks.append((ids[: t + 1], block))
                    add(row, "dup", float(sum(block[i] == block[i + 1] for i in range(3)) > 0))
                    for d in range(1, 4):
                        add(row, f"func_share|h{d}", float(block[d] in func))
                        add(row, f"ref_func_share|h{d}", float(ref[d] in func))
                        if block[d] != ref[d]:
                            kd = kind(block[d], ref[d], ids[: t + 1])
                            for k in KINDS:
                                add(row, f"wrong_{k}|h{d}", float(kd == k))
                # judge: NLL of d1..d3 given prefix + t0, batched
                for i in range(0, len(blocks), args.batch_size):
                    chunk = blocks[i: i + args.batch_size]
                    seqs = [p + b for p, b in chunk]
                    width = max(len(s) for s in seqs)
                    x = torch.zeros(len(seqs), width, dtype=torch.long)
                    m = torch.zeros(len(seqs), width, dtype=torch.long)
                    for j, s in enumerate(seqs):
                        x[j, : len(s)] = torch.tensor(s)
                        m[j, : len(s)] = 1
                    with torch.no_grad():
                        lp = torch.log_softmax(judge(input_ids=x.to(device), attention_mask=m.to(device)).logits.float(), -1)
                    for j, (p, b) in enumerate(chunk):
                        pos = len(p)                    # index of t0; d1..d3 at pos+1..pos+3, predicted from pos..pos+2
                        nll = -sum(float(lp[j, pos + q, b[q + 1]]) for q in range(3)) / 3
                        add(row, "fluency", nll)
                        t = len(p) - 1
                        if b[1] != ids[t + 2]:
                            add(row, "fluency_wrong", nll)
                rows.append(row)
            stats[name][ds] = rows
            print(f"{name} {ds}: fluency {pooled(rows, 'fluency'):.3f} (wrong {pooled(rows, 'fluency_wrong'):.3f})  "
                  f"dup {100 * pooled(rows, 'dup'):.2f}%  func h1 {100 * pooled(rows, 'func_share|h1'):.1f}% "
                  f"(ref {100 * pooled(rows, 'ref_func_share|h1'):.1f}%)  wrong h1: " + " ".join(
                      f"{k} {100 * pooled(rows, f'wrong_{k}|h1'):.1f}%" for k in KINDS), flush=True)
        del model
        torch.cuda.empty_cache()

    keys = sorted({k for rows in next(iter(stats.values())).values() for r in rows for k in r})
    res = {"runs": {n: {ds: {k: pooled(rows, k) for k in keys} for ds, rows in v.items()} for n, v in stats.items()},
           "pairs": {}}
    for pair in args.pairs:
        a, b = pair.split("-")
        res["pairs"][pair] = {ds: {k: boot_diff(stats[a][ds], stats[b][ds], k) for k in keys} for ds in args.datasets}
        for ds in args.datasets:
            v = res["pairs"][pair][ds]
            f = lambda k, s=100: f"{s * v[k][0]:+.2f}{'*' if v[k][1] > 0 or v[k][2] < 0 else ''}"
            print(f"\n{pair} [{ds}]  fluency {f('fluency', 1)} (wrong {f('fluency_wrong', 1)}) nats  dup {f('dup')} pts")
            for d in range(1, 4):
                print(f"  h{d}: func share {f(f'func_share|h{d}')}  wrong kinds " +
                      " ".join(f"{k} {f(f'wrong_{k}|h{d}')}" for k in KINDS))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
