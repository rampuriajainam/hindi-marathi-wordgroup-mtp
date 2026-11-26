"""Representation coherence: are the hidden states of adjacent tokens in the same word group more similar than
those of adjacent tokens across a group boundary, and did group training make that gap bigger? (Brief: "coherent
phrase-level representations".)

    python scripts/group_coherence.py --runs R2=checkpoints/kaggle/R2 R3=checkpoints/kaggle/R3 ... \
        --groupers hi_rules_v0 stanza random --base R2 --out results/grouping/coherence.json

Teacher-forced pass over the IndicCorp eval sentences (~500). For each layer L in --layers the backbone hidden
states are mean-centred (over all eval tokens of that model and layer: LM states are anisotropic) and, for every
adjacent token pair (t, t+1) that lies in two *different words* (pairs inside one word are dropped: subword
continuation is not grouping), the cosine similarity is added to "within" if the two words are in the same group
and to "across" otherwise. coherence = mean(within) - mean(across). Distance is fixed at 1 token, so "neighbours
are similar anyway" cannot create the gap.

Differences between runs (vs --base, and the listed --pairs) use a paired bootstrap over sentences (2,000
resamples, 95% CI). Positive control: R6a / R4a were trained on `random` groups, so if training moves
representations at all they should gain coherence on the `random` labels.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def coherence(s):
    """s: per-sentence rows [sum_in, n_in, sum_out, n_out] -> pooled coherence."""
    a = sum(r[0] for r in s); b = sum(r[1] for r in s); c = sum(r[2] for r in s); d = sum(r[3] for r in s)
    return a / b - c / d if b and d else float("nan")


def boot_diff(sa, sb, n=2000, seed=0):
    rng = random.Random(seed)
    idx = range(len(sa))
    ds = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        ds.append(coherence([sa[i] for i in pick]) - coherence([sb[i] for i in pick]))
    ds.sort()
    return coherence(sa) - coherence(sb), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    ap.add_argument("--groupers", nargs="+", default=["hi_rules_v0", "stanza", "random"])
    ap.add_argument("--layers", nargs="+", default=["4", "8", "12", "16", "h1", "h2", "h3"],
                    help="backbone layer numbers, and hD = extra head D's state (labels shifted to its targets)")
    ap.add_argument("--base", default="R2")
    ap.add_argument("--pairs", nargs="*", default=[], help="A-B pairs to compare, e.g. R3-R6a")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    from mtp.data.corpus import load_split
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    texts = [t for t in load_split("hi", "eval", args.n) if t.strip()]
    labels = {}   # grouper -> per sentence: group id per token (computed once, tokenizer shared by all runs)
    word_ids = None
    stats = {}    # run -> grouper -> layer -> per-sentence rows
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        model, tok, cfg = load_run(run_dir, device=args.device)
        device = next(model.parameters()).device
        amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else __import__("contextlib").nullcontext
        if word_ids is None:
            word_ids = [r["group_id"] for r in label_batch(texts, tok, get_grouper("words", "hi"), 128)]
            for g in args.groupers:
                gr = get_grouper(g, "hi")
                if hasattr(gr, "prefetch"):
                    gr.prefetch(texts)
                labels[g] = [r["group_id"] for r in label_batch(texts, tok, gr, 128)]
            enc = [r["input_ids"] for r in label_batch(texts, tok, get_grouper("words", "hi"), 128)]
        # pass 1: hidden states per layer (kept on CPU in fp32, mean-centred per layer)
        hs = {L: [] for L in args.layers}
        dec = model._decoder()
        with torch.no_grad():
            for ids in enc:
                x = torch.tensor([ids], device=device)
                with amp():
                    out = dec(input_ids=x, output_hidden_states=True)
                    heads = model(input_ids=x).aux["head_hidden"] if any(L.startswith("h") for L in args.layers) else None
                for L in args.layers:
                    h = heads[int(L[1:])][0] if L.startswith("h") else out.hidden_states[int(L)][0]
                    hs[L].append(h.float().cpu().half())
        stats[name] = {g: {} for g in args.groupers}
        for L in args.layers:
            mu = torch.cat(hs[L]).float().mean(0)
            for g in args.groupers:
                rows = []
                shift = int(L[1:]) + 1 if L.startswith("h") else 0   # head D at t predicts token t+D+1
                for h, w, gid in zip(hs[L], word_ids, labels[g]):
                    if shift:
                        h, w, gid = h[:len(h) - shift], w[shift:], gid[shift:]
                    h = torch.nn.functional.normalize(h.float() - mu, dim=-1)
                    cos = (h[:-1] * h[1:]).sum(-1).tolist()
                    r = [0.0, 0, 0.0, 0]
                    for t, c in enumerate(cos):
                        if min(w[t], w[t + 1], gid[t], gid[t + 1]) < 0 or w[t] == w[t + 1]:
                            continue
                        if gid[t] == gid[t + 1]:
                            r[0] += c; r[1] += 1
                        else:
                            r[2] += c; r[3] += 1
                    rows.append(r)
                stats[name][g][L] = rows
        del model, hs
        torch.cuda.empty_cache()
        print(f"{name}: " + "  ".join(f"{g} " + "/".join(f"{coherence(stats[name][g][L]):.4f}" for L in args.layers)
                                      for g in args.groupers), flush=True)

    res = {"layers": args.layers, "n_sentences": len(texts), "coherence": {}, "vs_base": {}, "pairs": {}}
    for name in stats:
        res["coherence"][name] = {g: {L: coherence(stats[name][g][L]) for L in args.layers} for g in args.groupers}
        res["n_pairs"] = {g: {"within": sum(r[1] for r in stats[name][g][args.layers[0]]),
                              "across": sum(r[3] for r in stats[name][g][args.layers[0]])} for g in args.groupers}
    comps = [(n, args.base) for n in stats if n != args.base and args.base in stats] + \
            [tuple(p.split("-")) for p in args.pairs]
    for a, b in comps:
        key = f"{a}-{b}"
        res["pairs"][key] = {g: {L: boot_diff(stats[a][g][L], stats[b][g][L]) for L in args.layers} for g in args.groupers}
    print("\ncoherence = within-group minus across-boundary cosine (centred), adjacent cross-word pairs")
    for g in args.groupers:
        print(f"\n[{g}]  layers {args.layers}   n within/across = {res['n_pairs'][g]['within']}/{res['n_pairs'][g]['across']}")
        for name in stats:
            print(f"  {name:6s} " + "  ".join(f"{res['coherence'][name][g][L]:+.4f}" for L in args.layers))
        for key, v in res["pairs"].items():
            print(f"  {key:10s} " + "  ".join(f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if lo > 0 or hi < 0 else ' '}"
                                            for d, lo, hi in (v[g][L] for L in args.layers)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
