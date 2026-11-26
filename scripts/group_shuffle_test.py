"""The shuffle test of Karthika et al. (2025, arXiv 2501.03988, Section 4.1) applied to our models: is a sentence
shuffled *by word groups* closer to the original than one shuffled *by words*, and is that gap bigger for models
trained with word groups?

    python scripts/group_shuffle_test.py --runs R2=checkpoints/kaggle/R2 R3=... --units karthika hi_rules_v0 random \
        --pairs R3-R6a --out results/grouping/shuffle_test.json

FLORES-200 hi devtest (the paper's data). For each sentence, --k seeded shuffles of each kind:
  word shuffle            all whitespace words permuted;
  group shuffle (grouper) the grouper's groups permuted, words inside a group kept in order.
Per model, two measures (the paper used sentence-BERT similarity; an LM gives both of these):
  sim   cosine between the mean-pooled, mean-centred layer-L states of the original and the shuffled sentence;
  nll   head-0 loss per token of the shuffled sentence (how ungrammatical it looks to the model).
Gap (per unit) = sim(group shuffle) - sim(word shuffle), and nll(word shuffle) - nll(group shuffle): positive =
groups are the units the model treats as cohesive. Model differences in gap: paired bootstrap over sentences.
Positive control: R6a / R4a were trained on `random` groups, so their gap with --units random is where a pure
training effect should show.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def shuffled(units, rng):
    u = list(units)
    for _ in range(10):                     # avoid the identity permutation when there is a choice
        rng.shuffle(u)
        if u != list(units) or len(set(map(tuple, units))) < 2:
            break
    return " ".join(w for g in u for w in g)


def boot(vals_a, vals_b, n=2000, seed=0):
    rng = random.Random(seed)
    idx = range(len(vals_a))
    d = sum(a - b for a, b in zip(vals_a, vals_b)) / len(vals_a)
    ds = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        ds.append(sum(vals_a[i] - vals_b[i] for i in pick) / len(pick))
    ds.sort()
    return d, ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--units", nargs="+", default=["karthika", "hi_rules_v0", "random"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--base", default="R2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    from mtp.data.corpus import load_split
    from mtp.data.grouping.base import get_grouper
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    texts = [t for t in load_split("hi", "flores", args.n) if len(t.split()) >= 4]
    groups = {}
    for u in args.units:
        g = get_grouper(u, "hi")
        if hasattr(g, "prefetch"):
            g.prefetch(texts)
        groups[u] = [g.group_words(t) for t in texts]
    variants = []   # per sentence: {"orig": text, "word": [..k], unit: [..k]}
    for i, t in enumerate(texts):
        v = {"orig": t, "word": [shuffled([[w] for w in t.split()], random.Random(f"w{i}.{j}")) for j in range(args.k)]}
        for u in args.units:
            v[u] = [shuffled(groups[u][i], random.Random(f"{u}{i}.{j}")) for j in range(args.k)]
        variants.append(v)
    kinds = ["word"] + args.units
    flat = [(i, kd, j, s) for i, v in enumerate(variants) for kd in ["orig"] + kinds
            for j, s in enumerate([v[kd]] if kd == "orig" else v[kd])]

    per = {}   # run -> {"sim": {kind: [per sentence]}, "nll": {kind: [...]}}
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        model, tok, cfg = load_run(run_dir, device=args.device)
        device = next(model.parameters()).device
        amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else __import__("contextlib").nullcontext
        tok.padding_side = "right"
        emb, nll = {}, {}
        with torch.no_grad():
            for b in range(0, len(flat), args.batch):
                chunk = flat[b:b + args.batch]
                enc = tok([c[3] for c in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128)
                ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
                with amp():
                    out = model._decoder()(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                    logits = model._lm_head()(out.last_hidden_state).float()
                h = out.hidden_states[args.layer].float()
                m = mask.unsqueeze(-1).float()
                pooled = ((h * m).sum(1) / m.sum(1)).cpu()
                lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
                mm = mask[:, 1:].float()
                tok_nll = (-(lp * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu()
                for c, e, n_ in zip(chunk, pooled, tok_nll):
                    emb[c[:3]] = e
                    nll[c[:3]] = float(n_)
        mu = torch.stack([emb[(i, "orig", 0)] for i in range(len(texts))]).mean(0)
        cosf = lambda a, b: float(torch.nn.functional.cosine_similarity(a - mu, b - mu, dim=0))
        sim = {kd: [sum(cosf(emb[(i, "orig", 0)], emb[(i, kd, j)]) for j in range(args.k)) / args.k
                    for i in range(len(texts))] for kd in kinds}
        nl = {kd: [sum(nll[(i, kd, j)] for j in range(args.k)) / args.k for i in range(len(texts))] for kd in kinds}
        nl["orig"] = [nll[(i, "orig", 0)] for i in range(len(texts))]
        per[name] = {"sim": sim, "nll": nl}
        del model
        torch.cuda.empty_cache()
        avg = lambda xs: sum(xs) / len(xs)
        print(f"{name}: sim " + " ".join(f"{kd} {avg(sim[kd]):.4f}" for kd in kinds) +
              " | nll orig %.3f " % avg(nl["orig"]) + " ".join(f"{kd} {avg(nl[kd]):.3f}" for kd in kinds), flush=True)

    avg = lambda xs: sum(xs) / len(xs)
    res = {"n_sentences": len(texts), "k": args.k, "layer": args.layer, "units": args.units, "models": {}, "pairs": {}}
    gap = {}
    for name, p in per.items():
        gap[name] = {u: {"sim": [a - b for a, b in zip(p["sim"][u], p["sim"]["word"])],
                         "nll": [b - a for a, b in zip(p["nll"][u], p["nll"]["word"])]} for u in args.units}
        res["models"][name] = {"sim": {k: avg(v) for k, v in p["sim"].items()}, "nll": {k: avg(v) for k, v in p["nll"].items()},
                               "gap": {u: {m: boot(gap[name][u][m], [0.0] * len(texts)) for m in ("sim", "nll")} for u in args.units}}
    comps = [(n, args.base) for n in per if n != args.base and args.base in per] + [tuple(x.split("-")) for x in args.pairs]
    for a, b in comps:
        res["pairs"][f"{a}-{b}"] = {u: {m: boot(gap[a][u][m], gap[b][u][m]) for m in ("sim", "nll")} for u in args.units}
    star = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]{'*' if t[1] > 0 or t[2] < 0 else ' '}"
    for u in args.units:
        print(f"\nunit = {u}: gap (group shuffle vs word shuffle; + = groups cohesive)   sim | nll")
        for name in per:
            g = res["models"][name]["gap"][u]
            print(f"  {name:6s} {star(g['sim'])}  {star(g['nll'])}")
        for key, v in res["pairs"].items():
            print(f"  {key:10s} {star(v[u]['sim'])}  {star(v[u]['nll'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
