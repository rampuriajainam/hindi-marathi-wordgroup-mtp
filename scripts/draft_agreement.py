"""
Offline screen of cheaper draft word predictors (DR-0 / DR-3 / DR-4): no decoding.

    python scripts/draft_agreement.py --run_dir <Rsd_seq run> --counts results/drafter/vocab_rank_hi.json \\
        --topn 2000 4000 8000 16000 --ranks 64 128 256 512 --combo 256:8000

Teacher-forced forward of the run on the IndicCorp eval sentences. For head d the chain state s_d (aux
"head_hidden") is scored by the full LM head and by each candidate predictor (mtp/model/draft_lm.py). Reported per
head: top-1 agreement with the full LM head (the drafts the candidate would change), top-1 accuracy against the
real token t+d+1, and the matmul size relative to the full head. Writes results/{run_name}/draft_agreement.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`


def candidates(weight, counts, topn, ranks, combos):
    from mtp.model.draft_lm import make_draft_lm, svd_factors

    out = [make_draft_lm(weight, "full")]
    out += [make_draft_lm(weight, "topn", n=n, counts=counts) for n in topn]
    factors = {r: svd_factors(weight, r) for r in sorted(set(ranks) | {int(c.split(":")[0]) for c in combos})}
    out += [make_draft_lm(weight, "lowrank", rank=r, factors=factors[r]) for r in ranks]
    for c in combos:
        r, n = (int(x) for x in c.split(":"))
        out.append(make_draft_lm(weight, "lowrank", rank=r, n=n, counts=counts, factors=factors[r]))
    return out


def cost(pred, vocab, hidden):
    if pred.name == "full":
        return 1.0
    if hasattr(pred, "A"):
        return pred.B.shape[0] * (pred.A.shape[0] + hidden) / (vocab * hidden)
    return pred.weight.shape[0] / vocab


def agreement(model, tokenizer, texts, preds, amp, device, max_length=128):
    import torch

    k = model.num_heads
    stats = {p.name: {"agree": [0] * (k - 1), "correct": [0] * (k - 1)} for p in preds}
    total = [0] * (k - 1)
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
            if len(ids) < 3:
                continue
            x = torch.tensor([ids], device=device)
            with amp():
                out = model(x)
                states = out.aux["head_hidden"]                     # [h, s_1, ..., s_{k-1}], each [1, T, H]
                for d in range(1, k):
                    n = len(ids) - d - 1                            # positions with a target t+d+1
                    if n <= 0:
                        continue
                    s = states[d][0, :n]
                    target = x[0, d + 1:d + 1 + n]
                    full = preds[0](s).argmax(-1)
                    total[d - 1] += n
                    for p in preds:
                        top = full if p is preds[0] else p(s).argmax(-1)
                        stats[p.name]["agree"][d - 1] += int((top == full).sum())
                        stats[p.name]["correct"][d - 1] += int((top == target).sum())
    return {name: {"agreement": [a / t for a, t in zip(v["agree"], total)],
                   "top1": [c / t for c, t in zip(v["correct"], total)]} for name, v in stats.items()}, total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--counts", default=None, help="vocab_rank JSON (scripts/vocab_rank.py) for top-N variants")
    ap.add_argument("--topn", type=int, nargs="*", default=[])
    ap.add_argument("--ranks", type=int, nargs="*", default=[])
    ap.add_argument("--combo", nargs="*", default=[], help="RANK:N low-rank restricted to the top N tokens")
    ap.add_argument("--dataset", default="indiccorp_eval")
    ap.add_argument("--n", type=int, default=None, help="cap sentences")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from mtp.data.corpus import load_split
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    model, tokenizer, cfg = load_run(args.run_dir, device=args.device)
    model.eval()
    device = next(model.parameters()).device
    amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else __import__("contextlib").nullcontext
    counts = None
    if args.counts:
        counts = {int(i): c for i, c in json.loads(Path(args.counts).read_text(encoding="utf-8"))["counts"].items()}
    weight = model._lm_head().weight
    preds = candidates(weight, counts, args.topn, args.ranks, args.combo)
    split = {"indiccorp_eval": "eval", "indiccorp_eval_small": "eval_small", "flores_hi": "flores", "flores_mr": "flores"}
    texts = load_split(cfg.lang, split[args.dataset], args.n)
    t0 = time.perf_counter()
    res, total = agreement(model, tokenizer, texts, preds, amp, device, max_length=cfg.data.max_length)
    vocab, hidden = weight.shape
    out = {"run_name": cfg.run_name, "dataset": args.dataset, "n_sentences": len(texts), "positions_per_head": total,
           "candidates": [{"name": p.name, "relative_matmul": cost(p, vocab, hidden), **res[p.name]} for p in preds]}
    dest = Path(args.out) if args.out else ROOT / "results" / cfg.run_name / "draft_agreement.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(texts)} sentences, {total} positions per head, {time.perf_counter() - t0:.0f}s -> {dest}")
    for c in out["candidates"]:
        print(f"  {c['name']:18s} matmul x{c['relative_matmul']:.3f}  agreement "
              + " / ".join(f"{100 * a:.1f}" for a in c["agreement"]) + "  top-1 "
              + " / ".join(f"{100 * a:.1f}" for a in c["top1"]))


if __name__ == "__main__":
    main()
