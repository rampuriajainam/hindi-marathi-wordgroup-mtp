"""Does a Hindi phrase table (corpus n-grams) add accepted tokens on top of the heads and prompt lookup? (Club 1)

    python scripts/phrase_table_oracle.py --run_dir checkpoints/kaggle/Rsd_seq --train_n 200000 --out results/drafter/phrase_table_Rsd_seq.json

Same setting as group_stop_oracle.py: greedy continuations of --n IndicCorp eval prompts, chain/parallel head drafts
from one teacher-forced pass. At each position t (token t+1 = the committed head-0 token) three drafters propose up
to K tokens for t+2..:
  heads  : the model's draft heads;
  lookup : prompt lookup, the latest earlier match of the last 2 tokens in the text so far (as in the engine);
  table  : greedy chain through a trigram->next table built from --train_n training sentences (bigram backoff),
           firing only where the top continuation has p >= --min_p and count >= --min_count.
A step that verifies several drafts (a tree, verify cost ~flat) accepts the longest matching one. Reported: tokens
per step for heads, heads+lookup, heads+table, heads+lookup+table, and what the table's winning tokens are."""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POSTP = set("ने को से में पर का की के तक लिए बारे ओर द्वारा साथ बाद पहले".split())
AUX = set("है हैं था थी थे हो गया गई गए रहा रही रहे जा सकता सकती सकते चुका चुकी चुके दिया दी दिए लिया ली लिए".split())


def build_table(seqs, min_p, min_count):
    tri = collections.defaultdict(collections.Counter)
    bi = collections.defaultdict(collections.Counter)
    for s in seqs:
        for i in range(1, len(s)):
            bi[s[i - 1]][s[i]] += 1
            if i >= 2:
                tri[(s[i - 2], s[i - 1])][s[i]] += 1

    def best(c):
        tok, n = c.most_common(1)[0]
        tot = sum(c.values())
        return tok if n >= min_count and n / tot >= min_p else None

    return ({k: b for k, c in tri.items() if (b := best(c)) is not None},
            {k: b for k, c in bi.items() if (b := best(c)) is not None})


def table_draft(ctx, tri, bi, K):
    a, b, out = ctx[-2], ctx[-1], []
    for _ in range(K):
        nxt = tri.get((a, b), bi.get(b))
        if nxt is None:
            break
        out.append(nxt)
        a, b = b, nxt
    return out


def lookup_draft(ctx, K, n=2):
    key = ctx[-n:]
    for i in range(len(ctx) - n - 1, -1, -1):
        if ctx[i:i + n] == key:
            return ctx[i + n:i + n + K]
    return []


def match(draft, gold):
    j = 0
    while j < len(draft) and j < len(gold) and draft[j] == gold[j]:
        j += 1
    return j


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--new_tokens", type=int, default=96)
    ap.add_argument("--train_n", type=int, default=200000)
    ap.add_argument("--min_p", type=float, default=0.5)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    from mtp.data.corpus import load_split
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.eval.spec_decode import greedy_generate, make_prompts
    from mtp.model.checkpoint import load_run

    model, tok, cfg = load_run(args.run_dir, device=args.device)
    device = next(model.parameters()).device
    amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else __import__("contextlib").nullcontext
    K = len(model.extra_heads)

    train = load_split(cfg.lang, "train", args.train_n)
    tri, bi = build_table(tok(train, add_special_tokens=False)["input_ids"], args.min_p, args.min_count)
    print(f"table from {len(train)} sentences: {len(tri)} trigram + {len(bi)} bigram contexts")

    embed, lm_head = model.base_model.get_input_embeddings(), model._lm_head()
    combos = {"heads": ("heads",), "heads+lookup": ("heads", "lookup"), "heads+table": ("heads", "table"),
              "heads+lookup+table": ("heads", "lookup", "table"), "table_only": ("table",)}
    tot = {c: collections.Counter() for c in combos}
    wins = collections.Counter()
    examples = collections.Counter()
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    with torch.no_grad():
        for p in make_prompts(load_split(cfg.lang, "eval"), tok, n=args.n):
            new, _ = greedy_generate(model, tok, p, args.new_tokens, amp=amp, ignore_eos=True)
            full = p + [int(t) for t in new]
            ids = torch.tensor([full], device=device)
            with amp():
                if model.head_type == "seq":
                    out = model(input_ids=ids, extra_heads=False)
                    s, tokn, drafts = out.hidden[0], out.logits[0][0].argmax(-1), []
                    for head in model.extra_heads[:K]:
                        s = head(s, embed(tokn).to(s.dtype))
                        tokn = lm_head(s).argmax(-1)
                        drafts.append(tokn)
                else:
                    out = model(input_ids=ids)
                    drafts = [out.logits[d + 1][0].argmax(-1) for d in range(K)]
            D = torch.stack(drafts, 0).T.cpu().tolist()                     # [T][K]
            start, end = len(p) - 1, len(full) - 2 - K
            acc = {}
            for t in range(start, end):
                ctx, gold = full[:t + 2], full[t + 2:t + 2 + K]
                td = table_draft(ctx, tri, bi, K)
                acc[t] = {"heads": match(D[t], gold), "lookup": match(lookup_draft(ctx, K), gold), "table": match(td, gold)}
                if acc[t]["table"] > max(acc[t]["heads"], acc[t]["lookup"]):
                    w = vocab[gold[acc[t]["heads"]]].lstrip("▁")
                    wins["postposition" if w in POSTP else "auxiliary" if w in AUX else "other"] += 1
                    examples[" ".join(vocab[x].lstrip("▁") for x in [ctx[-1]] + td[:acc[t]["table"]])] += 1
            for c, srcs in combos.items():
                t = start
                while t < end:
                    a = max(acc[t][s_] for s_ in srcs)
                    tot[c]["steps"] += 1
                    tot[c]["toks"] += 1 + a
                    t += 1 + a
    res = {"run": cfg.run_name, "K": K, "train_n": len(train), "min_p": args.min_p, "min_count": args.min_count,
           "tokens_per_step": {c: v["toks"] / v["steps"] for c, v in tot.items()},
           "table_wins_by_type": dict(wins), "top_table_wins": examples.most_common(30)}
    for c, v in res["tokens_per_step"].items():
        print(f"  {c:20s} tokens/step {v:.3f}")
    print("table beats heads+lookup at:", dict(wins))
    for e, n in res["top_table_wins"][:20]:
        print(f"    {e}  x{n}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
