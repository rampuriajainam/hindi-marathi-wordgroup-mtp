"""Fit the `random_tokens` group-length histogram (tokens per word) and check how far it is from word boundaries.

    python scripts/fit_token_histogram.py --lang mr --tokenizer smallstepai/Misal-1B-instruct-v0.1 --n 5000

Writes mtp/data/grouping/histograms_tokens_words_{lang}.json. Design check on the eval split: tokens per group for
`words` and `random_tokens`, and how many `random_tokens` group starts fall on a word start (and how many word starts
it reproduces): the two groupings should differ in where boundaries fall, not in how many there are.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="mr")
    ap.add_argument("--tokenizer", default="smallstepai/Misal-1B-instruct-v0.1")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from mtp.data.corpus import load_split
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.token_random import HISTOGRAM_FILE, TokenRandomGrouper, fit_histogram
    from mtp.data.grouping.words import WordGrouper

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train = [t for t in load_split(args.lang, "train", args.n) if t.strip()]
    hist = fit_histogram(tok, train)
    out = Path(str(HISTOGRAM_FILE).format(lang=args.lang))
    out.write_text(json.dumps({"tokenizer": args.tokenizer, "sentences": len(train), "histogram": hist}, indent=1) + "\n",
                   encoding="utf-8")
    mean = sum(k * v for k, v in hist.items())
    print(f"histogram ({len(train)} {args.lang} train sentences): " + ", ".join(f"{k}: {v:.4f}" for k, v in hist.items())
          + f" | mean {mean:.3f} tokens per group -> {out}")

    texts = [t for t in load_split(args.lang, "eval", None) if t.strip()]
    rnd = TokenRandomGrouper(lang=args.lang, tokenizer=tok)
    rows_w = label_batch(texts, tok, WordGrouper(lang=args.lang), 128)
    rows_r = label_batch(texts, tok, rnd, 128)
    n_tok = starts_w = starts_r = both = 0
    for w, r in zip(rows_w, rows_r):
        for gw, gr, sw, sr in zip(w["group_id"], r["group_id"], w["group_start"], r["group_start"]):
            if gw < 0:
                continue
            n_tok += 1
            starts_w += sw
            starts_r += sr
            both += sw and sr
    print(f"eval ({len(texts)} sentences, {n_tok} tokens): tokens/group words {n_tok / starts_w:.3f}, "
          f"random_tokens {n_tok / starts_r:.3f} | random_tokens starts on a word start {100 * both / starts_r:.1f}% "
          f"(chance = share of tokens that start a word: {100 * starts_w / n_tok:.1f}%) | word starts kept "
          f"{100 * both / starts_w:.1f}% | boundary decisions that differ {100 * (starts_w + starts_r - 2 * both) / n_tok:.1f}% of tokens")


if __name__ == "__main__":
    main()
