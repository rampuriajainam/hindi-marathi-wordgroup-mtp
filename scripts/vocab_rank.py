"""
Token frequency ranking for top-N draft vocabularies (DR-3).

    python scripts/vocab_rank.py --lang hi --n 100000                       # IndicCorp train text
    python scripts/vocab_rank.py --lang hi --texts /kaggle/working/sd/hi_sd.jsonl   # self-distilled text

Counts the tokenizer's ids over the texts and writes results/drafter/vocab_rank_{lang}.json:
{"source", "n_texts", "n_tokens", "coverage": {N: share of tokens in the top N}, "counts": {id: count}}.
The drafts are the model's own text, so the self-distilled file is the better source when available; IndicCorp
train text is the fallback (the self-distillation prompts come from it).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

MODELS = {"hi": "LingoIITGN/ganga-1b", "mr": "smallstepai/Misal-1B-instruct-v0.1"}


def read_texts(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out.append(row["text"] if isinstance(row, dict) else str(row))
        except json.JSONDecodeError:
            out.append(line)
    return out


def count_tokens(texts, tokenizer, batch=512):
    counts = Counter()
    for i in range(0, len(texts), batch):
        for ids in tokenizer(texts[i:i + batch], add_special_tokens=False)["input_ids"]:
            counts.update(ids)
    return counts


def coverage(counts, ns=(1000, 2000, 4000, 8000, 16000)):
    total = sum(counts.values())
    ranked = [c for _, c in counts.most_common()]
    return {str(n): sum(ranked[:n]) / total for n in ns}


def main(argv=None):
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="hi", choices=sorted(MODELS))
    ap.add_argument("--texts", default=None, help="jsonl/txt of texts (e.g. the self-distilled file)")
    ap.add_argument("--n", type=int, default=100000, help="IndicCorp train sentences when --texts is not given")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.texts:
        texts, source = read_texts(args.texts), str(args.texts)
    else:
        from mtp.data.corpus import load_split

        texts, source = load_split(args.lang, "train", args.n), f"IndicCorpV2 train ({args.lang}, first {args.n})"
    tok = AutoTokenizer.from_pretrained(MODELS[args.lang])
    counts = count_tokens(texts, tok)
    out = {"source": source, "n_texts": len(texts), "n_tokens": sum(counts.values()), "coverage": coverage(counts),
           "counts": {str(i): c for i, c in counts.most_common()}}
    dest = Path(args.out) if args.out else ROOT / "results" / "drafter" / f"vocab_rank_{args.lang}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(texts)} texts, {out['n_tokens']} tokens, {len(counts)} distinct -> {dest}")
    print("coverage: " + ", ".join(f"top {n}: {100 * v:.1f}%" for n, v in out["coverage"].items()))


if __name__ == "__main__":
    main()
