"""Build a word-completion lexicon (mtp/eval/lexicon.py) from IndicCorp train text for a run's tokenizer.

  python scripts/build_lexicon.py --tokenizer Qwen/Qwen2.5-1.5B --lang hi --n_train 200000 --out results/lexicon/qwen_hi.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from mtp.data.corpus import load_split
    from mtp.eval.lexicon import build_lexicon, save_lexicon

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sents = load_split(a.lang, "train", n=a.n_train)     # rows >= 1000: disjoint from the eval prompts
    trie = build_lexicon(tok, sents, min_count=a.min_count, top=a.top)
    save_lexicon(a.out, trie, {"tokenizer": a.tokenizer, "lang": a.lang, "n_train": len(sents),
                               "min_count": a.min_count, "top": a.top})
    print(f"{len(trie)} prefixes from {len(sents)} sentences -> {a.out}")


if __name__ == "__main__":
    main()
