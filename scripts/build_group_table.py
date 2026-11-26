"""Build a phrase table of multi-word entries for tree decoding (mtp/eval/lexicon.py, max_words > 1).

Entries are the N most frequent multi-word units of IndicCorp train text, in one of four kinds:
  groups   word groups of --grouper (hi_rules_v1, karthika, mr_rules_v1)
  ngrams   word n-grams with the groups' word-length histogram (--ref_grouper), ignoring group boundaries
  random   groups of a random grouper with a matched size histogram (--grouper random_hi_v1, ...)
For every token prefix of an entry the table keeps its most frequent completions, in the lexicon's JSON format, so
`evaluate.py --lexicon FILE [FILE ...]` grafts them at the root and under head nodes like word completions.

  python scripts/build_group_table.py --kind groups --grouper karthika --tokenizer LingoIITGN/ganga-1b --lang hi \
      --size 50000 --tag gkg --out results/group_tables/ganga_hi_karthika_50k.json
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["groups", "ngrams", "random"], required=True)
    ap.add_argument("--grouper", help="groups/random: the grouper; ngrams: unused")
    ap.add_argument("--ref_grouper", default="hi_rules_v1", help="ngrams: grouper whose length histogram to match")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--size", type=int, default=50000)
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--max_words", type=int, default=3, help="decode-time: prefixes may start at the last N word starts")
    ap.add_argument("--tag", required=True, help="policy-name tag, letters only (e.g. grl, gkg, ngr, rnd)")
    ap.add_argument("--karthika_cache", default=str(ROOT / "checkpoints/boundary_cache/hi_karthika_train"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not a.tag.isalpha():
        raise SystemExit("--tag must be letters only")

    from transformers import AutoTokenizer

    import group_table_probe as g
    from mtp.data.corpus import load_split
    from mtp.eval.lexicon import save_lexicon

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sents = [s if isinstance(s, str) else s["text"] for s in load_split(a.lang, "train", n=a.n_train)]
    if a.kind == "ngrams":
        ref = g.count_groups(g.grouped(a.ref_grouper, a.lang, sents, a.karthika_cache))
        hist = Counter(len(w) for w, _ in g.top_entries(ref, a.size))
        entries = g.ngram_entries(sents, dict(hist))
    else:
        entries = g.top_entries(g.count_groups(g.grouped(a.grouper, a.lang, sents, a.karthika_cache)), a.size)
    trie = g.build_table(tok, entries)
    lens = Counter(len(w) for w, _ in entries)
    save_lexicon(a.out, trie, {"tag": a.tag, "max_words": a.max_words, "kind": a.kind, "grouper": a.grouper,
                               "ref_grouper": a.ref_grouper if a.kind == "ngrams" else None,
                               "tokenizer": a.tokenizer, "lang": a.lang, "n_train": len(sents), "entries": len(entries),
                               "min_count": entries[-1][1] if entries else 0,
                               "words_per_entry": {str(k): lens[k] for k in sorted(lens)}})
    print(f"{a.kind} {a.grouper or ''}: {len(entries)} entries, {len(trie)} prefixes -> {a.out}")


if __name__ == "__main__":
    main()
