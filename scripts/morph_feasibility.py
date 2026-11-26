"""Can MorphTok's morpheme segmentations (karthika95/morphTok) give Marathi word-internal groups? No model, no training.

    python scripts/morph_feasibility.py --csv checkpoints/morphtok/marathi.csv --lang mr \
        --tokenizer smallstepai/Misal-1B-instruct-v0.1 --out results/grouping/morph_feasibility_mr.json

On the eval texts (IndicCorp eval + FLORES devtest of --lang):
  coverage   share of word occurrences (punctuation stripped) found in the dictionary, and of those, the share
             with >= 2 morphemes;
  located    share of in-dictionary morpheme boundaries that can be placed in the surface word. Segments are
             matched right to left as suffixes of the word, because sandhi changes the join (वाघिणीसह = वाघीण+सह):
             the suffix usually survives, the stem's last vowel may not;
  aligned    share of located boundaries that fall on a token boundary of --tokenizer (a group boundary inside a
             token cannot be used), and the share of word-internal token boundaries that are morpheme boundaries;
  reach      tokens per word / per morpheme, and how often token t+1 is in the same morpheme as token t.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUNCT = "।॥.,!?;:\"'()[]{}“”‘’-–—…/"
SIGN = lambda ch: "ऀ" <= ch <= "ः" or "ा" <= ch <= "्" or "ॢ" <= ch <= "ॣ"   # dependent signs, virama


def load_dict(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w, seg = row["word"].strip(), row["segmentation"].strip()
            if w and seg:
                d[w] = [s for s in seg.split("+") if s]
    return d


def locate(word, segs):
    """Character offsets (inside `word`) of the morpheme boundaries, matching segments right to left as suffixes.
    Returns (boundaries, n_wanted): boundaries found, and how many the segmentation has (len(segs) - 1)."""
    cuts, end = [], len(word)
    for s in reversed(segs[1:]):
        stem = word[:end]
        if stem.endswith(s) and len(stem) > len(s):
            end -= len(s)
            cuts.append(end)
        else:
            break
    return sorted(cuts), len(segs) - 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lang", default="mr")
    ap.add_argument("--tokenizer", default="smallstepai/Misal-1B-instruct-v0.1")
    ap.add_argument("--n", type=int, default=None, help="sentences per split (default all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from mtp.data.corpus import load_split

    lex = load_dict(args.csv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    res = {"dictionary_words": len(lex), "dictionary_multi_morpheme": sum(len(v) > 1 for v in lex.values()),
           "splits": {}}
    for split in ("eval", "flores"):
        texts = [t for t in load_split(args.lang, split, args.n) if t.strip()]
        c = collections.Counter()
        examples = []
        for text in texts:
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            tok_bounds = {s for s, e in enc["offset_mapping"] if e > s}            # token start chars
            starts = sorted(tok_bounds)
            pos = 0
            for raw in text.split():
                wstart = text.index(raw, pos)
                pos = wstart + len(raw)
                lead = len(raw) - len(raw.lstrip(PUNCT))
                word = raw.strip(PUNCT)
                if not word:
                    continue
                w0 = wstart + lead
                c["words"] += 1
                internal = [s for s in starts if w0 < s < w0 + len(word)]            # token boundaries inside the word
                c["word_internal_token_bounds"] += len(internal)
                c["tokens_in_words"] += len(internal) + 1
                segs = lex.get(word)
                if segs is None:
                    continue
                c["covered"] += 1
                if len(segs) < 2:
                    continue
                c["multi"] += 1
                cuts, wanted = locate(word, segs)
                c["bounds_wanted"] += wanted
                c["bounds_located"] += len(cuts)
                abs_cuts = {w0 + x for x in cuts}
                hit = abs_cuts & set(internal)
                c["bounds_aligned"] += len(hit)
                # sandhi tolerance: a token boundary up to 2 characters before the cut counts if only vowel signs /
                # virama / anusvara lie between (औषधा|ंचा located, tokens ▁औषध|ांचा)
                for x in cuts:
                    a = w0 + x
                    c["bounds_aligned_loose"] += any(
                        a - 2 <= b <= a and all(SIGN(ch) for ch in text[b:a]) for b in internal)
                c["internal_bounds_in_multi_words"] += len(internal)
                c["internal_bounds_that_are_morph"] += len(hit)
                c["morphemes_in_multi"] += len(cuts) + 1
                if len(examples) < 12:
                    pieces = [t for t in tok.convert_ids_to_tokens(tok(word, add_special_tokens=False)["input_ids"])]
                    examples.append({"word": word, "morphtok": "+".join(segs), "located_cuts": cuts,
                                     "tokens": pieces})
        r = {
            "sentences": len(texts), "words": c["words"],
            "coverage": c["covered"] / c["words"],
            "multi_morpheme_share_of_covered": c["multi"] / max(c["covered"], 1),
            "multi_morpheme_share_of_all_words": c["multi"] / c["words"],
            "located": c["bounds_located"] / max(c["bounds_wanted"], 1),
            "aligned_with_token_boundary": c["bounds_aligned"] / max(c["bounds_located"], 1),
            "aligned_with_sandhi_tolerance": c["bounds_aligned_loose"] / max(c["bounds_located"], 1),
            "usable_boundaries_per_100_words_tolerant": 100 * c["bounds_aligned_loose"] / c["words"],
            "internal_token_bounds_that_are_morph": c["internal_bounds_that_are_morph"] / max(c["internal_bounds_in_multi_words"], 1),
            "tokens_per_word": c["tokens_in_words"] / c["words"],
            "word_internal_token_share": c["word_internal_token_bounds"] / c["tokens_in_words"],
            "usable_boundaries_per_100_words": 100 * c["bounds_aligned"] / c["words"],
            "examples": examples,
        }
        res["splits"][split] = r
        print(f"[{split}] {r['sentences']} sentences, {r['words']} words | coverage {100 * r['coverage']:.1f}% | "
              f"multi-morpheme {100 * r['multi_morpheme_share_of_covered']:.1f}% of covered "
              f"({100 * r['multi_morpheme_share_of_all_words']:.1f}% of all words) | boundaries located "
              f"{100 * r['located']:.1f}% | on a token boundary {100 * r['aligned_with_token_boundary']:.1f}% | "
              f"word-internal token bounds that are morph bounds {100 * r['internal_token_bounds_that_are_morph']:.1f}% | "
              f"tokens/word {r['tokens_per_word']:.2f}, word-internal tokens {100 * r['word_internal_token_share']:.1f}% | "
              f"usable boundaries per 100 words {r['usable_boundaries_per_100_words']:.1f} | with sandhi tolerance: "
              f"aligned {100 * r['aligned_with_sandhi_tolerance']:.1f}%, usable per 100 words "
              f"{r['usable_boundaries_per_100_words_tolerant']:.1f}")
        for ex in examples[:8]:
            print("   ", ex["word"], "=", ex["morphtok"], "| cuts", ex["located_cuts"], "| tokens", " ".join(ex["tokens"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
