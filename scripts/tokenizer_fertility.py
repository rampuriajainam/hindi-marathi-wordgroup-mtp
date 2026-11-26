"""
KT-2 (a): tokens per word of several tokenizers on the Hindi and Marathi evaluation sentences.

    python scripts/tokenizer_fertility.py [--out results/analysis/tokenizer_fertility.json] [--tokenizers ...]

Texts: the IndicCorp eval split (raw rows 0-999, blanks dropped: ~500 sentences) and FLORES devtest (1,012
sentences; set MTP_FLORES_DIR to a folder with flores_devtest_{hin_Deva,mar_Deva}.parquet), both loaded through
mtp/data/corpus.py. Only tokenizer files are downloaded (AutoTokenizer), never model weights. A tokenizer that is
gated or fails to load is recorded under "unavailable" with the error and skipped.

Per (tokenizer, language, dataset) the JSON holds: sentences, words (whitespace split), chars (code points incl.
spaces), tokens (no special tokens), tokens_per_word, tokens_per_char, single_token_word_share, the share of words
with 1 / 2 / 3 / 4+ tokens (tokens_per_word_dist), word_initial_token_share, and diagnostics.

Word-initial is defined from character offsets (return_offsets_mapping), not from a piece prefix such as the
SentencePiece "▁" or the byte-level "Ġ": each token's start is moved past any leading whitespace (SentencePiece-like
tokenizers such as ganga-1b include the leading space in the offsets of "▁word" tokens), the token is assigned to the
whitespace word that contains that start, and the FIRST token assigned to a word is word-initial; every later token of
that word is word-internal. Taking "first token of the word" rather than "start == word start" matters for byte-level
BPE tokenizers: when a token splits a multi-byte character, the fast tokenizer gives every piece of that character
the same character offsets, so several tokens can "start" at the word start (counted in
diagnostics.tokens_sharing_offsets). A whitespace-only token (Mistral emits a lone "▁" before every Devanagari
word, then one token per character) is assigned to the next word, which it opens, so it is that word's initial token
(diagnostics.whitespace_tokens counts them; empty-span tokens and trailing whitespace belong to no word,
diagnostics.unassigned_tokens). A word that contains no
token start (its text was covered by a token that started in an earlier word, i.e. a multi-word token) is counted in
diagnostics.words_without_token_start and as 0 tokens in the distribution's "0" bin. diagnostics.prefix_agreement
compares the offset rule with the piece-prefix rule ("▁" / "Ġ") where the tokenizer uses such a prefix.
"""

import argparse
import bisect
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

TOKENIZERS = [
    "LingoIITGN/ganga-1b",
    "smallstepai/Misal-1B-instruct-v0.1",
    "sarvamai/sarvam-1",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-4B-Base",
    "bigscience/bloom-1b7",
    "ai-forever/mGPT",
    "facebook/xglm-1.7B",
    "google/gemma-3-1b-pt",          # gated: used only if it loads without credentials
    "meta-llama/Llama-3.2-1B",       # gated
    "mistralai/Mistral-7B-v0.3",     # gated
]
DATASETS = [("hi", "eval", "indiccorp_eval"), ("hi", "flores", "flores"),
            ("mr", "eval", "indiccorp_eval"), ("mr", "flores", "flores")]
WORD_PREFIXES = ("▁", "Ġ")


def word_spans(text):
    """[(start, end)] of the whitespace-separated words of text."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def assign_tokens(text, offsets, words=None):
    """Map tokens to whitespace words from character offsets.

    Returns (word_of, initial): word_of[i] is the index of the word that token i belongs to (None for an
    empty-span token or trailing whitespace), initial[i] is True for the first token of its word, False for later
    ones, None when word_of[i] is None. A token's start is moved past leading whitespace first (ganga-style offsets);
    a whitespace-only token is assigned to the next word (it opens that word)."""
    words = word_spans(text) if words is None else words
    starts = [w[0] for w in words]
    word_of, initial, seen = [], [], set()
    for a, b in offsets:
        a0 = a
        while a < b and text[a].isspace():
            a += 1
        if a >= b:
            # whitespace-only token (e.g. Mistral's lone "▁" before every Devanagari word): it opens the next word
            w = bisect.bisect_left(starts, a0) if b > a0 else len(words)
            if w < len(words):
                word_of.append(w)
                initial.append(w not in seen)
                seen.add(w)
            else:
                word_of.append(None)
                initial.append(None)
            continue
        w = bisect.bisect_right(starts, a) - 1
        if w < 0 or a >= words[w][1]:      # cannot happen after the whitespace trim; kept as a guard
            word_of.append(None)
            initial.append(None)
            continue
        word_of.append(w)
        initial.append(w not in seen)
        seen.add(w)
    return word_of, initial


def tokens_per_word(word_of, n_words):
    """Number of tokens assigned to each word (0 for a word covered by a token that started earlier)."""
    counts = [0] * n_words
    for w in word_of:
        if w is not None:
            counts[w] += 1
    return counts


def shared_offsets(offsets):
    """Tokens whose (non-empty) offsets overlap the previous token's: pieces of one split character."""
    n = 0
    for (a0, b0), (a1, b1) in zip(offsets, offsets[1:]):
        if b1 > a1 and a1 < b0:
            n += 1
    return n


def summarize(texts, tok):
    """Fertility stats of tokenizer `tok` (a HF fast tokenizer) on `texts`."""
    n_words = n_tokens = n_chars = 0
    dist = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
    n_init = n_ws = n_shared = n_unassigned = 0
    agree = checked = 0
    for text in texts:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        words = word_spans(text)
        word_of, initial = assign_tokens(text, offsets, words)
        counts = tokens_per_word(word_of, len(words))
        n_words += len(words)
        n_tokens += len(ids)
        n_chars += len(text)
        for c in counts:
            dist[str(c) if c < 4 else "4+"] += 1
        n_init += sum(1 for x in initial if x)
        n_unassigned += sum(1 for x in initial if x is None)
        n_ws += sum(1 for a, b in offsets if b > a and not text[a:b].strip())
        n_shared += shared_offsets(offsets)
        pieces = tok.convert_ids_to_tokens(ids)
        for p, x in zip(pieces, initial):
            if x is None or not isinstance(p, str):
                continue
            checked += 1
            agree += p.startswith(WORD_PREFIXES) == x
    uses_prefix = None
    if checked:
        uses_prefix = agree / checked
    return {
        "sentences": len(texts), "words": n_words, "chars": n_chars, "tokens": n_tokens,
        "tokens_per_word": n_tokens / n_words, "tokens_per_char": n_tokens / n_chars,
        "single_token_word_share": dist["1"] / n_words,
        "tokens_per_word_dist": {k: v / n_words for k, v in dist.items()},
        "word_initial_token_share": n_init / n_tokens,
        "diagnostics": {
            "whitespace_tokens": n_ws, "unassigned_tokens": n_unassigned, "tokens_sharing_offsets": n_shared,
            "words_without_token_start": dist["0"],
            "prefix_agreement": uses_prefix,
        },
    }


def load_texts():
    from mtp.data.corpus import load_split

    texts, skipped = {}, {}
    for lang, split, name in DATASETS:
        try:
            texts[(lang, name)] = load_split(lang, split)
        except Exception as e:  # noqa: BLE001 (recorded, not fatal)
            skipped[f"{lang}/{name}"] = f"{type(e).__name__}: {e}"[:300]
    return texts, skipped


def load_tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name, use_fast=True), None
    except Exception as e:  # noqa: BLE001 (gated or broken: recorded as unavailable)
        return None, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results/analysis/tokenizer_fertility.json"))
    ap.add_argument("--tokenizers", nargs="*", default=TOKENIZERS)
    args = ap.parse_args()

    texts, skipped = load_texts()
    print("texts:", {f"{k[0]}/{k[1]}": len(v) for k, v in texts.items()}, "skipped:", skipped, flush=True)
    results, unavailable = [], {}
    for name in args.tokenizers:
        t0 = time.time()
        tok, err = load_tokenizer(name)
        if tok is None:
            unavailable[name] = err
            print(f"{name}: unavailable ({err})", flush=True)
            continue
        if not getattr(tok, "is_fast", False):
            unavailable[name] = "no fast tokenizer (offsets unavailable)"
            continue
        for (lang, ds), tx in texts.items():
            row = {"tokenizer": name, "tokenizer_class": type(tok).__name__, "vocab_size": len(tok),
                   "lang": lang, "dataset": ds}
            row.update(summarize(tx, tok))
            results.append(row)
        print(f"{name}: done in {time.time() - t0:.1f}s", flush=True)

    out = {
        "description": "KT-2 (a): tokens per word on the eval sentences; word-initial from character offsets "
                       "(see scripts/tokenizer_fertility.py docstring).",
        "datasets": {f"{k[0]}/{k[1]}": len(v) for k, v in texts.items()},
        "datasets_skipped": skipped,
        "unavailable": unavailable,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| tokenizer | lang | dataset | tok/word | 1-tok words | 2 | 3 | 4+ | tok/char | word-initial tokens |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        d = r["tokens_per_word_dist"]
        print(f"| {r['tokenizer']} | {r['lang']} | {r['dataset']} | {r['tokens_per_word']:.2f} | "
              f"{r['single_token_word_share']:.1%} | {d['2']:.1%} | {d['3']:.1%} | {d['4+']:.1%} | "
              f"{r['tokens_per_char']:.3f} | {r['word_initial_token_share']:.1%} |")
    if unavailable:
        print("\nunavailable:", "; ".join(unavailable))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
