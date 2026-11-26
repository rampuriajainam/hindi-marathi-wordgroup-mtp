"""Offline probe: can a word-completion lexicon draft the inside of Hindi words for a high-fertility tokenizer?

No model needed. Reads the greedy outputs saved in an eval JSON (per_prompt prompt_ids + ids), builds a trie
word-prefix tokens -> most frequent completions from IndicCorp train text, and simulates exact decoding where the only
drafts are the trie's completions of the word in progress (the verifier's own next token is the root, as in the tree).

  python scripts/lexicon_probe.py --eval results/Rsd_qwen_hi_k4_frozen_sd_seq/eval_indiccorp_eval.json \
      --tokenizer Qwen/Qwen2.5-1.5B --lang hi --n_train 200000
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_trie(tok, sentences, min_count=3, top=4):
    words = Counter()
    for s in sentences:
        for w in s.split():
            words[w] += 1
    comp = defaultdict(Counter)                      # prefix tokens -> Counter of completions (tuples)
    n_multi = 0
    for w, c in words.items():
        if c < min_count:
            continue
        ids = tuple(tok.encode(" " + w, add_special_tokens=False))
        if len(ids) < 2:
            continue
        n_multi += 1
        for i in range(1, len(ids)):
            comp[ids[:i]][ids[i:]] += c
    trie = {p: [k for k, _ in cs.most_common(top)] for p, cs in comp.items()}
    return trie, len(words), n_multi


def word_starts(tok, ids):
    toks = tok.convert_ids_to_tokens(ids)
    return [t.startswith(("Ġ", "▁", " ")) for t in toks]


def simulate(seq, n_prompt, starts, trie, chains):
    """Returns (steps, drafted tokens accepted, steps with any candidate)."""
    i, steps, acc, had = n_prompt, 0, 0, 0             # token i is the root (verifier's own token)
    while i < len(seq):
        steps += 1
        j = i
        while j > 0 and not starts[j]:
            j -= 1
        prefix = tuple(seq[j:i + 1])
        best = 0
        cands = trie.get(prefix, [])[:chains]
        had += bool(cands)
        for c in cands:
            m = 0
            while m < len(c) and i + 1 + m < len(seq) and seq[i + 1 + m] == c[m]:
                m += 1
            best = max(best, m)
        acc += best
        i += 1 + best
    return steps, acc, had


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--policy", default="fixed_k")
    ap.add_argument("--out")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from mtp.data.corpus import load_split
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sents = load_split(a.lang, "train", n=a.n_train)
    sents = [s if isinstance(s, str) else s["text"] for s in sents]
    d = json.load(open(a.eval, encoding="utf-8"))
    e = next(x for x in d["spec_decode"] if x["policy"] == a.policy)
    pp = e["per_prompt"]
    res = {"eval": a.eval, "tokenizer": a.tokenizer, "n_train": len(sents), "heads_tok_per_step": e["mean_accepted_len"]}
    for min_count in (2, 5):
        trie, n_words, n_multi = build_trie(tok, sents, min_count=min_count)
        for chains in (1, 4):
            S = A = H = T = 0
            internal = internal_hit = 0
            for pids, ids in zip(pp["prompt_ids"], pp["ids"]):
                seq = list(pids) + list(ids)
                st = word_starts(tok, seq)
                s, acc, had = simulate(seq, len(pids), st, trie, chains)
                S += s; A += acc; H += had; T += len(ids)
                for i in range(len(pids), len(seq)):         # word-internal next-token accuracy of the trie
                    if not st[i]:
                        internal += 1
                        j = i - 1
                        while j > 0 and not st[j]:
                            j -= 1
                        c = trie.get(tuple(seq[j:i]), [])[:chains]
                        internal_hit += any(x[0] == seq[i] for x in c)
            key = f"min{min_count}_chains{chains}"
            res[key] = {"trie_prefixes": len(trie), "vocab_words": n_words, "multi_token_words": n_multi,
                        "tok_per_step": T / S, "steps_with_candidate": H / S,
                        "word_internal_share": internal / T, "word_internal_top_acc": internal_hit / max(internal, 1)}
            print(key, json.dumps(res[key]))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
