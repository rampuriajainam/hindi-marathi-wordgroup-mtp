"""Offline probe: are linguistic word groups better draft units than frequent n-grams or random groups?

No model needed. Each table holds multi-word entries (>= 2 words) counted in IndicCorp train text; for every token
prefix of an entry (starting at its first word) it keeps the most frequent completions. At decode time the drafter
tries the suffixes of the text that start at one of the last `max_words` word starts and proposes up to `chains`
completions, longest prefix first, below the root (the verifier's own token), as the lexicon does.

Tables of the same size N (entries):
  groups:<grouper>   most frequent multi-word groups of a grouper (hi_rules_v1, karthika, mr_rules_v1)
  ngrams             most frequent word n-grams ignoring group boundaries, with the same word-length histogram
  random:<grouper>   most frequent multi-word groups of a random grouper with a matched size histogram

Base: the per-step spans saved in an eval JSON (exact decoding, so the output is the greedy text for every policy).
Combined: at each root the step accepts max(base, table) drafted tokens. Where the walk lands inside a base step, the
base is assumed to reach the end of that step (it cannot be replayed without the model), so the gain is an estimate.

CAUTION: this replay only drafts at the ROOT. Grafting under head nodes, where most of the gain is, needs the model:
for the Qwen lexicon this replay gives +0.016 tokens/step against +0.35 measured. Use it as a lower bound only; the
real numbers come from `evaluate.py --lexicon <tables from scripts/build_group_table.py>`.

  python scripts/group_table_probe.py --eval results/Rsd_hi_k4_frozen_sd_seq/eval_indiccorp_eval.json \
      --tokenizer LingoIITGN/ganga-1b --lang hi --base tree_static_n64+pld4x6 \
      --tables groups:hi_rules_v1 groups:karthika ngrams random:random_hi_v1 random:random_karthika \
      --out results/group_tables/ganga_hi_indiccorp.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAX_WORDS = 6


def karthika_groups_from_cache(path, n):
    """Word groups of the first n rows of the karthika boundary cache (ganga-1b token labels). The last group of a
    truncated row is dropped."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("LingoIITGN/ganga-1b")
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    d = load_from_disk(str(path))
    out = []
    for r in d.select(range(min(n, len(d)))):
        words = r["text"].split()
        gid_of_word, w = [], -1
        for t, g in zip(r["input_ids"], r["group_id"]):
            if g < 0:
                continue
            if pieces[t].startswith("\u2581") or w < 0:
                w += 1
                gid_of_word.append(g)
        if w + 1 > len(words):
            continue
        groups, cur, last = [], [], None
        for i, g in enumerate(gid_of_word):
            if g != last and cur:
                groups.append(cur)
                cur = []
            cur.append(words[i])
            last = g
        truncated = len(gid_of_word) < len(words)
        if cur and not truncated:
            groups.append(cur)
        out.append(groups)
    return out


def grouped(name, lang, sents, cache=None):
    if name == "karthika":
        return karthika_groups_from_cache(cache, len(sents))
    from mtp.data.grouping.base import get_grouper
    if name == "random_karthika":
        g = get_grouper(name, lang=lang, seed=0)
    else:
        g = get_grouper(name, lang=lang) if not name.startswith("random") else get_grouper(name, lang=lang, seed=0)
    return [g.group_words(s) for s in sents]


def count_groups(groups_per_sent):
    c = Counter()
    for gs in groups_per_sent:
        for g in gs:
            if 2 <= len(g) <= MAX_WORDS:
                c[tuple(g)] += 1
    return c


def top_entries(counter, n):
    return counter.most_common(n)


def ngram_entries(sents, length_hist, min_prefix=3):
    """Top-n_k word k-grams for each k in length_hist (apriori: a k-gram is counted only if its (k-1)-prefix was
    seen >= min_prefix times)."""
    split = [s.split() for s in sents]
    prev, out = None, []
    for k in range(2, max(length_hist) + 1):
        c = Counter()
        for ws in split:
            for i in range(len(ws) - k + 1):
                g = tuple(ws[i:i + k])
                if prev is None or prev.get(g[:-1], 0) >= min_prefix:
                    c[g] += 1
        out += c.most_common(length_hist.get(k, 0))
        prev = c
    return out


def build_table(tok, entries, top=4):
    comp = defaultdict(Counter)
    enc = {}
    for words, c in entries:
        ids = tuple(tok.encode(" " + " ".join(words), add_special_tokens=False))
        enc[words] = ids
        for i in range(1, len(ids)):
            comp[ids[:i]][ids[i:]] += c
    return {p: [list(k) for k, _ in cs.most_common(top)] for p, cs in comp.items()}


def table_draft(seq, i, trie, flags, chains, max_words=3, max_prefix=16):
    starts, j, lo = [], i, max(0, i - max_prefix)
    while j >= lo and len(starts) < max_words:
        if flags[seq[j]]:
            starts.append(j)
        j -= 1
    out = []
    for j in reversed(starts):                       # longest prefix first
        for c in trie.get(tuple(seq[j:i + 1]), []):
            if c not in out:
                out.append(c)
    return out[:chains]


def match(c, seq, k):
    m = 0
    while m < len(c) and k + m < len(seq) and seq[k + m] == c[m]:
        m += 1
    return m


def walk(pp, trie, flags, chains, use_base, use_table):
    """Returns tokens, steps, steps where the table beat the base, extra tokens from the table."""
    T = S = wins = extra = 0
    for pids, ids, spans in zip(pp["prompt_ids"], pp["ids"], pp["spans"]):
        seq = list(pids) + list(ids)
        n0 = len(pids)
        end_of = {}
        pos = n0
        for s in spans:                              # base step covering each position ends at pos+s-1
            for q in range(pos, pos + s):
                end_of[q] = pos + s - 1
            pos += s
        i = n0
        while i < len(seq):
            b = (end_of.get(i, i) - i) if use_base else 0
            t = 0
            if use_table:
                for c in table_draft(seq, i, trie, flags, chains):
                    t = max(t, match(c, seq, i + 1))
            a = max(b, t)
            if t > b:
                wins += 1
                extra += t - b
            S += 1
            T += 1 + a
            i += 1 + a
    return T, S, wins, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--base", default="tree_static_n64+pld4x6")
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000])
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--karthika_cache", default=str(ROOT / "checkpoints/boundary_cache/hi_karthika_train"))
    ap.add_argument("--out")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from mtp.data.corpus import load_split
    from mtp.eval.lexicon import word_start_flags
    random.seed(0)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    flags = word_start_flags(tok)
    sents = load_split(a.lang, "train", n=a.n_train)
    sents = [s if isinstance(s, str) else s["text"] for s in sents]
    evals = {}
    for p in a.eval:
        d = json.load(open(p, encoding="utf-8"))
        e = next(x for x in d["spec_decode"] if x["policy"] == a.base)
        evals[p] = e
    res = {"tokenizer": a.tokenizer, "lang": a.lang, "base": a.base, "n_train": len(sents), "chains": a.chains,
           "note": "estimate: inside a base step the base is assumed to reach that step's end", "evals": {}}
    for p, e in evals.items():
        T, S, _, _ = walk(e["per_prompt"], {}, flags, a.chains, True, False)
        res["evals"][p] = {"base_reported": e["mean_accepted_len"], "base_replayed": T / S, "tables": {}}
        print(f"{Path(p).parent.name}/{Path(p).name}: base {a.base} reported {e['mean_accepted_len']:.3f}, "
              f"replayed {T / S:.3f}")

    counters, cache_groups = {}, {}
    for spec in a.tables:
        kind, _, name = spec.partition(":")
        if kind in ("groups", "random"):
            gs = grouped(name, a.lang, sents, a.karthika_cache)
            counters[spec] = count_groups(gs)
            print(f"{spec}: {len(counters[spec])} distinct multi-word groups")
    ref = next((s for s in a.tables if s.startswith("groups:")), None)
    for N in a.sizes:
        for spec in a.tables:
            if spec == "ngrams":
                hist = Counter(len(w) for w, _ in top_entries(counters[ref], N))
                entries = ngram_entries(sents, dict(hist))
            else:
                entries = top_entries(counters[spec], N)
            trie = build_table(tok, entries)
            lens = Counter(len(w) for w, _ in entries)
            for p, e in evals.items():
                pp = e["per_prompt"]
                T0, S0, _, _ = walk(pp, trie, flags, a.chains, True, False)
                Tt, St, _, _ = walk(pp, trie, flags, a.chains, False, True)
                Tc, Sc, wins, extra = walk(pp, trie, flags, a.chains, True, True)
                r = {"entries": len(entries), "min_count": entries[-1][1] if entries else 0,
                     "words_per_entry": {k: lens[k] for k in sorted(lens)},
                     "table_only_tok_per_step": Tt / St, "combined_tok_per_step": Tc / Sc,
                     "gain_tok_per_step": Tc / Sc - T0 / S0, "steps_table_wins": wins / Sc,
                     "extra_tokens_share": extra / Tc}
                res["evals"][p]["tables"][f"{spec}@{N}"] = r
                print(f"  N={N:6d} {spec:24s} {Path(p).name:26s} table-only {r['table_only_tok_per_step']:.3f}  "
                      f"combined {r['combined_tok_per_step']:.3f} (+{r['gain_tok_per_step']:.3f})  "
                      f"wins {100 * r['steps_table_wins']:.1f}% of steps, min count {r['min_count']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
