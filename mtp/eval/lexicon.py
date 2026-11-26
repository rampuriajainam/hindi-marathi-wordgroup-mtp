"""
Word-completion lexicon branches for tree decoding: a Hindi/Marathi word list as a drafter for the inside of words.

With a high-fertility tokenizer (Qwen2.5: ~4.9 tokens per Hindi word) most tokens continue a word that has already
started, and the tokens that can follow a word prefix form a small closed set (the word's stem and its inflections).
build_lexicon counts words in corpus text and stores, for every proper token prefix of a word, its most frequent
completions. LexiconLookup wraps a TreePolicy (or a PromptLookup around one):

  extra_chains(ids)  completions of the word in progress at the root (ids ends with the root t0), plus the wrapped
                     policy's own chains (prompt lookup);
  extend(...)        completions under draft-tree nodes: a head drafts the first piece of the next word, the lexicon
                     the rest. Only the first `ext_nodes` head nodes are extended, `ext` completions each.

Verification is unchanged, so the output stays identical to greedy decoding. No model call is needed for the chains.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def word_start_flags(tokenizer):
    """[vocab] bool: the token begins a new word (it carries the leading-space marker) or is a special token."""
    toks = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    special = set(tokenizer.all_special_ids)
    return [i in special or t is None or t.startswith(("Ġ", "▁", " ", "Ċ")) for i, t in enumerate(toks)]


def build_lexicon(tokenizer, sentences, min_count=2, top=4, min_prob=0.02):
    """{prefix token tuple: [completion token tuples, most frequent first]} for every proper prefix of every
    multi-token word seen at least min_count times. A completion is kept if it has >= min_prob of its prefix's mass."""
    words = Counter()
    for s in sentences:
        words.update(s.split())
    comp = defaultdict(Counter)
    for w, c in words.items():
        if c < min_count:
            continue
        ids = tuple(tokenizer.encode(" " + w, add_special_tokens=False))
        for i in range(1, len(ids)):
            comp[ids[:i]][ids[i:]] += c
    trie = {}
    for p, cs in comp.items():
        total = sum(cs.values())
        keep = [list(k) for k, c in cs.most_common(top) if c >= min_prob * total]
        if keep:
            trie[p] = keep
    return trie


def save_lexicon(path, trie, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"policy": "lexicon", **(meta or {}),
                   "trie": {",".join(map(str, p)): v for p, v in trie.items()}}, f)


def load_lexicon(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {tuple(int(x) for x in p.split(",")): v for p, v in d["trie"].items()}


def load_lexicon_meta(path):
    """The file's metadata without the trie: tag (policy-name prefix, default "lex") and max_words (default 1)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d.pop("trie", None)
    return d


class LexiconLookup:
    """max_words = 1: completions of the word in progress (the lexicon). max_words > 1: a phrase table whose entries
    span several words (scripts/build_group_table.py); prefixes may start at any of the last max_words word starts,
    the longest prefix's completions first. tag names the policy (<tree>+<tag><chains>e<ext>x<ext_nodes>)."""

    def __init__(self, base, trie, word_start, chains=4, ext=2, ext_nodes=12, max_prefix=12, max_words=1, tag="lex"):
        self.base, self.trie, self.word_start = base, trie, word_start
        self.chains, self.ext, self.ext_nodes, self.max_prefix = int(chains), int(ext), int(ext_nodes), int(max_prefix)
        self.max_words = int(max_words)
        self.name = f"{getattr(base, 'name', 'tree')}+{tag}{self.chains}e{self.ext}x{self.ext_nodes}"
        longest = max((len(c) for v in trie.values() for c in v), default=0)
        self.max_nodes = getattr(base, "max_nodes", 0) + (self.chains + self.ext * self.ext_nodes) * longest

    def tree(self, head_logits, context_ids, step_state):
        return self.base.tree(head_logits, context_ids, step_state)

    def completions(self, ids, n):
        """Up to n completions of the word in progress at the end of ids (most frequent first)."""
        if n <= 0:
            return []
        lo = max(0, len(ids) - self.max_prefix)
        starts = [j for j in range(len(ids) - 1, lo - 1, -1) if self.word_start[ids[j]]][:self.max_words]
        out = []
        for j in reversed(starts):                      # longest prefix (earliest word start) first
            for c in self.trie.get(tuple(ids[j:]), []):
                if c not in out:
                    out.append(c)
        return out[:n]

    def extra_chains(self, ids):
        ids = list(ids)
        own = self.base.extra_chains(ids) if hasattr(self.base, "extra_chains") else []
        return own + self.completions(ids, self.chains)

    def extend(self, context, tokens, parents, depths, n_head_rows):
        """[(row, chains)] for ext_nodes head rows (1..n_head_rows-1; row 0 is the root, done by extra_chains).
        Rows on the most likely draft paths come first (smallest sum of sibling ranks along the path, deeper rows
        first on a tie): there a completion extends the draft beyond the depth the heads reach.
        context: tokens before the root."""
        seen, cost = {}, [0] * n_head_rows
        for row in range(1, n_head_rows):                       # sibling index = the head's rank for that node
            rank = seen.get(parents[row], 0)
            seen[parents[row]] = rank + 1
            cost[row] = cost[parents[row]] + rank
        order = sorted(range(1, n_head_rows), key=lambda r: (cost[r], -depths[r]))[:self.ext_nodes]
        out = []
        for row in order:
            path, r = [], row
            while r >= 0:
                path.append(tokens[r])
                r = parents[r]
            chains = self.completions(list(context[-self.max_prefix:]) + path[::-1], self.ext)
            if chains:
                out.append((row, chains))
        return out
