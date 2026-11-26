"""
Prompt lookup (n-gram) branches for tree decoding (#39 P2-9, #33): extra draft chains copied from the
text so far, grafted onto the head tree and verified in the same pass by generate_tree.

At each step the context is prompt + generated tokens + t0 (the root). Its last `ngram` tokens (3,
then 2 as a fallback) are searched for earlier in the context; for up to `chains` distinct matches
(most recent first) the `length` tokens that followed become a chain under the root. A chain token
that equals an existing child (a head draft) reuses that node, so the tree only grows by new tokens.
Verification is unchanged, so the output stays identical to greedy decoding.

PromptLookup wraps any TreePolicy (tree_policy.py): .tree() is the head tree, .extra_chains() the copies.
"""


class PromptLookup:
    def __init__(self, base, chains=4, length=6, ngram=(3, 2)):
        self.base, self.chains, self.length, self.ngram = base, int(chains), int(length), tuple(ngram)
        self.name = f"{getattr(base, 'name', 'tree')}+pld{self.chains}x{self.length}"
        self.max_nodes = getattr(base, "max_nodes", 0) + self.chains * self.length

    def tree(self, head_logits, context_ids, step_state):
        return self.base.tree(head_logits, context_ids, step_state)

    def extra_chains(self, ids):
        """ids: context tokens ending with the root t0. Returns up to `chains` token lists (each the
        continuation after an earlier occurrence of the context's last n tokens), longest-match n first."""
        ids = list(ids)
        out, seen = [], set()
        for n in self.ngram:
            if len(ids) <= n:
                continue
            key = ids[-n:]
            for start in range(len(ids) - n - 1, -1, -1):          # most recent earlier occurrence first
                if ids[start:start + n] == key:
                    cont = tuple(ids[start + n:start + n + self.length])
                    if cont and cont not in seen:
                        seen.add(cont)
                        out.append(list(cont))
                        if len(out) >= self.chains:
                            return out
        return out
