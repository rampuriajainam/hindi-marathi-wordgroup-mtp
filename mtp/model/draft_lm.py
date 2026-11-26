"""
Cheaper word predictors for the draft chain (drafter experiments DR-3 / DR-4).

The draft heads end in the frozen LM head (V x H, V = 30k for ganga-1b): one matrix-vector product per draft
step. These replacements trade a little draft accuracy for a smaller product; the full model still verifies
every draft, so decoding output is unchanged whatever they predict.

  FullDraftLM(weight)                 logits = W s                                 (the reference)
  TopNDraftLM(weight, ids)            logits over the N kept token ids only        (N x H)
  LowRankDraftLM(A, B, ids=None)      logits = A (B s), A: V' x r, B: r x H        (r (V' + H)), optional top-N rows

Every predictor returns logits over the FULL vocabulary ([..., V], -inf outside its subset), so the draft-token
code and the DraftPolicy / TreePolicy contracts are unchanged. `.score(s)` gives the subset logits and
`.vocab_ids` (or None) maps them back, for callers that want to skip the scatter.
"""

import torch


class FullDraftLM:
    name = "full"

    def __init__(self, weight):
        self.weight = weight                         # V x H
        self.vocab_size = weight.shape[0]
        self.vocab_ids = None

    def score(self, s):
        return s @ self.weight.T

    def __call__(self, s):
        return self.score(s)


class TopNDraftLM:
    def __init__(self, weight, ids):
        self.vocab_size = weight.shape[0]
        self.vocab_ids = torch.as_tensor(ids, dtype=torch.long, device=weight.device)
        self.weight = weight.index_select(0, self.vocab_ids)          # N x H
        self.name = f"top{len(self.vocab_ids)}"

    def score(self, s):
        return s @ self.weight.T

    def __call__(self, s):
        return scatter_full(self.score(s), self.vocab_ids, self.vocab_size)


class LowRankDraftLM:
    def __init__(self, A, B, ids=None, vocab_size=None):
        self.vocab_size = vocab_size or A.shape[0]
        self.vocab_ids = None if ids is None else torch.as_tensor(ids, dtype=torch.long, device=A.device)
        self.A = A if ids is None else A.index_select(0, self.vocab_ids)   # V' x r
        self.B = B                                                           # r x H
        self.name = f"lowrank{B.shape[0]}" + ("" if ids is None else f"_top{len(self.vocab_ids)}")

    def score(self, s):
        return (s @ self.B.T) @ self.A.T

    def __call__(self, s):
        z = self.score(s)
        return z if self.vocab_ids is None else scatter_full(z, self.vocab_ids, self.vocab_size)


def scatter_full(sub_logits, ids, vocab_size):
    """[..., N] logits of token ids -> [..., V] with -inf elsewhere."""
    full = sub_logits.new_full((*sub_logits.shape[:-1], vocab_size), float("-inf"))
    full[..., ids] = sub_logits
    return full


@torch.no_grad()
def svd_factors(weight, rank):
    """Best rank-r factorisation of the LM head (fp32 SVD): W ~ A B, A = U_r S_r (V x r), B = V_r^T (r x H)."""
    w = weight.detach().float()
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    return (U[:, :rank] * S[:rank]).to(weight.dtype), Vh[:rank].to(weight.dtype).contiguous()


def topn_ids(counts, n):
    """Ids of the n most frequent tokens; counts: sequence or dict id -> count."""
    items = counts.items() if isinstance(counts, dict) else enumerate(counts)
    return [int(i) for i, _ in sorted(items, key=lambda kv: (-kv[1], int(kv[0])))[:n]]


def make_draft_lm(weight, kind="full", n=None, rank=None, counts=None, factors=None):
    """kind: full | topn | lowrank. counts: token frequencies (topn, or lowrank restricted to top n);
    factors: precomputed (A, B) for lowrank (else computed by SVD)."""
    if kind == "full":
        return FullDraftLM(weight)
    ids = topn_ids(counts, n) if n else None
    if kind == "topn":
        if ids is None:
            raise ValueError("topn needs counts and n")
        return TopNDraftLM(weight, ids)
    if kind == "lowrank":
        A, B = factors if factors is not None else svd_factors(weight, rank)
        return LowRankDraftLM(A, B, ids=ids, vocab_size=weight.shape[0])
    raise ValueError(f"unknown draft LM kind {kind!r}")
