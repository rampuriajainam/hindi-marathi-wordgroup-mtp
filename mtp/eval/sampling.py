"""
Sampling mode for tree decoding (NEXT_PHASE A2): lossless speculative SAMPLING with deterministic drafts.

At a tree node, draw y from head 0's distribution (temperature T, nucleus top_p); if y is a child, move
there, else stop and y is the next step's root. Every emitted token is then a sample from the verifier
given its prefix, so the output distribution is exactly plain ancestral sampling.

To test that exactly, the randomness is tied to the generated position, not to the step: u[i] (one
uniform per position, from a seed) picks token i by inverse CDF, whichever step emits it. Tree sampling
must then output the same tokens as plain one-token-per-pass sampling with the same uniforms
(greedy_generate(..., sample=(T, top_p), sample_seed=s)), as exact tree decoding must equal greedy.
"""

import torch


def position_uniforms(seed: int, n: int) -> list[float]:
    """n uniforms in [0, 1) from a CPU generator: u[i] decides the token at generated index i."""
    return torch.rand(n, generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64).tolist()


def sample_token(logits: torch.Tensor, u: float, temperature: float, top_p: float = 1.0) -> int:
    """Inverse-CDF draw from softmax(logits / T) restricted to the nucleus top_p. Deterministic in u.

    Tokens sorted by probability, descending (stable: ties by token id); the nucleus is the shortest
    prefix whose mass reaches top_p (at least one token; top_p >= 1 = no truncation). -inf logits
    (masked EOS) have probability 0 and are never drawn."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    p = torch.softmax(logits.float() / temperature, dim=-1)
    probs, order = torch.sort(p, descending=True, stable=True)
    cdf = probs.cumsum(0)
    n = int((probs > 0).sum())
    if top_p < 1.0:
        n = min(n, int((cdf < top_p).sum()) + 1)
    n = max(n, 1)
    i = int((cdf[:n] <= u * float(cdf[n - 1])).sum())   # first kept index whose CDF exceeds the target
    return int(order[min(i, n - 1)])                      # min: u ~ 1 with rounding


class SampledTree:
    """A TreePolicy decoded in sampling mode (generate_tree reads `sample`). Same tree, same lookup chains."""

    def __init__(self, base, temperature: float, top_p: float = 1.0):
        if getattr(base, "accept_eps", None) is not None:
            raise ValueError("sampling and typical acceptance cannot be combined")
        self.base = base
        self.sample = (float(temperature), float(top_p))
        self.name = f"{getattr(base, 'name', 'tree')}+samp{temperature:g}p{top_p:g}"
        if hasattr(base, "max_nodes"):
            self.max_nodes = base.max_nodes
        if hasattr(base, "extra_chains"):        # generate_tree grafts chains only if the policy has the method
            self.extra_chains = base.extra_chains

    def tree(self, head_logits, context_ids, step_state):
        return self.base.tree(head_logits, context_ids, step_state)
