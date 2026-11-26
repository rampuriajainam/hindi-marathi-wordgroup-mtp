"""
Typical acceptance (Medusa, Cai et al. 2024) for self-speculative decoding: LOSSY.

The exact engine accepts a draft only if it is head 0's argmax. Here a draft x is also accepted when
head 0 finds it plausible enough:

    p0(x) >= min(eps, delta * exp(-H(p0)))        delta = sqrt(eps)   (Medusa's default pairing)

with p0 head 0's softmax at that position and H its entropy (nats). A confident verifier (low H)
keeps the bar at eps; an unsure one lowers it, so plausible alternatives pass where the next
token is open. The first token of every step (head 0's argmax) is unchanged; once a draft is
rejected, the rest of the step is dropped as in exact decoding. The output is no longer greedy.

`Typical(eps)` is FixedK drafting + this rule; the engine reads `policy.accept_eps`.
Names: typical_<eps> (e.g. typical_0.1), see parse_typical. `TypicalTree(tree, eps)` is the same rule
in the tree decoder (generate_tree), named <tree>+typ<eps>.
"""

import math

import torch

from mtp.eval.draft_policy import FixedK


def typical_threshold(logits: torch.Tensor, eps: float, delta: float | None = None) -> tuple[torch.Tensor, float]:
    """(head 0's probabilities [V], the acceptance threshold) for one position's logits [V]."""
    delta = math.sqrt(eps) if delta is None else delta
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    h = float(-(p * logp).nan_to_num().sum())        # -inf logits (masked EOS) contribute 0
    return p, min(eps, delta * math.exp(-h))


def typical_accepts(logits: torch.Tensor, token: int, eps: float, delta: float | None = None) -> bool:
    """True if `token` passes typical acceptance under head 0's `logits` [V] (argmax always passes)."""
    if int(logits.argmax()) == token:
        return True
    p, thr = typical_threshold(logits, eps, delta)
    return float(p[token]) >= thr


class Typical(FixedK):
    """Propose all k-1 drafts; accept with the typical rule (lossy)."""

    def __init__(self, eps: float, delta: float | None = None):
        self.accept_eps = float(eps)
        self.accept_delta = delta
        self.name = f"typical_{eps:g}"


class TypicalTree:
    """A TreePolicy verified with the typical rule (generate_tree reads `accept_eps`): at a node whose
    children miss head 0's argmax, follow the most likely child that passes. Wrap PromptLookup inside
    (TypicalTree(PromptLookup(tree))), as evaluate.py does: the lookup chains are then children like any."""

    def __init__(self, base, eps: float, delta: float | None = None):
        if getattr(base, "sample", None) is not None:
            raise ValueError("typical acceptance and sampling cannot be combined")
        self.base = base
        self.accept_eps = float(eps)
        self.accept_delta = delta
        self.name = f"{getattr(base, 'name', 'tree')}+typ{eps:g}"
        if hasattr(base, "max_nodes"):
            self.max_nodes = base.max_nodes
        if hasattr(base, "extra_chains"):        # generate_tree grafts chains only if the policy has the method
            self.extra_chains = base.extra_chains

    def tree(self, head_logits, context_ids, step_state):
        return self.base.tree(head_logits, context_ids, step_state)


def parse_typical(name: str):
    """'typical_0.1' -> Typical(0.1); None if `name` is not a typical policy."""
    if not name.startswith("typical_"):
        return None
    return Typical(float(name[len("typical_"):]))
