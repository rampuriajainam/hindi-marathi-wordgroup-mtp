"""
Repetition-penalty mode for tree decoding: exact speculative decoding of repetition-penalised greedy output.

Greedy decoding with the CTRL / Hugging Face repetition penalty r: before the argmax, every token already in the
context (prompt + generated so far) has its logit divided by r if positive, multiplied by r if negative. The rule is
deterministic and depends only on the context, so a tree verifier stays exact: at each node the next token is the
argmax of head 0's logits after applying the penalty with that node's own context (the prefix plus the path to the
node). generate_tree(policy=PenalizedTree(...)) must then output exactly greedy_generate(..., penalty=r).
"""

import torch


def apply_penalty(logits: torch.Tensor, context, penalty: float) -> torch.Tensor:
    """logits [V] -> a copy with the CTRL repetition penalty applied to the tokens in `context`."""
    if not penalty or penalty == 1.0 or not context:
        return logits
    out = logits.clone()
    ids = torch.tensor(sorted(set(int(t) for t in context)), device=logits.device)
    v = out[ids]
    out[ids] = torch.where(v > 0, v / penalty, v * penalty)
    return out


class PenalizedTree:
    """A TreePolicy decoded as repetition-penalised greedy (generate_tree reads `penalty`). Same tree and chains."""

    def __init__(self, base, penalty: float):
        if getattr(base, "accept_eps", None) is not None or getattr(base, "sample", None) is not None:
            raise ValueError("the repetition penalty mode cannot be combined with typical acceptance or sampling")
        if penalty <= 0:
            raise ValueError(f"penalty must be > 0, got {penalty}")
        self.base = base
        self.penalty = float(penalty)
        self.name = f"{getattr(base, 'name', 'tree')}+rp{penalty:g}"
        if hasattr(base, "max_nodes"):
            self.max_nodes = base.max_nodes
        if hasattr(base, "extra_chains"):
            self.extra_chains = base.extra_chains

    def tree(self, head_logits, context_ids, step_state):
        return self.base.tree(head_logits, context_ids, step_state)
