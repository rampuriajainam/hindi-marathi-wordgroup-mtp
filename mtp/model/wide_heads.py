"""
Sequential heads with a wide working state: head_type "seq_wide" (drafter idea: build, read out, fold back, reiterate).

Rsd_seq drafts in the backbone's own H numbers:  s_d = SeqBlock_d(s_{d-1}, e(x_{t+d})),  logits_d = lm_head(s_d).
seq_wide keeps a bigger state z of D >= H numbers and reads one word out of it per step:

    z_0 = [h_t ; SiLU(U h_t)]                         U: H -> D-H, small random init
    z_d = z_{d-1} + SiLU(F_d [z_{d-1}; e(x_{t+d})] + b_d)   F_d: D+H -> D, zero-init (or one F shared by every step)
    r_d = P z_d                                       P: D -> H, initialised to "take the first H numbers"
    logits_d = lm_head(r_d)

At init F = 0 and P ignores the extra numbers, so the model drafts exactly like Rsd_seq; U is random (not zero) so the
extra numbers carry a signal and get gradient from the first step. U goes through a SiLU: a linear U h_t would only
re-express h_t at step 0 (F's first layer could compute it), while SiLU(U h_t) holds features a single SeqBlock can't
(wide_expand: linear keeps U h_t, to test whether that matters). With D = H
and per-step F there is no U and no P, and the model is exactly Rsd_seq. A verify pass gives a fresh h_t and z_0 is
rebuilt from it, so nothing stale carries over between steps.

Decoding engines use the model like head_type "seq" (head_type stays "seq"); they draft through seq_state / seq_step /
seq_read (tree_decode) or draft_chain (spec_decode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtp.model.heads import MTPModel


class WideStep(nn.Module):
    """z + SiLU(W [z; e] + b) with z of size D and e of size H; zero-initialised (starts as the identity in z)."""

    def __init__(self, wide_dim, hidden_size, dtype=None, device=None):
        super().__init__()
        self.linear = nn.Linear(wide_dim + hidden_size, wide_dim, dtype=dtype, device=device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z, e):
        return z + F.silu(self.linear(torch.cat([z, e], dim=-1)))


class WideMTPModel(MTPModel):
    def __init__(self, base_model, num_heads, wide_mult=2.0, wide_shared=False, wide_expand="silu", **kw):
        kw.pop("head_type", None)
        super().__init__(base_model, num_heads, head_type="seq", **kw)
        if kw.get("boundary_probes") or kw.get("contrastive_dim"):
            raise ValueError("seq_wide does not support boundary probes / contrastive heads")
        lm_head = self._lm_head()
        H = lm_head.in_features
        D = int(round(wide_mult * H))
        if D < H:
            raise ValueError(f"wide_mult {wide_mult} gives D = {D} < H = {H}")
        if wide_expand not in ("silu", "linear"):
            raise ValueError(f"wide_expand must be silu or linear, got {wide_expand!r}")
        factory = {"dtype": lm_head.weight.dtype, "device": lm_head.weight.device}
        self.wide_expand = wide_expand
        self.wide_dim, self.wide_shared = D, bool(wide_shared)
        self.extra_heads = nn.ModuleList([WideStep(D, H, **factory) for _ in range(1 if wide_shared else num_heads - 1)])
        self.wide_up = self.wide_read = None
        if D > H:
            self.wide_up = nn.Linear(H, D - H, bias=False, **factory)
            nn.init.normal_(self.wide_up.weight, std=0.02)
            self.wide_read = nn.Linear(D, H, bias=False, **factory)
            with torch.no_grad():
                self.wide_read.weight.zero_()
                self.wide_read.weight[:, :H] = torch.eye(H, **factory)

    # ---------------------------------------------------------------------------------------------- the three steps
    def seq_state(self, h):
        """z_0 from the backbone state h [..., H]."""
        if self.wide_up is None:
            return h
        u = self.wide_up(h)
        return torch.cat([h, F.silu(u) if self.wide_expand == "silu" else u], dim=-1)

    def seq_step(self, d, z, e):
        """z_d from z_{d-1} and the embedding of the token drafted at depth d-1 (d = 1 .. k-1)."""
        return self.extra_heads[0 if self.wide_shared else d - 1](z, e)

    def seq_read(self, z):
        """The H-dim state the LM head reads."""
        return z if self.wide_read is None else self.wide_read(z)

    # ---------------------------------------------------------------------------------------------- training
    def forward(self, input_ids, attention_mask=None, use_cache: bool = False, extra_heads: bool = True,
                **kw):
        out = super().forward(input_ids, attention_mask=attention_mask, use_cache=use_cache, extra_heads=False, **kw)
        if not extra_heads:
            return out
        hidden = out.hidden
        lm_head = self._lm_head()
        a = self.backbone_grad
        h_in = hidden if a == 1.0 else (hidden.detach() if a == 0.0 else a * hidden + (1 - a) * hidden.detach())
        emb = self.base_model.get_input_embeddings()(input_ids).detach()
        z = self.seq_state(h_in)
        logits, head_hidden = list(out.logits), list(out.aux["head_hidden"])
        for d in range(1, self.num_heads):
            e = torch.zeros_like(emb)
            if emb.shape[1] > d:
                e[:, : emb.shape[1] - d] = emb[:, d:]                         # x_{t+d} at position t
            z = self.seq_step(d, z, e.to(z.dtype))
            r = self.seq_read(z)
            logits.append(lm_head(r))
            head_hidden.append(r)
        out.logits = logits
        out.aux["head_hidden"] = head_hidden
        return out

    # ---------------------------------------------------------------------------------------------- decoding
    @torch.no_grad()
    def draft_chain(self, h, t0: int, n: int, return_logits: bool = False):
        """As MTPModel.draft_chain, through the wide state."""
        embed, lm_head = self.base_model.get_input_embeddings(), self._lm_head()
        z, tok, out, logits = self.seq_state(h), t0, [], []
        for d in range(1, n + 1):
            e = embed(torch.tensor([tok], device=h.device))[0]
            z = self.seq_step(d, z, e.to(z.dtype))
            lg = lm_head(self.seq_read(z))
            tok = int(lg.argmax())
            out.append(tok)
            logits.append(lg)
        return (out, logits) if return_logits else out
