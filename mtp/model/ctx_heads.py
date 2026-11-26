"""
Sequential heads with context attention: head_type "seq_ctx" (drafter experiment DR-8, the continuous control).

Rsd_seq's chain drafts from the last hidden state h_t alone:  s_d = SeqBlock_d(s_{d-1}, e(x_{t+d})).
seq_ctx lets every future slot d read the whole context once more:

    q_d(t) = W_q h_t + Q_d                                    Q_d: learned slot embedding
    o_d(t) = MultiHeadAttn(q_d(t), K = W_k H_{<=t}, V = W_v H_{<=t})   causal: positions <= t only
    s_d    = SeqBlock_d(s_{d-1}, e(x_{t+d})) + W_o,d o_d(t)    W_o,d zero-initialised

H = the backbone's last hidden states (detached; the backbone is frozen in the Rsd line). With W_o zero-initialised the
model starts exactly as Rsd_seq, so any difference after training comes from the context read. The query depends only
on h_t, so at decode time o_d is computed once per step for each slot and shared by every tree node at depth d; the
projected keys / values of accepted positions are cached incrementally (ContextCache).

Decoding engines use the model like head_type "seq" (head_type stays "seq", `uses_context = True`); they keep a
ContextCache and pass it to draft_chain / ctx_reads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtp.model.heads import MTPModel, MTPOutput


class ContextCache:
    """Projected keys / values of the positions seen so far ([n_heads, T, d_head] each) for one sequence."""

    def __init__(self):
        self.k = self.v = None

    def append(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=1)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=1)

    def __len__(self):
        return 0 if self.k is None else self.k.shape[1]


class CtxMTPModel(MTPModel):
    uses_context = True

    def __init__(self, base_model, num_heads, ctx_dim=256, ctx_heads=4, **kw):
        kw.pop("head_type", None)
        super().__init__(base_model, num_heads, head_type="seq", **kw)
        if kw.get("boundary_probes") or kw.get("contrastive_dim"):
            raise ValueError("seq_ctx does not support boundary probes / contrastive heads")
        if ctx_dim % ctx_heads:
            raise ValueError(f"ctx_dim {ctx_dim} must be divisible by ctx_heads {ctx_heads}")
        lm_head = self._lm_head()
        H = lm_head.in_features
        factory = {"dtype": lm_head.weight.dtype, "device": lm_head.weight.device}
        self.ctx_dim, self.ctx_heads = ctx_dim, ctx_heads
        self.ctx_q = nn.Linear(H, ctx_dim, bias=False, **factory)
        self.ctx_k = nn.Linear(H, ctx_dim, bias=False, **factory)
        self.ctx_v = nn.Linear(H, ctx_dim, bias=False, **factory)
        self.ctx_slots = nn.Parameter(torch.randn(num_heads - 1, ctx_dim, **factory) * 0.02)
        self.ctx_out = nn.ModuleList([nn.Linear(ctx_dim, H, bias=False, **factory) for _ in range(num_heads - 1)])
        for o in self.ctx_out:
            nn.init.zeros_(o.weight)

    # ---------------------------------------------------------------------------------------------- shapes
    def _split(self, x):
        """[..., T, ctx_dim] -> [..., n_heads, T, d_head]"""
        *lead, T, _ = x.shape
        return x.view(*lead, T, self.ctx_heads, self.ctx_dim // self.ctx_heads).transpose(-3, -2)

    def _merge(self, x):
        """[..., n_heads, T, d_head] -> [..., T, ctx_dim]"""
        x = x.transpose(-3, -2)
        return x.reshape(*x.shape[:-2], self.ctx_dim)

    # ---------------------------------------------------------------------------------------------- training
    def forward(self, input_ids, attention_mask=None, use_cache: bool = False, extra_heads: bool = True,
                **kw) -> MTPOutput:
        out = super().forward(input_ids, attention_mask=attention_mask, use_cache=use_cache, extra_heads=False, **kw)
        if not extra_heads:
            return out
        hidden = out.hidden
        lm_head = self._lm_head()
        a = self.backbone_grad
        h_in = hidden if a == 1.0 else (hidden.detach() if a == 0.0 else a * hidden + (1 - a) * hidden.detach())
        ctx = hidden.detach()
        K, V = self._split(self.ctx_k(ctx)), self._split(self.ctx_v(ctx))        # [B, nh, T, dh]
        q_base = self.ctx_q(ctx)                                                  # [B, T, ctx_dim]
        emb = self.base_model.get_input_embeddings()(input_ids).detach()
        s = h_in
        logits, head_hidden = list(out.logits), list(out.aux["head_hidden"])
        for d, head in enumerate(self.extra_heads, start=1):
            q = self._split(q_base + self.ctx_slots[d - 1])
            o = self._merge(F.scaled_dot_product_attention(q, K, V.to(q.dtype), is_causal=True))   # [B, T, ctx_dim]
            e = torch.zeros_like(emb)
            if emb.shape[1] > d:
                e[:, : emb.shape[1] - d] = emb[:, d:]                             # x_{t+d} at position t
            s = head(s, e.to(s.dtype)) + self.ctx_out[d - 1](o).to(s.dtype)
            logits.append(lm_head(s))
            head_hidden.append(s)
        out.logits = logits
        out.aux["head_hidden"] = head_hidden
        return out

    # ---------------------------------------------------------------------------------------------- decoding
    @torch.no_grad()
    def ctx_project(self, hidden_rows):
        """hidden_rows [T, H] of newly seen positions -> (k, v) [n_heads, T, d_head] to append to a ContextCache."""
        return self._split(self.ctx_k(hidden_rows)), self._split(self.ctx_v(hidden_rows))

    @torch.no_grad()
    def ctx_reads(self, h, cache, n=None):
        """Per slot d = 1..n, the context read W_o,d o_d for the current position (query from h [H]): list of [H]."""
        n = self.num_heads - 1 if n is None else n
        q = self._split((self.ctx_q(h)[None] + self.ctx_slots[:n]).unsqueeze(1))     # [n, nh, 1, dh]
        k, v = cache.k[None].to(q.dtype), cache.v[None].to(q.dtype)                  # [1, nh, T, dh]
        o = self._merge(F.scaled_dot_product_attention(q, k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)))[:, 0]
        return [self.ctx_out[d](o[d]) for d in range(n)]

    @torch.no_grad()
    def draft_chain(self, h, t0: int, n: int, return_logits: bool = False, ctx=None):
        """As MTPModel.draft_chain, with the context reads added (ctx: ContextCache of the positions <= t)."""
        if ctx is None or len(ctx) == 0:
            raise ValueError("seq_ctx drafting needs the context cache (ctx=ContextCache)")
        reads = self.ctx_reads(h, ctx, n)
        embed, lm_head = self.base_model.get_input_embeddings(), self._lm_head()
        s, tok, out, logits = h, t0, [], []
        for d, head in enumerate(self.extra_heads[:n]):
            e = embed(torch.tensor([tok], device=h.device))[0]
            s = head(s, e.to(s.dtype)) + reads[d].to(s.dtype)
            lg = lm_head(s)
            tok = int(lg.argmax())
            out.append(tok)
            logits.append(lg)
        return (out, logits) if return_logits else out
