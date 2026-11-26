"""
Pre-cast linear weights for fp16-autocast inference (drafter experiments DR-1 / DR-2).

On a T4 the runs keep fp32 weights and compute under fp16 autocast. Autocast only caches the fp32 -> fp16 cast of
weights that require grad, so for a frozen model every nn.Linear converts its weight again on every call: DR-1
measured the LM head at 2.01 ms per draft step with the cast vs 0.54 ms with a weight converted once (about 72% of
the draft chain's GPU time), and the same cost hits every layer of every verify and greedy pass.

`fast_linear(model)` swaps in a cached fp16 copy of every nn.Linear weight (and bias) for the duration of the block and
restores the fp32 tensors afterwards. Under autocast a linear layer casts its weight to fp16 anyway, so outputs are
bit-identical; embeddings and norms stay fp32, so the residual stream is unchanged. A linear layer whose weight is
shared with an embedding (tied LM head: Qwen2.5, Gemma) cannot have its weight swapped without changing the embedding, so
its forward is redirected to the cached copy instead (F.linear on the fp16 weight; the shared fp32 tensor is untouched).
For a 152k-vocabulary tied head that cast is the single most expensive one. Use it only together with fp16 autocast:
outside autocast an fp32 input meets an fp16 weight. The fp32 copies stay in memory (one extra fp16 copy of the linear
weights).
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_linears(model):
    """(untied nn.Linear layers, nn.Linear layers whose weight is an embedding's), each module once."""
    tied = {m.weight.data_ptr() for m in model.modules() if isinstance(m, nn.Embedding)}
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return [m for m in lin if m.weight.data_ptr() not in tied], [m for m in lin if m.weight.data_ptr() in tied]


def _linears(model):
    return _split_linears(model)[0]


def _cached(m, dtype):
    cache = getattr(m, "_fast_linear_cache", None)
    if cache is None or cache[0].dtype != dtype or cache[0].shape != m.weight.shape:
        cache = (m.weight.detach().to(dtype), None if m.bias is None else m.bias.detach().to(dtype))
        m._fast_linear_cache = cache
    return cache


@contextlib.contextmanager
def fast_linear(model, dtype=torch.float16):
    """Within the block, every nn.Linear of `model` computes from a cached `dtype` copy of its weight (untied layers by
    a weight swap, tied ones by a redirected forward). Yields the number of layers covered."""
    layers, tied = _split_linears(model)
    saved = []
    with torch.no_grad():
        for m in layers:
            cache = _cached(m, dtype)
            saved.append((m.weight.data, None if m.bias is None else m.bias.data))
            m.weight.data = cache[0]
            if m.bias is not None:
                m.bias.data = cache[1]
        for m in tied:
            w, b = _cached(m, dtype)
            m.forward = lambda x, w=w, b=b: F.linear(x, w, b)    # instance attribute: shadows the class forward
    try:
        yield len(layers) + len(tied)
    finally:
        for m, (w, b) in zip(layers, saved):
            m.weight.data = w
            if b is not None:
                m.bias.data = b
        for m in tied:
            m.__dict__.pop("forward", None)


def clear_cache(model):
    for m in model.modules():
        if hasattr(m, "_fast_linear_cache"):
            del m._fast_linear_cache


def with_fast_linear(amp, model, dtype=torch.float16):
    """Wrap an autocast context factory `amp` (e.g. lambda: autocast_ctx(cfg)) so it also swaps in the fp16 weights."""
    @contextlib.contextmanager
    def ctx():
        with amp(), fast_linear(model, dtype):
            yield
    return ctx
