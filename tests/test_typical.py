import math

import pytest
import torch

from mtp.eval.draft_policy import FixedK
from mtp.eval.spec_decode import generate, greedy_generate
from mtp.eval.typical import Typical, parse_typical, typical_accepts, typical_threshold
from tests.test_spec_decode import TOK, V, CountingModel


def test_argmax_always_accepted():
    logits = torch.tensor([0.0, 9.0, 1.0])
    assert typical_accepts(logits, 1, eps=1e-9)


def test_confident_verifier_rejects_unlikely_draft():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])       # p(0) ~ 1
    assert not typical_accepts(logits, 2, eps=0.3)


def test_unsure_verifier_accepts_plausible_alternative():
    logits = torch.log(torch.tensor([0.55, 0.30, 0.10, 0.05]))
    p, thr = typical_threshold(logits, eps=0.1)
    h = -(p * p.log()).sum().item()
    assert thr == pytest.approx(min(0.1, math.sqrt(0.1) * math.exp(-h)))
    assert typical_accepts(logits, 1, eps=0.1)          # 0.30 >= ~0.1: lossy accept
    assert not typical_accepts(logits, 3, eps=0.1)      # 0.05 < ~0.1


def test_threshold_monotone_in_eps():
    logits = torch.randn(50, generator=torch.Generator().manual_seed(0))
    thr = [typical_threshold(logits, e)[1] for e in (0.3, 0.2, 0.1, 0.05, 0.02)]
    assert thr == sorted(thr, reverse=True)


def test_masked_eos_ignored():
    logits = torch.tensor([2.0, 1.0, float("-inf")])
    p, thr = typical_threshold(logits, eps=0.1)
    assert math.isfinite(thr) and p[2] == 0


def test_parse_typical():
    pol = parse_typical("typical_0.05")
    assert isinstance(pol, Typical) and pol.accept_eps == 0.05 and pol.name == "typical_0.05"
    assert parse_typical("fixed_k") is None


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("use_cache", [True, False])
def test_strict_typical_equals_exact(right, use_cache):
    """Wrong drafts get p ~ 0.005 under the fake verifier: eps 0.3 rejects them, so the output is greedy."""
    model = CountingModel(num_heads=3, right=right)
    ids, st = generate(model, TOK, [7], 12, Typical(0.3), use_cache=use_cache)
    ref, sx = generate(model, TOK, [7], 12, FixedK(), use_cache=use_cache)
    assert ids == ref == greedy_generate(model, TOK, [7], 12, use_cache=use_cache)[0]
    assert st["spans"] == sx["spans"]


@pytest.mark.parametrize("use_cache", [True, False])
def test_loose_typical_accepts_wrong_drafts(use_cache):
    model = CountingModel(num_heads=3, right=False)
    ids, st = generate(model, TOK, [7], 12, Typical(1e-4), use_cache=use_cache)
    ref, sx = generate(model, TOK, [7], 12, FixedK(), use_cache=use_cache)
    assert sx["mean_accepted_len"] == 1.0 and st["mean_accepted_len"] == 3.0
    assert st["accepted"] == st["proposed"] and len(ids) == 12
    assert ids[:3] == [8, 10, 11]             # t0 = 8, then the wrong head 1 / 2 drafts from 7 (x+3, x+4) accepted
    assert ids != ref


def test_typical_tree_wrapper():
    from mtp.eval.prompt_lookup import PromptLookup
    from mtp.eval.tree_policy import StaticTree
    from mtp.eval.typical import TypicalTree

    base = StaticTree.from_widths((2, 1))
    base.name = "tree_static_n4"
    t = TypicalTree(PromptLookup(base, chains=2, length=3), 0.2)
    assert t.name == "tree_static_n4+pld2x3+typ0.2" and t.accept_eps == 0.2 and t.accept_delta is None
    assert t.max_nodes == 4 + 6 and t.tree(None, None, None) == base.nodes and t.extra_chains([1, 2, 1, 2]) == [[1, 2]]
    plain = TypicalTree(base, 0.05, delta=0.3)
    assert plain.name == "tree_static_n4+typ0.05" and plain.accept_delta == 0.3 and not hasattr(plain, "extra_chains")
