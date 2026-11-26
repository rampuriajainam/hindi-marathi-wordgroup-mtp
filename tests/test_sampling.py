"""Sampling mode: sample_token is an exact inverse-CDF nucleus sampler, and tree sampling outputs exactly
what plain one-token-per-pass sampling outputs with the same per-position uniforms."""

import random
from types import SimpleNamespace

import pytest
import torch

from mtp.eval import tree_decode as td
from mtp.eval.prompt_lookup import PromptLookup
from mtp.eval.sampling import SampledTree, position_uniforms, sample_token
from mtp.eval.spec_decode import greedy_generate
from mtp.eval.tree_policy import StaticTree
from mtp.eval.typical import TypicalTree
from tests.test_spec_decode import _tiny_model
from tests.test_tree_decode import garbage_tree, policies, prompts


def test_position_uniforms_deterministic_and_seeded():
    a = position_uniforms(3, 100)
    assert a == position_uniforms(3, 100) and a != position_uniforms(4, 100)
    assert len(a) == 100 and all(0 <= x < 1 for x in a)
    assert position_uniforms(3, 10) == a[:10]          # u[i] does not depend on how many are drawn


def test_small_top_p_or_temperature_gives_argmax():
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        logits = torch.randn(50, generator=gen) * 3
        for u in (0.0, 0.3, 0.999999):
            assert sample_token(logits, u, 1.0, top_p=1e-9) == int(logits.argmax())
            assert sample_token(logits, u, 1e-4, top_p=1.0) == int(logits.argmax())


def test_frequencies_match_the_distribution():
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 0.0])
    for t in (1.0, 0.7):
        p = torch.softmax(logits / t, -1)
        n = 20000
        counts = torch.bincount(torch.tensor([sample_token(logits, u, t) for u in position_uniforms(0, n)]),
                                minlength=5).float()
        assert torch.allclose(counts / n, p, atol=0.012)
        grid = torch.bincount(torch.tensor([sample_token(logits, (i + 0.5) / 1000, t) for i in range(1000)]),
                              minlength=5).float()
        assert torch.allclose(grid / 1000, p, atol=0.002)   # inverse CDF: a uniform grid gives p exactly


def test_nucleus_truncation():
    gen = torch.Generator().manual_seed(1)
    for _ in range(10):
        logits = torch.randn(30, generator=gen) * 2
        p = torch.softmax(logits, -1)
        probs, order = torch.sort(p, descending=True, stable=True)
        n = int((probs.cumsum(0) < 0.6).sum()) + 1
        nucleus = set(order[:n].tolist())
        got = [sample_token(logits, (i + 0.5) / 500, 1.0, top_p=0.6) for i in range(500)]
        assert set(got) == nucleus                          # every nucleus token reachable, nothing outside
        kept = probs[:n] / probs[:n].sum()
        freq = torch.tensor([got.count(t) for t in order[:n].tolist()], dtype=torch.float) / 500
        assert torch.allclose(freq, kept, atol=0.005)       # renormalised within the nucleus


def test_masked_token_never_drawn_and_edges():
    logits = torch.tensor([1.0, 1.0, float("-inf"), 1.0])
    assert all(sample_token(logits, u, 1.0) != 2 for u in (0.0, 0.5, 1 - 1e-12, 0.99999999999999))
    assert sample_token(logits, 1 - 1e-16, 1.0) == 3      # u ~ 1: the last token with mass
    assert sample_token(logits, 0.0, 1.0) == 0              # ties ordered by token id
    with pytest.raises(ValueError):
        sample_token(logits, 0.5, 0.0)


def test_wrapper_names_delegation_and_exclusivity():
    base = StaticTree.from_widths((2, 1, 1))
    base.name = "tree_static_n4"
    s = SampledTree(PromptLookup(base, chains=2, length=3), 0.7, 0.9)
    assert s.name == "tree_static_n4+pld2x3+samp0.7p0.9" and s.sample == (0.7, 0.9)
    assert s.max_nodes == base.max_nodes + 6 and s.extra_chains([1, 2, 1, 2]) == [[1, 2]]
    assert not hasattr(SampledTree(base, 1.0, 1.0), "extra_chains")
    assert SampledTree(base, 1.0, 1.0).name == "tree_static_n4+samp1p1"
    with pytest.raises(ValueError):
        SampledTree(TypicalTree(base, 0.1), 0.7, 0.9)
    with pytest.raises(ValueError):
        TypicalTree(SampledTree(base, 0.7, 0.9), 0.1)


@pytest.mark.parametrize("head_type", ["resblock", "seq"])
def test_tree_sampling_equals_plain_sampling(tiny_model_dir, tmp_path, head_type):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
    accepted = 0
    for j, prompt in enumerate(prompts(tok, 6, 11)):
        for setting in ((1.0, 1.0), (0.7, 0.9), (0.3, 0.8)):
            seed = 100 * j + int(10 * setting[0])
            ref, _ = greedy_generate(model, tok, prompt, 16, sample=setting, sample_seed=seed)
            for base in policies()[:4]:
                trace = []
                ids, st = td.generate_tree(model, tok, prompt, 16, SampledTree(base, *setting), sample_seed=seed,
                                           trace=trace)
                assert ids == ref, (base.name, setting, prompt)
                assert st["tokens"] == sum(st["spans"]) == len(ref) and "lossy_accepts" not in st
                accepted += sum(t["accepted"] for t in trace)
    assert accepted > 0                                    # multi-token steps happened


@pytest.mark.parametrize("head_type", ["resblock", "seq"])
def test_tree_sampling_with_prompt_lookup_equals_plain_sampling(tiny_model_dir, tmp_path, head_type):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
    rng, grafted = random.Random(7), 0
    for j in range(8):
        unit = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 4))]
        prompt = unit * rng.randint(2, 4)
        for setting in ((0.7, 0.9), (0.05, 1.0)):      # near-greedy: the lookups hit
            ref, _ = greedy_generate(model, tok, prompt, 16, sample=setting, sample_seed=j)
            for base in (StaticTree.from_widths((1, 1, 1)), garbage_tree(2)):
                trace = []
                policy = SampledTree(PromptLookup(base, chains=4, length=6), *setting)
                ids, _ = td.generate_tree(model, tok, prompt, 16, policy, sample_seed=j, trace=trace)
                assert ids == ref, (setting, prompt)
                grafted += sum(len(t["tokens"]) > len(t["nodes"]) + 1 for t in trace)
    assert grafted > 0


def test_seeds_change_the_output_and_sample_none_is_greedy(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    prompt = [5, 6, 7]
    outs = {tuple(greedy_generate(model, tok, prompt, 16, sample=(1.0, 1.0), sample_seed=s)[0]) for s in range(5)}
    assert len(outs) > 1
    tree = StaticTree.from_widths((3, 2, 1))
    a = {tuple(td.generate_tree(model, tok, prompt, 16, SampledTree(tree, 1.0, 1.0), sample_seed=s)[0]) for s in range(5)}
    assert a == outs
    assert td.generate_tree(model, tok, prompt, 16, tree, sample_seed=3)[0] == greedy_generate(model, tok, prompt, 16)[0]


def test_sampling_eos_handling(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    tree = StaticTree.from_widths((3, 2, 1))
    for j, prompt in enumerate(prompts(tok, 4, 12)):
        plain, _ = greedy_generate(model, tok, prompt, 16, sample=(1.0, 1.0), sample_seed=j)
        eos_tok = SimpleNamespace(eos_token_id=plain[3])            # stops at its first occurrence
        ref, _ = greedy_generate(model, eos_tok, prompt, 16, sample=(1.0, 1.0), sample_seed=j)
        ids, _ = td.generate_tree(model, eos_tok, prompt, 16, SampledTree(tree, 1.0, 1.0), sample_seed=j)
        assert ids == ref and ids[-1] == plain[3] and len(ids) <= 4
        ref, _ = greedy_generate(model, eos_tok, prompt, 16, sample=(1.0, 1.0), sample_seed=j, ignore_eos=True)
        ids, _ = td.generate_tree(model, eos_tok, prompt, 16, SampledTree(tree, 1.0, 1.0), sample_seed=j,
                                  ignore_eos=True)
        assert ids == ref and len(ids) == 16 and plain[3] not in ids


def test_sample_and_typical_cannot_combine(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    with pytest.raises(ValueError):
        td.generate_tree(model, tok, [5, 6], 8, TypicalTree(StaticTree.from_widths((2, 1, 1)), 0.1), sample=(1.0, 1.0))
