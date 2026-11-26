"""Tree speculative decoding (generate_tree): output identical to greedy, oracle-consistent acceptance,
cache = prefix + accepted path."""

import random
from types import SimpleNamespace

import pytest
import torch

from mtp.eval import tree_decode as td
from mtp.eval.spec_decode import greedy_generate
from mtp.eval.tree_policy import EntropyTree, StaticTree, accepted_length, nodes_from_prefixes, widths_tree
from tests.test_spec_decode import _tiny_model


def garbage_tree(seed, max_nodes=15, k=4, max_rank=20):
    """A prefix-closed tree of random (mostly wrong) ranks."""
    rng = random.Random(seed)
    prefixes = set()
    while len(prefixes) < max_nodes:
        p = tuple(rng.randrange(max_rank) for _ in range(rng.randint(1, k - 1)))
        prefixes.update(p[:j] for j in range(1, len(p) + 1))
    return StaticTree(nodes_from_prefixes(prefixes))


def policies():
    entropy = EntropyTree(medians=[0.0, 1e9, 1e9], trees={1: widths_tree((2, 2, 1))}, fallback=widths_tree((1, 1, 1)))
    return [StaticTree.from_widths((3, 2, 1)), StaticTree.from_widths((1, 1, 1)), StaticTree.from_widths((5, 1, 1)),
            garbage_tree(0), garbage_tree(1), entropy]


def prompts(tok, n, seed):
    rng = random.Random(seed)
    return [[rng.randrange(4, len(tok)) for _ in range(rng.randint(1, 8))] for _ in range(n)]


@pytest.mark.parametrize("head_type", ["linear", "resblock"])
def test_tree_outputs_identical_to_greedy_on_20_prompts(tiny_model_dir, tmp_path, head_type):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
    for prompt in prompts(tok, 20, 1):
        ref, _ = greedy_generate(model, tok, prompt, 16)
        for policy in policies():
            ids, st = td.generate_tree(model, tok, prompt, 16, policy)
            assert ids == ref, (policy.name, prompt)
            assert st["tokens"] == len(ref) == sum(st["spans"]) and st["forward_passes"] <= len(ref) + 1
            assert len(st["nodes"]) == st["steps"]


def test_acceptance_matches_the_offline_oracle(tiny_model_dir, tmp_path):
    """Per step, the engine accepts exactly accepted_length(nodes, ranks of the greedy continuation)."""
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    checked = 0
    for prompt in prompts(tok, 10, 2):
        ref, _ = greedy_generate(model, tok, prompt, 24)
        for policy in (StaticTree.from_widths((3, 2, 1)), StaticTree.from_widths((8, 4, 2)), garbage_tree(3)):
            trace = []
            ids, _ = td.generate_tree(model, tok, prompt, 24, policy, trace=trace)
            assert ids == ref
            for step in trace:
                ranks = []
                for d in range(1, len(step["head_logits"])):
                    if step["start"] + d >= len(ref):
                        break
                    order = step["head_logits"][d].argsort(descending=True).tolist()
                    ranks.append(order.index(ref[step["start"] + d]))
                assert step["accepted"] == accepted_length(step["nodes"], ranks)
                checked += 1
    assert checked > 50


def test_cache_holds_prefix_plus_path(tiny_model_dir, tmp_path, monkeypatch):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    lengths, real = [], td._keep_path

    def spy(cache, n_tree, rows):
        before = cache.get_seq_length() - n_tree
        real(cache, n_tree, rows)
        lengths.append((cache.get_seq_length(), before + len(rows)))

    monkeypatch.setattr(td, "_keep_path", spy)
    td.generate_tree(model, tok, [5, 6, 7], 20, StaticTree.from_widths((3, 2, 1)))
    assert lengths and all(a == b for a, b in lengths)


def test_ignore_eos_runs_to_max_new_tokens(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    for prompt in prompts(tok, 5, 3):
        eos_tok = SimpleNamespace(eos_token_id=greedy_generate(model, tok, prompt, 1)[0][0])
        assert len(td.generate_tree(model, eos_tok, prompt, 16, StaticTree.from_widths((3, 2, 1)))[0]) == 1
        ref, _ = greedy_generate(model, eos_tok, prompt, 16, ignore_eos=True)
        ids, _ = td.generate_tree(model, eos_tok, prompt, 16, StaticTree.from_widths((3, 2, 1)), ignore_eos=True)
        assert ids == ref and len(ids) == 16


def test_seq_heads_tree_identical_to_greedy(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "seq")
    for prompt in prompts(tok, 20, 4):
        ref, _ = greedy_generate(model, tok, prompt, 16)
        for policy in policies():
            ids, st = td.generate_tree(model, tok, prompt, 16, policy)
            assert ids == ref, (policy.name, prompt)


def test_seq_heads_acceptance_matches_conditional_oracle(tiny_model_dir, tmp_path):
    """Seq heads: ranks are conditional on the path, i.e. head d's rank of the greedy token given the
    greedy tokens before it (what collect_ranks measures teacher-forced). The engine must accept
    accepted_length(nodes, those ranks) per step, and the chain tree must reproduce draft_chain."""
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "seq")
    embed, lm_head = model.base_model.get_input_embeddings(), model._lm_head()
    checked = 0
    for prompt in prompts(tok, 8, 5):
        ref, _ = greedy_generate(model, tok, prompt, 24)
        for policy in (StaticTree.from_widths((3, 2, 1)), StaticTree.from_widths((1, 1, 1)), garbage_tree(6)):
            trace = []
            assert td.generate_tree(model, tok, prompt, 24, policy, trace=trace)[0] == ref
            for step in trace:
                s, t, ranks = step["hidden"], ref[step["start"]], []
                with torch.no_grad():
                    for d in range(1, model.num_heads):
                        if step["start"] + d >= len(ref):
                            break
                        s = model.extra_heads[d - 1](s, embed(torch.tensor([t]))[0])
                        lg = lm_head(s)
                        t = ref[step["start"] + d]
                        ranks.append(int((lg > lg[t]).sum()))
                assert step["accepted"] == accepted_length(step["nodes"], ranks)
                checked += 1
    assert checked > 40


def test_tree_mask():
    mask = td._tree_mask([-1, 0, 0, 1], prefix_len=2, device="cpu")[0, 0]
    assert mask.tolist() == [[True, True, True, False, False, False],
                             [True, True, True, True, False, False],
                             [True, True, True, False, True, False],
                             [True, True, True, True, False, True]]


def test_batched_draft_tokens_match_tree_policy():
    from mtp.eval.tree_policy import draft_tokens

    torch.manual_seed(0)
    logits = [torch.randn(50) for _ in range(4)]
    for policy in policies():
        nodes = policy.nodes if hasattr(policy, "nodes") else policy.fallback
        assert td._draft_tokens(nodes, logits) == draft_tokens(nodes, logits)


# ----------------------------------------------------------------------------- typical acceptance (lossy)

def _check_lossy_trace(trace, eps):
    """Every accepted token is head 0's argmax at its parent or passes the typical threshold there.
    Returns how many were not the argmax."""
    from mtp.eval.typical import typical_threshold

    n = 0
    for step in trace:
        for a, b in zip(step["path"], step["path"][1:]):
            logits, tok = step["verify_logits"][a], step["tokens"][b]
            if int(logits.argmax()) != tok:
                p, thr = typical_threshold(logits, eps)
                assert float(p[tok]) >= thr
                n += 1
    return n


@pytest.mark.parametrize("head_type", ["resblock", "seq"])
def test_lossy_tree_accepts_more_and_only_plausible_tokens(tiny_model_dir, tmp_path, head_type):
    from mtp.eval.typical import TypicalTree

    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
    exact_tok = exact_steps = lossy_tok = lossy_steps = differ = 0
    for prompt in prompts(tok, 8, 21):
        ref, _ = greedy_generate(model, tok, prompt, 16)
        for base in (StaticTree.from_widths((3, 2, 1)), garbage_tree(4)):
            _, sx = td.generate_tree(model, tok, prompt, 16, base)
            trace = []
            ids, st = td.generate_tree(model, tok, prompt, 16, TypicalTree(base, 1e-4), trace=trace)
            assert st["tokens"] == len(ids) == sum(st["spans"])     # may stop early: a lossy draft can be EOS
            assert st["lossy_accepts"] == _check_lossy_trace(trace, 1e-4) and "lossy_accepts" not in sx
            exact_tok, exact_steps = exact_tok + sx["tokens"], exact_steps + sx["steps"]
            lossy_tok, lossy_steps = lossy_tok + st["tokens"], lossy_steps + st["steps"]
            differ += ids != ref
    assert differ > 0 and lossy_tok / lossy_steps > exact_tok / exact_steps


@pytest.mark.parametrize("head_type", ["resblock", "seq"])
def test_lossy_tree_with_nothing_passing_is_greedy(tiny_model_dir, tmp_path, head_type):
    """delta huge: the threshold is min(eps=1, ...) = 1, which no non-argmax token reaches."""
    from mtp.eval.typical import TypicalTree

    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
    for prompt in prompts(tok, 6, 22):
        ref, _ = greedy_generate(model, tok, prompt, 16)
        for base in policies():
            ids, st = td.generate_tree(model, tok, prompt, 16, TypicalTree(base, 1.0, delta=1e9))
            assert ids == ref and st["lossy_accepts"] == 0


@pytest.mark.parametrize("head_type", ["resblock", "seq"])
def test_lossy_tree_with_prompt_lookup_inside(tiny_model_dir, tmp_path, head_type):
    """Canonical nesting (evaluate.py): TypicalTree(PromptLookup(tree)); the chains are grafted and judged."""
    from mtp.eval.prompt_lookup import PromptLookup
    from mtp.eval.typical import TypicalTree

    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
    rng, grafted = random.Random(7), 0
    for _ in range(6):
        unit = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 4))]
        prompt = unit * rng.randint(2, 4)
        ref, _ = greedy_generate(model, tok, prompt, 16)
        lookup = PromptLookup(StaticTree.from_widths((1, 1, 1)), chains=4, length=6)
        assert td.generate_tree(model, tok, prompt, 16, TypicalTree(lookup, 1.0, delta=1e9))[0] == ref
        trace = []
        _, st = td.generate_tree(model, tok, prompt, 16, TypicalTree(lookup, 0.05), trace=trace)
        assert st["lossy_accepts"] == _check_lossy_trace(trace, 0.05)
        grafted += sum(len(t["tokens"]) > len(t["nodes"]) + 1 for t in trace)
    assert grafted > 0


def test_wrapped_static_tree_keeps_seq_fast_path(tiny_model_dir, tmp_path, monkeypatch):
    """Seq heads skip draft_chain for a StaticTree; a TypicalTree / SampledTree around one must too."""
    from mtp.eval.sampling import SampledTree
    from mtp.eval.typical import TypicalTree

    model, tok = _tiny_model(tiny_model_dir, tmp_path, "seq")
    tree = StaticTree.from_widths((3, 2, 1))
    ref = td.generate_tree(model, tok, [5, 6, 7], 12, tree)[0]

    def boom(*a, **kw):
        raise AssertionError("draft_chain called for a static tree")

    monkeypatch.setattr(model, "draft_chain", boom)
    assert td.generate_tree(model, tok, [5, 6, 7], 12, TypicalTree(tree, 1.0, delta=1e9))[0] == ref
    td.generate_tree(model, tok, [5, 6, 7], 12, SampledTree(tree, 0.7, 0.9))
    with pytest.raises(AssertionError, match="draft_chain"):                 # an EntropyTree needs the logits
        td.generate_tree(model, tok, [5, 6, 7], 12, TypicalTree(policies()[-1], 0.1))
