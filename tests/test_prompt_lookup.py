"""Prompt-lookup (n-gram) branches grafted onto tree decoding: still identical to greedy."""

import random

from mtp.eval import tree_decode as td
from mtp.eval.prompt_lookup import PromptLookup
from mtp.eval.spec_decode import greedy_generate
from mtp.eval.tree_policy import StaticTree
from tests.test_spec_decode import _tiny_model


def test_extra_chains_most_recent_match_first_and_fallback():
    pl = PromptLookup(StaticTree.from_widths((1, 1, 1)), chains=3, length=4)
    #            0  1  2  3  4  5  6  7  8  9 10
    ids = [1, 2, 3, 9, 8, 1, 2, 3, 7, 2, 3]
    # last 3 = [7, 2, 3]: no earlier match; last 2 = [2, 3]: matches at 6 (-> 7 2 3) and 1 (-> 9 8 1 2)
    assert pl.extra_chains(ids) == [[7, 2, 3], [9, 8, 1, 2]]
    assert pl.extra_chains([5, 6]) == []
    assert pl.name == "static_tree+pld3x4" and pl.max_nodes == 3 + 12


def test_graft_reuses_head_nodes_and_respects_depth():
    tokens, parents, depths = [10, 11, 12], [-1, 0, 1], [0, 1, 2]     # root 10 -> 11 -> 12 (head chain)
    td._graft(tokens, parents, depths, [[11, 13, 14], [20, 21]], max_depth=2)
    assert tokens == [10, 11, 12, 13, 20, 21]                          # 11 reused; 14 cut by max_depth
    assert parents == [-1, 0, 1, 1, 0, 4] and depths == [0, 1, 2, 2, 1, 2]


def test_prompt_lookup_tree_identical_to_greedy(tiny_model_dir, tmp_path):
    grafted = 0
    for head_type in ("resblock", "seq"):
        model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
        rng = random.Random(7)
        for _ in range(10):
            unit = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 4))]
            prompt = unit * rng.randint(2, 4)                          # repetitive, so lookups hit
            ref, _ = greedy_generate(model, tok, prompt, 16)
            for base in (StaticTree.from_widths((1, 1, 1)), StaticTree.from_widths((3, 2, 1))):
                trace = []
                ids, st = td.generate_tree(model, tok, prompt, 16, PromptLookup(base, chains=4, length=6), trace=trace)
                assert ids == ref, (head_type, base.nodes, prompt)
                grafted += sum(len(t["tokens"]) > len(t["nodes"]) + 1 for t in trace)
    assert grafted > 0                                                 # lookups hit and chains were verified
