"""Prompt lookup without heads (an empty head tree): a lookup-only baseline, still identical to greedy."""

import json
import random

from mtp.eval import tree_decode as td
from mtp.eval.prompt_lookup import PromptLookup
from mtp.eval.spec_decode import greedy_generate
from mtp.eval.tree_policy import StaticTree, load_policy
from tests.test_spec_decode import _tiny_model


def test_empty_tree_policy_loads(tmp_path):
    p = tmp_path / "tree_policy_static_n0.json"
    p.write_text(json.dumps({"policy": "static_tree", "nodes": []}), encoding="utf-8")
    pol = load_policy(p)
    assert pol.nodes == [] and pol.max_nodes == 0


def test_lookup_only_identical_to_greedy(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "seq")
    rng = random.Random(3)
    grafted = 0
    for _ in range(8):
        unit = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 4))]
        prompt = unit * rng.randint(2, 4)
        ref, _ = greedy_generate(model, tok, prompt, 16)
        trace = []
        ids, st = td.generate_tree(model, tok, prompt, 16, PromptLookup(StaticTree([]), chains=4, length=6), trace=trace)
        assert ids == ref, prompt
        grafted += sum(len(t["tokens"]) > 1 for t in trace)
    assert grafted > 0


def test_static_unwraps_prompt_lookup():
    from mtp.eval.tree_decode import _static
    from mtp.eval.typical import TypicalTree

    assert _static(PromptLookup(StaticTree([]), chains=4, length=6))
    assert _static(TypicalTree(PromptLookup(StaticTree.from_widths((2, 1, 1))), 0.5))
