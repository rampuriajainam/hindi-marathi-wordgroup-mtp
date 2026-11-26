"""Repetition-penalty mode: tree decoding must equal penalised greedy decoding exactly."""

import random

import pytest
import torch

from mtp.eval import tree_decode as td
from mtp.eval.penalty import PenalizedTree, apply_penalty
from mtp.eval.prompt_lookup import PromptLookup
from mtp.eval.spec_decode import greedy_generate
from mtp.eval.tree_policy import StaticTree
from mtp.eval.typical import TypicalTree
from tests.test_spec_decode import _tiny_model


def test_apply_penalty_ctrl_rule():
    lg = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_penalty(lg, [0, 1, 1], 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0, 0.5]
    assert lg.tolist() == [2.0, -2.0, 1.0, 0.5]                 # input untouched
    assert apply_penalty(lg, [0], 1.0) is lg


def test_penalised_tree_identical_to_penalised_greedy(tiny_model_dir, tmp_path):
    differs = 0
    for head_type in ("resblock", "seq"):
        model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
        rng = random.Random(11)
        for _ in range(6):
            unit = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 4))]
            prompt = unit * rng.randint(2, 4)
            plain, _ = greedy_generate(model, tok, prompt, 16)
            for r in (1.3, 3.0):
                ref, _ = greedy_generate(model, tok, prompt, 16, penalty=r)
                differs += ref != plain
                for base in (StaticTree.from_widths((1, 1, 1)), StaticTree.from_widths((3, 2, 1))):
                    for pol in (PenalizedTree(base, r), PenalizedTree(PromptLookup(base, chains=4, length=6), r)):
                        ids, _ = td.generate_tree(model, tok, prompt, 16, pol)
                        assert ids == ref, (head_type, r, pol.name, prompt)
    assert differs > 0                                         # the penalty changed some outputs


def test_penalty_rejects_other_modes():
    with pytest.raises(ValueError):
        PenalizedTree(TypicalTree(StaticTree.from_widths((1, 1, 1)), 0.5), 1.2)
    assert td._static(PenalizedTree(PromptLookup(StaticTree([])), 1.2))
