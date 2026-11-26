"""fast_linear (DR-2): cached low-precision linear weights under autocast give bit-identical outputs."""

import random

import torch

from mtp.eval.draft_policy import FixedK
from mtp.eval.precast import fast_linear, with_fast_linear
from mtp.eval.spec_decode import generate, greedy_generate
from mtp.eval.tree_decode import generate_tree
from mtp.eval.tree_policy import StaticTree
from tests.test_spec_decode import _tiny_model

BF16 = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731  (CPU stand-in for fp16 autocast on T4)


def test_forward_bit_identical_and_weights_restored(tiny_model_dir, tmp_path):
    for head_type in ("resblock", "seq"):
        model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
        x = torch.tensor([[5, 6, 7, 8, 9, 10]])
        with torch.no_grad(), BF16():
            ref = [lg.clone() for lg in model(x).logits]
        with torch.no_grad(), BF16(), fast_linear(model, torch.bfloat16) as n:
            got = model(x).logits
            assert n > 0 and model._lm_head().weight.dtype == torch.bfloat16
        assert all(torch.equal(a, b) for a, b in zip(ref, got))
        assert model._lm_head().weight.dtype == torch.float32                      # restored after the block
        emb = model.base_model.get_input_embeddings().weight
        assert emb.dtype == torch.float32                                         # embeddings untouched


def test_decoding_identical_with_fast_linear(tiny_model_dir, tmp_path):
    for head_type in ("resblock", "seq"):
        model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
        fast = with_fast_linear(BF16, model, torch.bfloat16)
        rng = random.Random(0)
        for _ in range(5):
            p = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 6))]
            assert greedy_generate(model, tok, p, 12, amp=BF16)[0] == greedy_generate(model, tok, p, 12, amp=fast)[0]
            assert generate(model, tok, p, 12, FixedK(), amp=BF16)[0] == generate(model, tok, p, 12, FixedK(), amp=fast)[0]
            tree = StaticTree.from_widths((3, 2, 1))
            assert generate_tree(model, tok, p, 12, tree, amp=BF16)[0] == generate_tree(model, tok, p, 12, tree, amp=fast)[0]
