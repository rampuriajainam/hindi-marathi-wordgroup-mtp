"""seq_wide heads: start exactly at seq heads, decoding matches teacher forcing, output == greedy."""

import random
from pathlib import Path

import pytest
import torch
import yaml

from mtp.config import load_config
from mtp.eval.draft_policy import FixedK
from mtp.eval.spec_decode import generate, greedy_generate
from mtp.eval.tree_decode import _seq_draft_tokens, generate_tree
from mtp.eval.tree_policy import StaticTree

ROOT = Path(__file__).resolve().parents[1]


def _build(tiny_model_dir, tmp_path, head_type, perturb=False, seed=0, **extra):
    from mtp.model.build import build_model

    d = yaml.safe_load((ROOT / "configs" / "Rsd_seq_wide.yaml").read_text(encoding="utf-8"))
    d.update(model_name=str(tiny_model_dir), num_heads=4, head_type=head_type, dtype="fp32", **extra)
    path = tmp_path / f"{head_type}.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    torch.manual_seed(seed)
    model, tok = build_model(load_config(path), "cpu")
    if perturb:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not name.startswith("base_model."):
                    p.add_(torch.randn_like(p) * 0.5)
    return model.eval(), tok


WIDE = [dict(wide_mult=2.0, wide_shared=False), dict(wide_mult=1.5, wide_shared=True), dict(wide_mult=1.0, wide_shared=True),
        dict(wide_mult=2.0, wide_shared=False, wide_expand="linear")]


@pytest.mark.parametrize("opts", [dict(wide_mult=2.0, wide_shared=False), dict(wide_mult=2.0, wide_shared=True)])
def test_starts_exactly_as_seq_heads(tiny_model_dir, tmp_path, opts):
    seq, _ = _build(tiny_model_dir, tmp_path, "seq")
    wide, _ = _build(tiny_model_dir, tmp_path, "seq_wide", **opts)
    x = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
    with torch.no_grad():
        a, b = seq(x).logits, wide(x).logits                # F zero-init, P = first H numbers: identical outputs
    assert len(a) == len(b) == 4 and all(torch.allclose(u, v, atol=1e-6) for u, v in zip(a, b))
    assert wide.wide_dim == 2 * wide.wide_read.out_features and len(wide.extra_heads) == (1 if opts["wide_shared"] else 3)


def test_mult_one_is_rsd_seq(tiny_model_dir, tmp_path):
    """D = H with per-step F: the same parameters as seq heads, the same outputs for any weights."""
    seq, _ = _build(tiny_model_dir, tmp_path, "seq", perturb=True)
    wide, _ = _build(tiny_model_dir, tmp_path, "seq_wide", wide_mult=1.0, wide_shared=False)
    assert wide.wide_up is None and wide.wide_read is None
    wide.extra_heads.load_state_dict(seq.extra_heads.state_dict())
    x = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
    with torch.no_grad():
        assert all(torch.allclose(u, v, atol=1e-6) for u, v in zip(seq(x).logits, wide(x).logits))


@pytest.mark.parametrize("opts", WIDE)
def test_draft_chain_matches_teacher_forcing(tiny_model_dir, tmp_path, opts):
    model, _ = _build(tiny_model_dir, tmp_path, "seq_wide", perturb=True, **opts)
    x = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13]])
    with torch.no_grad():
        out = model(x)
        for t in range(1, 5):
            first = model.draft_chain(out.hidden[0, t], int(x[0, t + 1]), 1, return_logits=True)[1][0]
            assert torch.allclose(first, out.logits[1][0, t], atol=1e-4)


@pytest.mark.parametrize("opts", WIDE)
def test_wide_decoding_identical_to_greedy(tiny_model_dir, tmp_path, opts):
    model, tok = _build(tiny_model_dir, tmp_path, "seq_wide", perturb=True, **opts)
    rng = random.Random(1)
    for _ in range(8):
        p = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 7))]
        ref, _ = greedy_generate(model, tok, p, 14)
        assert generate(model, tok, p, 14, FixedK())[0] == ref
        for tree in (StaticTree.from_widths((3, 2, 1)), StaticTree.from_widths((1, 1, 1))):
            assert generate_tree(model, tok, p, 14, tree)[0] == ref


@pytest.mark.parametrize("opts", WIDE)
def test_tree_chain_matches_draft_chain(tiny_model_dir, tmp_path, opts):
    model, _ = _build(tiny_model_dir, tmp_path, "seq_wide", perturb=True, **opts)
    x = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.no_grad():
        out = model(x)
        h, t0 = out.hidden[0, -1], int(out.logits[0][0, -1].argmax())
        assert model.draft_chain(h, t0, 3) == _seq_draft_tokens(model, h, t0, StaticTree.from_widths((1, 1, 1)).nodes)


def test_train_save_and_load_roundtrip(tiny_model_dir, tmp_path, tiny_tok):
    import importlib.util

    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model
    from mtp.model.checkpoint import load_run

    spec = importlib.util.spec_from_file_location("train_wide", ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    d = yaml.safe_load((ROOT / "configs" / "Rsd_seq_wide.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_wide", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32", wide_mult=2.0,
             wide_shared=True, log_every=1, eval_every=3, save_every=3)
    d["optim"].update(max_steps=3, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0", train_file=None, eval_split="eval_small")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    texts = ["मैं कल बाजार जा रहा था। वह घर से आया था", "बच्चे पार्क में खेल रहे हैं। के बारे बात कर"]
    examples = label_batch(texts, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    up, rd = model.wide_up.weight.detach().clone(), model.wide_read.weight.detach().clone()
    run_dir = tmp_path / "runs" / "tiny_wide"
    train.train(cfg, model, tok, examples, examples, run_dir)
    assert not torch.equal(rd, model.wide_read.weight.detach())          # the read-out of the extra numbers trains
    assert not torch.equal(up, model.wide_up.weight.detach())            # and so does the expansion (no dead dims)
    loaded, _, _ = load_run(run_dir, device="cpu")
    assert type(loaded).__name__ == "WideMTPModel" and loaded.wide_shared
    assert torch.equal(loaded.wide_up.weight, model.wide_up.weight.detach())
