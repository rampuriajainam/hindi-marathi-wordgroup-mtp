"""seq_ctx heads (drafter DR-8): start exactly at seq heads, decoding matches teacher forcing, output == greedy."""

import random
from pathlib import Path

import torch
import yaml

from mtp.config import load_config
from mtp.eval.draft_policy import FixedK
from mtp.eval.spec_decode import context_cache, generate, greedy_generate
from mtp.eval.tree_decode import generate_tree
from mtp.eval.tree_policy import StaticTree

ROOT = Path(__file__).resolve().parents[1]


def _build(tiny_model_dir, tmp_path, head_type, perturb=False, seed=0):
    from mtp.model.build import build_model

    d = yaml.safe_load((ROOT / "configs" / "Rsd_seq_ctx.yaml").read_text(encoding="utf-8"))
    d.update(model_name=str(tiny_model_dir), num_heads=4, head_type=head_type, dtype="fp32", ctx_dim=8, ctx_heads=2)
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


def test_starts_exactly_as_seq_heads(tiny_model_dir, tmp_path):
    seq, _ = _build(tiny_model_dir, tmp_path, "seq")
    ctx, _ = _build(tiny_model_dir, tmp_path, "seq_ctx")
    with torch.no_grad():
        ctx.extra_heads.load_state_dict({k: v + 0.1 * torch.randn_like(v) for k, v in seq.extra_heads.state_dict().items()})
        seq.extra_heads.load_state_dict(ctx.extra_heads.state_dict())
        x = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
        a, b = seq(x).logits, ctx(x).logits                       # ctx_out is zero-init: identical outputs
    assert len(a) == len(b) == 4 and all(torch.allclose(u, v, atol=1e-6) for u, v in zip(a, b))
    assert any(k.startswith("ctx_") for k in ctx.head_state_dict())


def test_decode_time_reads_match_teacher_forcing(tiny_model_dir, tmp_path):
    model, _ = _build(tiny_model_dir, tmp_path, "seq_ctx", perturb=True)
    x = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13]])
    embed = model.base_model.get_input_embeddings()
    with torch.no_grad():
        out = model(x)
        for t in range(1, 5):                                     # chain state s_d at position t, true tokens forced
            cache = context_cache(model, out.hidden[0, : t + 1])
            reads = model.ctx_reads(out.hidden[0, t], cache)
            s = out.hidden[0, t]
            for d in range(1, model.num_heads):
                s = model.extra_heads[d - 1](s, embed(x[0, t + d: t + d + 1])[0]) + reads[d - 1]
                assert torch.allclose(s, out.aux["head_hidden"][d][0, t], atol=1e-5), (t, d)
            first = model.draft_chain(out.hidden[0, t], int(x[0, t + 1]), 1, return_logits=True, ctx=cache)[1][0]
            assert torch.allclose(first, out.logits[1][0, t], atol=1e-4)


def test_ctx_decoding_identical_to_greedy(tiny_model_dir, tmp_path):
    model, tok = _build(tiny_model_dir, tmp_path, "seq_ctx", perturb=True)
    rng = random.Random(1)
    for _ in range(10):
        p = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 7))]
        ref, _ = greedy_generate(model, tok, p, 14)
        assert generate(model, tok, p, 14, FixedK())[0] == ref
        for tree in (StaticTree.from_widths((3, 2, 1)), StaticTree.from_widths((1, 1, 1))):
            assert generate_tree(model, tok, p, 14, tree)[0] == ref


def test_ctx_tree_chain_matches_draft_chain(tiny_model_dir, tmp_path):
    """The (1,1,1) tree drafts exactly what draft_chain drafts (shared context reads per depth)."""
    from mtp.eval.tree_decode import _seq_draft_tokens

    model, _ = _build(tiny_model_dir, tmp_path, "seq_ctx", perturb=True)
    x = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.no_grad():
        out = model(x)
        cache = context_cache(model, out.hidden[0])
        h, t0 = out.hidden[0, -1], int(out.logits[0][0, -1].argmax())
        chain = model.draft_chain(h, t0, 3, ctx=cache)
        tree = _seq_draft_tokens(model, h, t0, StaticTree.from_widths((1, 1, 1)).nodes, cache)
    assert chain == tree


def test_train_save_and_load_roundtrip(tiny_model_dir, tmp_path, tiny_tok):
    import importlib.util

    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model
    from mtp.model.checkpoint import load_run

    spec = importlib.util.spec_from_file_location("train_ctx", ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    d = yaml.safe_load((ROOT / "configs" / "Rsd_seq_ctx.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_ctx", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32", ctx_dim=8, ctx_heads=2,
             log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=2, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0", train_file=None, eval_split="eval_small")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    texts = ["मैं कल बाजार जा रहा था। वह घर से आया था", "बच्चे पार्क में खेल रहे हैं। के बारे बात कर"]
    examples = label_batch(texts, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    before = model.ctx_out[0].weight.detach().clone()
    run_dir = tmp_path / "runs" / "tiny_ctx"
    train.train(cfg, model, tok, examples, examples, run_dir)
    assert not torch.equal(before, model.ctx_out[0].weight.detach())         # the context path trains
    loaded, _, _ = load_run(run_dir, device="cpu")
    assert type(loaded).__name__ == "CtxMTPModel"
    assert torch.equal(loaded.ctx_out[0].weight, model.ctx_out[0].weight.detach())
