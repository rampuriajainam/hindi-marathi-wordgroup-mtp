"""Drafter experiment tools (DR-0 / DR-3 / DR-4): compare_spec, draft word predictors, draft_agreement, vocab_rank."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mtp.model.draft_lm import (FullDraftLM, LowRankDraftLM, TopNDraftLM, make_draft_lm, scatter_full, svd_factors,
                                topn_ids)

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_json(path, policy, tokens, steps, speedup=1.5):
    rec = {"run_name": "x", "dataset": "indiccorp_eval", "spec_decode": [{
        "policy": policy, "mean_accepted_len": sum(tokens) / sum(steps), "speedup": speedup,
        "accept_rate_per_head": [0.5, 0.3, 0.1], "per_prompt": {"tokens": tokens, "steps": steps}}]}
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def test_compare_spec_identical_and_known_difference(tmp_path):
    cs = _load("compare_spec")
    a = _eval_json(tmp_path / "a.json", "fixed_k", [64, 64, 64, 64], [32, 30, 34, 32])
    d, lo, hi = cs.main([str(a), str(a)])
    assert d == lo == hi == 0
    b = _eval_json(tmp_path / "b.json", "tree_x", [64, 64, 64, 64], [16, 16, 16, 16])
    d, lo, hi = cs.main([str(a), f"{b}:tree_x"])
    assert d == pytest.approx(4.0 - 2.0) and lo <= d <= hi


def test_compare_spec_spec_parsing():
    cs = _load("compare_spec")
    assert cs.split_spec("D:/r/eval_x.json") == ("D:/r/eval_x.json", "fixed_k")
    assert cs.split_spec("D:/r/eval_x.json:tree_static_n40") == ("D:/r/eval_x.json", "tree_static_n40")
    assert cs.split_spec("results/a/eval.json") == ("results/a/eval.json", "fixed_k")


def test_draft_lms_agree_with_the_full_head():
    torch.manual_seed(0)
    W, s = torch.randn(50, 8), torch.randn(5, 8)
    full = FullDraftLM(W)
    assert torch.allclose(full(s), s @ W.T)
    ids = list(range(50))
    assert torch.equal(TopNDraftLM(W, ids)(s).argmax(-1), full(s).argmax(-1))     # all ids kept = full
    A, B = svd_factors(W, 8)                                                       # full rank = exact
    assert torch.allclose(LowRankDraftLM(A, B)(s), full(s), atol=1e-4)
    sub = TopNDraftLM(W, [3, 7, 11])(s)
    assert torch.isinf(sub[:, 0]).all() and sub.argmax(-1).tolist() == [[3, 7, 11][i] for i in sub[:, [3, 7, 11]].argmax(-1)]
    lr = make_draft_lm(W, "lowrank", rank=4, n=10, counts=list(range(50)))
    assert lr.A.shape == (10, 4) and lr(s).shape == (5, 50) and lr.name == "lowrank4_top10"


def test_topn_ids_and_scatter():
    assert topn_ids({5: 3, 2: 9, 7: 9, 1: 1}, 2) == [2, 7]
    full = scatter_full(torch.tensor([[1.0, 2.0]]), torch.tensor([4, 0]), 6)
    assert full[0, 4] == 1 and full[0, 0] == 2 and torch.isinf(full[0, 1])


def test_draft_agreement_on_tiny_seq_model(tiny_model_dir, tmp_path):
    import contextlib

    from tests.test_spec_decode import _tiny_model

    da = _load("draft_agreement")
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "seq")
    W = model._lm_head().weight
    preds = [make_draft_lm(W, "full"), make_draft_lm(W, "lowrank", rank=W.shape[1])]
    texts = ["मैं कल बाजार जा रहा था। वह घर से आया था", "बच्चे पार्क में खेल रहे हैं। के बारे बात कर"]
    res, total = da.agreement(model, tok, texts, preds, contextlib.nullcontext, "cpu")
    assert all(t > 0 for t in total)
    assert res["full"]["agreement"] == [1.0] * (model.num_heads - 1)
    assert min(res["lowrank16"]["agreement"]) > 0.99                              # full-rank SVD = full head


def test_vocab_rank_counts(tiny_tok):
    vr = _load("vocab_rank")
    counts = vr.count_tokens(["मैं कल बाजार", "मैं कल"], tiny_tok)
    assert sum(counts.values()) == 5 and counts.most_common(1)[0][1] == 2
    cov = vr.coverage(counts, ns=(1, 3))
    assert cov["1"] == pytest.approx(2 / 5) and cov["3"] == pytest.approx(1.0)


def test_profile_draft_runs_on_cpu(tiny_model_dir, tmp_path):
    import contextlib

    from tests.test_spec_decode import _tiny_model

    pd = _load("profile_draft")
    model, _ = _tiny_model(tiny_model_dir, tmp_path, "seq")
    res, table = pd.profile(model, contextlib.nullcontext, reps=3, warmup=1, counts={i: 50 - i for i in range(50)},
                            topn=(10,), ranks=(4,))
    assert {"embed", "seqblock", "lm_head", "lm_head_fp16", "argmax", "sync", "draft_chain"} <= set(res)
    assert set(res["predictors"]) == {"top10", "lowrank4"} and table is None


def test_wide_sizing_counts_and_r2(tiny_model_dir):
    from transformers import AutoModelForCausalLM

    from mtp.model.build import load_tokenizer

    ws = _load("wide_sizing")
    torch.manual_seed(0)
    Y = torch.randn(400, 3) @ torch.randn(3, 12)                  # rank 3 in 12 dims
    assert ws.pcs(Y)["99%"] <= 3
    X = torch.randn(600, 4)
    Yl = torch.cat([X @ torch.randn(4, 4) for _ in range(2)], dim=1)
    pred = ws.ridge(X[:500], Yl[:500])(X[500:])
    assert ws.r2(pred, Yl[500:], 4)["overall"] > 0.99            # exactly linear: ridge recovers it
    res = ws.sizing(X[:500], Yl[:500], X[500:], Yl[500:], 4, ws.logit_metric(torch.randn(20, 4)), mlp_epochs=2)
    assert set(res) == {"pcs_single", "pcs_stacked", "linear", "mlp"} and res["linear"]["pcs_pred"]["95%"] <= 4
    tok = load_tokenizer(str(tiny_model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    Xc, Yc = ws.collect(model, tok, ["मैं कल बाजार जा रहा था", "वह घर से आया था और फिर"], 3, torch.device("cpu"))
    assert Yc.shape == (len(Xc), 3 * Xc.shape[1]) and torch.equal(Yc[1, : Xc.shape[1]], Xc[2])


def test_decode_by_group_type(tiny_tok):
    from mtp.data.grouping.base import GROUP_TYPES, get_grouper

    dg = _load("decode_by_group_type")
    s = dg.summarize([("single", "start", 1, None), ("aux_chain", "start", 3, True), ("aux_chain", "inside", 2, False)])
    assert s["all"]["steps"] == 3 and s["all"]["accepted_len"] == 2.0 and s["aux_chain"]["gi"] == 0.5
    assert s["aux_chain/inside"]["accepted_len"] == 2.0 and s["single"]["gi"] is None
    grouper = get_grouper("hi_rules_v0")
    prompt = list(tiny_tok("मैं कल")["input_ids"])
    new = list(tiny_tok(" बाजार जा रहा था और वह घर से आया")["input_ids"])
    spans = [1] * (len(new) % 3) + [3] * (len(new) // 3)
    recs = dg.step_records(prompt, new, spans, tiny_tok, grouper)
    assert len(recs) == len(spans) and {r[0] for r in recs} <= set(GROUP_TYPES)
    assert all(r[2] == s for r, s in zip(recs, spans))
