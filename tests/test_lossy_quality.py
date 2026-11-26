"""scripts/lossy_quality.py (judge with another tokenizer, human row, output names) and scripts/lossy_table.py.
All data here is synthetic and lives in tmp_path."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from conftest import WORDS, make_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lq = _load("lossy_quality")
lt = _load("lossy_table")


def make_char_tokenizer(bos=True):
    """A judge with a different tokenizer: one token per character (spaces included), optional BOS."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    chars = sorted(set(" ".join(WORDS)) | set("abcdefghijklmnopqrstuvwxyz0123456789 "))
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    if bos:
        tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>")


def tiny_model(vocab_size, seed=0):
    import torch
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(seed)
    return MistralForCausalLM(MistralConfig(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=False)).eval()


# ------------------------------------------------------------------ pure functions

def test_count_words_and_slug_and_names():
    assert lq.count_words(" मैं कल  बाजार\nजा ") == 4
    assert lq.count_words("") == 0
    assert lq.judge_slug("sarvamai/sarvam-1") == "sarvam-1"
    assert lq.judge_slug("bigscience/bloom-1b7") == "bloom-1b7"
    assert lq.judge_slug("C:\\models\\my judge:v2/") == "my_judge_v2"
    assert lq.quality_name("indiccorp_eval", lq.DEFAULT_JUDGE) == "quality_indiccorp_eval.json"
    assert lq.quality_name("flores_mr", "sarvamai/sarvam-1") == "quality_flores_mr.sarvam-1.json"
    assert lq.quality_name("flores_hi", lq.DEFAULT_JUDGE, suffix="self text") == "quality_flores_hi.self_text.json"


def test_continuation_span_second_tokenizer():
    run, judge = make_tiny_tokenizer(), make_char_tokenizer(bos=True)
    prompt = run("मैं कल बाजार")["input_ids"]
    gen = run(" जा रहा था")["input_ids"]
    p_text, full, ok = lq.decode_split(run, prompt, gen)
    assert ok and p_text == "मैं कल बाजार" and full == "मैं कल बाजार जा रहा था"
    for use_offsets in (True, False):
        ids, start, end = lq.continuation_span(judge, p_text, full, use_offsets=use_offsets)
        assert ids[0] == judge.bos_token_id                       # the judge's own BOS is kept as context
        assert judge.decode(ids[start:end]) .replace(" ", "") == "जारहाथा"
        assert end - start == len(" जा रहा था") and end == len(ids)
    assert not lq.same_tokenizer(run, judge) and lq.same_tokenizer(run, make_tiny_tokenizer())


def test_continuation_span_never_counts_appended_eos():
    from tokenizers import processors

    judge = make_char_tokenizer(bos=False)
    judge._tokenizer.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 3)])
    for use_offsets in (True, False):
        ids, start, end = lq.continuation_span(judge, "ab", "ab cd", use_offsets=use_offsets)
        assert ids[-1] == 3 and (start, end) == (2, 5)


def test_text_level_equals_id_level_for_the_same_tokenizer():
    """Round-tripping tokens: text-level scoring with the run's own tokenizer gives the id-level numbers."""
    tok = make_tiny_tokenizer()
    model = tiny_model(len(tok))
    sents = ["मैं कल बाजार जा रहा था।", "वह घर से आया था भारत एक विशाल", "बच्चे पार्क में खेल रहे हैं।"]
    pairs = []
    for s in sents:
        w = s.split()
        pairs.append((tok(" ".join(w[:2]))["input_ids"], tok(" " + " ".join(w[2:]))["input_ids"]))
    entry = {"policy": "fixed_k", "mean_accepted_len": 1.0, "per_prompt": {"prompt_ids": [p for p, _ in pairs],
                                                                            "ids": [g for _, g in pairs]}}
    a = lq.score_entries([entry], model, "cpu", 2, run_tok=tok, judge_tok=tok, text_level=False)[0]
    b = lq.score_entries([entry], model, "cpu", 2, run_tok=tok, judge_tok=tok, text_level=True)[0]
    assert b["ppl"] == pytest.approx(a["ppl"], rel=1e-5) and b["ppl_word"] == pytest.approx(a["ppl_word"], rel=1e-5)
    assert b["ppl_norep"] is None and "nll_norep_sum" not in b["per_prompt"]
    assert a["per_prompt"]["n_words"] == [len(s.split()) - 2 for s in sents]
    assert a["ppl_word"] == pytest.approx(math.exp(sum(a["per_prompt"]["nll_sum"]) / sum(a["per_prompt"]["n_words"])))


def test_text_level_char_judge_and_left_truncation():
    run, judge = make_tiny_tokenizer(), make_char_tokenizer()
    model = tiny_model(len(judge))
    prompt, gen = run("मैं कल बाजार")["input_ids"], run(" जा रहा था")["input_ids"]
    entry = {"policy": "typical_0.1", "per_prompt": {"prompt_ids": [prompt], "ids": [gen]}}
    r = lq.score_entries([entry], model, "cpu", run_tok=run, judge_tok=judge, text_level=True)[0]
    assert r["per_prompt"]["n_tok"] == [len(" जा रहा था")] and r["per_prompt"]["n_words"] == [3]
    span = lq.continuation_span(judge, "मैं कल बाजार", "मैं कल बाजार जा रहा था")
    full = lq.nll_spans(model, [span], "cpu", 0)[0]
    cut = lq.nll_spans(model, [span], "cpu", 0, max_len=len(span[0]) - 3)[0]
    assert full[1] == cut[1] and full[0] == pytest.approx(r["per_prompt"]["nll_sum"][0], rel=1e-5)


def test_human_continuations_rebuild_on_toy_corpus():
    from mtp.eval.spec_decode import make_prompts

    tok = make_tiny_tokenizer()
    texts = [" ".join(WORDS[i: i + n]) for i, n in [(0, 5), (0, 20), (3, 25), (1, 12), (2, 30)]]
    prompts, conts, fb = lq.human_continuations(texts, tok, n=10, max_new_tokens=6)
    assert prompts == make_prompts(texts, tok, n=10) and fb == 0 and len(prompts) >= 2
    for p, c in zip(prompts, conts):
        assert 1 <= len(c) <= 6
        src = next(t for t in texts if tok(t)["input_ids"][: len(p)] == p)
        assert p + c == tok(src)["input_ids"][: len(p) + len(c)]   # consistent with tokenising the sentence
    rec = {"spec_decode": [{"policy": "fixed_k", "per_prompt": {"prompt_ids": prompts, "ids": [[5] * 8] * len(prompts)}}]}
    h, why = lq.human_entry(rec, tok, texts)
    assert why is None and h["policy"] == "human" and h["_human"]["max_new_tokens"] == 8
    rec["spec_decode"][0]["per_prompt"]["prompt_ids"] = [p[:-1] for p in prompts]
    h, why = lq.human_entry(rec, tok, texts)
    assert h is None and "differ" in why


def test_run_info_and_dataset_language(tmp_path):
    (tmp_path / "config.yaml").write_text("run_name: toy\nlang: mr\nmodel_name: smallstepai/Misal-1B-instruct-v0.1\n",
                                          encoding="utf-8")
    info = lq.run_info(tmp_path, {"run_name": "toy"}, configs_dir=tmp_path / "none")
    assert info == {"tokenizer": "smallstepai/Misal-1B-instruct-v0.1", "lang": "mr"}
    assert lq.run_info(tmp_path / "x", {"run_name": "?", "model_name": "m", "lang": "hi"}) == {"tokenizer": "m", "lang": "hi"}
    assert lq.dataset_language("flores_mr", "hi") == "mr" and lq.dataset_language("indiccorp_eval", "mr") == "mr"
    assert lq.reference_policy([{"policy": "tree_x+typ0.1"}, {"policy": "tree_x"}]) == "tree_x"


# ------------------------------------------------------------------ CLI default path

OLD_FIELDS = {"policy", "ppl", "ppl_norep", "rep4", "identical_rate", "mean_accepted_len", "speedup"}
OLD_PER_PROMPT = {"nll_sum", "n_tok", "nll_norep_sum", "n_norep", "rep4", "div"}


def test_default_cli_writes_the_old_fields(tiny_model_dir, tmp_path, monkeypatch):
    """With no new flags: same file name and every field of the original scorer, with the original values."""
    import torch
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(lq, "DEFAULT_JUDGE", str(tiny_model_dir))
    monkeypatch.setattr(lq, "DEFAULT_TOKENIZER", str(tiny_model_dir))
    tok = make_tiny_tokenizer()
    prompts = [tok("मैं कल बाजार जा")["input_ids"], tok("वह घर से")["input_ids"]]
    greedy = [tok(" रहा था। वह घर रहा था। वह घर")["input_ids"], tok(" आया था भारत")["input_ids"]]
    lossy = [greedy[0][:3] + tok(" के बारे")["input_ids"], greedy[1]]
    rec = {"run_name": "toy_run", "dataset": "indiccorp_eval", "spec_decode": [
        {"policy": p, "mean_accepted_len": m, "speedup": s, "per_prompt": {"prompt_ids": prompts, "ids": ids}}
        for p, m, s, ids in [("fixed_k", 1.5, 1.2, greedy), ("typical_0.1", 1.8, 1.4, lossy)]]}
    run = tmp_path / "toy_run"
    run.mkdir()
    (run / "eval_indiccorp_eval.json").write_text(json.dumps(rec), encoding="utf-8")
    lq.main(["--results", str(run), "--batch_size", "1", "--device", "cpu"])
    assert sorted(p.name for p in run.iterdir()) == ["eval_indiccorp_eval.json", "quality_indiccorp_eval.json"]
    out = json.loads((run / "quality_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert {"run_name", "dataset", "judge", "policies"} <= set(out) and out["text_level"] is False
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, torch_dtype=torch.float32).eval()
    for r, e in zip(out["policies"], rec["spec_decode"]):
        assert OLD_FIELDS <= set(r) and OLD_PER_PROMPT <= set(r["per_prompt"])
        nll = lq.nll_batch(model, list(zip(prompts, e["per_prompt"]["ids"])), "cpu")
        assert r["ppl"] == pytest.approx(math.exp(sum(x[0] for x in nll) / sum(x[1] for x in nll)), rel=1e-5)
        assert r["rep4"] == pytest.approx(sum(lq.rep4(p, g) for p, g in zip(prompts, e["per_prompt"]["ids"])) / 2)
        assert r["mean_accepted_len"] == e["mean_accepted_len"] and r["speedup"] == e["speedup"]
    assert out["policies"][0]["identical_rate"] == 1.0 and out["policies"][1]["per_prompt"]["div"] == [3, None]


# ------------------------------------------------------------------ table

def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_table_on_synthetic_fixture(tmp_path):
    run = tmp_path / "SYNTH_run"
    run.mkdir()
    pols = ["tree_s+typ0.1", "fixed_k", "tree_s+samp0.7p0.9", "tree_s", "tree_s+typ0.5", "typical_0.02"]
    _write(run / "eval_toy.json", {"run_name": "SYNTH_run", "dataset": "toy", "spec_decode": [
        {"policy": p, "mean_accepted_len": 2.0 + i, "speedup": 1.0 + i, "tokens_per_sec": 10.0 * i,
         "match_rate": 0.5, "per_prompt": {"tokens": [64, 64], "steps": [32, 32]},
         **({"lossy_accepts": 16} if "+typ" in p else {}),
         **({"baseline": "sample_0.7_0.9"} if "samp" in p else {})} for i, p in enumerate(pols)]})
    _write(run / "quality_toy.json", {"policies": [   # old-style file: no ppl_word, no human
        {"policy": p, "ppl": 3.0, "ppl_norep": 4.0, "rep4": 0.1, "identical_rate": 0.0,
         "per_prompt": {"div": [None, 2, 4]}} for p in pols]})
    _write(run / "quality_toy.sarvam-1.json", {"policies": [
        {"policy": p, "ppl": 7.0, "ppl_word": 9.0, "rep4": 0.1} for p in pols]
        + [{"policy": "human", "ppl": 20.0, "ppl_word": 30.0, "rep4": 0.0, "mean_new_tokens": 12.0}]})
    rows = lt.build_rows([run])
    assert [r["policy"] for r in rows] == ["fixed_k", "tree_s", "typical_0.02", "tree_s+typ0.5", "tree_s+typ0.1",
                                           "tree_s+samp0.7p0.9", "human"]
    by = {r["policy"]: r for r in rows}
    assert by["fixed_k"]["mean_div"] == 3.0 and by["fixed_k"]["ppl_word"] is None
    assert by["tree_s+typ0.1"]["lossy_share"] == 0.25 and by["fixed_k"]["lossy_share"] is None  # 16 / (128 - 64)
    assert by["tree_s+samp0.7p0.9"]["reference"] == "sample_0.7_0.9" and by["tree_s"]["reference"] == "greedy"
    assert by["tree_s"]["judges"]["sarvam-1"] == {"ppl_word": 9.0, "ppl": 7.0}
    h = by["human"]       # only under the extra judge: run-token fields taken from there
    assert h["tok_step"] is None and h["ppl"] is None and h["rep4"] == 0.0 and h["mean_new_tokens"] == 12.0
    assert h["judges"]["sarvam-1"] == {"ppl_word": 30.0, "ppl": 20.0}
    md = lt.to_markdown(rows)
    assert "### SYNTH_run | toy" in md and "sarvam-1 ppl_word" in md and md.count("\n| ") == len(rows) + 1
    assert lt.policy_eps("typical_0.02") == 0.02 and lt.policy_eps("tree+pld4x6+typ0.1") == 0.1
    assert lt.policy_base("tree+pld4x6+typ0.1") == "tree+pld4x6"
