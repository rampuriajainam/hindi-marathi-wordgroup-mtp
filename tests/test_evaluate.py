import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from mtp.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["मैं कल बाजार जा रहा था।", "बच्चे पार्क में खेल रहे हैं।", "वह घर से आया था",
         "भारत एक विशाल और विविधतापूर्ण देश है।", "के बारे बात कर"]
SECTION10_KEYS = {"run_name", "dataset", "step", "grouper", "git_commit", "per_head", "spec_decode"}
PER_HEAD_KEYS = {"head", "loss", "ppl", "top1", "top5", "n", "top1_in_group", "top1_at_boundary",
                 "loss_in_group", "loss_at_boundary", "n_in_group", "n_at_boundary"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def evaluate():
    return _load("evaluate_script", ROOT / "scripts" / "evaluate.py")


@pytest.fixture
def tiny_run(tmp_path, tiny_model_dir, tiny_tok):
    """A tiny R2-style run trained for 2 steps by scripts/train.py (CPU, seconds)."""
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model

    train = _load("train_script_eval", ROOT / "scripts" / "train.py")
    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_R2", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32",
             log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=2, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    examples = label_batch(TEXTS, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    run_dir = tmp_path / "runs" / "tiny_R2"
    train.train(cfg, model, tok, examples, examples, run_dir)
    return run_dir


def rec_tok_dir(run_dir):
    """The tokenizer the tiny run was built with (its config's model_name)."""
    import yaml

    return yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8"))["model_name"]


def fake_texts(name, cfg, n=None):
    texts = TEXTS if name.startswith("indiccorp") else TEXTS[::-1]
    return texts[:n] if n else texts


def test_writes_section10_json_matching_evaluate_heads(evaluate, tiny_run, tmp_path):
    from mtp.data.collate import Collator
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.eval.head_accuracy import evaluate_heads
    from mtp.model.checkpoint import load_run

    out = tmp_path / "results"
    written = evaluate.evaluate_run(tiny_run, ["indiccorp_eval", "flores_hi"], out_dir=out, device="cpu",
                                    dump_tokens=True, texts_fn=fake_texts)
    assert set(written) == {"indiccorp_eval", "flores_hi"}

    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert set(rec) == SECTION10_KEYS
    assert (rec["run_name"], rec["dataset"], rec["step"], rec["grouper"]) == ("tiny_R2", "indiccorp_eval", 2, "hi_rules_v0")
    assert rec["git_commit"] and rec["spec_decode"] is None
    assert [r["head"] for r in rec["per_head"]] == [0, 1, 2]
    assert all(PER_HEAD_KEYS <= set(r) for r in rec["per_head"])

    # Same numbers as calling evaluate_heads directly on the loaded run.
    model, tok, cfg = load_run(tiny_run, device="cpu")
    examples = label_batch(TEXTS, tok, get_grouper("hi_rules_v0"))
    ref = evaluate_heads(model, examples, Collator(tok.pad_token_id), cfg, "cpu")
    for got, exp in zip(rec["per_head"], ref):
        for key in PER_HEAD_KEYS:
            if exp[key] is None or isinstance(exp[key], int):
                assert got[key] == exp[key], key
            else:
                assert got[key] == pytest.approx(exp[key], abs=1e-6), key

    # FLORES got its own texts; the token dump carries sentence text.
    flores = json.loads((out / "tiny_R2" / "eval_flores_hi.json").read_text(encoding="utf-8"))
    assert flores["dataset"] == "flores_hi"
    dump = [json.loads(l) for l in (out / "tiny_R2" / "tokens_indiccorp_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dump) == sum(r["n"] for r in rec["per_head"])
    assert dump[0]["sent_id"] == 0 and dump[0]["head"] == 0


def test_rerun_keeps_other_sections(evaluate, tiny_run, tmp_path):
    out = tmp_path / "results"
    path = out / "tiny_R2" / "eval_indiccorp_eval.json"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", texts_fn=fake_texts)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["spec_decode"] = [{"policy": "fixed_k", "speedup": 1.3}]  # as if OM-6 ran in another session
    path.write_text(json.dumps(rec), encoding="utf-8")

    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", texts_fn=fake_texts, n=3)
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["spec_decode"] == [{"policy": "fixed_k", "speedup": 1.3}]
    assert again["per_head"][0]["n"] < rec["per_head"][0]["n"]  # per_head recomputed on 3 sentences


def test_unknown_grouper_evaluates_without_split(evaluate, tiny_run, tmp_path):
    with pytest.warns(UserWarning, match="not registered"):
        evaluate.evaluate_run(tiny_run, ["indiccorp_eval_small"], grouper_name="unregistered_test_grouper",
                              out_dir=tmp_path, device="cpu", texts_fn=fake_texts)
    rec = json.loads((tmp_path / "tiny_R2" / "eval_indiccorp_eval_small.json").read_text(encoding="utf-8"))
    assert rec["grouper"] is None
    assert rec["per_head"][0]["top1_in_group"] is None and rec["per_head"][0]["n_in_group"] == 0


def test_hindi_grouper_on_marathi_dataset_evaluates_without_split(evaluate, tiny_run, tmp_path):
    """A Hindi run on flores_mr: hi_rules_v0 refuses lang "mr" (ValueError); that must not crash."""
    with pytest.warns(UserWarning, match="cannot group 'mr'"):
        evaluate.evaluate_run(tiny_run, ["flores_hi", "flores_mr"], grouper_name="hi_rules_v0",
                              out_dir=tmp_path, device="cpu", texts_fn=fake_texts)
    hi = json.loads((tmp_path / "tiny_R2" / "eval_flores_hi.json").read_text(encoding="utf-8"))
    mr = json.loads((tmp_path / "tiny_R2" / "eval_flores_mr.json").read_text(encoding="utf-8"))
    assert hi["grouper"] == "hi_rules_v0" and mr["grouper"] is None
    assert mr["per_head"][0]["n_in_group"] == 0


def test_resolve_ignore_eos(evaluate):
    from types import SimpleNamespace

    frozen_instruct = SimpleNamespace(freeze_backbone=True, model_name="smallstepai/Misal-1B-instruct-v0.1")
    lora_instruct = SimpleNamespace(freeze_backbone=False, model_name="smallstepai/Misal-1B-instruct-v0.1")
    frozen_base = SimpleNamespace(freeze_backbone=True, model_name="LingoIITGN/ganga-1b")
    assert evaluate.resolve_ignore_eos("auto", frozen_instruct) is True     # Rsd_mr
    assert evaluate.resolve_ignore_eos("auto", lora_instruct) is False      # R9
    assert evaluate.resolve_ignore_eos("auto", frozen_base) is False        # Rsd (Hindi)
    from_r8 = SimpleNamespace(freeze_backbone=True, model_name="smallstepai/Misal-1B-instruct-v0.1",
                              init_lora_from="/kaggle/input/mtp-run-r8/R8_mr_ntp")
    assert evaluate.resolve_ignore_eos("auto", from_r8) is False            # Rsd_r8: LoRA from R8
    assert evaluate.resolve_ignore_eos("on", frozen_base) is True
    assert evaluate.resolve_ignore_eos("off", frozen_instruct) is False


def test_grouper_for_lang(evaluate):
    spec = "hi:hi_rules_v0, mr:mr_rules_v1"
    assert evaluate.grouper_for_lang(spec, "hi") == "hi_rules_v0"
    assert evaluate.grouper_for_lang(spec, "mr") == "mr_rules_v1"
    assert evaluate.grouper_for_lang("hi:hi_rules_v0", "mr") is None   # -> the run's data.grouper
    assert evaluate.grouper_for_lang("hi_rules_v0", "mr") == "hi_rules_v0"
    assert evaluate.grouper_for_lang("", "hi") is None


def test_flores_alias_and_per_language_grouper(evaluate, tiny_run, tmp_path):
    """`flores` = FLORES in the run's language; a per-language --grouper picks the run's entry."""
    written = evaluate.evaluate_run(tiny_run, ["flores"], grouper_name="mr:mr_rules_v1,hi:hi_rules_v0",
                                    out_dir=tmp_path, device="cpu", texts_fn=fake_texts)
    assert set(written) == {"flores_hi"}
    rec = json.loads((tmp_path / "tiny_R2" / "eval_flores_hi.json").read_text(encoding="utf-8"))
    assert rec["dataset"] == "flores_hi" and rec["grouper"] == "hi_rules_v0"


SPEC_KEYS = {"policy", "mean_accepted_len", "accept_rate_per_head", "tokens_per_sec", "greedy_tokens_per_sec",
             "speedup", "outputs_match_greedy", "group_integrity", "n_prompts", "match_rate", "max_mismatch_margin", "fp32_check",
             "mean_new_tokens", "ignore_eos", "per_prompt", "wall", "greedy_wall"}
LONG = ["मैं कल बाजार जा रहा था। वह घर से आया था भारत एक विशाल और विविधतापूर्ण देश है।",
        "बच्चे पार्क में खेल रहे हैं। के बारे बात कर वह घर से आया था मैं कल बाजार जा रहा था।"]


def long_texts(name, cfg, n=None):
    return LONG + TEXTS


def test_spec_decode_section(evaluate, tiny_run, tmp_path):
    out = tmp_path / "results"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True,
                          policies=["fixed_k", "confidence_cut"], tau=0.2, max_new_tokens=8, texts_fn=long_texts)
    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["per_head"] is not None and [e["policy"] for e in rec["spec_decode"]] == ["fixed_k", "confidence_cut"]
    for e in rec["spec_decode"]:
        assert set(e) == SPEC_KEYS
        assert e["outputs_match_greedy"] is True and e["n_prompts"] == 2   # only LONG has > 16 words
        assert len(e["accept_rate_per_head"]) == 2 and e["mean_accepted_len"] >= 1.0

    # stored outputs re-score to the run's own GI with the same grouper (scripts/rescore_gi.py)
    from transformers import AutoTokenizer

    rescore = _load("rescore_gi_script", ROOT / "scripts" / "rescore_gi.py")
    tok = AutoTokenizer.from_pretrained(rec_tok_dir(tiny_run))
    dest, gi = rescore.rescore_file(out / "tiny_R2" / "eval_indiccorp_eval.json", tok, "hi_rules_v0", "hi")
    assert dest.name == "gi_hi_rules_v0_indiccorp_eval.json" and dest.exists()
    for e, g in zip(rec["spec_decode"], gi["policies"]):
        pp = e["per_prompt"]
        assert [sum(sp) for sp in pp["spans"]] == pp["tokens"] == [len(i) for i in pp["ids"]]
        assert g["gi_hits"] == pp["gi_hits"] and g["gi_spans"] == pp["gi_spans"]
        assert g["group_integrity"] == e["group_integrity"]

    # spec-only session: per_head is kept from the first run, spec_decode replaced
    before = rec["per_head"]
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", heads=False,
                          spec_decode=True, policies=["fixed_k"], max_new_tokens=4, texts_fn=long_texts)
    again = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert again["per_head"] == before and [e["policy"] for e in again["spec_decode"]] == ["fixed_k"]


def test_spec_decode_skips_single_head_run(evaluate, tmp_path, tiny_model_dir, tiny_tok):
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model

    train = _load("train_script_eval_r0", ROOT / "scripts" / "train.py")
    d = yaml.safe_load((ROOT / "configs" / "R0.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_R0", model_name=str(tiny_model_dir), dtype="fp32", log_every=1, eval_every=1, save_every=1)
    d["optim"].update(max_steps=1, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    examples = label_batch(TEXTS, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    run_dir = tmp_path / "runs" / "tiny_R0"
    train.train(cfg, model, tok, examples, examples, run_dir)

    with pytest.warns(UserWarning, match="one head"):
        evaluate.evaluate_run(run_dir, ["indiccorp_eval"], out_dir=tmp_path / "res", device="cpu",
                              spec_decode=True, texts_fn=long_texts)
    rec = json.loads((tmp_path / "res" / "tiny_R0" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["spec_decode"] is None and len(rec["per_head"]) == 1


def test_cli_errors(evaluate, tiny_run):
    with pytest.raises(SystemExit):
        evaluate.main(["--run_dir", str(tiny_run), "--datasets", "nope"])
    with pytest.raises(SystemExit, match="not available"):
        evaluate.main(["--run_dir", str(tiny_run), "--spec_decode", "--policies", "no_such_policy"])
    with pytest.raises(SystemExit, match="nothing to do"):
        evaluate.main(["--run_dir", str(tiny_run), "--no_heads"])


def test_frozen_self_distillation_run_is_lossless(evaluate, tmp_path, tiny_model_dir, tiny_tok):
    """An Rsd-style run (freeze_backbone + data.train_file) goes through train.py -> load_run ->
    evaluate.py --spec_decode, and head 0 of the loaded run IS the untouched base model."""
    import torch
    from transformers import AutoModelForCausalLM

    from mtp.data.collate import Collator
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model
    from mtp.model.checkpoint import load_run

    train = _load("train_script_eval_rsd", ROOT / "scripts" / "train.py")
    sd_file = tmp_path / "sd.jsonl"            # stands in for gen_selfdistill.py output
    sd_file.write_text("\n".join(json.dumps({"text": t}) for t in TEXTS * 4) + "\n", encoding="utf-8")
    d = yaml.safe_load((ROOT / "configs" / "Rsd.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_Rsd", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32",
             log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=3, batch_size=2)
    d["data"].update(cache_dir=None, train_file=str(sd_file), eval_n=None)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.freeze_backbone is True

    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    examples = label_batch([json.loads(l)["text"] for l in sd_file.read_text(encoding="utf-8").splitlines()],
                           tiny_tok, get_grouper("hi_rules_v0"))
    run_dir = tmp_path / "runs" / "tiny_Rsd"
    train.train(cfg, model, tok, examples, examples, run_dir)

    loaded, tok2, _ = load_run(run_dir, device="cpu")
    pristine = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    batch = Collator(tok2.pad_token_id)(label_batch(TEXTS, tok2, get_grouper("hi_rules_v0")))
    with torch.no_grad():
        head0 = loaded(batch["input_ids"], batch["attention_mask"]).logits[0]
        base = pristine(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        heads = loaded(batch["input_ids"], batch["attention_mask"]).logits[1:]
    torch.testing.assert_close(head0, base, rtol=0, atol=0)          # lossless: exactly the base model
    assert not torch.equal(heads[0], head0)                           # the extra heads did train

    out = tmp_path / "res"
    evaluate.evaluate_run(run_dir, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True,
                          policies=["fixed_k"], max_new_tokens=8, texts_fn=long_texts)
    rec = json.loads((out / "tiny_Rsd" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["spec_decode"][0]["outputs_match_greedy"] is True and len(rec["per_head"]) == 3


def test_dumps_round_trips_and_keeps_number_lists_on_one_line(evaluate):
    rec = {"run_name": "x", "per_head": [{"head": 0, "top1": 0.5}],
           "spec_decode": [{"policy": "fixed_k", "accept_rate_per_head": [0.3, None, 1e-05],
                            "per_prompt": {"ids": [[1, 2, 3], [4]], "match": [True, False], "spans": [[]]}}]}
    text = evaluate.dumps(rec)
    assert json.loads(text) == rec
    assert '"ids": [\n' in text and "[1, 2, 3]" in text and "[true, false]" in text and "[0.3, null, 1e-05]" in text


def test_tree_policy_through_evaluate_run(evaluate, tiny_run, tmp_path):
    """--tree_policies: a TreePolicy JSON is verified with generate_tree and reported like any policy."""
    from mtp.eval.tree_policy import StaticTree, save_policy

    path = tmp_path / "tree_policy_static_n15.json"
    save_policy(StaticTree.from_widths((3, 2, 1)), path)
    out = tmp_path / "results"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True, heads=False,
                          policies=["fixed_k"], tree_policies=[str(path)], max_new_tokens=8, texts_fn=long_texts)
    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    chain, tree = rec["spec_decode"]
    assert tree["policy"] == "tree_static_n15" and set(tree) == SPEC_KEYS | {"mean_tree_nodes"}
    assert tree["outputs_match_greedy"] is True and 0 < tree["mean_tree_nodes"] <= 15
    assert tree["per_prompt"]["ids"] == chain["per_prompt"]["ids"]      # both equal greedy


def test_prompt_lookup_flag_adds_tree_plus_lookup_policies(evaluate, tiny_run, tmp_path):
    from mtp.eval.tree_policy import StaticTree, save_policy

    path = tmp_path / "tree_policy_static_n15.json"
    save_policy(StaticTree.from_widths((3, 2, 1)), path)
    out = tmp_path / "results"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True, heads=False,
                          policies=["fixed_k"], tree_policies=[str(path)], prompt_lookup=(4, 6), max_new_tokens=8,
                          texts_fn=long_texts)
    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert [e["policy"] for e in rec["spec_decode"]] == ["fixed_k", "tree_static_n15", "tree_static_n15+pld4x6"]
    assert all(e["outputs_match_greedy"] for e in rec["spec_decode"])


def test_tree_typical_and_sample_flags(evaluate, tiny_run, tmp_path, monkeypatch):
    """--tree_typical / --tree_sample: one lossy / sampled variant of every tree policy (incl. lookup ones);
    exact entries unchanged, sampled ones checked against plain sampling, which runs once per setting."""
    from mtp.eval import spec_decode
    from mtp.eval.tree_policy import StaticTree, save_policy

    path = tmp_path / "tree_policy_static_n15.json"
    save_policy(StaticTree.from_widths((3, 2, 1)), path)
    calls, real = [], spec_decode.greedy_generate

    def spy(*a, **kw):
        calls.append(kw.get("sample"))
        return real(*a, **kw)

    monkeypatch.setattr(spec_decode, "greedy_generate", spy)
    out = tmp_path / "results"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True, heads=False,
                          policies=["typical_0.1"], tree_policies=[str(path)], prompt_lookup=(4, 6),
                          tree_typical=[1e-4], tree_sample=[(0.7, 0.9)], max_new_tokens=8, texts_fn=long_texts)
    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    names = [e["policy"] for e in rec["spec_decode"]]
    assert names == ["typical_0.1", "tree_static_n15", "tree_static_n15+pld4x6", "tree_static_n15+typ0.0001",
                     "tree_static_n15+pld4x6+typ0.0001", "tree_static_n15+samp0.7p0.9",
                     "tree_static_n15+pld4x6+samp0.7p0.9"]
    by = {e["policy"]: e for e in rec["spec_decode"]}
    assert set(by["typical_0.1"]) == SPEC_KEYS                                       # chain typical untouched
    for name in names[1:3]:
        assert set(by[name]) == SPEC_KEYS | {"mean_tree_nodes"} and by[name]["outputs_match_greedy"] is True
    for name in names[3:5]:
        e = by[name]
        assert set(e) == SPEC_KEYS | {"mean_tree_nodes", "lossy_accepts"} and e["lossy_accepts"] > 0
        assert e["mean_accepted_len"] > by["tree_static_n15"]["mean_accepted_len"]
    for name in names[5:]:
        e = by[name]
        assert set(e) == SPEC_KEYS | {"mean_tree_nodes", "baseline"} and e["baseline"] == "sample_0.7_0.9"
        assert e["outputs_match_greedy"] is True and e["match_rate"] == 1.0 and e["max_mismatch_margin"] is None
        assert [sum(sp) for sp in e["per_prompt"]["spans"]] == e["per_prompt"]["tokens"] == \
            [len(i) for i in e["per_prompt"]["ids"]] and e["per_prompt"]["prompt_ids"] == by[names[1]]["per_prompt"]["prompt_ids"]
    assert by[names[5]]["per_prompt"]["ids"] == by[names[6]]["per_prompt"]["ids"] != by[names[1]]["per_prompt"]["ids"]
    n_prompts = by[names[1]]["n_prompts"]
    assert calls.count((0.7, 0.9)) == n_prompts and calls.count(None) == n_prompts + 1   # + 1: warm-up


def test_sampled_fp32_check_uses_sampled_reference(tiny_run):
    """Under amp the fp32 re-check of a sampled policy compares with fp32 plain sampling (same seeds)."""
    import contextlib

    from mtp.eval.sampling import SampledTree
    from mtp.eval.spec_decode import evaluate_spec_decode
    from mtp.eval.tree_policy import StaticTree
    from mtp.eval.typical import TypicalTree
    from mtp.model.checkpoint import load_run

    model, tok, _ = load_run(tiny_run, device="cpu")
    prompts = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    tree = StaticTree.from_widths((3, 2, 1))
    res = evaluate_spec_decode(model, tok, prompts, [tree, SampledTree(tree, 1.0, 1.0), TypicalTree(tree, 1e-4)],
                               max_new_tokens=8, amp=contextlib.nullcontext, fp32_check_n=2)
    exact, sampled, lossy = res
    assert exact["fp32_check"]["outputs_match_greedy"] is True and "baseline" not in exact
    assert sampled["baseline"] == "sample_1_1" and sampled["fp32_check"]["outputs_match_greedy"] is True
    assert sampled["fp32_check"]["n_prompts"] == 2 and sampled["match_rate"] == 1.0
    assert "lossy_accepts" in lossy and "baseline" not in lossy


def test_tree_sample_flag_parsing(evaluate, tiny_run, monkeypatch):
    assert evaluate.parse_sample("0.7,0.9") == (0.7, 0.9)
    seen = {}
    monkeypatch.setattr(evaluate, "evaluate_run", lambda *a, **kw: seen.update(kw))
    evaluate.main(["--run_dir", str(tiny_run), "--spec_decode", "--tree_typical", "0.5", "0.02",
                   "--tree_sample", "0.7,0.9", "1.0,1.0"])
    assert seen["tree_typical"] == [0.5, 0.02] and seen["tree_sample"] == [(0.7, 0.9), (1.0, 1.0)]
    monkeypatch.undo()
    for bad in ("0.7", "0,0.9", "0.7,1.5", "a,b"):
        with pytest.raises(SystemExit):
            evaluate.main(["--run_dir", str(tiny_run), "--spec_decode", "--tree_sample", bad])
