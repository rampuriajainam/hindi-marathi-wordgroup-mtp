"""Second backbone (docs/plan/SECOND_MODEL.md): the frozen self-distilled sequential-head pipeline (Rsd_seq) on a
Qwen2-style model (tied embeddings, q/k/v bias, no BOS, pad = EOS) and a Llama-style model (sarvam-1: BOS added,
untied). Tiny random models built from configs, CPU, no network. Real tokenizers only if already cached."""

import importlib.util
import json
import random
from pathlib import Path

import pytest
import torch
import yaml

from mtp.config import load_config
from tests.conftest import make_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["मैं कल बाजार जा रहा था। वह घर से आया था", "भारत एक विशाल और विविधतापूर्ण देश है।",
         "बच्चे पार्क में खेल रहे हैं। के बारे बात कर"]
BF16 = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731  (CPU stand-in for fp16 autocast on T4)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tokenizer(arch):
    """Stand-ins with the conventions of the real tokenizers. Qwen2.5: byte-level BPE (a Devanagari character can be
    split over several tokens), no BOS, pad = EOS = <|endoftext|>; a qwen2 model folder loads it as Qwen2Tokenizer.
    sarvam-1: SentencePiece-style pieces with '<s>' prepended."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    if arch == "qwen2":
        from tests.conftest import WORDS

        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator([" ".join(WORDS)] * 5, trainers.BpeTrainer(
            vocab_size=300, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False))
        return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok = make_tiny_tokenizer()
    tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", tok.bos_token_id)])
    return tok


def _base(arch, vocab):
    torch.manual_seed(0)
    if arch == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        return Qwen2ForCausalLM(Qwen2Config(       # Qwen2: q/k/v bias, tied lm_head, padded vocabulary
            vocab_size=vocab + 13, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=True))
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(LlamaConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False, bos_token_id=2, eos_token_id=3))


@pytest.fixture(params=["qwen2", "llama"])
def arch_dir(request, tmp_path):
    tok = _tokenizer(request.param)
    path = tmp_path / f"tiny_{request.param}"
    _base(request.param, len(tok)).save_pretrained(path)
    tok.save_pretrained(path)
    return request.param, path


def _cfg(tmp_path, model_dir, steps=40):
    """configs/Rsd_seq.yaml on the tiny model: frozen backbone, seq heads, no boundary cache, no grouper."""
    d = yaml.safe_load((ROOT / "configs" / "Rsd_seq.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_sd_seq", model_name=str(model_dir), dtype="fp32", log_every=10, eval_every=1000,
             save_every=1000)
    d["optim"].update(max_steps=steps, batch_size=4, lr=1e-2)
    d["data"].update(cache_dir=None, grouper=None, train_file=None)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    return load_config(tmp_path / "c.yaml")


def _prompts(tok, n, seed):
    rng = random.Random(seed)
    bos = [tok.bos_token_id] if tok("x")["input_ids"][:1] == [tok.bos_token_id] else []
    return [bos + [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 6))] for _ in range(n)]


@pytest.fixture
def trained(arch_dir, tmp_path):
    """build_model -> greedy self-distillation text -> scripts/train.py (frozen backbone, 40 steps)."""
    from mtp.eval.spec_decode import greedy_generate
    from mtp.model.build import build_model

    arch, model_dir = arch_dir
    train = _load("train_second_model", ROOT / "scripts" / "train.py")
    cfg = _cfg(tmp_path, model_dir)
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    model.eval()
    examples = []
    for p in _prompts(tok, 24, 0):
        ids = p + greedy_generate(model, tok, p, 20, ignore_eos=True)[0]
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    before = {n: q.detach().clone() for n, q in model.named_parameters()}
    run_dir = tmp_path / "runs" / cfg.run_name
    train.train(cfg, model, tok, examples[:20], examples[20:], run_dir)
    changed = {n for n, q in model.named_parameters() if not torch.equal(q.detach(), before[n])}
    assert changed and all(n.startswith("extra_heads.") for n in changed)      # backbone (and lm_head) frozen
    return arch, model.eval(), tok, run_dir


def test_wrapper_finds_heads_and_embeddings(arch_dir, tmp_path):
    from mtp.model.build import build_model

    arch, model_dir = arch_dir
    model, tok = build_model(_cfg(tmp_path, model_dir), "cpu")
    lm_head, emb = model._lm_head(), model.base_model.get_input_embeddings()
    assert (lm_head.weight is emb.weight) == (arch == "qwen2")                 # Qwen2 ties lm_head to the embeddings
    assert lm_head.in_features == 32 and model.extra_heads[0].linear.in_features == 64
    if arch == "qwen2":
        assert tok.pad_token_id == tok.eos_token_id and tok.bos_token_id is None
    else:
        assert tok("भारत")["input_ids"][0] == tok.bos_token_id and tok.pad_token_id is not None
    lora = [n for n, _ in model.named_parameters() if "lora_A" in n]
    assert any("q_proj" in n for n in lora) and any("v_proj" in n for n in lora)
    out = model(torch.tensor([[5, 6, 7, 8]]))
    assert len(out.logits) == 4 and out.logits[1].shape == out.logits[0].shape


def test_frozen_seq_heads_train_and_decode_losslessly(trained):
    """Chain, tree and tree + prompt lookup all give exactly the greedy output, with some drafts accepted."""
    from mtp.eval.draft_policy import FixedK
    from mtp.eval.prompt_lookup import PromptLookup
    from mtp.eval.spec_decode import generate, greedy_generate
    from mtp.eval.tree_decode import generate_tree
    from mtp.eval.tree_policy import StaticTree

    arch, model, tok, _ = trained
    tree = StaticTree.from_widths((3, 2, 1))
    policies = [FixedK(), tree, PromptLookup(tree, chains=2, length=4)]
    tokens = steps = 0
    for p in _prompts(tok, 8, 1):
        for ignore_eos in (True, False):
            ref, _ = greedy_generate(model, tok, p, 16, ignore_eos=ignore_eos)
            for pol in policies:
                dec = generate_tree if hasattr(pol, "tree") else generate
                ids, st = dec(model, tok, p, 16, pol, ignore_eos=ignore_eos)
                assert ids == ref, (arch, pol.name, p)
                tokens += st["tokens"]
                steps += st["steps"]
    assert tokens > 1.2 * steps                     # drafts were accepted: cache rollback paths were exercised


def test_fast_linear_bit_identical(trained):
    """fast_linear also covers a tied lm_head (Qwen2) via a redirected forward; the shared embedding stays fp32."""
    from mtp.eval.draft_policy import FixedK
    from mtp.eval.precast import fast_linear, with_fast_linear
    from mtp.eval.spec_decode import generate, greedy_generate
    from mtp.eval.tree_decode import generate_tree
    from mtp.eval.tree_policy import StaticTree

    arch, model, tok, _ = trained
    x = torch.tensor([[5, 6, 7, 8, 9, 10]])
    lm_head, emb = model._lm_head(), model.base_model.get_input_embeddings()
    with torch.no_grad(), BF16():
        ref = [lg.clone() for lg in model(x).logits]
    with torch.no_grad(), BF16(), fast_linear(model, torch.bfloat16):
        got = model(x).logits
        assert emb.weight.dtype == torch.float32
        assert ("forward" in lm_head.__dict__) == (arch == "qwen2")
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    assert "forward" not in lm_head.__dict__ and lm_head.weight.dtype == torch.float32
    fast = with_fast_linear(BF16, model, torch.bfloat16)
    tree = StaticTree.from_widths((3, 2, 1))
    for p in _prompts(tok, 4, 2):
        assert greedy_generate(model, tok, p, 12, amp=BF16)[0] == greedy_generate(model, tok, p, 12, amp=fast)[0]
        assert generate(model, tok, p, 12, FixedK(), amp=BF16)[0] == generate(model, tok, p, 12, FixedK(), amp=fast)[0]
        assert generate_tree(model, tok, p, 12, tree, amp=BF16)[0] == generate_tree(model, tok, p, 12, tree, amp=fast)[0]


def test_evaluate_run_without_grouper(trained, tmp_path):
    """evaluate.py on such a run with no grouper (data.grouper null): GI is None, outputs match greedy."""
    from mtp.eval.tree_policy import StaticTree, save_policy

    evaluate = _load("evaluate_second_model", ROOT / "scripts" / "evaluate.py")
    _, _, _, run_dir = trained
    path = tmp_path / "tree_policy_static_n6.json"
    save_policy(StaticTree.from_widths((3, 2, 1)), path)
    out = tmp_path / "results"
    texts = lambda name, cfg, n=None: [t + " " + t for t in TEXTS]  # noqa: E731
    evaluate.evaluate_run(run_dir, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True,
                          policies=["fixed_k"], tree_policies=[str(path)], prompt_lookup=(2, 4), max_new_tokens=8,
                          ignore_eos="on", texts_fn=texts)
    rec = json.loads((out / "tiny_sd_seq" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["grouper"] is None and len(rec["per_head"]) == 4
    assert [e["policy"] for e in rec["spec_decode"]] == ["fixed_k", "tree_static_n6", "tree_static_n6+pld2x4"]
    for e in rec["spec_decode"]:
        assert e["outputs_match_greedy"] is True and e["group_integrity"] is None

    wps = _load("words_per_step", ROOT / "scripts" / "words_per_step.py")
    rows = wps.main([str(out / "tiny_sd_seq" / "eval_indiccorp_eval.json"), "--run_dir", str(run_dir)])
    assert len(rows) == 3
    for row, e in zip(rows, rec["spec_decode"]):
        assert row["per_prompt_steps"] == e["per_prompt"]["steps"] and len(row["per_prompt_words"]) == e["n_prompts"]
        assert row["tokens_per_step"] == pytest.approx(e["mean_accepted_len"])
        if row["tokens_per_word"]:
            assert row["words_per_step"] == pytest.approx(row["tokens_per_step"] / row["tokens_per_word"])
            assert row["words_per_sec"] / row["greedy_words_per_sec"] == pytest.approx(e["speedup"])


def test_continuation_words_counts_whitespace_words():
    wps = _load("words_per_step_cw", ROOT / "scripts" / "words_per_step.py")
    tok = _tokenizer("llama")                           # adds <s>; decode drops it
    p, c = tok("भारत एक")["input_ids"], tok("विशाल देश है।", add_special_tokens=False)["input_ids"]
    assert wps.continuation_words(tok, p, c) == 3


# ------------------------------------------------------------------ real tokenizers (only if cached)

def _cached_tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:  # noqa: BLE001  (not in the HF cache: no network in tests)
        pytest.skip(f"{name} tokenizer not cached")


@pytest.mark.parametrize("name", ["Qwen/Qwen2.5-1.5B", "sarvamai/sarvam-1"])
def test_real_tokenizer_groups_and_group_integrity(name):
    """Byte-level BPE (Qwen) splits a Devanagari character over several tokens; the token -> group labels still
    put every token in the group of the character it belongs to, and the spec-decode GI path agrees with align.py."""
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.eval.spec_decode import group_integrity_counts, make_prompts, token_group_ids

    tok = _cached_tokenizer(name)
    grouper = get_grouper("hi_rules_v1", "hi")
    for text in TEXTS:
        ex = label_batch([text], tok, grouper)[0]
        ids = ex["input_ids"]
        assert token_group_ids(ids, tok, grouper) == ex["group_id"]
        real = [g for g, t in zip(ex["group_id"], ids) if t != tok.bos_token_id]
        assert min(real) == 0 and real == sorted(real)                       # every real token has a group, in order
        assert max(real) + 1 == len(grouper.group_words(text))
        hits, spans = group_integrity_counts(ids[:3], ids[3:], [2] * ((len(ids) - 3) // 2), tok, grouper)
        assert 0 <= hits <= spans
    prompts = make_prompts([t + " " + t for t in TEXTS], tok, n=3, min_words=3, max_words=4)
    has_bos = tok("x")["input_ids"][0] == tok.bos_token_id
    assert all((p[0] == tok.bos_token_id) == has_bos for p in prompts)
