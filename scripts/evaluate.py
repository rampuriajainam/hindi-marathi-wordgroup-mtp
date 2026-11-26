"""
Official evaluation of a trained run (OM-7): per-head loss / ppl / top-k with the in-group /
at-boundary split (mtp.eval.head_accuracy), written per INTERFACES §10.

    python scripts/evaluate.py --run_dir /kaggle/input/mtp-run-r2 \
        --datasets indiccorp_eval flores_hi [--step N] [--grouper hi_rules_v0] [--dump_tokens]

--run_dir is a run folder (config.yaml + step_N/), e.g. a published mtp-run-<ID> dataset.
Writes results/{run_name}/eval_{dataset}.json (commit these) and, with --dump_tokens,
results/{run_name}/tokens_{dataset}.jsonl (large: keep out of git).

Datasets (INTERFACES §4): indiccorp_eval, indiccorp_eval_small (in the run's language),
flores_hi, flores_mr, and flores = FLORES in the run's language (written as flores_{lang}).
FLORES is gated: accept the terms on huggingface.co/datasets/facebook/flores and log in (HF_TOKEN).

--grouper takes one name or one per run language, e.g. "hi:hi_rules_v0,mr:mr_rules_v1", so Hindi
and Marathi runs can share one command line (a language not listed uses the run's data.grouper).

--spec_decode adds the §10 spec_decode section (mtp.eval.spec_decode): prompts are the first 8-16
words of the first --spec_n sentences of each dataset, batch size 1, outputs checked against greedy.
--tree_typical EPS ... adds a lossy (typical acceptance) variant of every tree policy per eps;
--tree_sample T,TOP_P ... a sampling-mode variant per setting, checked against plain sampling.
--tree_penalty R ... a repetition-penalised greedy variant per R, checked against penalised greedy.
--ignore_eos auto (default) masks EOS for head 0 when the run has a frozen backbone on an *instruct*
base model (frozen Misal-1B stops after one sentence, #39); on / off force it. Recorded per entry.
Re-running with the same run, step and grouper updates only the sections it computes, so per-head
and spec-decode numbers can come from separate sessions (--no_heads skips the per-head part).
"""

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

from mtp.config import cfg_get  # noqa: E402

DATASETS = {
    # name -> (lang or None for the run's lang, load_split split)
    "indiccorp_eval": (None, "eval"),
    "indiccorp_eval_small": (None, "eval_small"),
    "flores_hi": ("hi", "flores"),
    "flores_mr": ("mr", "flores"),
    "flores": (None, "flores"),   # the run's language; renamed flores_{lang} in evaluate_run
}


def dataset_lang(name, cfg):
    lang, _ = DATASETS[name]
    return lang or cfg.lang


def load_texts(name, cfg, n=None):
    from mtp.data.corpus import load_split

    _, split = DATASETS[name]
    return load_split(dataset_lang(name, cfg), split, n)


def grouper_for_lang(spec, lang):
    """Grouper name for a run in `lang`: `spec` is one name, or "lang:name,lang:name" (None if
    `lang` is not listed, i.e. use the run's config)."""
    if not spec or ":" not in spec:
        return spec or None
    by_lang = dict(part.strip().split(":", 1) for part in spec.split(",") if part.strip())
    return by_lang.get(lang) or None


def fp16_amp(cfg, model, fast_linear=False):
    """Autocast factory for runs that use fp16 autocast (T4), else None. fast_linear: also compute every linear layer
    from a cached fp16 weight copy (mtp/eval/precast.py; bit-identical, skips the per-call weight cast)."""
    from mtp.device import autocast_ctx, uses_fp16_autocast

    if not uses_fp16_autocast(cfg):
        return None
    amp = lambda: autocast_ctx(cfg)  # noqa: E731
    if fast_linear:
        from mtp.eval.precast import with_fast_linear

        amp = with_fast_linear(amp, model)
    return amp


def resolve_ignore_eos(value, cfg):
    """auto: on for a frozen backbone on an instruct base model (it keeps stopping at EOS after one
    sentence); LoRA-tuned runs (R9), frozen runs whose LoRA comes from a tuned run (init_lora_from,
    e.g. Rsd_r8 from R8) and base models (ganga-1b) run to max_new_tokens anyway."""
    if value in (True, "on"):
        return True
    if value in (False, "off", None):
        return False
    return (bool(cfg_get(cfg, "freeze_backbone", False)) and not cfg_get(cfg, "init_lora_from")
            and "instruct" in str(cfg.model_name).lower())


def resolve_grouper(name, lang):
    """(grouper, name) or (None, None) if `name` is unset, not registered yet (e.g. mr_rules_v1
    before JI-7) or does not support `lang` (hi_rules_v0 on flores_mr): the run is then evaluated
    without the in-group / at-boundary split."""
    from mtp.data.grouping.base import get_grouper

    if not name:
        return None, None
    try:
        return get_grouper(name, lang), name
    except KeyError:
        warnings.warn(f"grouper {name!r} is not registered yet; evaluating without the in-group split")
    except ValueError as e:  # language not supported by this grouper
        warnings.warn(f"grouper {name!r} cannot group {lang!r} ({e}); evaluating without the in-group split")
    return None, None


def make_examples(texts, tokenizer, grouper, max_length):
    if grouper is not None:
        from mtp.data.grouping.align import label_batch

        return label_batch(texts, tokenizer, grouper, max_length=max_length)
    enc = tokenizer(list(texts), truncation=True, max_length=max_length)
    return [{"input_ids": list(i), "attention_mask": list(m)} for i, m in zip(enc["input_ids"], enc["attention_mask"])]


def write_result(path, record):
    """Write record, keeping sections of an existing file for the same run / step / grouper
    that this call did not compute (e.g. spec_decode from another session)."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if all(old.get(k) == record[k] for k in ("run_name", "dataset", "step", "grouper")):
            for key in ("per_head", "spec_decode"):
                if record.get(key) is None and old.get(key) is not None:
                    record[key] = old[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(record) + "\n", encoding="utf-8")


_FLAT_LIST = re.compile(r"\[\s*((?:[-0-9.eE+]+|true|false|null)(?:,\s*(?:[-0-9.eE+]+|true|false|null))*)\s*\]")


def dumps(record):
    """json.dumps with indent 2, but lists of plain numbers / bools on one line: per-prompt token
    ids would otherwise take a line each and make the eval JSON ~3x larger."""
    text = json.dumps(record, indent=2, ensure_ascii=False)
    return _FLAT_LIST.sub(lambda m: "[" + re.sub(r",\s*", ", ", m.group(1)) + "]", text)


def print_table(name, per_head):
    print(f"\n{name}")
    print(f"  {'head':>4} {'n':>7} {'loss':>7} {'ppl':>8} {'top1':>6} {'top5':>6} {'in-grp':>7} {'bound':>7}")
    pct = lambda v: f"{100 * v:6.1f}" if v is not None else "     -"  # noqa: E731
    for r in per_head:
        print(f"  {r['head']:>4} {r['n']:>7} {r['loss']:>7.3f} {r['ppl']:>8.2f} {pct(r['top1'])} {pct(r['top5'])}"
              f" {pct(r['top1_in_group'])} {pct(r['top1_at_boundary'])}")


def print_spec(name, entries):
    print(f"\n{name}: speculative decoding")
    for e in entries:
        acc = " / ".join(f"{100 * a:.0f}%" if a is not None else "-" for a in e["accept_rate_per_head"])
        gi = f"{100 * e['group_integrity']:.0f}%" if e["group_integrity"] is not None else "-"
        ref = e.get("baseline", "greedy")   # sampling-mode entries: plain sampling with the same seeds
        print(f"  {e['policy']:<15} accepted len {e['mean_accepted_len']:.2f} | accept/head {acc} | "
              f"{e['tokens_per_sec']:.1f} vs {ref} {e['greedy_tokens_per_sec']:.1f} tok/s = x{e['speedup']:.2f} | "
              f"match {ref} {e['outputs_match_greedy']} ({100 * e['match_rate']:.0f}% of prompts"
              + (f", max margin at a divergence {e['max_mismatch_margin']:.3f}" if e["max_mismatch_margin"] is not None else "")
              + ")" + (f" | fp32 re-check {e['fp32_check']['outputs_match_greedy']} on {e['fp32_check']['n_prompts']}"
                       + (f" (margin {e['fp32_check']['max_mismatch_margin']:.2e})"
                          if e["fp32_check"].get("max_mismatch_margin") is not None else "")
                       if e["fp32_check"] else "")
              + f" | group integrity {gi} | {e['n_prompts']} prompts")


def make_policies(names, tau):
    from mtp.eval.draft_policy import POLICIES, get_policy
    from mtp.eval.typical import parse_typical

    missing = [p for p in names if p not in POLICIES and parse_typical(p) is None]
    if missing:
        raise SystemExit(f"draft polic(ies) {missing} not available; have {sorted(POLICIES)} and typical_<eps>")
    return [parse_typical(p) or (get_policy(p, tau=tau) if p == "confidence_cut" else get_policy(p)) for p in names]


def load_tree_policies(paths, prompt_lookup=None, typical=(), sample=(), penalty=()):
    """TreePolicy JSON files (mtp/eval/tree_policy.py save_policy; e.g. results/{run}/tree_policy_entropy_n25.json),
    relative to the working directory or the repo. Each is named after its file: tree_entropy_n25.
    typical (eps values) / sample ((T, top_p) pairs): also every tree, incl. its lookup variant, decoded with
    typical acceptance (TypicalTree, <name>+typ<eps>) / in sampling mode (SampledTree, <name>+samp<T>p<top_p>)."""
    from mtp.eval.tree_policy import load_policy

    out = []
    for p in paths:
        path = Path(p) if Path(p).exists() else ROOT / p
        if not path.exists():
            raise SystemExit(f"tree policy {p} not found")
        policy = load_policy(path)
        policy.name = "tree_" + path.stem.replace("tree_policy_", "")
        out.append(policy)
    if prompt_lookup:   # (chains, length): also each tree + copied n-gram branches (mtp/eval/prompt_lookup.py)
        from mtp.eval.prompt_lookup import PromptLookup

        out += [PromptLookup(p, chains=prompt_lookup[0], length=prompt_lookup[1]) for p in list(out)]
    if typical or sample or penalty:   # decode-mode wrappers go outside PromptLookup: lookup chains follow the same rule
        from mtp.eval.sampling import SampledTree
        from mtp.eval.typical import TypicalTree

        exact = list(out)
        out += [TypicalTree(p, eps) for eps in typical for p in exact]
        out += [SampledTree(p, t, top_p) for t, top_p in sample for p in exact]
        if penalty:
            from mtp.eval.penalty import PenalizedTree

            out += [PenalizedTree(p, r) for r in penalty for p in exact]
    return out


def parse_sample(text):
    """'0.7,0.9' -> (0.7, 0.9): --tree_sample's temperature,top_p."""
    try:
        t, top_p = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected T,top_p (e.g. 0.7,0.9), got {text!r}") from None
    if t <= 0 or not 0 < top_p <= 1:
        raise argparse.ArgumentTypeError(f"need T > 0 and 0 < top_p <= 1, got {text!r}")
    return t, top_p


def evaluate_run(run_dir, datasets, step=None, grouper_name=None, out_dir=ROOT / "results", dump_tokens=False,
                 spec_decode=False, policies=("fixed_k", "confidence_cut"), tau=0.5, spec_n=200, max_new_tokens=64,
                 fp32_check_n=20, ignore_eos="auto", tree_policies=(), prompt_lookup=None, tree_typical=(), tree_sample=(),
                 fast_linear=False, tree_penalty=(), lexicon=None, lexicon_cfg=(4, 2, 12), half=False,
                 heads=True, bench_verify=False, device="auto", batch_size=None, n=None, texts_fn=load_texts):
    """Evaluate one run on each dataset; returns {dataset: path of eval_{dataset}.json}.
    heads=False skips the per-head section (e.g. a session that only adds spec_decode)."""
    import torch

    from mtp.data.collate import Collator
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.eval.head_accuracy import evaluate_heads
    from mtp.eval.spec_decode import bench_verify_pass, evaluate_spec_decode, make_prompts
    from mtp.model.checkpoint import load_run

    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}")
    if not heads and not spec_decode and not bench_verify:
        raise SystemExit("nothing to do: --no_heads without --spec_decode or --bench_verify")
    draft_policies = make_policies(policies, tau) + load_tree_policies(
        tree_policies, prompt_lookup, tree_typical, tree_sample, tree_penalty) if spec_decode else []

    model, tokenizer, cfg = load_run(run_dir, device=device, step=step)
    if half:   # plain fp16 weights, no autocast: timings only (greedy and drafts both run in fp16; no fp32 re-check)
        if not torch.cuda.is_available():
            raise SystemExit("--half needs a GPU")
        model.half()
    if lexicon and spec_decode:   # word-completion chains (mtp/eval/lexicon.py) on every exact tree policy
        from mtp.eval.lexicon import LexiconLookup, load_lexicon, load_lexicon_meta, word_start_flags

        flags = word_start_flags(tokenizer)
        chains, ext, ext_nodes = lexicon_cfg
        exact_trees = [p for p in draft_policies if getattr(p, "name", "").startswith("tree_") and all(
            t not in p.name for t in ("+typ", "+samp", "+rp"))]
        for path in ([lexicon] if isinstance(lexicon, str) else lexicon):   # one file or several (phrase tables)
            trie, meta = load_lexicon(path), load_lexicon_meta(path)
            draft_policies += [LexiconLookup(p, trie, flags, chains=chains, ext=ext, ext_nodes=ext_nodes,
                                             max_words=meta.get("max_words", 1), tag=meta.get("tag", "lex"))
                               for p in exact_trees]
    device = next(model.parameters()).device
    if batch_size:
        cfg.optim.batch_size = batch_size
    max_length = cfg_get(cfg, "data.max_length", 128)
    grouper_name = grouper_for_lang(grouper_name, cfg.lang) or cfg_get(cfg, "data.grouper")
    datasets = [f"flores_{cfg.lang}" if d == "flores" else d for d in datasets]
    collator = Collator(tokenizer.pad_token_id)
    ignore_eos = resolve_ignore_eos(ignore_eos, cfg)
    print(f"run {cfg.run_name} | step {model.loaded_step} | heads {model.num_heads} | device {device}"
          + (" | ignore_eos" if ignore_eos and spec_decode else ""))
    if spec_decode and model.num_heads < 2:
        warnings.warn(f"{cfg.run_name} has one head (no drafts): skipping speculative decoding")
        draft_policies = []

    written = {}
    if bench_verify:   # once per run: verify-pass cost vs positions scored (chain drafts / tree nodes)
        texts = texts_fn(datasets[0], cfg, n)
        prefix = [t for text in texts for t in tokenizer(text)["input_ids"]][:64]
        amp_b = fp16_amp(cfg, model, fast_linear)
        bench = {"run_name": cfg.run_name, "step": model.loaded_step, "device": str(device),
                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "fp16_autocast": amp_b is not None, **bench_verify_pass(model, prefix, amp=amp_b)}
        path = Path(out_dir) / cfg.run_name / "bench_verify.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bench, indent=2) + "\n", encoding="utf-8")
        written["bench_verify"] = path
        print(f"\nverify-pass cost ({bench['gpu'] or device}, prefix {bench['prefix_len']} tokens): "
              f"greedy 1-token pass {bench['greedy_ms']:.1f} ms")
        for b in bench["passes"]:
            print(f"  {b['positions']:>3} positions: {b['ms']:7.1f} ms  (x{b['x_greedy']:.2f} greedy, p90 {b['ms_p90']:.1f})")
        if bench["draft_ms"] is not None:
            print(f"  + draft chain ({model.num_heads - 1} sequential steps): {bench['draft_ms']:.1f} ms per decoding step")
    if not heads and not spec_decode:
        return written
    for name in datasets:
        t0 = time.perf_counter()
        texts = texts_fn(name, cfg, n)
        grouper, used_grouper = resolve_grouper(grouper_name, dataset_lang(name, cfg))
        out = Path(out_dir) / cfg.run_name
        per_head = spec = None
        if heads:
            examples = make_examples(texts, tokenizer, grouper, max_length)
            dump = out / f"tokens_{name}.jsonl" if dump_tokens else None
            per_head = evaluate_heads(model, examples, collator, cfg, device, top_k=(1, 5), dump_path=dump,
                                      texts=texts, tokenizer=tokenizer)
        if draft_policies:
            prompts = make_prompts(texts, tokenizer, n=spec_n)
            # fp16 autocast only where the run uses it (T4); the fp32 re-check then turns it off
            amp = None if half else fp16_amp(cfg, model, fast_linear)
            spec = evaluate_spec_decode(model, tokenizer, prompts, draft_policies, max_new_tokens=max_new_tokens,
                                        grouper=grouper, amp=amp, log_every=25, fp32_check_n=fp32_check_n,
                                        ignore_eos=ignore_eos)
        record = {
            "run_name": cfg.run_name, "dataset": name, "step": model.loaded_step, "grouper": used_grouper,
            "git_commit": cfg_get(cfg, "git_commit"), "per_head": per_head, "spec_decode": spec,
            **({"fast_linear": True} if fast_linear and spec and not half else {}),
            **({"half": True} if half else {}),
        }
        path = out / f"eval_{name}.json"
        write_result(path, record)
        written[name] = path
        took = f"{time.perf_counter() - t0:.0f}s"
        if per_head is not None:
            print_table(f"{name}: {len(texts)} sentences, grouper {used_grouper}, {took} -> {path}", per_head)
        if spec is not None:
            print_spec(f"{name} ({took} -> {path})", spec)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="run folder with config.yaml and step_N/")
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"], choices=sorted(DATASETS))
    ap.add_argument("--step", type=int, default=None, help="default: latest step")
    ap.add_argument("--grouper", default=None, help='one name or "hi:hi_rules_v0,mr:mr_rules_v1"; '
                    "default: data.grouper from the run's config")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--dump_tokens", action="store_true", help="also write results/{run}/tokens_{dataset}.jsonl")
    ap.add_argument("--no_heads", action="store_true", help="skip the per-head section (keeps an existing one)")
    ap.add_argument("--spec_decode", action="store_true", help="self-speculative decoding (OM-6)")
    ap.add_argument("--policies", nargs="+", default=["fixed_k", "confidence_cut"])
    ap.add_argument("--tree_policies", nargs="+", default=[],
                    help="TreePolicy JSON files verified with generate_tree, e.g. results/<run>/tree_policy_entropy_n25.json")
    ap.add_argument("--fast_linear", action="store_true",
                    help="fp16-autocast runs: compute linear layers from cached fp16 weights (bit-identical, faster)")
    ap.add_argument("--prompt_lookup", type=int, nargs=2, metavar=("CHAINS", "LENGTH"), default=None,
                    help="also run each tree policy with that many copied n-gram chains of that length grafted on")
    ap.add_argument("--tree_typical", type=float, nargs="+", metavar="EPS", default=[],
                    help="also run every tree policy (incl. --prompt_lookup ones) with typical acceptance (lossy), "
                         "one variant per eps: <tree>+typ<eps>")
    ap.add_argument("--tree_sample", type=parse_sample, nargs="+", metavar="T,TOP_P", default=[],
                    help="also run every tree policy (incl. --prompt_lookup ones) in sampling mode, one variant per "
                         "setting: <tree>+samp<T>p<top_p>, checked against plain sampling with the same seeds")
    ap.add_argument("--tree_penalty", type=float, nargs="+", metavar="R", default=[],
                    help="also run every tree policy (incl. --prompt_lookup ones) as repetition-penalised greedy "
                         "(CTRL penalty R): <tree>+rp<R>, checked against penalised greedy")
    ap.add_argument("--half", action="store_true",
                    help="cast the model to fp16 weights and decode without autocast (speed only; use a separate --out)")
    ap.add_argument("--lexicon", nargs="+", default=None,
                    help="word-completion lexicon JSON (scripts/build_lexicon.py) or phrase tables "
                         "(scripts/build_group_table.py): also run every exact tree policy with each file's chains, "
                         "<tree>+<tag><chains>e<ext>x<ext_nodes> (tag from the file, default lex)")
    ap.add_argument("--lexicon_cfg", type=int, nargs=3, metavar=("CHAINS", "EXT", "EXT_NODES"), default=[4, 2, 12],
                    help="completions at the root, completions per extended head node, head nodes extended")
    ap.add_argument("--tau", type=float, default=0.5, help="confidence_cut threshold")
    ap.add_argument("--spec_n", type=int, default=200, help="prompts per dataset (first 8-16 words of a sentence)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--bench_verify", action="store_true",
                    help="time one verify pass vs positions scored (1..64) -> results/{run}/bench_verify.json")
    ap.add_argument("--fp32_check_n", type=int, default=20,
                    help="under fp16 autocast, re-run this many prompts in fp32: outputs must equal greedy")
    ap.add_argument("--ignore_eos", choices=["auto", "on", "off"], default="auto",
                    help="mask EOS for head 0 in spec decode; auto = frozen backbone on an instruct model, "
                         "no init_lora_from")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=None, help="default: optim.batch_size of the run")
    ap.add_argument("--n", type=int, default=None, help="cap sentences per dataset (smoke tests)")
    args = ap.parse_args(argv)

    evaluate_run(args.run_dir, args.datasets, step=args.step, grouper_name=args.grouper, out_dir=args.out,
                 dump_tokens=args.dump_tokens, spec_decode=args.spec_decode, policies=args.policies, tau=args.tau,
                 tree_policies=args.tree_policies, prompt_lookup=args.prompt_lookup,
                 tree_typical=args.tree_typical, tree_sample=args.tree_sample, fast_linear=args.fast_linear,
                 tree_penalty=args.tree_penalty, lexicon=args.lexicon, lexicon_cfg=tuple(args.lexicon_cfg), half=args.half,
                 spec_n=args.spec_n, max_new_tokens=args.max_new_tokens, fp32_check_n=args.fp32_check_n,
                 ignore_eos=args.ignore_eos, heads=not args.no_heads, bench_verify=args.bench_verify,
                 device=args.device, batch_size=args.batch_size, n=args.n)


if __name__ == "__main__":
    main()
