"""
Quality of decoded outputs (exact, lossy, sampled), from the stored spec-decode outputs (no decoding).

    python scripts/lossy_quality.py --results results/R12_hi_k4_struct_x4 [--datasets indiccorp_eval flores_hi]
        [--judge NAME] [--tokenizer RUN_TOKENIZER] [--text_level auto|on|off] [--judge_dtype float32|float16]
        [--human_ref] [--lang hi|mr] [--suffix SLUG]

For every spec_decode policy in results/{run}/eval_{dataset}.json, per prompt:
  nll_sum / n_tok  NLL of the generated tokens given the prompt under a FIXED judge (default plain ganga-1b,
                   fp32), the same for every run, so perplexities compare across runs;
  rep4             share of generated 4-grams already seen earlier in prompt + output (degeneration, run tokens);
  div              first position where the output leaves the reference output (fixed_k = the run's greedy
                   output: exact decoding; else the first exact policy), None if identical.

Judge and run tokenizer. The run's tokenizer (--tokenizer; default: model_name recorded in the eval JSON, else
in configs/*.yaml with the same run_name, else in {run}/config.yaml, else LingoIITGN/ganga-1b) wrote the ids.
  id level    (judge tokenizer == run tokenizer): ids are fed to the judge as they are (the original scorer);
  text level  (--text_level auto: the tokenizers differ by name and vocab; or on): prompt and output are decoded
              with the run tokenizer, prompt + continuation is re-tokenised by the judge (its own BOS included),
              and the NLL of the judge tokens that end after the prompt's characters is summed (tokenizers
              without offsets: tokens after the common prefix of prompt and full text). `ppl` is then per JUDGE
              token and `ppl_norep` is None (the 4-gram mask lives in run-token space).
Every judge also reports tokenizer-independent numbers on the decoded continuation:
  ppl_word = exp(total NLL / total whitespace words), nll_per_char = total NLL / total characters (stripped).

--human_ref adds a pseudo-policy `human`: the rest of each prompt's source sentence. Prompts are rebuilt exactly
as scripts/evaluate.py --spec_decode did (its load_texts, make_prompts defaults, seed 0); if they differ from the
stored prompt_ids the row is skipped (reason in human_skipped). The continuation is the tail of tokenising the
whole sentence (prompt ids must be its prefix; else the rest is tokenised alone, counted in human.n_fallback),
capped at the longest stored output. Human continuations end with the sentence, so they are shorter: see
mean_new_tokens.

Writes results/{run}/quality_{dataset}.json for the default judge (ganga-1b) and quality_{dataset}.{slug}.json
for any other judge (slug from the judge name, or --suffix):
  {"run_name", "dataset", "judge", "tokenizer", "text_level", "judge_dtype", "div_reference",
   "policies": [{"policy", "ppl", "ppl_norep", "rep4", "identical_rate", "mean_accepted_len", "speedup",
                 "ppl_word", "nll_per_char", "n_text_fallback", "mean_new_tokens",
                 "per_prompt": {"nll_sum", "n_tok", "nll_norep_sum", "n_norep", "n_words", "n_chars", "rep4",
                                "div"}}],
   "human": {"n_prompts", "n_fallback", "max_new_tokens", "mean_new_tokens", "note"} | "human_skipped": reason}
(text level: no nll_norep_sum / n_norep; human: no div, identical_rate / speed fields None).
"""

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

DEFAULT_JUDGE = "LingoIITGN/ganga-1b"
DEFAULT_TOKENIZER = "LingoIITGN/ganga-1b"


def rep4(prompt, ids):
    """Share of the 4-grams that end in `ids` and already occurred earlier in prompt + ids."""
    seq = list(prompt) + list(ids)
    seen, hits, total = set(), 0, 0
    for i in range(3, len(seq)):
        g = tuple(seq[i - 3: i + 1])
        if i >= len(prompt):
            total += 1
            hits += g in seen
        seen.add(g)
    return hits / total if total else 0.0


def repeated_mask(prompt, ids):
    """Per generated token: True if the 4-gram ending at it already occurred earlier in prompt + ids."""
    seq = list(prompt) + list(ids)
    seen, out = set(), []
    for i in range(len(seq)):
        g = tuple(seq[i - 3: i + 1]) if i >= 3 else None
        if i >= len(prompt):
            out.append(g is not None and g in seen)
        if g is not None:
            seen.add(g)
    return out


def nll_batch(model, pairs, device):
    """[(prompt ids, generated ids)] -> [(sum NLL of the generated ids, count, sum NLL / count over the tokens
    that do not close a repeated 4-gram)] in one padded forward pass."""
    import torch

    seqs = [list(p) + list(g) for p, g in pairs]
    width = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    with torch.no_grad():
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tok = logp.gather(-1, ids[:, 1:].to(device)[..., None])[..., 0]
    out = []
    for i, (p, g) in enumerate(pairs):
        lo, hi = len(p) - 1, len(p) - 1 + len(g)          # positions predicting the generated tokens
        keep = torch.tensor([not r for r in repeated_mask(p, g)], device=device)
        span = tok[i, lo:hi]
        out.append((float(-span.sum()), len(g), float(-span[keep].sum()), int(keep.sum())))
    return out


# ---------------------------------------------------------------- text level (judge with its own tokenizer)

def count_words(text):
    """Whitespace words (a word cut by the token budget counts as one)."""
    return len(text.split())


def decode_split(tok, prompt_ids, gen_ids):
    """(prompt text, prompt + continuation text, consistent), decoded with the run tokenizer. consistent is False
    when decoding prompt + continuation does not start with the decoded prompt; the continuation is then decoded
    alone and appended (with a space if it does not start with one)."""
    p = tok.decode(list(prompt_ids), skip_special_tokens=True)
    full = tok.decode(list(prompt_ids) + list(gen_ids), skip_special_tokens=True)
    if full.startswith(p):
        return p, full, True
    c = tok.decode(list(gen_ids), skip_special_tokens=True)
    return p, p + ("" if not c or c[0].isspace() else " ") + c, False


def _strip_special_tail(ids, special):
    n = len(ids)
    while n and ids[n - 1] in special:
        n -= 1
    return n


def continuation_span(judge_tok, prompt_text, full_text, use_offsets=None):
    """Tokenise full_text with the judge (its own special tokens included) -> (ids, start, end): ids[start:end]
    are the continuation's tokens. With offsets (fast tokenizers): tokens whose character span ends after
    len(prompt_text) (special tokens with an empty span excluded). Without: tokens after the common prefix of
    the prompt's and the full text's ids. Trailing special tokens (an appended EOS) are never counted. start >= 1
    so every counted token has a context."""
    special = set(judge_tok.all_special_ids)
    if use_offsets is None:
        use_offsets = bool(getattr(judge_tok, "is_fast", False))
    if use_offsets:
        enc = judge_tok(full_text, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
        cut = len(prompt_text)
        idx = [i for i, (s, e) in enumerate(offs) if e > cut and not (s == e and ids[i] in special)]
        if not idx:
            return ids, len(ids), len(ids)
        start, end = idx[0], idx[-1] + 1
    else:
        ids = list(judge_tok(full_text)["input_ids"])
        p = list(judge_tok(prompt_text)["input_ids"])
        p = p[: _strip_special_tail(p, special)]
        start = 0
        while start < min(len(p), len(ids)) and p[start] == ids[start]:
            start += 1
        end = max(_strip_special_tail(ids, special), start)
    return ids, max(start, 1), max(end, 1)


def judge_context(model):
    cfg = model.config
    for k in ("max_position_embeddings", "n_positions", "seq_length"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


def nll_spans(model, items, device, pad_id=0, max_len=None):
    """[(ids, start, end)] -> [(sum NLL of ids[start:end], end - start)] in one right-padded, masked forward
    pass. Sequences longer than max_len are left-truncated (the counted span is kept)."""
    import torch

    seqs = []
    for ids, start, end in items:
        ids = list(ids)[:end]
        if max_len and len(ids) > max_len:
            drop = len(ids) - max_len
            ids, start = ids[drop:], max(start - drop, 1)
        seqs.append((ids, start, len(ids)))
    width = max(len(s[0]) for s in seqs)
    x = torch.full((len(seqs), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, (s, _, _) in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    with torch.no_grad():
        logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tok = logp.gather(-1, x[:, 1:].to(device)[..., None])[..., 0]
    return [(float(-tok[i, start - 1: end - 1].sum()), max(end - start, 0)) for i, (_, start, end) in enumerate(seqs)]


def judge_pad_id(judge_tok):
    for t in (judge_tok.pad_token_id, judge_tok.eos_token_id, judge_tok.unk_token_id):
        if t is not None:
            return t
    return 0


def same_tokenizer(a, b):
    """True if two tokenizers are the same by name or by vocabulary."""
    na, nb = getattr(a, "name_or_path", None), getattr(b, "name_or_path", None)
    if na and nb and na == nb:
        return True
    try:
        return a.get_vocab() == b.get_vocab()
    except Exception:  # noqa: BLE001
        return False


def judge_slug(name):
    """Filesystem-safe short name of a judge: 'sarvamai/sarvam-1' -> 'sarvam-1'."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).replace("\\", "/").rstrip("/").split("/")[-1]).strip("._")
    return s or "judge"


def quality_name(dataset, judge, suffix=None):
    """quality_{dataset}.json for the default judge (and no suffix), else quality_{dataset}.{slug}.json."""
    if suffix:
        return f"quality_{dataset}.{judge_slug(suffix)}.json"
    if judge == DEFAULT_JUDGE:
        return f"quality_{dataset}.json"
    return f"quality_{dataset}.{judge_slug(judge)}.json"


# ---------------------------------------------------------------- human reference

def human_continuations(texts, tokenizer, n, max_new_tokens, min_words=8, max_words=16, seed=0):
    """Mirror of mtp.eval.spec_decode.make_prompts (same rng draws): -> (prompt ids, continuation ids,
    n_fallback). The continuation is the tail of tokenising the whole sentence when the prompt ids are its
    prefix (ganga: no BOS/EOS, '▁word' tokens), else the remaining words tokenised alone (n_fallback += 1);
    trailing special tokens dropped, capped at max_new_tokens."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    rng = random.Random(seed)
    prompts, conts, fallback = [], [], 0
    for text in texts:
        words = text.split()
        w = rng.randint(min_words, max_words)
        if len(words) <= w:
            continue
        p = list(tokenizer(" ".join(words[:w]))["input_ids"])
        full = list(tokenizer(" ".join(words))["input_ids"])
        if full[: len(p)] == p:
            c = full[len(p):]
        else:
            fallback += 1
            c = list(tokenizer(" ".join(words[w:]), add_special_tokens=False)["input_ids"])
        c = c[: _strip_special_tail(c, special)]
        prompts.append(p)
        conts.append(c[:max_new_tokens])
        if len(prompts) >= n:
            break
    return prompts, conts, fallback


def _evaluate_module():
    """scripts/evaluate.py as a module (load_texts, DATASETS), by path: `import evaluate` may hit the HF package."""
    spec = importlib.util.spec_from_file_location("_mtp_scripts_evaluate", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_eval_texts(dataset, lang):
    """The texts scripts/evaluate.py used for `dataset` (its load_texts, n=None)."""
    from types import SimpleNamespace

    ev = _evaluate_module()
    if dataset not in ev.DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}")
    return ev.load_texts(dataset, SimpleNamespace(lang=lang), None)


def human_entry(rec, tokenizer, texts, max_new_tokens=None):
    """(pseudo spec_decode entry `human`, None) for an eval record, or (None, reason) if the prompts cannot be
    rebuilt exactly."""
    from mtp.eval.spec_decode import make_prompts

    entries = rec.get("spec_decode") or []
    if not entries:
        return None, "no spec_decode entries"
    stored = entries[0]["per_prompt"]["prompt_ids"]
    if max_new_tokens is None:
        max_new_tokens = max((len(g) for e in entries for g in e["per_prompt"]["ids"]), default=0)
    prompts, conts, fallback = human_continuations(texts, tokenizer, len(stored), max_new_tokens)
    if prompts != make_prompts(texts, tokenizer, n=len(stored)):
        return None, "internal: rebuilt prompts differ from make_prompts"
    if [list(p) for p in stored] != prompts:
        same = sum(list(a) == b for a, b in zip(stored, prompts))
        return None, (f"rebuilt prompts differ from the stored prompt_ids ({same}/{len(stored)} equal, "
                      f"{len(prompts)} rebuilt): different texts, tokenizer or make_prompts arguments")
    return {"policy": "human", "per_prompt": {"prompt_ids": prompts, "ids": conts},
            "_human": {"n_prompts": len(conts), "n_fallback": fallback, "max_new_tokens": max_new_tokens,
                       "mean_new_tokens": sum(map(len, conts)) / len(conts) if conts else None,
                       "note": "rest of the source sentence, tokenised with the run tokenizer; shorter than "
                               "the model outputs (the sentence ends)"}}, None


# ---------------------------------------------------------------- run info

def run_info(folder, rec, configs_dir=ROOT / "configs"):
    """{'tokenizer', 'lang'} of a run from what is recorded: the eval JSON (model_name / base_model / tokenizer,
    lang), {folder}/config.yaml, configs/*.yaml with the same run_name. Missing values are None."""
    info = {"tokenizer": next((rec[k] for k in ("model_name", "base_model", "tokenizer") if rec.get(k)), None),
            "lang": rec.get("lang")}
    if info["tokenizer"] and info["lang"]:
        return info
    import yaml

    for path in [Path(folder) / "config.yaml"] + sorted(Path(configs_dir).glob("*.yaml")):
        if not path.is_file():
            continue
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(cfg, dict):
            continue
        if path.parent != Path(folder) and cfg.get("run_name") != rec.get("run_name"):
            continue
        info["tokenizer"] = info["tokenizer"] or cfg.get("model_name")
        info["lang"] = info["lang"] or cfg.get("lang")
        if info["tokenizer"] and info["lang"]:
            break
    return info


def dataset_language(dataset, run_lang):
    """flores_xx -> xx; any other dataset is in the run's language."""
    m = re.fullmatch(r"flores_(\w+)", dataset)
    return m.group(1) if m else run_lang


# ---------------------------------------------------------------- scoring

def _is_exact(name):
    return not re.search(r"typ|samp", name)


def reference_policy(entries):
    """fixed_k (the run's greedy output) if present, else the first exact policy, else None."""
    names = [e["policy"] for e in entries]
    if "fixed_k" in names:
        return "fixed_k"
    return next((n for n in names if _is_exact(n)), None)


def score_entries(entries, model, device, batch_size=32, run_tok=None, judge_tok=None, text_level=False,
                  ref_ids=None):
    """One result dict per spec_decode-like entry (fields: module docstring). run_tok None: no text fields."""
    from mtp.eval.spec_decode import first_divergence

    if text_level and (run_tok is None or judge_tok is None):
        raise ValueError("text-level scoring needs the run tokenizer and the judge tokenizer")
    pad_id = judge_pad_id(judge_tok) if text_level else 0
    max_len = judge_context(model) if text_level else None
    res = []
    for e in entries:
        pp = e["per_prompt"]
        pairs = list(zip(pp["prompt_ids"], pp["ids"]))
        texts = words = chars = None
        if run_tok is not None:
            texts = [decode_split(run_tok, p, g) for p, g in pairs]
            conts = [full[len(pt):] for pt, full, _ in texts]
            words = [count_words(c) for c in conts]
            chars = [len(c.strip()) for c in conts]
        nll = []
        if text_level:
            spans = [continuation_span(judge_tok, pt, full) for pt, full, _ in texts]
            for i in range(0, len(spans), batch_size):
                nll += [(s, n, None, None) for s, n in nll_spans(model, spans[i: i + batch_size], device, pad_id,
                                                                  max_len)]
        else:
            for i in range(0, len(pairs), batch_size):
                nll += nll_batch(model, pairs[i: i + batch_size], device)
        reps = [rep4(p, g) for p, g in pairs]
        human = e["policy"] == "human"
        div = None
        if not human:
            div = [first_divergence(a, b) for a, b in zip(pp["ids"], ref_ids)] if ref_ids else [None] * len(pairs)
        tot, n = sum(x[0] for x in nll), sum(x[1] for x in nll)
        r = {"policy": e["policy"], "ppl": math.exp(tot / n) if n else None}
        if text_level:
            r["ppl_norep"] = None
        else:
            tot_nr, n_nr = sum(x[2] for x in nll), sum(x[3] for x in nll)
            r["ppl_norep"] = math.exp(tot_nr / n_nr) if n_nr else None
        r.update({"rep4": sum(reps) / len(reps) if reps else None,
                  "identical_rate": None if human else (sum(d is None for d in div) / len(div) if div else None),
                  "mean_accepted_len": e.get("mean_accepted_len"), "speedup": e.get("speedup")})
        if words is not None:
            nw, nc = sum(words), sum(chars)
            r["ppl_word"] = math.exp(tot / nw) if nw else None
            r["nll_per_char"] = tot / nc if nc else None
            r["n_text_fallback"] = sum(not ok for _, _, ok in texts)
        r["mean_new_tokens"] = sum(len(g) for _, g in pairs) / len(pairs) if pairs else None
        per = {"nll_sum": [x[0] for x in nll], "n_tok": [x[1] for x in nll]}
        if not text_level:
            per.update({"nll_norep_sum": [x[2] for x in nll], "n_norep": [x[3] for x in nll]})
        if words is not None:
            per.update({"n_words": words, "n_chars": chars})
        per["rep4"] = reps
        if not human:
            per["div"] = div
        r["per_prompt"] = per
        res.append(r)
    return res


def score_file(path, model, device, batch_size=32, run_tok=None, judge_tok=None, text_level=False, human=None):
    """-> (eval record, [policy results], reference policy). human: an extra `human` entry (human_entry)."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = list(rec.get("spec_decode") or [])
    ref = reference_policy(entries)
    ref_ids = next((e["per_prompt"]["ids"] for e in entries if e["policy"] == ref), None) if ref else None
    if human is not None:
        entries.append(human)
    return rec, score_entries(entries, model, device, batch_size, run_tok, judge_tok, text_level, ref_ids), ref


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="results/{run_name} folders with eval_*.json")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="any HF causal LM")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every eval_*.json in the folder")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tokenizer", default=None,
                    help=f"the run's tokenizer (default: the run's recorded model_name, else {DEFAULT_TOKENIZER})")
    ap.add_argument("--text_level", choices=["auto", "on", "off"], default="auto",
                    help="score decoded text with the judge's own tokenizer (auto: when the tokenizers differ)")
    ap.add_argument("--judge_dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--human_ref", action="store_true", help="add the `human` row (rest of the source sentence)")
    ap.add_argument("--lang", default=None, help="run language for --human_ref (default: the run's config)")
    ap.add_argument("--suffix", default=None, help="write quality_{dataset}.{suffix}.json")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16}[args.judge_dtype]
    model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=dtype).to(device).eval()
    judge_tok = AutoTokenizer.from_pretrained(args.judge)
    toks = {}
    for folder in map(Path, args.results):
        paths = [folder / f"eval_{d}.json" for d in args.datasets] if args.datasets else sorted(folder.glob("eval_*.json"))
        for path in paths:
            if not path.exists():
                print(f"skip {path} (missing)")
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            info = run_info(folder, rec)
            tok_name = args.tokenizer or info["tokenizer"] or DEFAULT_TOKENIZER
            if tok_name not in toks:
                toks[tok_name] = AutoTokenizer.from_pretrained(tok_name)
            run_tok = toks[tok_name]
            text_level = {"on": True, "off": False}.get(args.text_level)
            if text_level is None:
                text_level = not same_tokenizer(run_tok, judge_tok)
            human, skipped = None, None
            if args.human_ref:
                lang = dataset_language(rec["dataset"], args.lang or info["lang"])
                if not lang:
                    skipped = "run language unknown: pass --lang"
                else:
                    try:
                        human, skipped = human_entry(rec, run_tok, load_eval_texts(rec["dataset"], lang))
                    except Exception as e:  # noqa: BLE001
                        skipped = f"texts for {rec['dataset']} ({lang}) not loaded: {e!r}"
                if skipped:
                    print(f"  human row skipped for {rec['run_name']} | {rec['dataset']}: {skipped}", flush=True)
            rec, res, ref = score_file(path, model, device, args.batch_size, run_tok, judge_tok, text_level, human)
            out = {"run_name": rec["run_name"], "dataset": rec["dataset"], "judge": args.judge,
                   "tokenizer": tok_name, "text_level": text_level, "judge_dtype": args.judge_dtype,
                   "div_reference": ref, "policies": res}
            if human is not None:
                out["human"] = human["_human"]
            elif skipped:
                out["human_skipped"] = skipped
            dest = path.with_name(quality_name(rec["dataset"], args.judge, args.suffix))
            dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
            fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
            print(f"{rec['run_name']} | {rec['dataset']} -> {dest.name}: " + ", ".join(
                f"{r['policy']} ppl {fmt(r['ppl'], '.2f')} ppl_word {fmt(r.get('ppl_word'), '.2f')} "
                f"tok/step {fmt(r['mean_accepted_len'], '.2f')} rep4 {fmt(r['rep4'], '.3f')}"
                for r in res), flush=True)


if __name__ == "__main__":
    main()
