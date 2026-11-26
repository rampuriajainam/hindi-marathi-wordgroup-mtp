"""
Words per decoding step and per second from stored spec-decode outputs (second-model comparison,
docs/plan/SECOND_MODEL.md §5). Tokenizers differ (ganga ~1.1, sarvam-1 ~1.4, Qwen2.5 ~4.7 tokens per Hindi word),
so tokens/step is not comparable across models; words/step and words/sec are.

    python scripts/words_per_step.py results/<run>/eval_indiccorp_eval.json [...] [--tokenizer NAME] [--out x.json]

Per prompt (from the §10 per_prompt lists, no decoding run): words = whitespace words of
decode(prompt_ids + ids) minus those of decode(prompt_ids); steps = verification steps. Per policy:
  words_per_step = sum(words) / sum(steps)            (a step = one verify pass, the unit of the speed-up)
  tokens_per_word = sum(tokens) / sum(words)
  words_per_sec = tokens_per_sec / tokens_per_word    (tokens_per_sec is total tokens / total seconds, so this is
                                                       total words / total seconds); greedy_words_per_sec likewise
The speed-up is unchanged (same text, same seconds). Prompts are the same sentences and words for every model
(make_prompts picks 8-16 *words*), so per-prompt words/step can be paired across models (per_prompt_words).
--tokenizer defaults to the model_name in runs/<run_name>/config.yaml or --run_dir.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def continuation_words(tokenizer, prompt_ids, ids):
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    full = tokenizer.decode(list(prompt_ids) + list(ids), skip_special_tokens=True)
    return len(full.split()) - len(prompt.split())


def words_summary(entry, tokenizer):
    """One §10 spec_decode entry -> dict of word-level rates (see module docstring)."""
    pp = entry["per_prompt"]
    words = [continuation_words(tokenizer, p, i) for p, i in zip(pp["prompt_ids"], pp["ids"])]
    tokens, steps = sum(pp["tokens"]), sum(pp["steps"])
    tpw = tokens / sum(words) if sum(words) else None
    return {
        "policy": entry["policy"], "n_prompts": len(words),
        "tokens_per_step": tokens / steps if steps else None,
        "words_per_step": sum(words) / steps if steps else None,
        "tokens_per_word": tpw,
        "words_per_sec": entry["tokens_per_sec"] / tpw if tpw else None,
        "greedy_words_per_sec": entry["greedy_tokens_per_sec"] / tpw if tpw else None,
        "speedup": entry.get("speedup"),
        "mean_new_words": sum(words) / len(words) if words else None,
        "per_prompt_words": words, "per_prompt_steps": [int(s) for s in pp["steps"]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--run_dir", default=None, help="run folder whose config.yaml names the model")
    ap.add_argument("--out", default=None, help="write the summaries (with per-prompt lists) to this JSON")
    args = ap.parse_args(argv)

    from mtp.config import load_config
    from mtp.model.build import load_tokenizer

    rows, toks = [], {}
    for f in args.files:
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        name = args.tokenizer
        if name is None:
            cfg_path = Path(args.run_dir or ROOT / "runs" / rec["run_name"]) / "config.yaml"
            if not cfg_path.exists():
                raise SystemExit(f"{f}: pass --tokenizer or --run_dir (no {cfg_path})")
            name = load_config(cfg_path).model_name
        tok = toks.get(name) or toks.setdefault(name, load_tokenizer(name))
        for entry in rec.get("spec_decode") or []:
            if "ids" not in entry.get("per_prompt", {}):        # entries written before outputs were stored (#64)
                print(f"{rec['run_name']} {rec['dataset']} {entry['policy']}: no stored ids, skipped")
                continue
            s = words_summary(entry, tok)
            rows.append({"run_name": rec["run_name"], "dataset": rec["dataset"], "tokenizer": name, **s})
            f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
            print(f"{rec['run_name']:<34} {rec['dataset']:<15} {s['policy']:<28} tok/step {f(s['tokens_per_step'])} | "
                  f"words/step {f(s['words_per_step'])} | tok/word {f(s['tokens_per_word'])} | "
                  f"words/s {f(s['words_per_sec'], 1)} vs greedy {f(s['greedy_words_per_sec'], 1)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
