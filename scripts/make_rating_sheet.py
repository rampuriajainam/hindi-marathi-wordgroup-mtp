"""
Blind human-rating sheet for decoded outputs (greedy vs lossy vs sampled), from stored spec-decode outputs.

    python scripts/make_rating_sheet.py --eval results/lossy_tree/Rsd_hi_k4_frozen_sd_seq/eval_indiccorp_eval.json \
        --n 50 --out ratings/hi [--seed 0]

Picks n prompts (seeded), decodes each system's continuation with the run tokenizer, shuffles the systems per item,
and writes
  {out}_sheet.csv   one row per (item, output): item, prompt, label A-D, continuation, fluency, coherence, rank
                    (UTF-8 with BOM so Excel shows Devanagari); raters fill the last three columns
  {out}_key.json    item -> {label: policy}; keep it away from the raters
Scoring: scripts/score_ratings.py.
"""

import argparse
import csv
import json
import random
from pathlib import Path

SYSTEMS = {"greedy": "fixed_k", "lossy_tree": "tree_static_n64+pld4x6+typ0.5", "lossy_chain": "typical_0.02",
           "sampled_T0.7": "tree_static_n64+samp0.7p0.9"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None, help="default: LingoIITGN/ganga-1b, or Misal for a Marathi run")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    rec = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    name = args.tokenizer or ("smallstepai/Misal-1B-instruct-v0.1" if "mr" in rec["run_name"] else "LingoIITGN/ganga-1b")
    tok = AutoTokenizer.from_pretrained(name)
    pol = {e["policy"]: e["per_prompt"] for e in rec["spec_decode"]}
    missing = [p for p in SYSTEMS.values() if p not in pol]
    if missing:
        raise SystemExit(f"{args.eval}: missing policies {missing}")
    n_prompts = len(pol["fixed_k"]["prompt_ids"])
    rng = random.Random(args.seed)
    items = sorted(rng.sample(range(n_prompts), min(args.n, n_prompts)))
    rows, key = [], {}
    for k, i in enumerate(items, 1):
        prompt = tok.decode(pol["fixed_k"]["prompt_ids"][i], skip_special_tokens=True).strip()
        outs = [(s, tok.decode(pol[p]["ids"][i], skip_special_tokens=True).strip()) for s, p in SYSTEMS.items()]
        rng.shuffle(outs)
        key[str(k)] = {}
        for label, (system, text) in zip("ABCD", outs):
            key[str(k)][label] = system
            rows.append({"item": k, "prompt": prompt, "label": label, "continuation": text,
                         "fluency (1-5)": "", "coherence with prompt (1-5)": "", "rank in item (1=best)": ""})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Path(f"{out}_key.json").write_text(json.dumps({"eval": args.eval, "systems": SYSTEMS, "seed": args.seed,
                                                    "prompt_index": items, "key": key}, indent=1), encoding="utf-8")
    print(f"{len(items)} items x {len(SYSTEMS)} outputs -> {out}_sheet.csv (+ key)")


if __name__ == "__main__":
    main()
