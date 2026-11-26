"""
External baseline: Hugging Face `generate` on the same verifier, same prompts, same GPU session as evaluate.py.

    python scripts/hf_baseline.py --run_dir RUN --dataset indiccorp_eval [--n 200] [--max_new_tokens 64]
        [--out results] [--lookup 10]

The verifier (base model with the run's LoRA merged, heads unused) is timed with
  hf_greedy_fp16          model.half(), generate(do_sample=False)
  hf_pld_fp16             the same + prompt_lookup_num_tokens (HF's prompt-lookup decoding, exact)
  hf_greedy_autocast      fp32 weights + fp16 autocast (the configuration of evaluate.py, without --fast_linear)
  hf_pld_autocast         the same + prompt lookup
Every generation is forced to max_new_tokens (min_new_tokens), batch size 1, one warm-up prompt. Prompts are the
ones evaluate.py --spec_decode uses (load_texts, make_prompts seed 0). Writes {out}/{run}/hf_baseline_{dataset}.json
with tokens/s per configuration and the share of prompts whose PLD output equals the greedy output.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtp.eval.spec_decode import make_prompts
from mtp.model.checkpoint import load_run


def timed(model, prompts, n_new, ctx, **kw):
    outs, toks = [], 0
    dev = next(model.parameters()).device
    with ctx():
        model.generate(torch.tensor([prompts[0]], device=dev), max_new_tokens=8, min_new_tokens=8, do_sample=False,
                       **kw)                                   # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in prompts:
        with ctx():
            o = model.generate(torch.tensor([p], device=dev), max_new_tokens=n_new, min_new_tokens=n_new,
                               do_sample=False, **kw)
        new = o[0, len(p):].tolist()
        outs.append(new)
        toks += len(new)
    torch.cuda.synchronize()
    return toks / (time.perf_counter() - t0), outs


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dataset", default="indiccorp_eval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--lookup", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    import contextlib
    from scripts.evaluate import load_texts

    model, tok, cfg = load_run(args.run_dir)
    base = model.base_model
    if hasattr(base, "merge_and_unload"):
        base = base.merge_and_unload()
    base = base.float().eval()
    base.generation_config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompts = make_prompts(load_texts(args.dataset, cfg), tok, n=args.n, seed=0)
    auto = lambda: torch.autocast("cuda", dtype=torch.float16)
    res = {"run_name": cfg.run_name, "dataset": args.dataset, "n_prompts": len(prompts),
           "max_new_tokens": args.max_new_tokens, "lookup": args.lookup, "configs": {}}
    tps, outs = {}, {}
    tps["hf_greedy_autocast"], outs["g_a"] = timed(base, prompts, args.max_new_tokens, auto)
    tps["hf_pld_autocast"], outs["p_a"] = timed(base, prompts, args.max_new_tokens, auto,
                                                prompt_lookup_num_tokens=args.lookup)
    base = base.half()
    tps["hf_greedy_fp16"], outs["g_h"] = timed(base, prompts, args.max_new_tokens, contextlib.nullcontext)
    tps["hf_pld_fp16"], outs["p_h"] = timed(base, prompts, args.max_new_tokens, contextlib.nullcontext,
                                            prompt_lookup_num_tokens=args.lookup)
    for name, v in tps.items():
        res["configs"][name] = {"tokens_per_sec": v}
        print(f"{name:<22} {v:7.1f} tok/s", flush=True)
    res["pld_speedup_autocast"] = tps["hf_pld_autocast"] / tps["hf_greedy_autocast"]
    res["pld_speedup_fp16"] = tps["hf_pld_fp16"] / tps["hf_greedy_fp16"]
    res["pld_match_greedy_autocast"] = sum(a == b for a, b in zip(outs["p_a"], outs["g_a"])) / len(prompts)
    res["pld_match_greedy_fp16"] = sum(a == b for a, b in zip(outs["p_h"], outs["g_h"])) / len(prompts)
    print({k: round(v, 3) for k, v in res.items() if k.startswith("pld_")})
    out = Path(args.out) / cfg.run_name / f"hf_baseline_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
