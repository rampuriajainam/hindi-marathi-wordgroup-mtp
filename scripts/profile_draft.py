"""
Profile one draft step of a sequential-head run (DR-1): where do the ~2.3 ms per step go?

    python scripts/profile_draft.py --run_dir <Rsd_seq run> [--counts results/drafter/vocab_rank_hi.json]

Times (median of --reps after warm-up; CUDA events on GPU) under the run's autocast setting:
  embed        embedding lookup of one token
  seqblock     one SeqBlock step
  lm_head      LM head on one state, as decoding runs it (fp32 weight under fp16 autocast: re-cast every call)
  lm_head_fp16 LM head with a weight pre-cast to fp16 once
  argmax       argmax over the vocabulary
  sync         argmax + int() (CPU waits for the GPU)
  draft_chain  the full model.draft_chain(h, t0, k-1)
plus each cheaper word predictor (mtp/model/draft_lm.py: top-N rows, low-rank r) with pre-cast fp16 weights, and a
torch.profiler table of 20 draft_chain calls. Writes results/{run_name}/draft_profile.json.
"""

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`


def timer(fn, reps, warmup, device):
    import torch

    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        if device.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter()
            fn()
            times.append(1000 * (time.perf_counter() - t0))
    return statistics.median(times)


def profile(model, amp, reps=200, warmup=20, counts=None, topn=(4000, 8000, 16000), ranks=(128, 256, 512)):
    import torch

    from mtp.model.draft_lm import make_draft_lm, svd_factors

    device = next(model.parameters()).device
    embed, lm_head = model.base_model.get_input_embeddings(), model._lm_head()
    H = lm_head.weight.shape[1]
    half = torch.float16 if device.type == "cuda" else lm_head.weight.dtype
    h = torch.randn(H, device=device, dtype=lm_head.weight.dtype)
    tok = torch.tensor([5], device=device)
    w16 = lm_head.weight.detach().to(half)
    head = model.extra_heads[0]
    out = {}
    with torch.no_grad(), amp():
        e = embed(tok)[0]
        out["embed"] = timer(lambda: embed(tok), reps, warmup, device)
        out["seqblock"] = timer(lambda: head(h, e.to(h.dtype)), reps, warmup, device)
        out["lm_head"] = timer(lambda: lm_head(h), reps, warmup, device)
        out["lm_head_fp16"] = timer(lambda: h.to(half) @ w16.T, reps, warmup, device)
        lg = lm_head(h)
        out["argmax"] = timer(lambda: lg.argmax(), reps, warmup, device)
        out["sync"] = timer(lambda: int(lg.argmax()), reps, warmup, device)
        k1 = model.num_heads - 1
        out["draft_chain"] = timer(lambda: model.draft_chain(h, 5, k1), reps, warmup, device)
        preds = {}
        weight16 = lm_head.weight.detach().to(half)
        if counts:
            for n in topn:
                p = make_draft_lm(weight16, "topn", n=n, counts=counts)
                preds[p.name] = p
        for r in ranks:
            A, B = svd_factors(lm_head.weight.detach().float(), r)
            p = make_draft_lm(weight16, "lowrank", rank=r, factors=(A.to(half), B.to(half)))
            preds[p.name] = p
        h16 = h.to(half)
        out["predictors"] = {name: timer(lambda p=p: p.score(h16), reps, warmup, device) for name, p in preds.items()}
        table = None
        if device.type == "cuda":
            from torch.profiler import ProfilerActivity, profile as tprofile

            with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    model.draft_chain(h, 5, k1)
                torch.cuda.synchronize()
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    return out, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--counts", default=None)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.model.checkpoint import load_run

    model, _, cfg = load_run(args.run_dir, device=args.device)
    model.eval()
    amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else contextlib.nullcontext
    counts = None
    if args.counts:
        counts = {int(i): c for i, c in json.loads(Path(args.counts).read_text(encoding="utf-8"))["counts"].items()}
    res, table = profile(model, amp, reps=args.reps, counts=counts)
    device = next(model.parameters()).device
    rec = {"run_name": cfg.run_name, "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "fp16_autocast": uses_fp16_autocast(cfg), "ms": res, "profiler": table}
    dest = Path(args.out) if args.out else ROOT / "results" / cfg.run_name / "draft_profile.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    if table:
        print(table)
    print("->", dest)


if __name__ == "__main__":
    main()
