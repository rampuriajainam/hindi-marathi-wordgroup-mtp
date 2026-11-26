"""Oracle ceiling for "draft a whole word group, stop where the group ends" (no training).

    python scripts/group_stop_oracle.py --run_dir <run> --grouper hi_rules_v1 --merge 1 3 4 6 --out results/grouping/stop_oracle_<run>.json

Works for sequential (chain) and parallel (linear / resblock) heads. The model greedy-decodes `--n` Hindi prompts (so the text is what lossless decoding would produce). One teacher-forced
pass then gives, for every position t, m[t] = how many chain drafts (t+2 ...) match the greedy continuation, up to K.
Each stop policy fixes a draft length L[t] <= K; a step then accepts 1 + min(m[t], L[t]) tokens and the walk advances
by that amount (exact for lossless greedy decoding, since a chain draft at t depends only on the state at t).

Policies: fixed (L = K); group_end (L = tokens left in the group of the first drafted token, cap K, using the
true groups = a perfect stop signal); group_end_min2; random_len (same L distribution as group_end, shuffled
over positions = the control: does knowing the group end tell us anything beyond "draft shorter"?).
`--merge N` merges adjacent rule groups (words kept whole) until each has >= N tokens.
Speed is estimated from a step-cost model: t = verify + per_chain_step * L (constants below, from the T4 benches)."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERIFY_MS = 27.5      # one verify pass, fast_linear T4 (bench_verify_fast.json: flat in positions)
VERIFY_PER_POS = 0.07  # extra verify cost per drafted position beyond 4
CHAIN_MS = 0.8        # one chain step (seqblock + fp16 lm_head)


class MergedGrouper:
    """Merge adjacent groups of `base` until each has >= n_tokens tokens (words stay whole)."""

    def __init__(self, base, tokenizer, n_tokens):
        self.base, self.tok, self.n = base, tokenizer, n_tokens
        self.name, self.lang = f"{base.name}_min{n_tokens}", base.lang

    def group_words(self, sentence):
        out, cur, cnt = [], [], 0
        for g in self.base.group_words(sentence):
            cur += g
            cnt += len(self.tok(" ".join(g), add_special_tokens=False)["input_ids"])
            if cnt >= self.n:
                out.append(cur)
                cur, cnt = [], 0
        if cur:
            if out and cnt < self.n:
                out[-1] += cur
            else:
                out.append(cur)
        return out

    def group_types(self, sentence):
        return ["other"] * len(self.group_words(sentence))


def walk(m, L, start, end):
    """Steps/tokens from `start` to `end` with per-position accepted drafts m and draft lengths L."""
    t, steps, drafted, acc = start, 0, 0, 0
    while t < end:
        a = min(m[t], L[t])
        drafted += L[t]
        acc += a
        steps += 1
        t += 1 + a
    return steps, drafted, acc, t - start


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--grouper", default="hi_rules_v1")
    ap.add_argument("--merge", type=int, nargs="*", default=[1, 3, 4, 6])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--new_tokens", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    from mtp.data.corpus import load_split
    from mtp.data.grouping.base import get_grouper
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.eval.spec_decode import greedy_generate, make_prompts, token_group_ids
    from mtp.model.checkpoint import load_run

    model, tok, cfg = load_run(args.run_dir, device=args.device)
    device = next(model.parameters()).device
    amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else __import__("contextlib").nullcontext
    K = len(model.extra_heads)
    prompts = make_prompts(load_split(cfg.lang, "eval"), tok, n=args.n)
    base = get_grouper(args.grouper, cfg.lang)
    groupers = {f"min{n}": (base if n <= 1 else MergedGrouper(base, tok, n)) for n in args.merge}

    embed, lm_head = model.base_model.get_input_embeddings(), model._lm_head()
    seqs = []
    with torch.no_grad():
        for p in prompts:
            new, _ = greedy_generate(model, tok, p, args.new_tokens, amp=amp, ignore_eos=True)
            full = p + [int(t) for t in new]
            ids = torch.tensor([full], device=device)
            with amp():
                if model.head_type == "seq":                       # chain: each draft sees the previous one
                    out = model(input_ids=ids, extra_heads=False)
                    s = out.hidden[0]                              # [T, H]
                    tokn = out.logits[0][0].argmax(-1)             # head-0 token for t+1
                    drafts = []
                    for head in model.extra_heads[:K]:
                        s = head(s, embed(tokn).to(s.dtype))
                        tokn = lm_head(s).argmax(-1)
                        drafts.append(tokn)                        # drafts[d][t] = guess for token t+2+d
                else:                                              # parallel heads: head d+1 guesses t+2+d
                    out = model(input_ids=ids)
                    drafts = [out.logits[d + 1][0].argmax(-1) for d in range(K)]
            D = torch.stack(drafts, 0).cpu().tolist()              # [K][T]
            T = len(full)
            m = [0] * T
            for t in range(len(p) - 1, T - 2 - K):
                j = 0
                while j < K and D[j][t] == full[t + 2 + j]:
                    j += 1
                m[t] = j
            seqs.append((p, full, m))
    print(f"K={K}; {len(seqs)} sequences generated")

    res = {"run": cfg.run_name, "K": K, "n_prompts": len(seqs), "cost_model_ms": [VERIFY_MS, VERIFY_PER_POS, CHAIN_MS],
           "chain_acc_given_prefix": None, "groupers": {}}
    # accuracy of chain position j given all earlier drafts right (unconditional on groups)
    cnt = [0] * (K + 1)
    tot = 0
    for p, full, m in seqs:
        for t in range(len(p) - 1, len(full) - 2 - K):
            tot += 1
            for j in range(m[t] + 1):
                cnt[j] += 1
    res["chain_acc_given_prefix"] = [cnt[j + 1] / cnt[j] if cnt[j] else None for j in range(K)]
    rng = random.Random(0)
    for gname, g in groupers.items():
        policies = {k: dict(steps=0, drafted=0, acc=0, toks=0) for k in ("fixed", "group_end", "group_end_min2", "random_len")}
        glens, rej_at_start, rej_total, starts, ntok = [], 0, 0, 0, 0
        for p, full, m in seqs:
            T = len(full)
            gid = token_group_ids(full, tok, g)
            end = T - 2 - K
            Lg = [K] * T
            for t in range(len(p) - 1, end):
                i, left = t + 2, 0
                while i < T and gid[i] == gid[t + 2] and left < K:
                    left += 1
                    i += 1
                Lg[t] = max(1, left)
            # group length stats and where first rejection falls (group start vs inside)
            for t in range(len(p) - 1, end):
                if m[t] < K:
                    rej_total += 1
                    pos = t + 2 + m[t]
                    rej_at_start += pos < T and (pos == 0 or gid[pos] != gid[pos - 1])
            for t in range(len(p) - 1, end):
                starts += gid[t + 2] != gid[t + 1]
                ntok += 1
            glens += [Lg[t] for t in range(len(p) - 1, end)]
            pools = {"fixed": [K] * T, "group_end": Lg, "group_end_min2": [max(2, x) for x in Lg]}
            shuf = Lg[len(p) - 1:end]
            rng.shuffle(shuf)
            Lr = [K] * T
            Lr[len(p) - 1:end] = shuf
            pools["random_len"] = Lr
            for name, L in pools.items():
                s_, d_, a_, n_ = walk(m, L, len(p) - 1, end)
                for k, v in zip(("steps", "drafted", "acc", "toks"), (s_, d_, a_, n_)):
                    policies[name][k] += v
        row = {"mean_remaining_group_len_capK": sum(glens) / len(glens),
               "first_reject_at_group_start": rej_at_start / max(rej_total, 1),
               "base_rate_group_start": starts / max(ntok, 1)}
        for name, v in policies.items():
            tps = v["toks"] / v["steps"]
            dpl = v["drafted"] / v["steps"]
            step_ms = VERIFY_MS + VERIFY_PER_POS * max(0, dpl - 4) + CHAIN_MS * dpl
            row[name] = {"tokens_per_step": tps, "drafted_per_step": dpl, "accept_per_drafted": v["acc"] / max(v["drafted"], 1),
                         "est_ms_per_step": step_ms, "est_tokens_per_sec": 1000 * tps / step_ms}
        res["groupers"][gname] = row
        print(f"\n{gname}: mean remaining group len (cap {K}) {row['mean_remaining_group_len_capK']:.2f}; "
              f"first rejection lands on a group start {100 * row['first_reject_at_group_start']:.1f}% "
              f"(base rate of group starts {100 * row['base_rate_group_start']:.1f}%)")
        for name in policies:
            r = row[name]
            print(f"  {name:15s} tok/step {r['tokens_per_step']:.3f}  drafted/step {r['drafted_per_step']:.2f}  "
                  f"acc/drafted {r['accept_per_drafted']:.3f}  est tok/s {r['est_tokens_per_sec']:.1f}")
    print("\nchain accuracy given earlier drafts right:", [round(x, 3) for x in res["chain_acc_given_prefix"] if x is not None])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
