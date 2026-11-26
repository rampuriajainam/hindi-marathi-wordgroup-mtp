"""
Offline probe: repetition-controlled greedy decoding (deterministic, so exact speculative decoding still applies).

    python scripts/rep_probe.py --run_dir RUN --tree results/<run>/tree_policy_static_n64.json --n 100 \
        [--judge sarvamai/sarvam-1] [--out results/rep_probe/<run>.json]

For each rule (plain greedy; repetition penalty 1.2 / 1.1 in the CTRL style; no-repeat-4-gram blocking) it generates
64 tokens for n IndicCorp eval prompts (no timing: correctness of the probe only), and reports
  rep4          share of generated 4-grams already seen in prompt + output
  accepted      drafts a static tree would accept per step (teacher-forced, as fit_tree.py), with the heads' raw
                ranks, and with the same rule applied to the head logits (what an exact decoder with the rule would
                draft: the rule needs only the context, which is known when drafting)
  judge ppl     perplexity per whitespace word of the continuation given the prompt under an independent judge
Exact decoding with a rule verifies drafts against head 0's argmax AFTER the rule, which is the generated token,
so `accepted` is what such a decoder would accept per step on this text.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtp.data.corpus import load_split
from mtp.device import autocast_ctx
from mtp.eval.spec_decode import make_prompts
from mtp.eval.tree_policy import accepted_length, load_policy
from mtp.model.checkpoint import load_run


def apply_rule(logits, context, rule):
    """logits [V] (float), context: token ids so far (prompt + generated). Returns adjusted logits."""
    if rule == "greedy":
        return logits
    out = logits.clone()
    if rule.startswith("rp"):
        pen = float(rule[2:])
        ids = torch.tensor(sorted(set(context)), device=out.device)
        v = out[ids]
        out[ids] = torch.where(v > 0, v / pen, v * pen)
        return out
    if rule.startswith("nr"):
        n = int(rule[2:])
        if len(context) >= n - 1:
            prefix = tuple(context[-(n - 1):])
            ban = {context[i + n - 1] for i in range(len(context) - n + 1) if tuple(context[i:i + n - 1]) == prefix}
            if ban:
                out[torch.tensor(sorted(ban), device=out.device)] = float("-inf")
        return out
    raise ValueError(rule)


def rep4(prompt, new):
    full = list(prompt) + list(new)
    seen, rep, tot = set(), 0, 0
    for i in range(len(full) - 3):
        g = tuple(full[i:i + 4])
        if i + 4 > len(prompt):
            tot += 1
            rep += g in seen
        seen.add(g)
    return rep / max(tot, 1)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--rules", nargs="+", default=["greedy", "rp1.1", "rp1.2", "nr4"])
    ap.add_argument("--judge", default="sarvamai/sarvam-1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    model, tok, cfg = load_run(args.run_dir)
    model.eval()
    dev = next(model.parameters()).device
    k = model.num_heads
    tree = load_policy(args.tree).nodes
    prompts = make_prompts(load_split(cfg.lang, "eval"), tok, n=args.n, seed=1)
    res = {"run_name": cfg.run_name, "n_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
           "tree": args.tree, "rules": {}}
    texts = {}
    for rule in args.rules:
        accepted_raw, accepted_rule, reps, outs = [], [], [], []
        for p in prompts:
            ids = list(p)
            for _ in range(args.max_new_tokens):
                with autocast_ctx(cfg):
                    lg = model(torch.tensor([ids], device=dev)).logits[0][0, -1].float()
                ids.append(int(apply_rule(lg, ids, rule).argmax()))
            new = ids[len(p):]
            outs.append(new)
            reps.append(rep4(p, new))
            with autocast_ctx(cfg):
                out = model(torch.tensor([ids], device=dev))
            for t in range(len(p) - 1, len(ids) - k):
                r_raw, r_rule = [], []
                ctx = ids[:t + 2]                     # known when drafting at t: everything up to the root t0
                for d in range(1, k):
                    lg = out.logits[d][0, t].float()
                    tgt = ids[t + d + 1]
                    r_raw.append(int((lg > lg[tgt]).sum()))
                    adj = apply_rule(lg, ctx, rule)
                    r_rule.append(int((adj > adj[tgt]).sum()))
                accepted_raw.append(accepted_length(tree, r_raw))
                accepted_rule.append(accepted_length(tree, r_rule))
        texts[rule] = [(p, o) for p, o in zip(prompts, outs)]
        res["rules"][rule] = {"rep4": sum(reps) / len(reps),
                              "accepted_drafts_raw": sum(accepted_raw) / len(accepted_raw),
                              "accepted_drafts_rule_aware": sum(accepted_rule) / len(accepted_rule),
                              "examples": [tok.decode(o, skip_special_tokens=True) for o in outs[:3]]}
        print(rule, {a: round(b, 3) for a, b in res["rules"][rule].items() if a != "examples"}, flush=True)

    del model
    torch.cuda.empty_cache()
    if args.judge:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        jt = AutoTokenizer.from_pretrained(args.judge)
        jm = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.float16).to(dev).eval()
        for rule, pairs in texts.items():
            nll, words = 0.0, 0
            for p, o in pairs:
                ptxt = tok.decode(p, skip_special_tokens=True)
                otxt = tok.decode(o, skip_special_tokens=True)
                pi = jt(ptxt, return_tensors="pt").input_ids.to(dev)
                fi = jt(ptxt + otxt, return_tensors="pt").input_ids.to(dev)
                m = 0
                while m < min(pi.shape[1], fi.shape[1]) and pi[0, m] == fi[0, m]:
                    m += 1
                lp = torch.log_softmax(jm(fi).logits[0, :-1].float(), -1)
                tgt = fi[0, 1:]
                tok_lp = lp.gather(1, tgt[:, None])[:, 0]
                nll -= float(tok_lp[max(m - 1, 0):].sum())
                words += max(len(otxt.split()), 1)
            res["rules"][rule]["judge_ppl_word"] = math.exp(nll / words)
            print(rule, "judge ppl/word", round(res["rules"][rule]["judge_ppl_word"], 2), flush=True)
    out = Path(args.out or f"results/rep_probe/{cfg.run_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
