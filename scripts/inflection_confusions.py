"""When a draft head is wrong, is the right token an inflected form of the same stem? (Hindi, idea #9)

    python scripts/inflection_confusions.py --run_dir checkpoints/kaggle/R2 --out results/drafter/inflection_R2.json

Same setting as group_stop_oracle.py: the model greedy-decodes --n IndicCorp eval prompts; one teacher-forced pass
gives each draft head's top-k at every position (chain heads draft from their own previous drafts). Only draft
positions whose earlier drafts were all right are counted (the ones that decide acceptance).

Siblings: whole-word tokens with the same stem after stripping one Hindi inflectional ending (heuristic list below,
e.g. जाता / जाती / जाते -> जा, लड़का / लड़के / लड़कों -> लड़क). Reported: top-1 accuracy; for wrong drafts, how
often the target is in top-2/3/5, is a sibling of the top-1 token, and is a sibling already in top-k; the average
number of siblings per token (the tree width a sibling branch would cost); and the most frequent confusion pairs."""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUFFIXES = sorted("""ियाँ ियां ियों ियो ाएँ ाएं ाओं ुओं ओं ों ें ाँ ां ीं ता ती ते ना नी ने या यी ये ई ए ा ी े""".split(),
                  key=len, reverse=True)


def stem(word):
    for s in SUFFIXES:
        if word.endswith(s) and len(word) > len(s):
            return word[: -len(s)]
    return word


def is_deva(w):
    return bool(w) and all("ऀ" <= c <= "ॿ" for c in w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--new_tokens", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    from mtp.data.corpus import load_split
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.eval.spec_decode import greedy_generate, make_prompts
    from mtp.model.checkpoint import load_run

    model, tok, cfg = load_run(args.run_dir, device=args.device)
    device = next(model.parameters()).device
    amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else __import__("contextlib").nullcontext
    K = len(model.extra_heads)

    # sibling lexicon over whole-word Devanagari tokens
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    key = {}
    groups = collections.defaultdict(set)
    for i, t in enumerate(vocab):
        if t and t.startswith("▁") and is_deva(t[1:]):
            k = stem(t[1:])
            key[i] = k
            groups[k].add(i)
    sib = lambda a, b: a != b and a in key and b in key and key[a] == key[b]

    embed, lm_head = model.base_model.get_input_embeddings(), model._lm_head()
    st = [collections.Counter() for _ in range(K)]
    pairs = collections.Counter()
    with torch.no_grad():
        for p in make_prompts(load_split(cfg.lang, "eval"), tok, n=args.n):
            new, _ = greedy_generate(model, tok, p, args.new_tokens, amp=amp, ignore_eos=True)
            full = p + [int(t) for t in new]
            ids = torch.tensor([full], device=device)
            with amp():
                if model.head_type == "seq":
                    out = model(input_ids=ids, extra_heads=False)
                    s, tokn, tops = out.hidden[0], out.logits[0][0].argmax(-1), []
                    for head in model.extra_heads[:K]:
                        s = head(s, embed(tokn).to(s.dtype))
                        lg = lm_head(s)
                        tops.append(lg.topk(args.topk, -1).indices)
                        tokn = tops[-1][:, 0]
                else:
                    out = model(input_ids=ids)
                    tops = [out.logits[d + 1][0].topk(args.topk, -1).indices for d in range(K)]
            tops = [x.cpu().tolist() for x in tops]                     # [K][T][topk]
            T = len(full)
            for t in range(len(p) - 1, T - 2 - K):
                for d in range(K):
                    cand, tgt = tops[d][t], full[t + 2 + d]
                    c = st[d]
                    c["n"] += 1
                    if cand[0] == tgt:
                        c["top1"] += 1
                        continue
                    c["wrong"] += 1
                    r = cand.index(tgt) + 1 if tgt in cand else 0
                    for k in (2, 3, 5):
                        c[f"in_top{k}"] += 0 < r <= k
                    c["tgt_word"] += tgt in key
                    c["pred_word"] += cand[0] in key
                    if sib(cand[0], tgt):
                        c["sibling_of_top1"] += 1
                        c["sibling_in_topk"] += r > 0
                        c["sib_group_size"] += len(groups[key[tgt]])
                        pairs[(vocab[cand[0]][1:], vocab[tgt][1:])] += 1
                    break                                               # later drafts are rejected anyway

    res = {"run": cfg.run_name, "K": K, "n_prompts": args.n, "heads": []}
    for d, c in enumerate(st):
        w = max(c["wrong"], 1)
        row = {"head": d + 1, "n": c["n"], "top1": c["top1"] / max(c["n"], 1), "wrong": c["wrong"],
               "of_wrong": {"in_top2": c["in_top2"] / w, "in_top3": c["in_top3"] / w, "in_top5": c["in_top5"] / w,
                            "target_is_sibling_of_top1": c["sibling_of_top1"] / w,
                            "sibling_target_already_in_topk": c["sibling_in_topk"] / max(c["sibling_of_top1"], 1),
                            "both_whole_words": None},
               "mean_sibling_group_size": c["sib_group_size"] / max(c["sibling_of_top1"], 1)}
        res["heads"].append(row)
        o = row["of_wrong"]
        print(f"draft {d + 1}: n {c['n']}  top-1 {100 * row['top1']:.1f}%  | of wrong ({c['wrong']}): in top-2 "
              f"{100 * o['in_top2']:.1f}%  top-3 {100 * o['in_top3']:.1f}%  top-5 {100 * o['in_top5']:.1f}%  "
              f"sibling of top-1 {100 * o['target_is_sibling_of_top1']:.1f}% (of those, already in top-{args.topk} "
              f"{100 * o['sibling_target_already_in_topk']:.1f}%; family size {row['mean_sibling_group_size']:.1f})")
    res["top_confusions"] = [[a, b, n] for (a, b), n in pairs.most_common(40)]
    print("\nmost frequent sibling confusions (draft -> correct):")
    for a, b, n in res["top_confusions"][:25]:
        print(f"  {a} -> {b}  x{n}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
