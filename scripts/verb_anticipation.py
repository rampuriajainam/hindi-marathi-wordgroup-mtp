"""Club 3 pre-check: is the clause-final verb a big error source for draft heads, and does the frozen model
already know it in advance? (Decides whether a verb-anticipation head is worth training.)

    python scripts/verb_anticipation.py --dump dump_results/R2_hi_mtp_k4_resblock/tokens_indiccorp_eval.jsonl \
        --run_dir checkpoints/kaggle/R2 --n_train 3000 --out results/grouping/verb_anticipation.json

Part A (ceiling, from an evaluate.py --dump_tokens file): every IndicCorp eval sentence is parsed (stanza, HDTB);
target tokens are labelled root_verb (first token of the sentence's root VERB), verb (first token of any other
main verb, VM), verb_cont (later tokens of a VM word), auxiliary (VAUX), postposition (PSP), other. Per head: share of targets and top-1 by label, and the share of all
head errors that fall on each label.

Part B (already encoded?): linear probes on frozen hidden states (layers --layers) at positions *before* the
root verb predict the verb's lemma (top --classes lemmas + "other"), trained on --n_train train sentences, tested
on the eval sentences, reported by distance to the verb (tokens). Baselines: majority class and the lemma prior.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_all(texts, nlp, max_words=120):
    import stanza

    docs = nlp.bulk_process([stanza.Document([], text=[t.split()[:max_words]]) for t in texts])
    out = []
    for t, d in zip(texts, docs):
        ws = [w for s in d.sentences for tok in s.tokens for w in tok.words]
        root = next((w for w in ws if w.head == 0), None)
        info = {"root": None, "xpos": [w.xpos for w in ws]}
        if root is not None and root.upos == "VERB":
            aux = [w.id for w in ws if w.head == root.id and w.xpos == "VAUX"]
            info.update({"root": root.id - 1, "lemma": root.lemma, "aux": [a - 1 for a in aux]})
        out.append(info)
    return out


def word_of_token(text, tok):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=128)
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    wid = []
    for a, b in enc["offset_mapping"]:
        hit = [i for i, (s, e) in enumerate(spans) if a < e and b > s]
        wid.append(hit[0] if hit else -1)
    return enc["input_ids"], wid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import stanza
    import torch
    from transformers import AutoTokenizer

    from mtp.data.corpus import load_split
    from mtp.device import native_bf16

    amp_dtype = torch.bfloat16 if native_bf16() else torch.float16   # emulated bf16 on a T4 is ~6x slower
    nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True,
                          download_method=stanza.DownloadMethod.REUSE_RESOURCES, use_gpu=True, verbose=False)
    tok = AutoTokenizer.from_pretrained("LingoIITGN/ganga-1b", use_fast=True)
    res = {}

    # ---------- Part A: error share on the main verb (from the dump)
    sents = {}
    rows = []
    with open(args.dump, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sents[r["sent_id"]] = r["text"]
            rows.append((r["sent_id"], r["head"], r["target_idx"], r["correct"]))
    sids = sorted(sents)
    parses = dict(zip(sids, parse_all([sents[s] for s in sids], nlp)))
    lab = {}
    for s in sids:
        _, wid = word_of_token(sents[s], tok)
        p = parses[s]
        L = []
        xp = p["xpos"]
        for i, w in enumerate(wid):
            x = xp[w] if 0 <= w < len(xp) else None
            first = i == 0 or wid[i - 1] != w
            if p["root"] is not None and w == p["root"] and first:
                L.append("root_verb")              # first token of the sentence's main (root) verb
            elif x == "VM":
                L.append("verb" if first else "verb_cont")   # any main verb (all clauses)
            elif x == "VAUX":
                L.append("auxiliary")
            elif x == "PSP":
                L.append("postposition")
            else:
                L.append("other")
        lab[s] = L
    st = collections.defaultdict(lambda: [0, 0])
    for s, h, ti, ok in rows:
        if ti < len(lab[s]):
            for c in (lab[s][ti], "all"):
                st[(h, c)][0] += 1
                st[(h, c)][1] += bool(ok)
    A = {}
    for h in sorted({h for h, _ in st}):
        n_all, ok_all = st[(h, "all")]
        A[h] = {c: {"share": v[0] / n_all, "top1": v[1] / v[0], "share_of_errors": (v[0] - v[1]) / (n_all - ok_all)}
                for (hh, c), v in st.items() if hh == h}
    res["part_a"] = A
    res["sentences_with_verb_root"] = sum(p["root"] is not None for p in parses.values()) / len(parses)
    print(f"Part A ({len(sids)} eval sentences, {100 * res['sentences_with_verb_root']:.0f}% have a VERB root)")
    for h, d in A.items():
        print(f"  head {h}: " + "  ".join(f"{c} {100 * v['share']:.1f}% of targets, top-1 {100 * v['top1']:.1f}%, "
                                         f"{100 * v['share_of_errors']:.1f}% of errors" for c, v in sorted(d.items())))

    # ---------- Part B: probe the upcoming verb lemma from frozen states
    from mtp.model.checkpoint import load_run

    model, _, cfg = load_run(args.run_dir, device=args.device)
    device = next(model.parameters()).device
    train_texts = [t for t in load_split("hi", "train", args.n_train) if t.strip()]
    tparses = parse_all(train_texts, nlp)
    lemmas = collections.Counter(p["lemma"] for p in tparses if p["root"] is not None)
    top = [l for l, _ in lemmas.most_common(args.classes)]
    cls = {l: i for i, l in enumerate(top)}
    other = len(top)
    res["lemma_coverage_topk"] = sum(lemmas[l] for l in top) / max(sum(lemmas.values()), 1)

    def collect(texts, ps):
        X = {L: [] for L in args.layers}
        y, dist = [], []
        dec = model._decoder()
        for t, p in zip(texts, ps):
            if p["root"] is None:
                continue
            ids, wid = word_of_token(t, tok)
            first = next((i for i, w in enumerate(wid) if w == p["root"]), None)
            if first is None or first < 2:
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                hs = dec(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states
            for L in args.layers:
                X[L].append(hs[L][0, :first].float().cpu())
            y += [cls.get(p["lemma"], other)] * first
            dist += list(range(first, 0, -1))           # tokens from position t to the verb's first token
        return {L: torch.cat(v) for L, v in X.items()}, torch.tensor(y), torch.tensor(dist)

    Xtr, ytr, _ = collect(train_texts, tparses)
    Xev, yev, dev = collect([sents[s] for s in sids], [parses[s] for s in sids])
    prior = torch.bincount(ytr, minlength=other + 1).float()
    maj = int(prior.argmax())
    bins = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]
    B = {"n_train_positions": len(ytr), "n_eval_positions": len(yev), "classes": args.classes,
         "majority_acc": float((yev == maj).float().mean()), "layers": {}}
    for L in args.layers:
        mu, sd = Xtr[L].mean(0), Xtr[L].std(0) + 1e-5
        xtr, xev = ((Xtr[L] - mu) / sd).to(device), ((Xev[L] - mu) / sd).to(device)
        W = torch.nn.Linear(xtr.shape[1], other + 1).to(device)
        opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=1e-2)
        for ep in range(8):
            perm = torch.randperm(len(ytr))
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                loss = torch.nn.functional.cross_entropy(W(xtr[b]), ytr[b].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = W(xev).argmax(-1).cpu()
        acc = {f"{a}-{b}": float((pred[(dev >= a) & (dev <= b)] == yev[(dev >= a) & (dev <= b)]).float().mean())
               for a, b in bins if ((dev >= a) & (dev <= b)).any()}
        acc["all"] = float((pred == yev).float().mean())
        B["layers"][L] = acc
    res["part_b"] = B
    print(f"\nPart B: verb lemma probe ({args.classes} lemmas cover {100 * res['lemma_coverage_topk']:.0f}% of verb roots; "
          f"{len(ytr)} train / {len(yev)} eval positions). Majority-class acc {100 * B['majority_acc']:.1f}%")
    for L, acc in B["layers"].items():
        print(f"  layer {L}: " + "  ".join(f"dist {k}: {100 * v:.1f}%" for k, v in acc.items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
