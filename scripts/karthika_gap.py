"""Why is our reimplementation of the Karthika et al. (2025) grouping coarser than their reported counts?

    python scripts/karthika_gap.py --out results/grouping/karthika_gap.json

Their paper: FLORES-200 devtest Hindi, 25,643 words -> 18,980 groups (1.35 words per group). Parse the same text once
with our stanza pipeline, then count groups with each rule switched off or narrowed:
  psp / vaux / rp   join a PSP / VAUX / RP word to the preceding word when its head is that word (or, with
                    head_ext, that word's head: chains such as जाने के लिए);
  comp = all        join x to the next word for every compound*/flat* arc or NNC/NNPC tag (our implementation);
  comp = paper      only NNC/NNPC tags (pof_cn), and compound arcs from a NOUN/ADJ to a VERB (pof, conjunct verbs);
  comp = nnc        only NNC/NNPC tags;  comp = none.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAPER = {"words": 25643, "groups": 18980}


def partition_count(text, sent_words, psp=True, vaux=True, rp=True, head_ext=True, comp="all"):
    """Number of groups and which rule joined each consecutive pair, for one sentence."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    n = len(spans)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    fired = {}
    on = {"PSP": psp, "VAUX": vaux, "RP": rp}
    for sent in sent_words:
        mapped = {}
        for w in sent:
            a, b = w["dspan"]
            hit = [i for i, (s, e) in enumerate(spans) if a < e and b > s]
            mapped[w["id"]] = hit[0]
        for x, y in zip(sent, sent[1:]):
            if "PUNCT" in (x.get("upos"), y.get("upos")):
                continue
            heads = (x["id"], x.get("head")) if head_ext else (x["id"],)
            rule = None
            if on.get(y.get("xpos")) and y.get("head") in heads:
                rule = y["xpos"]
            if rule is None and comp != "none":
                rel = x.get("deprel") or ""
                nnc = x.get("xpos") in {"NNC", "NNPC"}
                yheads = (y["id"], y.get("head")) if head_ext else (y["id"],)
                if x.get("head") in yheads:
                    if comp == "all" and (nnc or rel.startswith(("compound", "flat"))):
                        rule = "NNC" if nnc else rel.split(":")[0]
                    elif comp in ("paper", "nnc") and nnc:
                        rule = "NNC"
                    elif comp == "paper" and rel.startswith("compound") and x.get("upos") in {"NOUN", "ADJ"} \
                            and y.get("upos") == "VERB":
                        rule = "pof"
            if rule:
                fired[rule] = fired.get(rule, 0) + 1
                parent[find(mapped[y["id"]])] = find(mapped[x["id"]])
    groups = 1 + sum(1 for i in range(1, n) if find(i) != find(i - 1)) if n else 0
    return groups, fired


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import stanza

    from mtp.data.corpus import load_split
    from mtp.data.grouping.karthika_grouper import KarthikaGrouper

    texts = [t.strip() for t in load_split("hi", "flores") if t.strip()]
    g = KarthikaGrouper(lang="hi")
    docs = g._load().bulk_process([stanza.Document([], text=[t.split()]) for t in texts])
    parsed = [(t, KarthikaGrouper.doc_words(t, d)) for t, d in zip(texts, docs)]
    n_words = sum(len(t.split()) for t in texts)
    variants = {
        "ours (all rules, comp=all)": {},
        "comp=paper (pof + pof_cn only)": {"comp": "paper"},
        "comp=nnc (pof_cn only)": {"comp": "nnc"},
        "comp=none": {"comp": "none"},
        "no head extension": {"head_ext": False},
        "comp=paper, no head extension": {"comp": "paper", "head_ext": False},
        "no RP": {"rp": False},
        "comp=paper, no RP": {"comp": "paper", "rp": False},
    }
    res = {"paper": PAPER, "flores_words": n_words, "sentences": len(texts), "variants": {}}
    print(f"FLORES devtest hi: {len(texts)} sentences, {n_words} whitespace words "
          f"(paper: {PAPER['words']} words -> {PAPER['groups']} groups, {PAPER['words'] / PAPER['groups']:.2f} w/g)")
    for name, kw in variants.items():
        total, fired = 0, {}
        for t, sw in parsed:
            c, f = partition_count(t, sw, **kw)
            total += c
            for k, v in f.items():
                fired[k] = fired.get(k, 0) + v
        res["variants"][name] = {"groups": total, "words_per_group": n_words / total,
                                 "vs_paper_groups": total / PAPER["groups"] - 1, "joins_by_rule": fired}
        print(f"  {name:34s} {total:6d} groups  {n_words / total:.3f} w/g  ({100 * (total / PAPER['groups'] - 1):+.1f}% vs paper)  joins {fired}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
