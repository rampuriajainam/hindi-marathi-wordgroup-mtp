"""Head accuracy by how the target token sits inside a Hindi word (from an evaluate.py --dump_tokens file).

    python scripts/token_structure_analysis.py --dump dump_results/R2_hi_mtp_k4_resblock/tokens_indiccorp_eval.jsonl --out results/grouping/token_structure_R2.json

Target classes: word_start (token begins with the SentencePiece word marker), continuation (mid-word piece),
split_akshara (continuation that begins with a vowel sign / virama-joined piece, i.e. BPE cut inside an akshara),
after_virama (previous token ended in a halant, the conjunct is split). Per head: top-1 and share of targets."""
import argparse, json, collections
from pathlib import Path

MATRA = set(chr(c) for c in list(range(0x093E, 0x094D)) + [0x0901, 0x0902, 0x0903, 0x093C])
VIRAMA = "\u094d"


def cls(tok, prev):
    c = []
    if tok.startswith("\u2581"):
        c.append("word_start")
    else:
        c.append("continuation")
        if tok and tok[0] in MATRA:
            c.append("split_akshara")
        if prev and prev.endswith(VIRAMA):
            c.append("after_virama")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    toks = collections.defaultdict(dict)
    rows = []
    with open(a.dump, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["head"] == 0:
                toks[r["sent_id"]][r["token_idx"]] = r["token"]
            rows.append((r["sent_id"], r["head"], r["target_idx"], r["correct"]))
    st = collections.defaultdict(lambda: [0, 0])
    for sid, h, ti, ok in rows:
        t = toks[sid].get(ti)
        if t is None:
            continue
        for c in cls(t, toks[sid].get(ti - 1)) + ["all"]:
            s = st[(h, c)]
            s[0] += 1
            s[1] += bool(ok)
    out = {}
    heads = sorted({h for h, _ in st})
    for h in heads:
        n = st[(h, "all")][0]
        out[h] = {c: {"n": v[0], "share": v[0] / n, "top1": v[1] / v[0]} for (hh, c), v in st.items() if hh == h}
    for h in heads:
        print(f"head {h}: " + "  ".join(f"{c} {100*v['share']:.1f}% of targets, top-1 {100*v['top1']:.1f}%" for c, v in out[h].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
