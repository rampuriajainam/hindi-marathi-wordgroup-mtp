"""Boundary caches for the `karthika` grouper (R11) and the size histogram for its random control (R11a).

    python scripts/build_karthika_cache.py --out boundary_cache --n_train 100000 --gpu

Parses with stanza in bulk (prefetch per chunk), then writes {out}/hi_karthika_train and hi_karthika_eval with
build_cache, the same layout train.py reads. Train texts = load_split("hi", "train", n_train), the first
n_train sentences R3 / R6a / R7s train on (12,500 steps x batch 8); eval = eval_small, as in their logged eval.
The random control's histogram is fitted on the first 5,000 train sentences (as random_hi_v1) and written to
mtp/data/grouping/histograms_karthika.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("boundary_cache"))
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from mtp.data.boundary_cache import build_cache
    from mtp.data.corpus import load_split
    from mtp.data.grouping.base import get_grouper
    from mtp.data.grouping.random_grouper import RandomGrouper
    from mtp.eval.group_statistics import sentence_digest

    tok = AutoTokenizer.from_pretrained("LingoIITGN/ganga-1b", use_fast=True)
    g = get_grouper("karthika", "hi", gpu=a.gpu)
    for split, n in (("eval_small", None), ("train", a.n_train)):
        texts = load_split("hi", split, n)
        t0 = time.time()
        for i in range(0, len(texts), a.chunk):
            g.prefetch(texts[i:i + a.chunk])
            print(f"{split}: parsed {min(i + a.chunk, len(texts))}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        name = "eval" if split == "eval_small" else "train"
        meta = build_cache(texts, tok, g, a.out / f"hi_karthika_{name}", 128, 64, True)
        print(f"{split}: cache {meta}", flush=True)
        if split == "train":
            fit = texts[:5000]
            hist = RandomGrouper.fit_from(g, fit).histogram
            path = ROOT / "mtp/data/grouping/histograms_karthika.json"
            path.write_text(json.dumps({"hi": {"grouper": "karthika", "n_sentences": len(fit),
                                               "source": "IndicCorpV2 hi train raw rows >= 1000",
                                               "sentence_sha256": sentence_digest(fit),
                                               "histogram": hist}}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            print("random_karthika histogram", hist)


if __name__ == "__main__":
    main()
