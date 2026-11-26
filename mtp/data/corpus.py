"""
Fixed text splits (INTERFACES §4). Nothing is downloaded at import time.

IndicCorpV2 separates documents with blank rows (every other row is blank).
Splits are defined on RAW row numbers and blank rows are then dropped:
  eval        raw rows 0-999   (~500 sentences)
  eval_small  raw rows 0-99    (~50 sentences; the laptop runs' held-out set)
  train       raw rows >= 1000, the first n NON-BLANK ones
  flores      FLORES-200 devtest (1,012 sentences)

FLORES is gated on the Hub. A local copy is used instead when one is found:
$MTP_FLORES_DIR/flores_devtest_{hin_Deva,mar_Deva}.parquet, or the same file name anywhere under
/kaggle/input (an attached dataset: Kaggle secrets, hence HF_TOKEN, are unavailable to versions
pushed with the Kaggle CLI). The copy holds facebook/flores devtest as is (column "sentence").
"""

import glob
import os


INDICCORP_SPLIT = {"hi": "hin_Deva", "mr": "mar_Deva"}


def local_flores_file(lang):
    """Path of a local FLORES devtest copy for `lang`, or None."""
    name = f"flores_devtest_{INDICCORP_SPLIT[lang]}.parquet"
    roots = [os.environ.get("MTP_FLORES_DIR")]
    for root in filter(None, roots):
        if os.path.isfile(os.path.join(root, name)):
            return os.path.join(root, name)
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits[0] if hits else None


def load_split(lang, split, n=None):
    from datasets import load_dataset

    if split == "flores":
        path = local_flores_file(lang)
        if path:
            import pyarrow.parquet as pq

            texts = pq.read_table(path, columns=["sentence"]).column("sentence").to_pylist()
        else:
            ds = load_dataset("facebook/flores", INDICCORP_SPLIT[lang], split="devtest")
            texts = [r["sentence"] for r in ds]
        return texts[:n] if n else texts
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split=INDICCORP_SPLIT[lang], streaming=True)
    if split in ("eval", "eval_small"):
        texts = [r["text"].strip() for r in raw.take(1000 if split == "eval" else 100)]
        texts = [t for t in texts if t]
        return texts[:n] if n else texts
    if split != "train":
        raise ValueError(f"unknown split {split!r}")
    texts = []
    for r in raw.skip(1000):
        t = r["text"].strip()
        if t:
            texts.append(t)
            if n and len(texts) >= n:
                break
    return texts
