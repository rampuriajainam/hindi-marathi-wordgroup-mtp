"""Validate every cached row without loading a language model or parser."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.boundary_cache import load_cache


def validate_cache(path, tokenizer_name=None, max_length=None):
    dataset, meta = load_cache(path, tokenizer_name, max_length)
    digest = hashlib.sha256()
    checked = 0
    keys = ("input_ids", "attention_mask", "group_start", "group_id")
    for batch in dataset.iter(batch_size=1024):
        for i, text in enumerate(batch["text"]):
            values = [batch[k][i] for k in keys]
            ids, masks, starts, groups = values
            if not text.strip() or not ids or len({len(v) for v in values}) != 1:
                raise ValueError(f"row {checked}: empty text or unequal label lengths")
            if len(ids) > meta["max_length"]:
                raise ValueError(f"row {checked}: exceeds max_length")
            previous = -1
            for token, mask, start, group in zip(ids, masks, starts, groups):
                if (type(token) is not int or token < 0 or mask not in (0, 1)
                        or start not in (0, 1) or type(group) is not int or group < -1):
                    raise ValueError(f"row {checked}: invalid token or boundary label")
                if group < 0:
                    if start:
                        raise ValueError(f"row {checked}: special/padding token starts a group")
                else:
                    if not mask or group < previous or group >= len(text.split()):
                        raise ValueError(f"row {checked}: invalid group index or mask")
                    if start != int(group != previous):
                        raise ValueError(f"row {checked}: group_start disagrees with group_id")
                    previous = group
            digest.update((json.dumps(text, ensure_ascii=False) + "\n").encode("utf-8"))
            checked += 1
    return {"metadata": meta, "validated_rows": checked,
            "text_sha256": digest.hexdigest(), "all_rows_valid": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--tokenizer", default="LingoIITGN/ganga-1b")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("results/grouping/boundary_cache_validation.json"))
    a = p.parse_args()
    if len({path.name for path in a.paths}) != len(a.paths):
        p.error("cache folder names must be unique")
    reports = {}
    for path in a.paths:
        reports[path.name] = validate_cache(path, a.tokenizer, a.max_length)
        print(f"{path.name}: {reports[path.name]['validated_rows']} valid rows", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"local_validation": True, "publication_verified": False,
                                "caches": reports}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
