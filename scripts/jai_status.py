"""Audit Jai's annotation progress from files, without running any models."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.annotation import read_jsonl, is_human_reviewed


def annotation_progress(packet, annotations):
    expected = {r["id"]: r["text"] for r in packet}
    seen = set()
    reviewed = []
    pending = []
    for row in annotations:
        if row["id"] in seen:
            raise ValueError(f"duplicate annotation ID: {row['id']}")
        seen.add(row["id"])
        if expected.get(row["id"]) != row["text"]:
            raise ValueError(f"annotation ID/text differs from packet: {row['id']}")
        (reviewed if is_human_reviewed(row) else pending).append(row["id"])
    return {"expected": len(packet), "reviewed": len(reviewed),
            "present_but_unreviewed": len(pending), "unreviewed_ids": pending,
            "remaining_ids": [r["id"] for r in packet if r["id"] not in set(reviewed)],
            "complete": len(reviewed) == len(packet),
            "status_counts": dict(Counter(r.get("annotation_status", "legacy_human") for r in annotations))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=Path("results/grouping/jai_progress.json"))
    a = p.parse_args()
    reports = {}
    for lang in ("hi", "mr"):
        packet = read_jsonl(a.root / f"data/annotation/{lang}_sentences.jsonl")
        active = a.root / f"data/gold/{lang}_gold.jsonl"
        fallback = a.root / "data/gold/mr_gold_reviewed_55.jsonl"
        source = active if active.exists() else fallback if lang == "mr" and fallback.exists() else None
        annotations = read_jsonl(source, True) if source else []
        reports[lang] = {"annotation_file": str(source.relative_to(a.root)) if source else None,
                         **annotation_progress(packet, annotations)}
    blind = a.root / "data/gold/hi_gold_om50.jsonl"
    reports["om_blind"] = annotation_progress(
        read_jsonl(a.root / "data/annotation/hi_om50_text_only.jsonl"),
        read_jsonl(blind, True) if blind.exists() else [],
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, report in reports.items():
        print(f"{name}: {report['reviewed']}/{report['expected']} reviewed; "
              f"{len(report['remaining_ids'])} remaining")


if __name__ == "__main__":
    main()
