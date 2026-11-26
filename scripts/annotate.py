"""JI-2: human annotation, resume by ID. Blind mode never displays rule groups."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.annotation import (
    read_jsonl, validate_record, parse_spans, is_human_reviewed, write_jsonl,
)
from mtp.data.grouping.base import GROUP_TYPES


def annotate_row(row, annotator, ask=input):
    print("\n" + row["id"] + " " + row["text"])
    print(" | ".join(f"{i}:{w}" for i, w in enumerate(row["text"].split())))
    while True:
        spans = ask(
            "Merge spans (e.g. 3-5 7-8), blank = all single; q = quit: "
        ).strip()
        if spans == "q":
            return None
        try:
            groups = parse_spans(spans, len(row["text"].split()))
            types = []
            for group in groups:
                if len(group) == 1:
                    types.append("single")
                    continue
                kind = ask(f'Type for {group} ({", ".join(GROUP_TYPES[1:])}): ').strip()
                if kind not in GROUP_TYPES[1:]:
                    raise ValueError("invalid multiword group type")
                types.append(kind)
            result = {k: row[k] for k in ("id", "source", "source_id", "text") if k in row}
            result.update(
                groups=groups,
                types=types,
                annotator=annotator,
                notes=ask("Notes: "),
                annotation_status="human_reviewed",
            )
            return validate_record(result)
        except ValueError as e:
            print(e)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--annotator", required=True)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--resume-from", type=Path,
                   help="Seed a NEW output with reviewed rows from a previous progress file")
    a = p.parse_args()
    rows = read_jsonl(a.input)
    if a.input.resolve() == a.out.resolve():
        p.error("input and output must differ")
    if a.resume_from and a.out.exists():
        p.error("--resume-from requires a new output; resume an existing output without it")
    if a.resume_from and not a.resume_from.exists():
        p.error("--resume-from file does not exist")
    previous = a.resume_from or a.out
    existing = read_jsonl(previous, True) if previous.exists() else []
    by_id = {r["id"]: r for r in rows}
    for row in existing:
        if row["id"] not in by_id or row["text"] != by_id[row["id"]]["text"]:
            p.error(f'annotation ID/text does not match input packet: {row["id"]}')
    if not a.resume_from and any(not is_human_reviewed(r) for r in existing):
        p.error("output contains pending labels; use --resume-from with a NEW output")
    existing = [r for r in existing if is_human_reviewed(r)]
    done = {r["id"] for r in existing}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.resume_from:
        write_jsonl(a.out, existing)
    print(f"Reviewed {len(done)}/{len(rows)}; remaining {len(rows) - len(done)}")
    try:
        with a.out.open("a", encoding="utf-8") as f:
            for row in rows:
                if row["id"] in done:
                    continue
                result = annotate_row(row, a.annotator)
                if result is None:
                    break
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    except (EOFError, KeyboardInterrupt):
        print("\nSaved completed annotations; rerun to resume.")


if __name__ == "__main__":
    main()
