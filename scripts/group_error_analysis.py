"""JI-9 token-dump analysis, target group type and target position in group.

Usage: --dump R2=results/R2/tokens_flores_hi.jsonl --dump R5=... . The same
sentences and token-index pairs must be present across runs. Type labels are
assigned with one shared grouper and tokenizer, independent of each run's own
cache, so comparison remains meaningful.
"""

import argparse
from collections import Counter
import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.data.grouping.align import label_tokens


def annotate_dump(rows, tokenizer, grouper, max_length=128):
    labels = {}
    counts = {}
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("dump requires original text")
        if text not in labels:
            lab = label_tokens(text, tokenizer, grouper, max_length)
            types = grouper.group_types(text)
            positions = {}
            for i, g in enumerate(lab["group_id"]):
                if g >= 0:
                    positions.setdefault(g, []).append(i)
            labels[text] = (lab, types, positions)
        lab, types, positions = labels[text]
        target = row["target_idx"]
        if target < 0 or target >= len(lab["group_id"]):
            raise ValueError("target_idx outside tokenizer labels")
        gid = lab["group_id"][target]
        if gid < 0:
            continue
        span = positions[gid]
        position = (
            "single"
            if len(span) == 1
            else (
                "first"
                if target == span[0]
                else "last" if target == span[-1] else "middle"
            )
        )
        key = (row["head"], types[gid], position)
        d = counts.setdefault(key, Counter())
        d["n"] += 1
        d["correct"] += bool(row["correct"])
    return [
        {
            "head": h,
            "group_type": g,
            "position": p,
            "n": d["n"],
            "top1": d["correct"] / d["n"],
        }
        for (h, g, p), d in sorted(counts.items())
    ]


def compare_examples(a, b):
    key = lambda r: (r["sent_id"], r["token_idx"], r["head"])
    aa = {key(r): r for r in a}
    bb = {key(r): r for r in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("duplicate token dump keys")
    if aa.keys() != bb.keys():
        raise ValueError("token dump keys differ")
    improved = []
    worse = []
    for k in aa:
        x, y = aa[k], bb[k]
        if x["text"] != y["text"] or x["target_idx"] != y["target_idx"]:
            raise ValueError("dump tokenization mismatch")
        if x["head"] < 1:
            continue
        if not x["correct"] and y["correct"]:
            improved.append(y)
        elif x["correct"] and not y["correct"]:
            worse.append(y)
    return {"improved": improved[:10], "worse": worse[:5]}


def compare_runs(raw, baseline="R2"):
    """Validate every paired dump, including controls, before reporting differences."""
    if baseline not in raw:
        raise ValueError(f"baseline {baseline!r} is missing from token dumps")
    return {
        name: compare_examples(raw[baseline], rows)
        for name, rows in raw.items() if name != baseline
    }


def markdown_report(report):
    lines = [
        f"# Group error analysis ({report['grouper']})", "",
        f"Baseline: {report['baseline']}. All runs have matching sentence/token/head keys and texts.", "",
        "| Run | Head | Target group type | Position | Tokens | Top-1 |",
        "|---|---:|---|---|---:|---:|",
    ]
    for name, rows in report["runs"].items():
        for row in rows:
            lines.append(f"| {name} | {row['head']} | {row['group_type']} | "
                         f"{row['position']} | {row['n']} | {row['top1']:.4f} |")
    for name, examples in report["comparisons"].items():
        lines.extend(["", f"## {name} vs {report['baseline']}", ""])
        for kind in ("improved", "worse"):
            lines.extend([f"{kind.capitalize()} far-head examples (up to {'10' if kind == 'improved' else '5'}):", ""])
            if not examples[kind]:
                lines.extend(["None in this matched dump.", ""])
            for row in examples[kind]:
                lines.append(f"- Sentence {row['sent_id']}, head {row['head']}, "
                             f"source token {row['token_idx']}, target token {row['target_idx']}: "
                             + row["text"].replace("\n", " "))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", action="append", required=True)
    p.add_argument("--lang", default="hi", choices=["hi", "mr"])
    p.add_argument("--grouper", default="hi_rules_v1")
    p.add_argument("--tokenizer", default="LingoIITGN/ganga-1b")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--baseline", default="R2", help="Name used in --dump NAME=PATH")
    p.add_argument(
        "--out", type=Path, default=Path("results/grouping/error_analysis.json")
    )
    a = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    g = get_grouper(a.grouper, a.lang)
    raw = {}
    reports = {}
    inputs = {}
    for spec in a.dump:
        if "=" not in spec:
            p.error("--dump must be NAME=PATH")
        name, path = spec.split("=", 1)
        if not name or name in raw:
            p.error("dump names must be nonempty and unique")
        inputs[name] = {"filename": Path(path).name,
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        raw[name] = [
            json.loads(s)
            for s in Path(path).read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        if not raw[name]:
            p.error(f"empty token dump: {name}")
    comparisons = compare_runs(raw, a.baseline)
    for name, rows in raw.items():
        reports[name] = annotate_dump(rows, tok, g, a.max_length)
    result = {
        "grouper": g.name,
        "tokenizer": a.tokenizer,
        "max_length": a.max_length,
        "baseline": a.baseline,
        "inputs": inputs,
        "runs": reports,
        "comparisons": comparisons,
        "examples": comparisons.get("R5") if a.baseline == "R2" else None,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    a.out.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
