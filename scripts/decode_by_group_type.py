"""
Decoding broken down by word-group type (OM-10): where does drafting accept more, and where do accepted spans respect
group boundaries? Uses the stored per-prompt outputs of finished spec-decode evals (no decoding).

    python scripts/decode_by_group_type.py --runs R2_hi_mtp_k4 R3_hi_k4_struct R6a_hi_k4_struct_random \
        R7s_hi_k4_struct_stanza --tokenizer LingoIITGN/ganga-1b --grouper hi_rules_v0 [--datasets indiccorp_eval]

Each decoding step emits head 0's token t0 and then the accepted drafts (a span of s >= 1 tokens). The step is filed
under the type of the word group t0 falls in (grouper.group_types on the decoded text), and under whether t0 starts
that group ("start") or continues it ("inside"). Per cell: steps, mean accepted length, and Group Integrity (share of
spans of >= 2 tokens that end on a group boundary; spans that end the generation are not judged, as in evaluate.py).

Writes results/tables/table6_group_types.md and results/group_types_{grouper}.json.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`


def step_records(prompt_ids, new, spans, tokenizer, grouper):
    """One (group type, 'start' | 'inside', accepted length, GI hit or None) per decoding step."""
    from mtp.eval.spec_decode import token_group_ids

    full = [int(t) for t in prompt_ids] + [int(t) for t in new]
    gid = token_group_ids(full, tokenizer, grouper)
    types = grouper.group_types(tokenizer.decode(full, skip_special_tokens=True))
    out, i = [], len(prompt_ids)
    for s in spans:
        end = i + s
        g = gid[i]
        if g >= 0 and g < len(types):
            where = "start" if i == 0 or gid[i - 1] != g else "inside"
            hit = None
            if s >= 2 and end < len(full) and gid[end - 1] >= 0 and gid[end] >= 0:
                hit = gid[end] != gid[end - 1]
            out.append((types[g], where, s, hit))
        i = end
    return out


def summarize(records):
    """{key: {"steps", "accepted_len", "gi", "gi_spans"}} for every type, type/where, and "all"."""
    cells = defaultdict(lambda: [0, 0, 0, 0])            # steps, tokens, gi hits, gi spans
    for typ, where, s, hit in records:
        for key in ("all", typ, f"{typ}/{where}", f"all/{where}"):
            c = cells[key]
            c[0] += 1
            c[1] += s
            if hit is not None:
                c[2] += hit
                c[3] += 1
    return {k: {"steps": n, "accepted_len": t / n, "gi": h / g if g else None, "gi_spans": g}
            for k, (n, t, h, g) in cells.items()}


def run_entry(path, policy, tokenizer, grouper):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = next((e for e in rec.get("spec_decode") or [] if e["policy"] == policy), None)
    pp = (entry or {}).get("per_prompt") or {}
    if "ids" not in pp:
        print(f"skip {path}: no per-prompt outputs for {policy}")
        return rec["run_name"], None
    records = []
    for prompt_ids, ids, sp in zip(pp["prompt_ids"], pp["ids"], pp["spans"]):
        records += step_records(prompt_ids, ids, sp, tokenizer, grouper)
    return rec["run_name"], summarize(records)


def markdown(table, types):
    lines = []
    for dataset, runs in table.items():
        lines += [f"### {dataset}", "",
                  "| run | " + " | ".join(f"{t} (share, acc. len, GI)" for t in types) + " | start / inside acc. len |",
                  "|---|" + "---|" * (len(types) + 1)]
        for run, s in runs.items():
            n = s["all"]["steps"]
            cells = []
            for t in types:
                c = s.get(t)
                cells.append("-" if not c else f"{100 * c['steps'] / n:.0f}%, {c['accepted_len']:.2f}, "
                             + ("-" if c["gi"] is None else f"{100 * c['gi']:.0f}%"))
            st, ins = s.get("all/start"), s.get("all/inside")
            cells.append(f"{st['accepted_len']:.2f} / {ins['accepted_len']:.2f}" if st and ins else "-")
            lines.append(f"| {run} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="results/{run} folders (names under results/)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--grouper", default="hi_rules_v0")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--policy", default="fixed_k")
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"])
    ap.add_argument("--out_md", default=str(ROOT / "results" / "tables" / "table6_group_types.md"))
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from mtp.data.grouping.base import GROUP_TYPES, get_grouper

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    grouper = get_grouper(args.grouper, args.lang)
    table = {}
    for dataset in args.datasets:
        table[dataset] = {}
        for run in args.runs:
            path = ROOT / "results" / run / f"eval_{dataset}.json"
            if path.exists():
                name, summary = run_entry(path, args.policy, tokenizer, grouper)
                if summary:
                    table[dataset][name] = summary
    types = [t for t in GROUP_TYPES if any(t in s for runs in table.values() for s in runs.values())]
    dest = ROOT / "results" / f"group_types_{args.grouper}.json"
    dest.write_text(json.dumps({"grouper": args.grouper, "policy": args.policy, "table": table}, indent=1) + "\n",
                    encoding="utf-8")
    md = (f"Decoding by word-group type ({args.grouper} types, policy {args.policy}): share of decoding steps whose "
          f"head-0 token falls in a group of that type, mean accepted length, Group Integrity.\n\n"
          + markdown(table, types))
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(md)
    print("->", dest, args.out_md)


if __name__ == "__main__":
    main()
