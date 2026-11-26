"""
Re-score Group Integrity of finished spec-decode evals with another grouper (no decoding).

    python scripts/rescore_gi.py --results results/R7s_hi_k4_struct_stanza --tokenizer LingoIITGN/ganga-1b \
        --grouper stanza [--datasets indiccorp_eval flores_hi]

Reads results/{run}/eval_{dataset}.json (spec_decode entries must carry per_prompt prompt_ids / ids /
spans: evals run after this was added) and writes results/{run}/gi_{grouper}_{dataset}.json:
{"run_name", "dataset", "grouper", "policies": [{"policy", "group_integrity", "gi_hits", "gi_spans"}]}.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`


def dataset_lang(dataset, default="hi"):
    return {"flores_hi": "hi", "flores_mr": "mr"}.get(dataset, default)


def rescore_file(path, tokenizer, grouper_name, lang):
    from mtp.data.grouping.base import get_grouper
    from mtp.eval.spec_decode import rescore_group_integrity

    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = rec.get("spec_decode") or []
    if not entries or "ids" not in (entries[0].get("per_prompt") or {}):
        raise SystemExit(f"{path}: no per-prompt outputs (re-run evaluate.py --spec_decode with this version)")
    grouper = get_grouper(grouper_name, lang)
    out = {"run_name": rec["run_name"], "dataset": rec["dataset"], "grouper": grouper_name,
           "policies": [{"policy": e["policy"], **rescore_group_integrity(e, tokenizer, grouper)} for e in entries]}
    dest = Path(path).with_name(f"gi_{grouper_name}_{rec['dataset']}.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return dest, out


def main(argv=None):
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/{run_name} folder with eval_*.json")
    ap.add_argument("--tokenizer", required=True, help="the run's model_name (its tokenizer made the ids)")
    ap.add_argument("--grouper", required=True)
    ap.add_argument("--lang", default=None, help="for indiccorp_eval*: the run's language (default hi)")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every eval_*.json in the folder")
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    folder = Path(args.results)
    paths = [folder / f"eval_{d}.json" for d in args.datasets] if args.datasets else sorted(folder.glob("eval_*.json"))
    for path in paths:
        dataset = path.stem[len("eval_"):]
        dest, out = rescore_file(path, tokenizer, args.grouper, dataset_lang(dataset, args.lang or "hi"))
        gis = ", ".join(f"{p['policy']} {100 * p['group_integrity']:.1f}%" for p in out["policies"]
                        if p["group_integrity"] is not None)
        print(f"{dataset}: GI with {args.grouper}: {gis} -> {dest}")


if __name__ == "__main__":
    main()
