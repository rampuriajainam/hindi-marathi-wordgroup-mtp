"""score_ratings: sheets covering different items; shared items count once (averaged) and give rater agreement."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_ratings  # noqa: E402

COLS = ["item", "prompt", "label", "continuation", "fluency (1-5)", "coherence with prompt (1-5)",
        "rank in item (1=best)"]


def write_sheet(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for item, label, flu, coh, rank in rows:
            w.writerow([item, "p", label, "c", flu, coh, rank])


def test_overlap_counts_once_and_gives_agreement(tmp_path):
    key = {"1": {"A": "x", "B": "y"}, "2": {"A": "y", "B": "x"}, "3": {"A": "x", "B": "y"}}
    (tmp_path / "hi_key.json").write_text(json.dumps({"key": key}), encoding="utf-8")
    # rater 1: items 1-2; rater 2: items 2-3; item 2 is shared
    write_sheet(tmp_path / "r1.csv", [(1, "A", 5, 5, 1), (1, "B", 1, 1, 2), (2, "A", 2, 2, 2), (2, "B", 4, 4, 1)])
    write_sheet(tmp_path / "r2.csv", [(2, "A", 4, 3, 2), (2, "B", 5, 5, 1), (3, "A", 4, 4, 1), (3, "B", 2, 2, 2)])
    out = tmp_path / "scores.json"
    score_ratings.main(["--key", str(tmp_path / "hi_key.json"), "--sheets", str(tmp_path / "r1.csv"),
                        str(tmp_path / "r2.csv"), "--out", str(out)])
    res = json.loads(out.read_text(encoding="utf-8"))
    # x: item 1 A=5, item 2 B=(4+5)/2=4.5, item 3 A=4 -> three continuations, each once
    assert res["systems"]["x"]["n"] == 3
    assert abs(res["systems"]["x"]["fluency"]["mean"] - (5 + 4.5 + 4) / 3) < 1e-9
    # x wins every item by (averaged) rank
    assert res["pairwise_win_rate"]["x > y"] == 1.0
    assert res["items_per_rater"] == {"r1.csv": 2, "r2.csv": 2}
    # only 2 shared continuations: too few for a correlation, so no agreement entry
    assert res["rater_agreement_fluency"] == {}
