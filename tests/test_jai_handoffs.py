import json
import pytest
from pathlib import Path
from scripts.group_error_analysis import annotate_dump, compare_examples, compare_runs
from mtp.data.grouping.base import get_grouper, check_partition
from mtp.data.grouping.random_refit import HindiV1RandomGrouper, MarathiV1RandomGrouper
from mtp.data.grouping.trankit_grouper import StanzaGrouper
from types import SimpleNamespace as N
from mtp.data.flores import load_flores_text


def test_named_refits_and_legacy_random_are_separate():
    for lang, name in [("hi", "random_hi_v1"), ("mr", "random_mr_v1")]:
        g = get_grouper(name, lang, histogram={2: 1})
        assert g.name == name and g.group_words("a b c") == [["a", "b"], ["c"]]
        check_partition("a b c", g.group_words("a b c"))
    with pytest.raises(ValueError):
        get_grouper("random_hi_v1", "mr", histogram={1: 1})


def test_error_types_target_position_and_comparison(tiny_tok):
    row = {
        "sent_id": 0,
        "text": "मैं जा रहा था।",
        "token_idx": 0,
        "target_idx": 2,
        "head": 1,
        "correct": False,
    }
    report = annotate_dump([row], tiny_tok, get_grouper("hi_rules_v1"))
    assert report == [
        {"head": 1, "group_type": "aux_chain", "position": "middle", "n": 1, "top1": 0}
    ]
    result = compare_examples([row], [{**row, "correct": True}])
    assert len(result["improved"]) == 1
    with pytest.raises(ValueError):
        compare_examples([row], [])


def test_error_analysis_checks_all_controls_without_r5():
    row = {"sent_id": 0, "text": "a b c", "token_idx": 0,
           "target_idx": 2, "head": 1, "correct": False}
    runs = {"R2": [row], "R3": [{**row, "correct": True}], "R6a": [row]}
    result = compare_runs(runs)
    assert result["R3"]["improved"] == runs["R3"]
    assert not result["R6a"]["improved"]
    with pytest.raises(ValueError, match="keys differ"):
        compare_runs({**runs, "R6a": []})
    with pytest.raises(ValueError, match="duplicate"):
        compare_runs({**runs, "R6a": [row, row]})
    with pytest.raises(ValueError, match="baseline"):
        compare_runs({"R3": [row]})


def test_stanza_expanded_word_offsets():
    word1 = N(id=1, head=0, deprel="root", upos="NOUN", feats=None)
    word2 = N(id=2, head=1, deprel="case", upos="ADP", feats=None)
    sentence = N(
        tokens=[
            N(start_char=0, end_char=2, words=[word1]),
            N(start_char=3, end_char=5, words=[word2]),
        ]
    )
    g = StanzaGrouper(pipeline=lambda _: N(sentences=[sentence]))
    assert g.group_words("घर से") == [["घर", "से"]]


def test_official_flores_archive_with_dot_prefix(tmp_path):
    import tarfile, io

    path = tmp_path / "flores.tar.gz"
    content = "नमस्ते दुनिया\nदूसरा वाक्य\n".encode()
    with tarfile.open(path, "w:gz") as t:
        info = tarfile.TarInfo("./flores200_dataset/devtest/hin_Deva.devtest")
        info.size = len(content)
        t.addfile(info, io.BytesIO(content))
    assert load_flores_text("hi", archive_path=path) == ["नमस्ते दुनिया", "दूसरा वाक्य"]


def test_partial_gold_cannot_select_final_grouper(tmp_path):
    import subprocess, sys

    row = {
        "id": "one",
        "text": "वह घर",
        "groups": [[0], [1]],
        "types": ["single", "single"],
        "annotator": "jai",
        "annotation_status": "human_reviewed",
    }
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "scores.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/score_groupers.py",
            "--gold",
            str(path),
            "--groupers",
            "hi_rules_v1",
            "--out",
            str(out),
        ],
        check=True,
    )
    result = json.loads(out.read_text())
    assert result["official_gold"] and not result["complete_gold"]
    assert result["best_by_boundary_f1"] is None
