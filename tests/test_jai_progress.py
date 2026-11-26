import pytest

from scripts.jai_status import annotation_progress
from scripts.validate_boundary_cache import validate_cache


def test_progress_uses_ids_and_review_status_not_prefix_count():
    packet = [{"id": str(i), "text": "a b"} for i in range(3)]
    rows = [{**packet[2], "annotator": "human", "annotation_status": "human_reviewed"},
            {**packet[0], "annotator": "human", "annotation_status": "needs_review"}]
    result = annotation_progress(packet, rows)
    assert result["reviewed"] == 1 and result["remaining_ids"] == ["0", "1"]
    assert not result["complete"]
    with pytest.raises(ValueError, match="differs"):
        annotation_progress(packet, [{**rows[0], "text": "changed"}])


def test_cache_validator_rejects_corrupt_boundary_labels(tmp_path, tiny_tok):
    from datasets import load_from_disk
    from mtp.data.boundary_cache import build_cache
    from mtp.data.grouping.base import get_grouper

    path = tmp_path / "cache"
    build_cache(["मैं जा रहा था।"], tiny_tok, get_grouper("hi_rules_v1"), path, max_length=16)
    assert validate_cache(path)["validated_rows"] == 1
    dataset = load_from_disk(str(path))
    bad = dataset.map(lambda row: {"group_start": [0] * len(row["group_start"])})
    corrupt = tmp_path / "corrupt"
    bad.save_to_disk(str(corrupt))
    (corrupt / "meta.json").write_text((path / "meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        validate_cache(corrupt)
