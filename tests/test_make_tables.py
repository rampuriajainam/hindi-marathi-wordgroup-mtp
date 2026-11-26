"""make_tables.py builds every table from the committed result JSONs (no network, no model)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("make_tables", ROOT / "scripts" / "make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tables_from_committed_results(tmp_path):
    mt = _load()
    mt.main(["--no_figures", "--out_dir", str(tmp_path)])
    names = sorted(p.name for p in (tmp_path / "tables").iterdir())
    assert names == sorted(f"table{i}_{n}.{e}" for i, n in ((1, "main_hi"), (2, "main_mr"), (3, "controls"),
                                                           (4, "decoding"), (5, "ablations")) for e in ("md", "tex"))
    dec = (tmp_path / "tables" / "table4_decoding.md").read_text(encoding="utf-8")
    e = mt.spec(mt.load("Rsd_seq", "indiccorp_eval"), "tree_static_n64")
    assert f"×{e['speedup']:.2f}" in dec                          # numbers come from the JSONs
    ctl = (tmp_path / "tables" / "table3_controls.md").read_text(encoding="utf-8")
    assert "R7s − R6a" in ctl and "✓" in ctl
    assert "toprule" in (tmp_path / "tables" / "table1_main_hi.tex").read_text(encoding="utf-8")


def test_paired_bootstrap_identical_runs_is_zero():
    mt = _load()
    a = ([3, 4, 5, 6], [5, 5, 6, 8])
    d, lo, hi = mt.paired_bootstrap(a, a, n=200)
    assert d == lo == hi == 0
