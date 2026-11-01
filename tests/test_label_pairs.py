"""Offline tests for scripts/label_pairs.py (B3 blind-labeling CLI).

Only the pure helpers and the `status` subcommand are exercised -- `label`
is an input()-driven human workflow and is explicitly out of scope for
automated tests (see module docstring).
"""
from __future__ import annotations

from scripts import label_pairs


def _rows(n: int, labeled: set[int] = frozenset()) -> list[dict]:
    return [
        {"entity": f"e{i}", "kind": "method", "candidate": f"c{i}",
         "cos": "0.9", "margin": "0.1",
         "label": "1" if i in labeled else "", "annotator": "yahor" if i in labeled else ""}
        for i in range(n)
    ]


def test_shuffled_order_is_a_permutation_and_deterministic():
    order_a = label_pairs.shuffled_order(10)
    order_b = label_pairs.shuffled_order(10)
    assert sorted(order_a) == list(range(10))
    assert order_a == order_b  # same seed every time -- resume walks the same sequence


def test_shuffled_order_is_not_just_the_identity():
    order = label_pairs.shuffled_order(20)
    assert order != list(range(20))


def test_is_labeled_accepts_only_valid_markers():
    assert label_pairs.is_labeled({"label": "1"})
    assert label_pairs.is_labeled({"label": "0"})
    assert label_pairs.is_labeled({"label": "?"})
    assert not label_pairs.is_labeled({"label": ""})
    assert not label_pairs.is_labeled({"label": "yes"})
    assert not label_pairs.is_labeled({})


def test_remaining_order_excludes_already_labeled_rows():
    rows = _rows(10, labeled={2, 5, 7})
    remaining = label_pairs.remaining_order(rows)
    assert set(remaining) == {0, 1, 3, 4, 6, 8, 9}
    assert len(remaining) == 7


def test_remaining_order_empty_when_everything_labeled():
    rows = _rows(5, labeled={0, 1, 2, 3, 4})
    assert label_pairs.remaining_order(rows) == []


def test_progress_summary_counts_each_label_value():
    rows = _rows(5)
    rows[0]["label"], rows[0]["annotator"] = "1", "yahor"
    rows[1]["label"], rows[1]["annotator"] = "1", "yahor"
    rows[2]["label"], rows[2]["annotator"] = "0", "yahor"
    rows[3]["label"], rows[3]["annotator"] = "?", "yahor"
    summary = label_pairs.progress_summary(rows)
    assert "4/5 labeled" in summary
    assert "1=2" in summary
    assert "0=1" in summary
    assert "?=1" in summary
    assert "1 remaining" in summary


def test_load_and_save_round_trip(tmp_path):
    path = tmp_path / "pairs.csv"
    rows = _rows(3)
    rows[1]["label"], rows[1]["annotator"] = "1", "yahor"
    label_pairs.save_pairs(path, rows)

    loaded = label_pairs.load_pairs(path)
    assert len(loaded) == 3
    assert loaded[1]["label"] == "1"
    assert loaded[1]["annotator"] == "yahor"
    assert loaded[0]["label"] == ""


def test_load_pairs_backfills_missing_annotator_column(tmp_path):
    """A file written before the annotator column existed (just entity/kind/
    candidate/cos/margin/label) must still load without a KeyError."""
    path = tmp_path / "pairs.csv"
    path.write_text(
        "entity,kind,candidate,cos,margin,label\n"
        "e0,method,c0,0.9,0.1,\n",
        encoding="utf-8",
    )
    rows = label_pairs.load_pairs(path)
    assert rows[0]["annotator"] == ""


def test_status_command_runs_without_stdin(tmp_path, capsys):
    path = tmp_path / "pairs.csv"
    label_pairs.save_pairs(path, _rows(4, labeled={0}))
    rc = label_pairs.main(["--file", str(path), "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1/4 labeled" in out
