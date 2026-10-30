"""Offline tests for scripts/label_cli.py (R9 human-review CLI).

Only the non-interactive subcommands (list/show/set) and the pure helper
functions are exercised -- `review` is an input()-driven human workflow and
is explicitly out of scope for automated tests (see module docstring).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import label_cli

FIXTURE = {
    "_provenance": "AI-drafted test fixture",
    "_scale": "0-2",
    "_scale_max": 2,
    "queries": [
        {
            "query": "retrieval augmented generation",
            "pooled_from": ["dense", "hybrid"],
            "judgments": {"P1": 2, "P2": 1},
            "judgments_meta": {
                "P1": {"provenance": "AI-drafted", "annotator": None, "note": None},
                "P2": {"provenance": "AI-drafted", "annotator": None, "note": None},
            },
        },
        {
            "query": "low rank adaptation",
            "pooled_from": ["dense"],
            "judgments": {"P3": 0},
            "judgments_meta": {
                "P3": {"provenance": "AI-drafted", "annotator": None, "note": None},
            },
        },
    ],
}


@pytest.fixture
def fixture_path(tmp_path):
    path = tmp_path / "ndcg_queries_v2.json"
    path.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_query_summary_lines():
    lines = label_cli.query_summary_lines(FIXTURE)
    assert len(lines) == 2
    assert "[0]" in lines[0] and "2 judgments" in lines[0]
    assert "retrieval augmented generation" in lines[0]


def test_query_detail_lines_shows_provenance_tag():
    lines = label_cli.query_detail_lines(FIXTURE, 0)
    assert any("P1: 2" in line and "AI-drafted" in line for line in lines)


def test_query_detail_lines_out_of_range_raises():
    with pytest.raises(ValueError):
        label_cli.query_detail_lines(FIXTURE, 99)


def test_set_judgment_overrides_value_and_records_annotator():
    data = json.loads(json.dumps(FIXTURE))  # deep copy
    label_cli.set_judgment(data, query_idx=0, paper_id="P1", relevance=0,
                            annotator="yahor", note="not actually relevant")
    assert data["queries"][0]["judgments"]["P1"] == 0
    meta = data["queries"][0]["judgments_meta"]["P1"]
    assert meta["provenance"] == "human-reviewed"
    assert meta["annotator"] == "yahor"
    assert meta["note"] == "not actually relevant"
    # Untouched judgment keeps its original AI-drafted provenance.
    assert data["queries"][0]["judgments_meta"]["P2"]["provenance"] == "AI-drafted"


def test_set_judgment_adds_a_new_paper_not_previously_judged():
    data = json.loads(json.dumps(FIXTURE))
    label_cli.set_judgment(data, query_idx=1, paper_id="NEWPAPER", relevance=2,
                            annotator="yahor")
    assert data["queries"][1]["judgments"]["NEWPAPER"] == 2
    assert data["queries"][1]["judgments_meta"]["NEWPAPER"]["annotator"] == "yahor"


@pytest.mark.parametrize("relevance", [-1, 3, 10])
def test_set_judgment_rejects_out_of_scale_relevance(relevance):
    data = json.loads(json.dumps(FIXTURE))
    with pytest.raises(ValueError):
        label_cli.set_judgment(data, query_idx=0, paper_id="P1", relevance=relevance,
                                annotator="yahor", scale_max=2)


def test_set_judgment_requires_annotator():
    data = json.loads(json.dumps(FIXTURE))
    with pytest.raises(ValueError):
        label_cli.set_judgment(data, query_idx=0, paper_id="P1", relevance=1, annotator="")


def test_set_judgment_out_of_range_query_index():
    data = json.loads(json.dumps(FIXTURE))
    with pytest.raises(ValueError):
        label_cli.set_judgment(data, query_idx=5, paper_id="P1", relevance=1, annotator="yahor")


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------
def test_load_save_roundtrip(fixture_path):
    data = label_cli.load(fixture_path)
    assert data["queries"][0]["judgments"]["P1"] == 2
    data["queries"][0]["judgments"]["P1"] = 1
    label_cli.save(fixture_path, data)
    reloaded = label_cli.load(fixture_path)
    assert reloaded["queries"][0]["judgments"]["P1"] == 1


# ---------------------------------------------------------------------------
# CLI subcommands (argparse, no stdin)
# ---------------------------------------------------------------------------
def test_cli_list(fixture_path, capsys):
    rc = label_cli.main(["--file", str(fixture_path), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "retrieval augmented generation" in out
    assert "low rank adaptation" in out


def test_cli_show(fixture_path, capsys):
    rc = label_cli.main(["--file", str(fixture_path), "show", "--query", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "P1: 2" in out


def test_cli_show_bad_index_returns_nonzero(fixture_path, capsys):
    rc = label_cli.main(["--file", str(fixture_path), "show", "--query", "99"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err


def test_cli_set_persists_override_to_disk(fixture_path, capsys):
    rc = label_cli.main([
        "--file", str(fixture_path), "set",
        "--query", "0", "--paper", "P1", "--relevance", "0",
        "--annotator", "yahor", "--note", "re-checked",
    ])
    assert rc == 0
    reloaded = label_cli.load(fixture_path)
    assert reloaded["queries"][0]["judgments"]["P1"] == 0
    meta = reloaded["queries"][0]["judgments_meta"]["P1"]
    assert meta == {"provenance": "human-reviewed", "annotator": "yahor", "note": "re-checked"}


def test_cli_set_invalid_relevance_returns_nonzero_and_does_not_write(fixture_path):
    before = fixture_path.read_text(encoding="utf-8")
    rc = label_cli.main([
        "--file", str(fixture_path), "set",
        "--query", "0", "--paper", "P1", "--relevance", "9",
        "--annotator", "yahor",
    ])
    assert rc == 1
    assert fixture_path.read_text(encoding="utf-8") == before


def test_cli_requires_a_subcommand(fixture_path):
    with pytest.raises(SystemExit):
        label_cli.main(["--file", str(fixture_path)])


def test_real_v2_skeleton_file_is_loadable_and_well_formed():
    """Sanity check against the actual eval/ndcg_queries_v2.json skeleton this
    implementer wrote -- confirms label_cli.py's schema assumptions match it."""
    path = Path(__file__).resolve().parent.parent / "eval" / "ndcg_queries_v2.json"
    data = label_cli.load(path)
    assert data["_scale_max"] == 2
    assert len(data["queries"]) >= 2
    for q in data["queries"]:
        assert set(q["judgments"]) == set(q["judgments_meta"])
        for v in q["judgments"].values():
            assert 0 <= v <= data["_scale_max"]
