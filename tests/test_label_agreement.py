"""Offline tests for scripts/label_agreement.py (E3) -- synthetic label data
only, no real files from the repo, no LLM/network call."""
from __future__ import annotations

import json
import math

from scripts import label_agreement


# ---------------------------------------------------------------------------
# load_labels -- flat JSON / ndcg_v2 JSON / CSV
# ---------------------------------------------------------------------------
def test_load_labels_flat_json(tmp_path):
    path = tmp_path / "flat.json"
    path.write_text(json.dumps({"P1": 2, "P2": 0, "P3": 1}), encoding="utf-8")
    assert label_agreement.load_labels(path) == {"P1": 2, "P2": 0, "P3": 1}


def test_load_labels_ndcg_v2_shape_keys_by_query_and_paper(tmp_path):
    path = tmp_path / "v2.json"
    path.write_text(json.dumps({
        "queries": [
            {"query": "rag for qa", "judgments": {"P1": 2, "P2": 0}},
            {"query": "lora", "judgments": {"P3": 1}},
        ]
    }), encoding="utf-8")
    labels = label_agreement.load_labels(path)
    assert labels == {"rag for qa::P1": 2, "rag for qa::P2": 0, "lora::P3": 1}


def test_load_labels_csv_with_header(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("id,label\nP1,2\nP2,0\nP3,1\n", encoding="utf-8")
    assert label_agreement.load_labels(path) == {"P1": 2, "P2": 0, "P3": 1}


def test_load_labels_csv_without_header(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("P1,2\nP2,0\n", encoding="utf-8")
    assert label_agreement.load_labels(path) == {"P1": 2, "P2": 0}


def test_load_labels_json_not_a_dict_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    try:
        label_agreement.load_labels(path)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_agreement
# ---------------------------------------------------------------------------
def test_compute_agreement_perfect_agreement():
    a = {"P1": 2, "P2": 0, "P3": 1, "P4": 2}
    b = {"P1": 2, "P2": 0, "P3": 1, "P4": 1}  # one disagreement
    report = label_agreement.compute_agreement(a, b)
    assert report["n_common"] == 4
    assert report["n_only_a"] == 0
    assert report["n_only_b"] == 0
    assert report["agreement_rate"] == 0.75
    assert 0.0 < report["kappa"] < 1.0  # some, but not perfect, agreement


def test_compute_agreement_total_disagreement_gives_low_kappa():
    a = {"P1": 2, "P2": 2, "P3": 0, "P4": 0}
    b = {"P1": 0, "P2": 0, "P3": 2, "P4": 2}
    report = label_agreement.compute_agreement(a, b)
    assert report["agreement_rate"] == 0.0
    assert report["kappa"] <= 0.0


def test_compute_agreement_reports_items_only_on_one_side():
    a = {"P1": 1, "P2": 2, "P5": 0}
    b = {"P1": 1, "P2": 2, "P6": 0}
    report = label_agreement.compute_agreement(a, b)
    assert report["n_common"] == 2
    assert report["only_in_a"] == ["P5"]
    assert report["only_in_b"] == ["P6"]
    assert report["agreement_rate"] == 1.0


def test_compute_agreement_no_overlap_is_zero_not_a_crash():
    report = label_agreement.compute_agreement({"P1": 1}, {"P2": 1})
    assert report["n_common"] == 0
    assert report["agreement_rate"] == 0.0
    assert math.isnan(report["kappa"])


def test_compute_agreement_fewer_than_two_common_items_kappa_is_nan():
    report = label_agreement.compute_agreement({"P1": 1}, {"P1": 1})
    assert report["n_common"] == 1
    assert math.isnan(report["kappa"])


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------
def test_main_writes_report_and_prints_summary(tmp_path, capsys):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps({"P1": 2, "P2": 0, "P3": 1}), encoding="utf-8")
    b_path.write_text(json.dumps({"P1": 2, "P2": 1, "P3": 1}), encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = label_agreement.main([str(a_path), str(b_path), "--out", str(out_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "n_common=3" in out
    assert "kappa=" in out

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["n_common"] == 3


def test_main_mixed_csv_and_json_inputs(tmp_path, capsys):
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.json"
    a_path.write_text("id,label\nP1,2\nP2,0\n", encoding="utf-8")
    b_path.write_text(json.dumps({"P1": 2, "P2": 0}), encoding="utf-8")

    rc = label_agreement.main([str(a_path), str(b_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_common=2" in out
