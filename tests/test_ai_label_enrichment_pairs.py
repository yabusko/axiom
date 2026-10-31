"""Offline tests for scripts/ai_label_enrichment_pairs.py's pure helpers.

The actual dual-labeling run (`run()`) makes real LLM calls and is not
exercised here -- only the deterministic acronym detector, the final-label
rule, cost estimation, and the shuffle/load/save helpers.
"""
from __future__ import annotations

from scripts import ai_label_enrichment_pairs as ai_label


def test_find_acronym_pairs_basic():
    abstract = "We study retrieval-augmented generation (RAG) for open-domain QA."
    pairs = ai_label.find_acronym_pairs(abstract)
    # hyphen folded to a space by the word tokenizer -- see module docstring
    assert ("retrieval augmented generation", "RAG") in pairs


def test_find_acronym_pairs_multiple_and_none():
    abstract = "Named entity recognition (NER) and part of speech (POS) tagging. Plain text."
    pairs = ai_label.find_acronym_pairs(abstract)
    assert ("Named entity recognition", "NER") in pairs
    assert ("part of speech", "POS") in pairs
    assert ai_label.find_acronym_pairs("") == []
    assert ai_label.find_acronym_pairs(None) == []


def test_acronym_evidence_for_matches_acronym_side():
    pairs = [("retrieval-augmented generation", "RAG")]
    assert ai_label.acronym_evidence_for("RAG", "Retrieval-Augmented Generation", pairs) != ""


def test_acronym_evidence_for_matches_expansion_side():
    pairs = [("retrieval-augmented generation", "RAG")]
    ev = ai_label.acronym_evidence_for("retrieval augmented generation", "RAG", pairs)
    assert ev != ""


def test_acronym_evidence_for_no_match():
    pairs = [("retrieval-augmented generation", "RAG")]
    assert ai_label.acronym_evidence_for("tokenization", "Tokenize", pairs) == ""


def test_acronym_evidence_for_empty_pairs():
    assert ai_label.acronym_evidence_for("x", "y", []) == ""


def test_compute_final_label_agreement():
    assert ai_label.compute_final_label("1", "1") == ("1", "agreed_1")
    assert ai_label.compute_final_label("0", "0") == ("0", "agreed_0")


def test_compute_final_label_disagreement():
    assert ai_label.compute_final_label("1", "0") == ("", "disagreement")


def test_compute_final_label_unsure_excludes_even_on_match():
    assert ai_label.compute_final_label("unsure", "unsure") == ("", "unsure")
    assert ai_label.compute_final_label("unsure", "1") == ("", "unsure")
    assert ai_label.compute_final_label("0", "unsure") == ("", "unsure")


def test_shuffled_order_is_a_permutation_and_deterministic():
    a = ai_label.shuffled_order(20)
    b = ai_label.shuffled_order(20)
    assert sorted(a) == list(range(20))
    assert a == b


def _row(entity="RAG", kind="method", evidence_span="uses RAG", candidate="Retrieval"):
    return {"entity": entity, "kind": kind, "evidence_span": evidence_span, "candidate": candidate}


def test_estimate_cost_is_positive_and_scales_with_row_count():
    import pytest
    one = ai_label.estimate_cost([_row()])
    ten = ai_label.estimate_cost([_row() for _ in range(10)])
    assert one > 0
    assert ten == pytest.approx(one * 10)


def test_load_and_save_round_trip(tmp_path):
    path = tmp_path / "pairs.csv"
    rows = [dict.fromkeys(ai_label.FIELDNAMES, "") for _ in range(2)]
    rows[0].update(entity="e0", kind="method", candidate="c0", cos="0.9", margin="0.1")
    ai_label.save_rows(path, rows)
    loaded = ai_label.load_rows(path)
    assert len(loaded) == 2
    assert loaded[0]["entity"] == "e0"


def test_load_rows_backfills_missing_columns(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("entity,kind,candidate,cos,margin,label\ne0,method,c0,0.9,0.1,\n",
                     encoding="utf-8")
    rows = ai_label.load_rows(path)
    assert rows[0]["acronym_evidence"] == ""
    assert rows[0]["label_a"] == ""
