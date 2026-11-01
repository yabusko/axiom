"""Tests for axiom.velocity — window split, share math, canonicalization, flags."""
from __future__ import annotations

import math

from axiom import db, velocity


def _add(conn, pid, year, concepts, venue="ACL"):
    db.insert_paper(conn, openalex_id=pid, title=pid, abstract="x",
                    publication_year=year, venue_id=None, venue=venue,
                    cited_by_count=0, doi=None)
    if concepts:
        db.insert_concepts(conn, pid, concepts)


# --- _windows ----------------------------------------------------------------
def test_windows_even_span():
    assert velocity._windows([2020, 2021, 2022, 2023]) == ((2020, 2021), (2022, 2023))


def test_windows_odd_span():
    # mid = 2020 + (2025-2020)//2 = 2022 → prior (2020,2022), recent (2023,2025)
    assert velocity._windows([2020, 2021, 2022, 2023, 2024, 2025]) == ((2020, 2022), (2023, 2025))


def test_windows_single_year_collapses():
    prior, recent = velocity._windows([2020])
    assert prior == recent == (2020, 2020)


# --- share / velocity math ---------------------------------------------------
def test_velocity_share_and_ratio(empty_conn):
    conn = empty_conn
    # prior window (2020-2021): X in 1 of 2 papers; recent (2022-2023): X in 2 of 2.
    _add(conn, "p1", 2020, [("X", 1)])
    _add(conn, "p2", 2021, [])
    _add(conn, "p3", 2022, [("X", 1)])
    _add(conn, "p4", 2023, [("X", 1)])
    conn.commit()

    analysis = velocity.compute_velocity(conn, min_freq=1)
    assert analysis.prior_window == (2020, 2021)
    assert analysis.recent_window == (2022, 2023)
    assert analysis.total_prior == 2
    assert analysis.total_recent == 2

    x = next(k for k in analysis.keywords if k.concept == "X")
    assert x.prior_count == 1 and x.recent_count == 2
    assert x.prior_share == 0.5 and x.recent_share == 1.0
    # log2(1.0 / 0.5) == 1.0, up to epsilon smoothing.
    assert math.isclose(x.velocity, 1.0, abs_tol=1e-3)
    assert x.low_confidence is False           # recent_count 2 >= min_freq 1


def test_low_confidence_flag(empty_conn):
    conn = empty_conn
    _add(conn, "p1", 2020, [("X", 1)])
    _add(conn, "p2", 2023, [("X", 1)])
    conn.commit()
    analysis = velocity.compute_velocity(conn, min_freq=5)
    x = next(k for k in analysis.keywords if k.concept == "X")
    assert x.recent_count == 1
    assert x.low_confidence is True            # 1 < 5


def test_level_zero_concepts_excluded(empty_conn):
    conn = empty_conn
    _add(conn, "p1", 2020, [("Computer science", 0), ("X", 1)])
    _add(conn, "p2", 2023, [("Computer science", 0), ("X", 1)])
    conn.commit()
    concepts = {k.concept for k in velocity.compute_velocity(conn, min_freq=1).keywords}
    assert "X" in concepts
    assert "Computer science" not in concepts   # level 0 dropped (min_concept_level=1)


def test_canonicalization_merges_synonyms(empty_conn):
    conn = empty_conn
    _add(conn, "p1", 2020, [("LoRA", 2)])
    _add(conn, "p2", 2023, [("Low-Rank Adaptation", 2)])
    conn.commit()
    db.upsert_canonical(conn, "Low-Rank Adaptation", "LoRA", source="auto")
    conn.commit()

    concepts = {k.concept for k in velocity.compute_velocity(conn, min_freq=1).keywords}
    assert "LoRA" in concepts
    assert "Low-Rank Adaptation" not in concepts   # merged into canonical LoRA
    lora = next(k for k in velocity.compute_velocity(conn, min_freq=1).keywords
                if k.concept == "LoRA")
    assert lora.prior_count == 1 and lora.recent_count == 1


def test_venue_and_year_filters(empty_conn):
    conn = empty_conn
    _add(conn, "p1", 2020, [("X", 1)], venue="ACL")
    _add(conn, "p2", 2023, [("X", 1)], venue="EMNLP")
    conn.commit()
    only_acl = velocity.compute_velocity(conn, venue="ACL", min_freq=1)
    # A single ACL paper spans one year → windows collapse.
    assert only_acl.insufficient_year_spread is True


def test_paper_ids_restricts_to_a_slice(empty_conn):
    """Stage 4, OD39: a field 'slice' shares the base corpus's db_path, so
    without this filter every field's trends would see the WHOLE corpus
    regardless of its own topic -- a real bug found running the real corpus
    for real (every field showed identical, unscoped rising concepts)."""
    conn = empty_conn
    _add(conn, "p1", 2020, [("X", 1)])
    _add(conn, "p2", 2023, [("X", 1)])
    _add(conn, "p3", 2020, [("Y", 1)])
    _add(conn, "p4", 2023, [("Y", 1)])
    conn.commit()

    sliced = velocity.compute_velocity(conn, min_freq=1, paper_ids={"p1", "p2"})
    concepts = {k.concept for k in sliced.keywords}
    assert concepts == {"X"}
    assert sliced.total_prior == 1 and sliced.total_recent == 1

    unsliced = velocity.compute_velocity(conn, min_freq=1)
    assert {k.concept for k in unsliced.keywords} == {"X", "Y"}


def test_empty_corpus_returns_empty(empty_conn):
    analysis = velocity.compute_velocity(empty_conn)
    assert analysis.keywords == []
    assert analysis.insufficient_year_spread is True


def test_sorted_descending_and_top_k(empty_conn):
    conn = empty_conn
    # Rising concept R (0→2) and fading concept F (2→0).
    _add(conn, "p1", 2020, [("F", 1)])
    _add(conn, "p2", 2020, [("F", 1)])
    _add(conn, "p3", 2023, [("R", 1)])
    _add(conn, "p4", 2023, [("R", 1)])
    conn.commit()
    kws = velocity.compute_velocity(conn, min_freq=1).keywords
    assert [k.velocity for k in kws] == sorted((k.velocity for k in kws), reverse=True)
    assert kws[0].concept == "R"

    top1 = velocity.compute_velocity(conn, min_freq=1, top_k=1).keywords
    assert len(top1) == 1
