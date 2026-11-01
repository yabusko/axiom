"""Tests for axiom.db — schema init, round-trips, FK-safe reset."""
from __future__ import annotations

import sqlite3

import pytest

from axiom import db


def test_init_creates_all_tables(empty_conn):
    names = {r["name"] for r in empty_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"papers", "concepts", "citation_edges", "paper_provenance",
            "concept_canonical", "paper_summaries", "reading_list",
            "review_queue"} <= names


def test_insert_paper_idempotent(empty_conn):
    for _ in range(2):
        db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                        publication_year=2020, venue_id=None, venue="ACL",
                        cited_by_count=3, doi=None)
    empty_conn.commit()
    n = empty_conn.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"]
    assert n == 1                              # INSERT OR REPLACE, no duplicate


def test_canonical_map_and_manual(empty_conn):
    db.upsert_canonical(empty_conn, "LoRA", "Low-Rank Adaptation", source="auto")
    db.upsert_canonical(empty_conn, "MT", "Machine Translation", source="manual")
    empty_conn.commit()
    assert db.canonical_map(empty_conn)["LoRA"] == "Low-Rank Adaptation"
    assert db.manual_canonical_concepts(empty_conn) == {"MT"}


def test_review_queue_roundtrip(empty_conn):
    rid = db.add_to_review_queue(
        empty_conn, gap_a_label="A", gap_b_label="B", title="t", claim="c",
        method_sketch="m", datasets=["d1"], supporting_paper_ids=["A1", "B1"])
    assert isinstance(rid, int)
    rows = db.list_review_queue(empty_conn)
    assert len(rows) == 1 and rows[0]["status"] == "pending"

    db.set_review_status(empty_conn, rid, "approved")
    assert db.list_review_queue(empty_conn, status="approved")[0]["id"] == rid
    assert db.list_review_queue(empty_conn, status="pending") == []


def test_set_review_status_rejects_invalid(empty_conn):
    rid = db.add_to_review_queue(
        empty_conn, gap_a_label="A", gap_b_label="B", title="t", claim="c",
        method_sketch="m", datasets=[], supporting_paper_ids=[])
    with pytest.raises(AssertionError):
        db.set_review_status(empty_conn, rid, "bogus")


def test_bookmarks_join_and_order(empty_conn):
    for pid, cites in (("W1", 5), ("W2", 9)):
        db.insert_paper(empty_conn, openalex_id=pid, title=pid, abstract="a",
                        publication_year=2020, venue_id=None, venue="ACL",
                        cited_by_count=cites, doi=None)
    empty_conn.commit()
    db.add_bookmark(empty_conn, "W1", added_at=__import__("datetime").datetime(2020, 1, 1))
    db.add_bookmark(empty_conn, "W2", added_at=__import__("datetime").datetime(2021, 1, 1))
    rows = db.list_bookmarks(empty_conn)
    assert [r["paper_id"] for r in rows] == ["W2", "W1"]   # most-recent first
    assert rows[0]["cited_by_count"] == 9                  # joined paper metadata

    db.remove_bookmark(empty_conn, "W2")
    assert [r["paper_id"] for r in db.list_bookmarks(empty_conn)] == ["W1"]


def test_summary_roundtrip(empty_conn):
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                    publication_year=2020, venue_id=None, venue="ACL",
                    cited_by_count=0, doi=None)
    empty_conn.commit()
    assert db.get_summary(empty_conn, "W1") is None
    db.save_summary(empty_conn, "W1", ["b1", "b2", "b3"], model="qwen")
    assert db.get_summary(empty_conn, "W1") == ["b1", "b2", "b3"]


def test_year_bounds(empty_conn):
    assert db.year_bounds(empty_conn) is None          # empty corpus
    for pid, yr in (("W1", 2019), ("W2", 2024)):
        db.insert_paper(empty_conn, openalex_id=pid, title=pid, abstract="a",
                        publication_year=yr, venue_id=None, venue="ACL",
                        cited_by_count=0, doi=None)
    empty_conn.commit()
    assert db.year_bounds(empty_conn) == (2019, 2024)


def test_papers_by_ids_empty(empty_conn):
    assert db.papers_by_ids(empty_conn, []) == {}


def test_reset_corpus_fk_safe(seeded_conn):
    # paper_concepts/concept_cooccurrence FK -> papers.openalex_id; reset_corpus must clear
    # them before papers or a non-empty agent-pipeline run blocks every future reset.
    seeded_conn.execute(
        "INSERT INTO paper_concepts (run_id, paper_id, name, kind, canonical, cosine, "
        "evidence_span, confidence, source, created_at) VALUES "
        "('r1', 'A1', 'LoRA', 'method', 'Low-Rank Adaptation', 0.98, 'uses LoRA', 0.9, "
        "'agent', '2026-01-01')"
    )
    seeded_conn.execute(
        "INSERT INTO concept_cooccurrence (run_id, paper_id, concept_a, concept_b) "
        "VALUES ('r1', 'A1', 'a', 'b')"
    )
    seeded_conn.commit()

    db.reset_corpus(seeded_conn)
    for table in ("papers", "concepts", "citation_edges", "review_queue",
                  "paper_concepts", "concept_cooccurrence"):
        n = seeded_conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        assert n == 0


def test_init_db_backfills_run_id_on_pre_existing_tables(tmp_path):
    """A database created before the run_id column existed (OD19/OD20 predates OD26) must
    still work after init_db -- CREATE TABLE IF NOT EXISTS is a no-op against a table that's
    already there, so the column has to be added by hand."""
    path = tmp_path / "pre_od26.db"
    conn = db.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE papers (openalex_id TEXT PRIMARY KEY, title TEXT, abstract TEXT,
                publication_year INTEGER, venue_id TEXT, venue TEXT,
                cited_by_count INTEGER DEFAULT 0, doi TEXT);
            CREATE TABLE paper_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT NOT NULL, name TEXT NOT NULL,
                kind TEXT NOT NULL, canonical TEXT NOT NULL, cosine REAL NOT NULL,
                evidence_span TEXT NOT NULL, confidence REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'agent', created_at DATETIME NOT NULL);
            CREATE TABLE concept_cooccurrence (
                paper_id TEXT NOT NULL, concept_a TEXT NOT NULL, concept_b TEXT NOT NULL);
        """)
        conn.commit()
        assert "run_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(paper_concepts)")}

        db.init_db(conn)  # must not raise

        pc_cols = {r["name"] for r in conn.execute("PRAGMA table_info(paper_concepts)")}
        cooc_cols = {r["name"] for r in conn.execute("PRAGMA table_info(concept_cooccurrence)")}
        assert "run_id" in pc_cols
        assert "run_id" in cooc_cols

        # the backfilled default lets an old-shape row round-trip without a NOT NULL error
        conn.execute(
            "INSERT INTO papers (openalex_id) VALUES ('W1')"
        )
        conn.execute(
            "INSERT INTO paper_concepts (paper_id, name, kind, canonical, cosine, "
            "evidence_span, confidence, created_at) VALUES "
            "('W1', 'x', 'method', 'X', 0.9, 'x', 0.9, '2026-01-01')"
        )
        conn.commit()
        row = conn.execute("SELECT run_id FROM paper_concepts").fetchone()
        assert row["run_id"] == ""
    finally:
        conn.close()


def test_init_db_migrates_a_real_pre_od26_database_end_to_end(tmp_path):
    """Reproduces exactly what happened against the real data/axiom.db: an old-shape
    database (no run_id column, no unique index) that ALSO already has pre-B4 duplicate
    (paper_id, canonical, kind) rows from before the matching cascade existed. init_db must
    get it to the current shape in one call -- backfill run_id AND dedupe -- without raising,
    and reset_corpus must still be FK-safe afterward."""
    path = tmp_path / "real_shape.db"
    conn = db.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE papers (openalex_id TEXT PRIMARY KEY, title TEXT, abstract TEXT,
                publication_year INTEGER, venue_id TEXT, venue TEXT,
                cited_by_count INTEGER DEFAULT 0, doi TEXT);
            CREATE TABLE paper_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT NOT NULL, name TEXT NOT NULL,
                kind TEXT NOT NULL, canonical TEXT NOT NULL, cosine REAL NOT NULL,
                evidence_span TEXT NOT NULL, confidence REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'agent', created_at DATETIME NOT NULL);
            CREATE TABLE concept_cooccurrence (
                paper_id TEXT NOT NULL, concept_a TEXT NOT NULL, concept_b TEXT NOT NULL);
        """)
        conn.execute("INSERT INTO papers (openalex_id) VALUES ('S0001')")
        # Three pre-B4 duplicates for the same (paper_id, canonical, kind) -- the real bug.
        for name in ("Tokenization", "tokenization", "Tokenize"):
            conn.execute(
                "INSERT INTO paper_concepts (paper_id, name, kind, canonical, cosine, "
                "evidence_span, confidence, created_at) VALUES "
                "('S0001', ?, 'method', 'Tokenization', 0.9, 'x', 0.9, '2026-01-01')",
                (name,),
            )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) c FROM paper_concepts").fetchone()["c"] == 3

        db.init_db(conn)  # must not raise -- this is exactly what crashed against the real db

        rows = conn.execute(
            "SELECT * FROM paper_concepts WHERE paper_id = 'S0001' AND canonical = 'Tokenization'"
        ).fetchall()
        assert len(rows) == 1  # deduped down to one
        assert rows[0]["run_id"] == ""  # backfilled

        # the unique index is now real: a fresh duplicate insert must fail going forward
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO paper_concepts (run_id, paper_id, name, kind, canonical, cosine, "
                "evidence_span, confidence, created_at) VALUES "
                "('r1', 'S0001', 'Tokenizing', 'method', 'Tokenization', 0.9, 'x', 0.9, '2026-01-01')"
            )
        conn.rollback()

        db.reset_corpus(conn)  # must not raise a foreign-key violation either
        assert conn.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM paper_concepts").fetchone()["c"] == 0
    finally:
        conn.close()
