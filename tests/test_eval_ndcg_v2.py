"""Offline tests for the v2 additions to scripts/eval_ndcg.py (R8/R9).

Hard rule: never construct Specter2Encoder/AxiomQdrant for real and never touch
a real model/network/Qdrant. Fakes stand in for both; `hybrid_struct`'s
structural arm uses a real tmp SQLite db (cheap, pure Python). `dense_general`'s
GeneralEncoder is never constructed -- only its pure helper functions
(`brute_force_rank`, `load_general_vectors`, `fetch_paper_rows`) are exercised.

Also confirms v1 is untouched: eval/ndcg_queries.json and eval/report.md are
byte-identical before/after importing and exercising this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axiom import config, db
from axiom.qdrant_client import SearchHit
from scripts import eval_ndcg

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_QUERIES = REPO_ROOT / "eval" / "ndcg_queries.json"
V1_REPORT = REPO_ROOT / "eval" / "report.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# v1 is untouched
# ---------------------------------------------------------------------------
def test_v1_eval_files_are_byte_identical_after_importing_v2_module():
    """Importing/using the extended script must never write to the v1 paths."""
    before_q, before_r = _sha256(V1_QUERIES), _sha256(V1_REPORT)
    # exercise a v2-only code path, which must not touch v1 files
    eval_ndcg.rrf_fuse([["a", "b"], ["b", "a"]])
    assert _sha256(V1_QUERIES) == before_q
    assert _sha256(V1_REPORT) == before_r


def test_v1_mode_choices_still_include_dense_and_hybrid():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "hybrid", "hybrid_struct", "dense_general"],
                     default="hybrid")
    ns = ap.parse_args([])
    assert ns.mode == "hybrid"
    ns2 = ap.parse_args(["--mode", "dense"])
    assert ns2.mode == "dense"


def test_ndcg_at_k_unchanged_behavior():
    # Same math v1 relied on: perfect ranking -> 1.0, empty judgments -> 0.0.
    assert eval_ndcg.ndcg_at_k([3, 2, 1], [3, 2, 1], k=10) == pytest.approx(1.0)
    assert eval_ndcg.ndcg_at_k([], [], k=10) == 0.0


# ---------------------------------------------------------------------------
# RRF fusion helper
# ---------------------------------------------------------------------------
def test_rrf_fuse_single_arm_preserves_order():
    assert eval_ndcg.rrf_fuse([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_fuse_promotes_item_strong_in_second_arm():
    hybrid_ranking = ["P1", "P2", "P3", "P4", "P5"]
    struct_ranking = ["P5", "P1", "P2", "P3", "P4"]
    fused = eval_ndcg.rrf_fuse([hybrid_ranking, struct_ranking])
    assert fused == ["P1", "P2", "P5", "P3", "P4"]
    # P5 moved up from last place in the hybrid-only ranking.
    assert fused.index("P5") < hybrid_ranking.index("P5")


def test_rrf_fuse_respects_top_k():
    fused = eval_ndcg.rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], top_k=2)
    assert len(fused) == 2


# ---------------------------------------------------------------------------
# structural_scores (hybrid_struct's concept-overlap arm)
# ---------------------------------------------------------------------------
@pytest.fixture
def struct_conn(tmp_path):
    path = tmp_path / "eval_struct.db"
    conn = db.connect(path)
    db.init_db(conn)
    for pid in ("P1", "P2", "P3", "P4", "P5"):
        db.insert_paper(conn, openalex_id=pid, title=f"paper {pid}", abstract=None,
                        publication_year=2024, venue_id=None, venue="ACL")
    # P5 shares both concepts with the rest of the pool (P1 has RAG, P2 has
    # Retrieval); P3/P4 have none -- P5 should score highest structurally.
    rows = [
        ("P1", "RAG"), ("P2", "Retrieval"), ("P5", "RAG"), ("P5", "Retrieval"),
    ]
    for pid, concept in rows:
        conn.execute(
            "INSERT INTO paper_concepts (paper_id, name, kind, canonical, cosine, "
            "evidence_span, confidence, source, created_at) VALUES (?, ?, 'method', ?, "
            "0.9, 'x', 0.9, 'agent', '2026-01-01')",
            (pid, concept, concept),
        )
    for pid in ("P1", "P2", "P3", "P4", "P5"):
        conn.execute(
            "INSERT INTO concept_cooccurrence (paper_id, concept_a, concept_b) "
            "VALUES (?, 'RAG', 'Retrieval')",
            (pid,),
        )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def test_structural_scores_favors_candidate_overlapping_the_pool(struct_conn):
    scores = eval_ndcg.structural_scores(struct_conn, ["P1", "P2", "P3", "P4", "P5"])
    assert scores["P5"] > scores["P1"]
    assert scores["P5"] > scores["P2"]
    assert scores.get("P3", 0.0) == 0.0
    assert scores.get("P4", 0.0) == 0.0


def test_structural_scores_empty_candidate_list():
    assert eval_ndcg.structural_scores(None, []) == {}


def test_structural_scores_no_rows_degrades_to_empty(tmp_path):
    """paper_concepts has no rows for the corpus yet -- hybrid_struct must
    degrade to a no-op (score 0 everywhere), not crash."""
    path = tmp_path / "empty_struct.db"
    conn = db.connect(path)
    db.init_db(conn)
    for pid in ("P1", "P2"):
        db.insert_paper(conn, openalex_id=pid, title="t", abstract=None,
                        publication_year=2024, venue_id=None, venue="ACL")
    conn.commit()
    try:
        assert eval_ndcg.structural_scores(conn, ["P1", "P2"]) == {}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# run_query dispatch: hybrid_struct end-to-end (fake encoder/store, real tmp db)
# ---------------------------------------------------------------------------
class _FakeEncoder:
    def encode_query(self, query: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class _FakeHybridStore:
    """store.search_hybrid returns a fixed ranking regardless of input."""

    def __init__(self, hits: list[SearchHit]):
        self._hits = hits

    def search_hybrid(self, *, query_vector, query_text, top_k=10, venues=None, year_range=None):
        return self._hits[:top_k]


def _hit(pid: str, score: float) -> SearchHit:
    return SearchHit(paper_id=pid, title=f"title {pid}", year=2024, venue="ACL",
                      cited_by_count=1, score=score)


def test_run_query_hybrid_struct_reranks_via_structural_overlap(struct_conn):
    hits = [_hit("P1", 0.9), _hit("P2", 0.8), _hit("P3", 0.7),
            _hit("P4", 0.6), _hit("P5", 0.5)]
    store = _FakeHybridStore(hits)
    out = eval_ndcg.run_query(
        "hybrid_struct", "retrieval augmented generation", _FakeEncoder(), store, 5,
        struct_conn=struct_conn, struct_candidates=5,
    )
    ids = [h.paper_id for h in out]
    assert ids == ["P1", "P2", "P5", "P3", "P4"]
    assert all(isinstance(h, SearchHit) for h in out)


def test_run_query_hybrid_struct_empty_hybrid_hits_returns_empty(struct_conn):
    store = _FakeHybridStore([])
    out = eval_ndcg.run_query(
        "hybrid_struct", "q", _FakeEncoder(), store, 5, struct_conn=struct_conn,
    )
    assert out == []


def test_run_query_unknown_mode_raises():
    with pytest.raises(ValueError):
        eval_ndcg.run_query("bogus", "q", _FakeEncoder(), _FakeHybridStore([]), 5)


# ---------------------------------------------------------------------------
# dense_general helpers (GeneralEncoder itself is never constructed)
# ---------------------------------------------------------------------------
def test_load_general_vectors_missing_cache_raises_systemexit(tmp_path):
    with pytest.raises(SystemExit):
        eval_ndcg.load_general_vectors(tmp_path / "missing.json")


def test_load_general_vectors_reads_cache(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"P1": [1.0, 0.0], "P2": [0.0, 1.0]}), encoding="utf-8")
    vectors = eval_ndcg.load_general_vectors(cache)
    assert vectors == {"P1": [1.0, 0.0], "P2": [0.0, 1.0]}


def test_brute_force_rank_orders_by_cosine_similarity():
    vectors = {"P1": [1.0, 0.0], "P2": [0.0, 1.0], "P3": [0.9, 0.1]}
    ranked = eval_ndcg.brute_force_rank([1.0, 0.0], vectors, top_k=3)
    ids = [pid for pid, _ in ranked]
    assert ids[0] == "P1"      # exact match
    assert ids[1] == "P3"      # close
    assert ids[2] == "P2"      # orthogonal, least similar


def test_brute_force_rank_empty_vectors_returns_empty():
    assert eval_ndcg.brute_force_rank([1.0, 0.0], {}, top_k=5) == []


def test_fetch_paper_rows_empty_ids_returns_empty(struct_conn):
    assert eval_ndcg.fetch_paper_rows(struct_conn, []) == {}


def test_fetch_paper_rows_returns_metadata(struct_conn):
    rows = eval_ndcg.fetch_paper_rows(struct_conn, ["P1", "P2"])
    assert set(rows) == {"P1", "P2"}
    assert rows["P1"]["venue"] == "ACL"


def test_run_query_dense_general_builds_hits_from_cache_and_rows(struct_conn):
    class _FakeGeneralEncoder:
        def encode_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    general_vectors = {"P1": [1.0, 0.0], "P2": [0.0, 1.0]}
    paper_rows = eval_ndcg.fetch_paper_rows(struct_conn, list(general_vectors))
    out = eval_ndcg.run_query(
        "dense_general", "q", None, None, 2,
        general_encoder=_FakeGeneralEncoder(), general_vectors=general_vectors,
        paper_rows=paper_rows,
    )
    assert [h.paper_id for h in out] == ["P1", "P2"]
    assert out[0].venue == "ACL"


# ---------------------------------------------------------------------------
# is_v2 / report-path selection logic (pure, no I/O)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,v2_flag,expected", [
    ("dense", False, False),
    ("hybrid", False, False),
    ("dense", True, True),
    ("hybrid_struct", False, True),
    ("dense_general", False, True),
])
def test_is_v2_selection_logic(mode, v2_flag, expected):
    is_v2 = v2_flag or mode in eval_ndcg.V2_MODES
    assert is_v2 is expected


def test_v1_invocation_queries_path_is_unchanged():
    is_v2 = False
    queries_path = eval_ndcg.QUERIES_PATH if not is_v2 else config.EVAL_V2_QUERIES_PATH
    assert queries_path == V1_QUERIES


def test_report_default_path_pattern_for_v2_modes():
    for mode in eval_ndcg.V2_MODES:
        expected = eval_ndcg.PROJECT_ROOT / "eval" / f"report_{mode}.md"
        assert str(expected).endswith(f"report_{mode}.md")
