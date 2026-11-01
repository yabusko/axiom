"""Offline tests for axiom/concept_graph.py (R7).

Builds a tiny fake `concept_cooccurrence` table in a tmp SQLite db (real
schema via axiom.db.init_db) -- no Qdrant, no torch, no network.

B5: exercises axiom's REAL `graph.pagerank` (promoted, weighted
power-iteration, see graph.py) directly -- no monkeypatch. An earlier
version of this file injected `nx.pagerank` in an autouse fixture under the
claim that `graph.pagerank` "does not exist yet"; that was already false by
the time this ran (OD19's promotion had landed), so the suite was silently
never exercising axiom's own implementation. Removed.
"""
from __future__ import annotations

import networkx as nx
import pytest

from axiom import concept_graph, db, graph as graphmod


def _seed(conn, rows):
    """rows: iterable of (paper_id, concept_a, concept_b) -- already a<b.

    `concept_cooccurrence.paper_id` FKs to `papers.openalex_id`, so each
    distinct paper_id gets a minimal placeholder paper row first.
    """
    for pid in {r[0] for r in rows}:
        db.insert_paper(conn, openalex_id=pid, title=f"paper {pid}", abstract=None,
                        publication_year=2024, venue_id=None, venue="ACL")
    conn.executemany(
        "INSERT INTO concept_cooccurrence (paper_id, concept_a, concept_b) VALUES (?, ?, ?)",
        list(rows),
    )
    conn.commit()


@pytest.fixture
def cooc_db(tmp_path):
    path = tmp_path / "concepts.db"
    conn = db.connect(path)
    try:
        db.init_db(conn)
        # Two clusters: {LoRA, Fine-tuning, Adapters} and {Translation, BLEU},
        # plus a single weak bridge (LoRA, Translation) so Louvain still
        # separates them, and LoRA<->Fine-tuning co-occurs in 3 papers (the
        # heaviest edge) so PageRank/weighted-degree favor LoRA.
        _seed(conn, [
            ("p1", "Fine-tuning", "LoRA"),
            ("p2", "Fine-tuning", "LoRA"),
            ("p3", "Fine-tuning", "LoRA"),
            ("p4", "Adapters", "LoRA"),
            ("p5", "Adapters", "Fine-tuning"),
            ("p6", "BLEU", "Translation"),
            ("p7", "BLEU", "Translation"),
            ("p8", "LoRA", "Translation"),   # weak cross-cluster bridge
        ])
    finally:
        conn.close()
    return path


def test_load_graph_builds_weighted_undirected_graph(cooc_db):
    g = concept_graph.load_graph(db_path=cooc_db)
    assert isinstance(g, nx.Graph)
    assert not g.is_directed()
    assert g.number_of_nodes() == 5  # Fine-tuning, LoRA, Adapters, BLEU, Translation
    assert g["Fine-tuning"]["LoRA"]["weight"] == 3
    assert g["Adapters"]["LoRA"]["weight"] == 1
    assert g["BLEU"]["Translation"]["weight"] == 2


def test_load_graph_empty_table_returns_empty_graph(tmp_path):
    path = tmp_path / "empty.db"
    conn = db.connect(path)
    db.init_db(conn)
    conn.close()
    g = concept_graph.load_graph(db_path=path)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


def test_load_graph_accepts_an_open_connection(cooc_db):
    conn = db.connect(cooc_db)
    try:
        g = concept_graph.load_graph(conn=conn)
        assert g.number_of_nodes() == 5
    finally:
        conn.close()


def test_detect_communities_separates_the_two_clusters(cooc_db):
    g = concept_graph.load_graph(db_path=cooc_db)
    node2c = concept_graph.detect_communities(g)
    assert node2c["Fine-tuning"] == node2c["LoRA"] == node2c["Adapters"]
    assert node2c["BLEU"] == node2c["Translation"]
    assert node2c["Fine-tuning"] != node2c["BLEU"]


def test_detect_communities_empty_graph(cooc_db):
    assert concept_graph.detect_communities(nx.Graph()) == {}


def test_communities_summary_largest_first_and_min_size_filter(cooc_db):
    g = concept_graph.load_graph(db_path=cooc_db)
    node2c = concept_graph.detect_communities(g)
    out = concept_graph.communities_summary(node2c, min_size=1)
    assert len(out) == 2
    assert out[0].size >= out[1].size
    assert all(c.size == len(c.members) for c in out)

    out_filtered = concept_graph.communities_summary(node2c, min_size=3)
    assert all(c.size >= 3 for c in out_filtered)


def test_central_concepts_calls_graph_pagerank_and_ranks_by_score(cooc_db):
    g = concept_graph.load_graph(db_path=cooc_db)
    out = concept_graph.central_concepts(g, top_k=3)
    assert len(out) == 3
    scores = [c.pagerank for c in out]
    assert scores == sorted(scores, reverse=True)
    # LoRA touches every other node in its cluster plus the cross-cluster
    # bridge -- highest degree in the graph, a strong PageRank candidate.
    assert out[0].concept == "LoRA"
    assert out[0].degree == 3            # Fine-tuning, Adapters, Translation
    assert out[0].weighted_degree == 5   # 3 + 1 + 1


def test_central_concepts_empty_graph_returns_empty_list():
    assert concept_graph.central_concepts(nx.Graph()) == []


def test_central_concepts_propagates_real_attribute_error_without_the_fixture(
    cooc_db, monkeypatch
):
    """If graph.pagerank truly doesn't exist (no fixture patching it), calling
    central_concepts raises AttributeError -- confirming concept_graph.py calls
    `graph.pagerank` exactly, not a private fallback, so the real gap is loud."""
    monkeypatch.delattr(graphmod, "pagerank", raising=False)
    g = concept_graph.load_graph(db_path=cooc_db)
    with pytest.raises(AttributeError):
        concept_graph.central_concepts(g, top_k=3)


def test_analyze_bundles_graph_communities_and_central(cooc_db):
    result = concept_graph.analyze(db_path=cooc_db, min_community_size=1, top_k=5)
    assert set(result.node2c) == {"Fine-tuning", "LoRA", "Adapters", "BLEU", "Translation"}
    assert len(result.communities) == 2
    assert len(result.central) == 5
