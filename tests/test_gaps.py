"""Tests for axiom.gaps — community detection, labels, centroids, gap scoring."""
from __future__ import annotations

import math

import numpy as np

from axiom import config, db, gaps, graph as graphmod
from tests.conftest import SEED_VECTORS


def _seed_agent_concepts(conn, paper_id: str, canonicals: list[str]) -> None:
    """Write `paper_concepts` rows for `paper_id`, one per canonical label."""
    rows = [(c, "method", c, 0.9, "evidence", 0.9, "agent", "2026-01-01T00:00:00+00:00")
            for c in canonicals]
    db.replace_paper_concepts(conn, "r1", paper_id, rows, [])


def test_detect_communities_splits_two_clusters(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    # All 8 corpus papers assigned; A* and B* land in different communities.
    assert set(node2c) == {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"}
    assert len({node2c["A2"], node2c["A3"], node2c["A4"]}) == 1
    assert len({node2c["B2"], node2c["B3"], node2c["B4"]}) == 1
    assert node2c["A2"] != node2c["B2"]


def test_community_labels_drop_generic(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    labels = gaps.community_labels(seeded_conn, node2c)
    all_labels = {lbl for lbls in labels.values() for lbl in lbls}
    # "Computer science" is in every community → generic → dropped as a label.
    assert "Computer science" not in all_labels
    assert "Information retrieval" in all_labels or "Machine translation" in all_labels


def test_community_labels_prefer_agent_concepts(seeded_conn):
    """OD34: a community with >= GAP_LABEL_MIN_AGENT_CONCEPTS distinct canonicals
    labels from `paper_concepts`, not the raw OpenAlex `concepts` table -- the
    other community (no agent data at all) still falls back to OpenAlex."""
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    cid_a, cid_b = node2c["A1"], node2c["B1"]
    assert cid_a != cid_b

    # Community A: 3 distinct agent canonicals across its 4 papers -> meets threshold.
    _seed_agent_concepts(seeded_conn, "A1", ["RAG core", "Dense retrieval"])
    _seed_agent_concepts(seeded_conn, "A2", ["Dense retrieval", "Reranking"])
    _seed_agent_concepts(seeded_conn, "A3", ["RAG core"])
    _seed_agent_concepts(seeded_conn, "A4", ["Reranking"])

    labels = gaps.community_labels(seeded_conn, node2c)
    assert set(labels[cid_a]) & {"RAG core", "Dense retrieval", "Reranking"}
    assert "Information retrieval" not in labels[cid_a]
    # Community B got no agent concepts -> unchanged OpenAlex fallback.
    assert "Machine translation" in labels[cid_b]


def test_community_labels_fall_back_when_agent_concepts_sparse(seeded_conn):
    """Fewer than GAP_LABEL_MIN_AGENT_CONCEPTS distinct agent canonicals in a
    community -> it still labels from OpenAlex `concepts`, not the sparse agent set."""
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    cid_a = node2c["A1"]

    # Only 1 distinct canonical across all of community A -- below the default
    # threshold (3), so GAP_LABEL_MIN_AGENT_CONCEPTS gates the fallback.
    _seed_agent_concepts(seeded_conn, "A1", ["Dense retrieval"])
    assert config.GAP_LABEL_MIN_AGENT_CONCEPTS > 1

    labels = gaps.community_labels(seeded_conn, node2c)
    assert "Information retrieval" in labels[cid_a]
    assert "Dense retrieval" not in labels[cid_a]


def test_centroids_are_means(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    centroids = gaps.community_centroids(node2c, SEED_VECTORS)
    for cid, c in centroids.items():
        # every seed vector is [1,0,0] or [1,1,0]; the mean keeps x==1.
        assert math.isclose(c[0], 1.0)


def test_cosine_zero_vector():
    assert gaps._cosine(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0


def test_analyze_gap_score_formula(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    analysis = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4)
    assert len(analysis.gaps) == 1
    cand = analysis.gaps[0]
    # centroids [1,0,0] vs [1,1,0] → cosine 1/sqrt(2); single A1->B1 bridge.
    assert math.isclose(cand.semantic_similarity, 1 / math.sqrt(2), abs_tol=1e-6)
    assert cand.inter_citations == 1
    assert math.isclose(cand.gap_score, cand.semantic_similarity / (1 + 1), abs_tol=1e-6)


# --- Review fix: junk-label filter + known-proper-noun casing ------------------
def test_is_junk_label_denylist_and_patterns():
    assert gaps._is_junk_label("bridging operation") is True
    assert gaps._is_junk_label("Bridging Operation") is True  # case-insensitive
    assert gaps._is_junk_label("98,734 theorems and proofs extracted from lean") is True
    assert gaps._is_junk_label("cai and yates (2013) dataset") is True
    assert gaps._is_junk_label("retrieval augmented generation") is False
    assert gaps._is_junk_label("gpt-4-turbo-2024-0409") is False  # a real model name, not junk


def test_normalize_casing_fixes_known_terms_only():
    assert gaps._normalize_casing("chatgpt") == "ChatGPT"
    assert gaps._normalize_casing("chatgpt-4o-latest") == "ChatGPT-4o-latest"
    assert gaps._normalize_casing("rag for question answering") == "RAG for question answering"
    # Ordinary lowercase phrases, the deliberate house style, stay untouched.
    assert gaps._normalize_casing("retrieve-and-read") == "retrieve-and-read"


# --- B1: similarity percentile --------------------------------------------------
def test_similarity_percentile_ranks_among_all_pairs():
    all_sims = [0.80, 0.90, 0.95, 0.98, 0.99]
    assert gaps.similarity_percentile(0.80, all_sims) == 20.0   # lowest -> 1/5
    assert gaps.similarity_percentile(0.99, all_sims) == 100.0  # highest -> 5/5
    assert gaps.similarity_percentile(0.95, all_sims) == 60.0   # 3/5 at-or-below


def test_similarity_percentile_empty_is_zero():
    assert gaps.similarity_percentile(0.9, []) == 0.0


def test_analyze_attaches_a_real_percentile(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    # Only one candidate pair in the seeded fixture (2 communities) -> it's
    # necessarily both the closest and the only one, 100th percentile.
    assert cand.similarity_percentile == 100.0


# --- B4: stable gap_id ---------------------------------------------------------
def test_gap_id_is_stable_and_order_independent():
    a = gaps.Community(cid=0, members=["P1", "P2"], labels=["X"], size=2)
    b = gaps.Community(cid=1, members=["P3", "P4"], labels=["Y"], size=2)
    cand_ab = gaps.GapCandidate(a=a, b=b, semantic_similarity=0.5, inter_citations=1, gap_score=0.25)
    cand_ba = gaps.GapCandidate(a=b, b=a, semantic_similarity=0.5, inter_citations=1, gap_score=0.25)
    assert cand_ab.gap_id == cand_ba.gap_id        # swapping a/b doesn't change the id
    assert len(cand_ab.gap_id) == 16


def test_gap_id_differs_for_different_member_sets():
    a = gaps.Community(cid=0, members=["P1", "P2"], labels=["X"], size=2)
    b = gaps.Community(cid=1, members=["P3", "P4"], labels=["Y"], size=2)
    c = gaps.Community(cid=2, members=["P5", "P6"], labels=["Z"], size=2)
    cand1 = gaps.GapCandidate(a=a, b=b, semantic_similarity=0.5, inter_citations=1, gap_score=0.25)
    cand2 = gaps.GapCandidate(a=a, b=c, semantic_similarity=0.5, inter_citations=1, gap_score=0.25)
    assert cand1.gap_id != cand2.gap_id


def test_gap_id_survives_reordering(seeded_conn):
    """The real regression this stage's B4 fixes: analyze() with a different
    resolution can reorder `gaps`, which used to silently repoint the old
    list-position gap_index at a different candidate. gap_id doesn't care
    about position."""
    g = graphmod.load_graph(seeded_conn)
    gaps_list = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps
    by_id = {c.gap_id: (frozenset(c.a.members), frozenset(c.b.members)) for c in gaps_list}
    # Re-running analyze() (even with identical inputs) must map the same
    # gap_id back to the same pair of member sets.
    gaps_list_again = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps
    for c in gaps_list_again:
        assert by_id[c.gap_id] == (frozenset(c.a.members), frozenset(c.b.members))


def test_analyze_respects_min_community_size(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    # Both clusters are size 4; require 5 → no candidate pairs survive.
    analysis = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=5)
    assert analysis.gaps == []


def test_analyze_empty_graph(empty_conn):
    import networkx as nx
    analysis = gaps.analyze(nx.DiGraph(), empty_conn, {})
    assert analysis.gaps == []
    assert analysis.communities == {}


# --- OD17 composite G-score --------------------------------------------------
def test_components_in_unit_range(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    assert set(cand.components) == {"similarity", "disconnection", "velocity", "authority"}
    for name, val in cand.components.items():
        assert 0.0 <= val <= 1.0, name
    assert 0.0 <= cand.g_score <= 1.0


def test_g_score_matches_weighted_components(seeded_conn, monkeypatch):
    weights = {"similarity": 0.5, "disconnection": 0.2, "velocity": 0.2, "authority": 0.1}
    monkeypatch.setattr(config, "load_calibration",
                        lambda: {"weights": weights, "tau": 0.5})
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    expected = sum(weights[k] * cand.components[k] for k in weights)
    assert math.isclose(cand.g_score, expected, abs_tol=1e-9)


def test_meets_threshold_follows_tau(seeded_conn, monkeypatch):
    weights = {"similarity": 0.4, "disconnection": 0.3, "velocity": 0.2, "authority": 0.1}

    monkeypatch.setattr(config, "load_calibration",
                        lambda: {"weights": weights, "tau": 0.0})
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    assert cand.meets_threshold is True        # τ=0 → everything passes
    g_val = cand.g_score

    monkeypatch.setattr(config, "load_calibration",
                        lambda: {"weights": weights, "tau": g_val + 0.01})
    cand2 = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    assert cand2.meets_threshold is False       # τ just above g_score → fails
