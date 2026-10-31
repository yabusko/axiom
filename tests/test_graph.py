"""Tests for axiom.graph — load, PageRank direction, influence, neighbors, stats."""
from __future__ import annotations

import networkx as nx

from axiom import graph as graphmod


def test_load_graph_flags_and_dangling(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    assert g.nodes["A1"]["in_corpus"] is True
    assert g.nodes["A1"]["cited_by_count"] == 100
    # WEXT1 is a referenced work outside the corpus: present, bare, not in-corpus.
    assert "WEXT1" in g
    assert g.nodes["WEXT1"].get("in_corpus", False) is False


def test_corpus_subgraph_excludes_external(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    sub = graphmod.corpus_subgraph(g)
    assert "WEXT1" not in sub
    assert sub.number_of_nodes() == 8


def test_pagerank_sums_to_one_and_ranks_cited(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    sub = graphmod.corpus_subgraph(g)
    pr = graphmod._pagerank(sub)
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    # A1 and B1 are the most-cited-within-corpus nodes; they should top their peers.
    assert pr["A1"] > pr["A4"]
    assert pr["B1"] > pr["B4"]


def test_pagerank_unweighted_edges_match_plain_networkx_pagerank(seeded_conn):
    """B5 regression: on a graph where every edge has the default weight of
    1 (none of graph.py's citation edges carry a `weight` attribute), the
    now-weighted `_pagerank` must be numerically IDENTICAL to the old
    unweighted behavior -- cross-checked here against networkx's own
    `pagerank` with `weight=None` (i.e. also ignoring any weight), which is
    exactly what the old out_degree/successor-count implementation computed.
    """
    import pytest

    g = graphmod.load_graph(seeded_conn)
    sub = graphmod.corpus_subgraph(g)
    ours = graphmod._pagerank(sub)
    reference = nx.pagerank(sub, alpha=0.85, max_iter=100, tol=1.0e-6, weight=None)
    assert ours.keys() == reference.keys()
    for node in ours:
        assert ours[node] == pytest.approx(reference[node], rel=1e-4, abs=1e-6)


def test_pagerank_is_actually_weighted():
    """B5: a heavier edge pulls more rank than a lighter one out of the same
    source -- the bug this fixes was that `_pagerank` never read the `weight`
    edge attribute at all, so this would previously have been a tie.
    """
    g = nx.DiGraph()
    g.add_edge("A", "B", weight=3)
    g.add_edge("A", "C", weight=1)
    g.add_edge("B", "A", weight=1)
    g.add_edge("C", "A", weight=1)
    pr = graphmod._pagerank(g)
    assert pr["B"] > pr["C"]


def test_pagerank_weighted_matches_hand_computed_fixed_point():
    """Hand-checked once (not recomputed every run, per B5's instruction):
    a symmetric directed triangle A->B->C->A with equal weight-1 edges has a
    uniform stationary distribution (1/3 each) regardless of alpha, since
    every node has exactly one successor receiving its entire rank."""
    g = nx.DiGraph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    g.add_edge("C", "A", weight=1)
    pr = graphmod._pagerank(g, alpha=0.85, max_iter=200, tol=1.0e-12)
    for node in ("A", "B", "C"):
        assert abs(pr[node] - 1.0 / 3.0) < 1e-6


def test_influence_local_in_degree(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    ranking = graphmod.influence(g, top_k=8)
    by_id = {r.paper_id: r for r in ranking}
    # A1 is cited by A2, A3, A4 (within corpus) → local_in_degree 3.
    assert by_id["A1"].local_in_degree == 3


def test_neighbors_direction_and_sort(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    nb = graphmod.neighbors(g, "A1")
    ref_ids = {r["paper_id"] for r in nb.references}
    citer_ids = {c["paper_id"] for c in nb.cited_by}
    assert ref_ids == {"B1", "WEXT1"}          # A1 cites B1 and the external
    assert citer_ids == {"A2", "A3", "A4"}     # A2/A3/A4 cite A1
    # cited_by sorted by global cited_by_count descending.
    counts = [c["cited_by_count"] for c in nb.cited_by]
    assert counts == sorted(counts, reverse=True)


def test_ego_subgraph_in_corpus_only(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    ego = graphmod.ego_subgraph(g, "A1", radius=1, in_corpus_only=True)
    assert "WEXT1" not in ego
    assert "A1" in ego


def test_stats(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    s = graphmod.stats(g)
    assert s["papers"] == 8
    assert s["external_nodes"] == 1            # WEXT1
    assert s["edges_in_corpus"] == 13          # 12 intra + 1 A1->B1 bridge
