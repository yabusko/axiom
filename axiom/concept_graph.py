"""Concept co-occurrence graph (R7, docs/agentic/ARCHITECTURE.md §1/§6).

Independent of `graph.py` (citation graph) / `gaps.py` (citation-graph
communities): this module builds a SEPARATE, undirected graph from the
agent-extracted `concept_cooccurrence` table (OD20 — agent concepts never
merge into the OpenAlex `concepts` table `gaps.py` reads). Nodes are canonical
concept names; an edge's weight is how many papers the pair co-occurred in.

Two analyses run on that graph, mirroring the citation-graph pattern exactly
so the two tracks stay consistent (OD19):
  - Louvain communities (`nx.community.louvain_communities`, same call shape
    `gaps.py.detect_communities` uses on the citation graph).
  - PageRank centrality via `graph.pagerank` (OD19: the citation graph's
    private `_pagerank` power-iteration, promoted to a public, shared
    `graph.pagerank(g, alpha, max_iter, tol)` — see the module-level NOTE
    below; this file calls that name as-is).

Pure-Python: networkx only (same as graph.py/gaps.py) — no torch.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import networkx as nx

from axiom import db
from axiom import graph as graphmod

# NOTE (patch for the main session, see .claude/agentic/patch_notes_eval_implementer.md):
# `graph.pagerank` does not exist yet — only the private `graph._pagerank` does.
# This module calls `graphmod.pagerank(g, alpha=alpha, max_iter=max_iter, tol=tol)`
# below exactly as if the promotion (OD19) has already landed. Until `graph.py`
# gains a one-line `pagerank = _pagerank` (or the body renamed), calling
# `central_concepts()` for real raises AttributeError; tests monkeypatch it.


@dataclass
class ConceptCommunity:
    """One Louvain cluster of co-occurring concepts, largest-first (cid 0 = biggest)."""
    cid: int
    members: list[str]
    size: int


@dataclass
class ConceptCentrality:
    """One concept's PageRank centrality on the co-occurrence graph."""
    concept: str
    pagerank: float
    degree: int          # number of distinct co-occurring partner concepts
    weighted_degree: int  # sum of co-occurrence counts over its edges


@dataclass
class ConceptGraphAnalysis:
    node2c: dict[str, int]
    communities: list[ConceptCommunity]
    central: list[ConceptCentrality]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def load_graph(conn: sqlite3.Connection | None = None,
                db_path=None) -> nx.Graph:
    """Build the undirected concept co-occurrence graph from SQLite.

    Nodes = canonical concept names that appear in `concept_cooccurrence`.
    Edge (a, b) weight = number of `concept_cooccurrence` rows for that pair
    (i.e. how many papers extracted both concepts together). `db_path` lets
    callers point at any DB (main corpus `data/axiom.db` or eval corpus
    `data/eval.db`, OD11-safe — this module never hardcodes a path); `conn`
    takes precedence if both are given.
    """
    own_conn = conn is None
    if own_conn:
        conn = db.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT concept_a, concept_b, COUNT(*) AS n "
            "FROM concept_cooccurrence GROUP BY concept_a, concept_b"
        ).fetchall()
    finally:
        if own_conn:
            conn.close()

    g = nx.Graph()
    for r in rows:
        a, b, n = r["concept_a"], r["concept_b"], r["n"]
        if g.has_edge(a, b):
            g[a][b]["weight"] += n
        else:
            g.add_edge(a, b, weight=n)
    return g


# ---------------------------------------------------------------------------
# Communities (Louvain)
# ---------------------------------------------------------------------------
def detect_communities(g: nx.Graph, resolution: float = 1.0,
                        seed: int = 42) -> dict[str, int]:
    """Louvain communities on the co-occurrence graph.

    Same library call `gaps.py.detect_communities` uses on the (undirected)
    citation graph: `nx.community.louvain_communities`, weighted by the edge
    `weight` attribute set in `load_graph`. Returns {concept: community_id},
    ids assigned largest-community-first (matches `gaps.py`'s convention).
    """
    if g.number_of_nodes() == 0:
        return {}
    comms = nx.community.louvain_communities(
        g, weight="weight", seed=seed, resolution=resolution
    )
    node2c: dict[str, int] = {}
    for cid, members in enumerate(sorted(comms, key=len, reverse=True)):
        for n in members:
            node2c[n] = cid
    return node2c


def communities_summary(node2c: dict[str, int],
                         min_size: int = 1) -> list[ConceptCommunity]:
    """Group `detect_communities`' output into `ConceptCommunity` rows.

    `GET /concepts/communities` calls this (via `analyze()` below, or
    directly after its own `detect_communities` call).
    """
    members: dict[int, list[str]] = {}
    for concept, cid in node2c.items():
        members.setdefault(cid, []).append(concept)
    out = [
        ConceptCommunity(cid=cid, members=sorted(ms), size=len(ms))
        for cid, ms in members.items()
        if len(ms) >= min_size
    ]
    out.sort(key=lambda c: c.size, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Centrality (PageRank)
# ---------------------------------------------------------------------------
def central_concepts(g: nx.Graph, top_k: int = 20, *,
                      alpha: float = 0.85, max_iter: int = 100,
                      tol: float = 1.0e-6) -> list[ConceptCentrality]:
    """Rank concepts by PageRank on the (weighted) co-occurrence graph.

    Calls `graph.pagerank(g, alpha, max_iter, tol)` (OD19's promoted, shared
    power-iteration implementation — see the module NOTE above) instead of a
    second copy, so the citation graph and this graph can never silently
    drift onto different PageRank math.

    `GET /concepts/central` calls this (via `analyze()` below, or directly).
    """
    if g.number_of_nodes() == 0:
        return []
    pr = graphmod.pagerank(g, alpha, max_iter, tol)
    weighted_deg = dict(g.degree(weight="weight"))
    deg = dict(g.degree())
    ranked = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        ConceptCentrality(
            concept=concept,
            pagerank=score,
            degree=int(deg.get(concept, 0)),
            weighted_degree=int(weighted_deg.get(concept, 0)),
        )
        for concept, score in ranked
    ]


# ---------------------------------------------------------------------------
# Convenience: one-call entrypoint for API wiring
# ---------------------------------------------------------------------------
def analyze(conn: sqlite3.Connection | None = None, db_path=None, *,
            min_community_size: int = 2, top_k: int = 20,
            resolution: float = 1.0) -> ConceptGraphAnalysis:
    """Build the graph and run both analyses in one call.

    Exposed for `axiom/agents/pipeline.py` and `api/main.py`'s
    `GET /concepts/communities` / `GET /concepts/central` to call without
    each re-deriving the graph-building/Louvain/PageRank sequence.
    """
    g = load_graph(conn, db_path)
    node2c = detect_communities(g, resolution=resolution)
    communities = communities_summary(node2c, min_size=min_community_size)
    central = central_concepts(g, top_k=top_k)
    return ConceptGraphAnalysis(node2c=node2c, communities=communities, central=central)
