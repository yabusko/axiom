"""Citation graph — the NetworkX-first graph seam (decision OD6, no Neo4j).

SQLite `citation_edges` is the durable source of truth; this module loads it into
an in-memory directed graph. Edge direction is `src -> dst` meaning *src cites
dst* (the citing paper points at the cited one).

Dangling externals are KEPT (decision: schema has no FK on dst_id): a referenced
work outside the corpus becomes a metadata-less node flagged `in_corpus=False`.
Influence metrics run on the corpus-only subgraph so those sinks don't distort
ranks; neighbor lookups still see them so "what does this paper cite" is honest.

Pure-Python (networkx only) — no torch — so it imports anywhere.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import networkx as nx

from axiom import config, db


@dataclass
class PaperRank:
    """One row of an influence ranking, ready to render."""
    paper_id: str
    title: str | None
    year: int | None
    venue: str | None
    cited_by_count: int        # global, from OpenAlex
    local_in_degree: int       # citations from *within* the corpus
    pagerank: float


@dataclass
class Neighbors:
    """A paper's immediate citation neighborhood."""
    references: list[dict]     # papers this one cites (successors); may be external
    cited_by: list[dict]       # corpus papers that cite this one (predecessors)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def load_graph(conn: sqlite3.Connection | None = None) -> nx.DiGraph:
    """Build the directed citation graph from SQLite.

    In-corpus papers get full node metadata (title/year/venue/cited_by_count)
    and `in_corpus=True`; dangling external targets get `in_corpus=False` and no
    metadata. Returns a DiGraph (edge src->dst = "src cites dst").
    """
    own_conn = conn is None
    if own_conn:
        conn = db.connect()
    try:
        papers = conn.execute(
            "SELECT openalex_id, title, publication_year, venue, cited_by_count "
            "FROM papers"
        ).fetchall()
        edges = conn.execute(
            "SELECT src_id, dst_id, year FROM citation_edges"
        ).fetchall()
    finally:
        if own_conn:
            conn.close()

    g = nx.DiGraph()
    for p in papers:
        g.add_node(
            p["openalex_id"],
            title=p["title"],
            year=p["publication_year"],
            venue=p["venue"],
            cited_by_count=p["cited_by_count"] or 0,
            in_corpus=True,
        )
    for e in edges:
        # Dangling target: create a bare external node the metrics will exclude.
        if e["dst_id"] not in g:
            g.add_node(e["dst_id"], in_corpus=False)
        g.add_edge(e["src_id"], e["dst_id"], year=e["year"])
    return g


def corpus_subgraph(g: nx.DiGraph) -> nx.DiGraph:
    """Subgraph induced by in-corpus nodes only (drops dangling externals)."""
    nodes = [n for n, d in g.nodes(data=True) if d.get("in_corpus")]
    return g.subgraph(nodes)


# ---------------------------------------------------------------------------
# Metrics & queries
# ---------------------------------------------------------------------------
def _pagerank(g: nx.DiGraph | nx.Graph, alpha: float = 0.85,
              max_iter: int = 100, tol: float = 1.0e-6) -> dict[str, float]:
    """Power-iteration PageRank in pure Python (no scipy/numpy dependency).

    Directed graphs (the citation graph): edge u->v contributes rank to v, so
    heavily-CITED nodes (many in-edges) score highest — the citation-influence
    reading `influence()` wants. Undirected graphs (e.g. the concept
    co-occurrence graph, OD19) are treated as symmetric: every edge contributes
    rank both ways via `degree`/`neighbors` instead of `out_degree`/`successors`.
    Dangling nodes (zero out-edges) redistribute their rank uniformly via teleport.
    """
    nodes = list(g.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    directed = g.is_directed()
    out_deg = dict(g.out_degree()) if directed else dict(g.degree())
    successors = g.successors if directed else g.neighbors
    dangling = [u for u in nodes if out_deg[u] == 0]
    x = {u: 1.0 / n for u in nodes}
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(nodes, 0.0)
        dangle = alpha * sum(xlast[d] for d in dangling) / n
        teleport = (1.0 - alpha) / n + dangle
        for u in nodes:
            if out_deg[u]:
                share = alpha * xlast[u] / out_deg[u]
                for v in successors(u):
                    x[v] += share
        for u in nodes:
            x[u] += teleport
        if sum(abs(x[u] - xlast[u]) for u in nodes) < tol:
            break
    return x


# Public alias (OD19): axiom/concept_graph.py calls this on the undirected
# concept co-occurrence graph; influence() below keeps using it on the
# directed citation graph. One implementation, two call sites.
pagerank = _pagerank


def influence(g: nx.DiGraph, top_k: int = 20) -> list[PaperRank]:
    """Rank in-corpus papers by PageRank over the corpus-only citation graph.

    PageRank runs on the corpus subgraph so metadata-less dangling sinks can't
    absorb rank. `local_in_degree` is how many corpus papers cite each one —
    a transparent companion to the (smoother) PageRank score.
    """
    sub = corpus_subgraph(g)
    if sub.number_of_nodes() == 0:
        return []
    pr = _pagerank(sub)
    ranked = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    out: list[PaperRank] = []
    for node_id, score in ranked:
        d = g.nodes[node_id]
        out.append(PaperRank(
            paper_id=node_id,
            title=d.get("title"),
            year=d.get("year"),
            venue=d.get("venue"),
            cited_by_count=d.get("cited_by_count", 0),
            local_in_degree=sub.in_degree(node_id),
            pagerank=score,
        ))
    return out


def _node_brief(g: nx.DiGraph, node_id: str) -> dict:
    d = g.nodes.get(node_id, {})
    return {
        "paper_id": node_id,
        "title": d.get("title"),
        "year": d.get("year"),
        "venue": d.get("venue"),
        "cited_by_count": d.get("cited_by_count", 0),
        "in_corpus": d.get("in_corpus", False),
    }


def neighbors(g: nx.DiGraph, paper_id: str) -> Neighbors:
    """Immediate citation neighborhood of `paper_id`.

    `references` = papers it cites (successors; some may be external).
    `cited_by`   = corpus papers that cite it (predecessors; always in-corpus,
    since only corpus papers carry out-edges).
    """
    if paper_id not in g:
        return Neighbors(references=[], cited_by=[])
    refs = [_node_brief(g, n) for n in g.successors(paper_id)]
    citers = [_node_brief(g, n) for n in g.predecessors(paper_id)]
    # Most globally-cited first for a sensible default order.
    refs.sort(key=lambda d: d["cited_by_count"], reverse=True)
    citers.sort(key=lambda d: d["cited_by_count"], reverse=True)
    return Neighbors(references=refs, cited_by=citers)


def top_influential_subgraph(g: nx.DiGraph, n: int = 25) -> nx.DiGraph:
    """Subgraph of the top-`n` in-corpus papers by PageRank, with the citation
    edges among them. Small enough to draw as a node-link diagram."""
    ids = {r.paper_id for r in influence(g, top_k=n)}
    return g.subgraph(ids).copy()


def ego_subgraph(
    g: nx.DiGraph, paper_id: str, radius: int = 1, in_corpus_only: bool = True
) -> nx.DiGraph:
    """Neighborhood around one paper out to `radius` hops (both citation
    directions). Drawable focus view; excludes metadata-less externals by
    default so the picture stays readable."""
    if paper_id not in g:
        return nx.DiGraph()
    nodes = {paper_id}
    frontier = {paper_id}
    for _ in range(max(radius, 0)):
        nxt: set[str] = set()
        for u in frontier:
            nxt |= set(g.successors(u)) | set(g.predecessors(u))
        nodes |= nxt
        frontier = nxt
    if in_corpus_only:
        nodes = {n for n in nodes if g.nodes[n].get("in_corpus")}
        nodes.add(paper_id)
    return g.subgraph(nodes).copy()


def stats(g: nx.DiGraph) -> dict:
    """Quick health numbers for the graph (used by the UI and diagnostics)."""
    sub = corpus_subgraph(g)
    n_corpus = sub.number_of_nodes()
    return {
        "papers": n_corpus,
        "external_nodes": g.number_of_nodes() - n_corpus,
        "edges_total": g.number_of_edges(),
        "edges_in_corpus": sub.number_of_edges(),
        "avg_in_corpus_out_degree": (
            sub.number_of_edges() / n_corpus if n_corpus else 0.0
        ),
    }
