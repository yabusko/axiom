"""Research-gap detection: where citation structure and meaning disagree.

The citation graph tells us a field's STRUCTURE (who builds on whom); the SPECTER2
embeddings tell us TOPICAL similarity (what's about the same thing). A research
gap is where the two disagree — two clusters of papers that are **semantically
close** (related problems) yet **barely cite each other** (the literatures never
connected). Those weak bridges between related sub-communities are candidate
openings for a thesis.

Pipeline:
  1. community detection (Louvain) on the undirected corpus citation graph
  2. label each community by its most *distinctive* OpenAlex concepts (TF-IDF
     across communities, so generic terms like "Computer science" don't win)
  3. an embedding centroid per community (mean of stored dense vectors)
  4. score community PAIRS: high centroid similarity + low inter-citation = gap

Pure-Python: networkx + numpy only (no torch, no scipy).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from axiom import config, db
from axiom import graph as graphmod
from axiom import velocity as velocitymod


@dataclass
class Community:
    cid: int
    members: list[str]
    labels: list[str]            # most distinctive concepts (topic label)
    size: int

    @property
    def label(self) -> str:
        return " · ".join(self.labels) if self.labels else f"cluster {self.cid}"


@dataclass
class GapCandidate:
    a: Community
    b: Community
    semantic_similarity: float   # cosine of centroids, ~[0,1]
    inter_citations: int         # citation edges between the two clusters
    gap_score: float             # OD9 heuristic: similarity / (1 + inter_citations)
    # OD17 composite score. Defaults keep GapCandidate constructible without the
    # extra signals (e.g. axiom/hypothesis.py builds candidates directly).
    g_score: float = 0.0                     # weighted blend of `components`, ~[0,1]
    components: dict = field(default_factory=dict)  # {similarity, disconnection, velocity, authority}
    meets_threshold: bool = False            # g_score >= calibrated τ


@dataclass
class GapAnalysis:
    node2c: dict[str, int]                 # paper_id -> community id (for coloring)
    communities: dict[int, Community]
    gaps: list[GapCandidate]


# ---------------------------------------------------------------------------
def detect_communities(g: nx.DiGraph, resolution: float = 1.0,
                       seed: int = 42) -> dict[str, int]:
    """Louvain communities on the undirected corpus citation graph.

    Returns {paper_id: community_id}, ids assigned largest-community-first.
    """
    sub = graphmod.corpus_subgraph(g).to_undirected()
    if sub.number_of_nodes() == 0:
        return {}
    comms = nx.community.louvain_communities(sub, seed=seed, resolution=resolution)
    node2c: dict[str, int] = {}
    for cid, members in enumerate(sorted(comms, key=len, reverse=True)):
        for n in members:
            node2c[n] = cid
    return node2c


def community_labels(conn, node2c: dict[str, int], top: int = 4,
                     min_agent_concepts: int = config.GAP_LABEL_MIN_AGENT_CONCEPTS
                     ) -> dict[int, list[str]]:
    """Most distinctive concepts per community via TF-IDF across communities.

    A concept that appears in every cluster (e.g. "Artificial intelligence")
    carries little signal; one concentrated in one cluster labels it well.

    OD34: labels prefer the agent pipeline's `paper_concepts.canonical` (the
    agent's own synonym-merged concept, e.g. "RAG" and "retrieval-augmented
    generation" both resolving to one label) over the raw OpenAlex `concepts`
    table, since the latter is coarse/noisy as user-facing copy (e.g. "Hop
    (telecommunications)"). A community only labels from `paper_concepts` when
    it has at least `min_agent_concepts` distinct canonical concepts; below
    that (not yet agent-processed, or genuinely sparse), it falls back to
    OpenAlex `concepts` per community, same as before this change.
    """
    agent_rows = conn.execute("SELECT paper_id, canonical FROM paper_concepts").fetchall()
    agent_tf: dict[int, Counter] = defaultdict(Counter)
    for r in agent_rows:
        cid = node2c.get(r["paper_id"])
        if cid is not None:
            agent_tf[cid][r["canonical"]] += 1

    openalex_rows = conn.execute("SELECT paper_id, concept FROM concepts").fetchall()
    canon = db.canonical_map(conn)
    openalex_tf: dict[int, Counter] = defaultdict(Counter)
    for r in openalex_rows:
        cid = node2c.get(r["paper_id"])
        if cid is not None:
            concept = canon.get(r["concept"], r["concept"])
            openalex_tf[cid][concept] += 1

    tf: dict[int, Counter] = {}
    for cid in set(agent_tf) | set(openalex_tf):
        agent_counts = agent_tf.get(cid, Counter())
        tf[cid] = agent_counts if len(agent_counts) >= min_agent_concepts else openalex_tf.get(cid, Counter())
    n_comm = len(tf) or 1
    # document frequency = number of communities containing each concept
    df: Counter = Counter()
    for cid, cnt in tf.items():
        for concept in cnt:
            df[concept] += 1
    # Concepts present in most communities (e.g. "Computer science",
    # "Artificial intelligence") carry no distinguishing signal — drop them.
    generic = {c for c, d in df.items() if d >= max(2, 0.6 * n_comm)}
    labels: dict[int, list[str]] = {}
    for cid, cnt in tf.items():
        scored = {
            c: count * math.log(1 + n_comm / df[c])
            for c, count in cnt.items() if c not in generic
        }
        if not scored:                       # fallback if everything was generic
            scored = {c: count for c, count in cnt.items()}
        labels[cid] = [c for c, _ in sorted(scored.items(),
                                            key=lambda kv: kv[1], reverse=True)[:top]]
    return labels


def community_centroids(node2c: dict[str, int],
                        vectors: dict[str, list[float]]) -> dict[int, np.ndarray]:
    """Mean dense embedding per community (its position in semantic space)."""
    acc: dict[int, list[list[float]]] = defaultdict(list)
    for paper_id, cid in node2c.items():
        v = vectors.get(paper_id)
        if v is not None:
            acc[cid].append(v)
    return {cid: np.asarray(vs, dtype=float).mean(axis=0)
            for cid, vs in acc.items() if vs}


def inter_community_citations(g: nx.DiGraph,
                              node2c: dict[str, int]) -> Counter:
    """Count citation edges running between (not within) communities."""
    counts: Counter = Counter()
    for u, v in g.edges():
        cu, cv = node2c.get(u), node2c.get(v)
        if cu is not None and cv is not None and cu != cv:
            counts[frozenset((cu, cv))] += 1
    return counts


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _velocity_map(conn) -> dict[str, float]:
    """{canonical concept -> velocity} across the whole corpus (OD10), for scoring."""
    analysis = velocitymod.compute_velocity(conn, top_k=10 ** 9)
    if analysis.insufficient_year_spread:
        return {}                              # neutral velocity everywhere
    return {k.concept: k.velocity for k in analysis.keywords}


def _mean_velocity(labels: list[str], vel: dict[str, float]) -> float:
    """Mean velocity of a community's label concepts (0.0 if none are known)."""
    vals = [vel[c] for c in labels if c in vel]
    return sum(vals) / len(vals) if vals else 0.0


def _mean_log_cited(g: nx.DiGraph, members: list[str]) -> float:
    """Mean of log1p(cited_by_count) over a community's papers — its 'authority'."""
    vals = [math.log1p(g.nodes[pid].get("cited_by_count", 0))
            for pid in members if pid in g.nodes]
    return sum(vals) / len(vals) if vals else 0.0


def analyze(g: nx.DiGraph, conn, vectors: dict[str, list[float]], *,
            min_community_size: int = 4, top_k: int = 15,
            resolution: float = 1.0) -> GapAnalysis:
    """Full gap analysis: communities, labels, and ranked gap candidates.

    A gap candidate is a pair of sizable communities that are semantically close
    (similar centroids) but weakly connected by citations. Two scores are
    attached: OD9's `gap_score = similarity / (1 + inter_citations)`, and OD17's
    composite `g_score` — a calibrated weighted blend of similarity,
    disconnection, concept velocity, and authority. Candidates are ranked by
    `g_score`; `meets_threshold` marks those at or above the calibrated τ.
    """
    node2c = detect_communities(g, resolution=resolution)
    if not node2c:
        return GapAnalysis({}, {}, [])

    labels = community_labels(conn, node2c)
    centroids = community_centroids(node2c, vectors)
    members: dict[int, list[str]] = defaultdict(list)
    for paper_id, cid in node2c.items():
        members[cid].append(paper_id)

    communities = {
        cid: Community(cid, members[cid], labels.get(cid, []), len(members[cid]))
        for cid in members
    }
    inter = inter_community_citations(g, node2c)
    calib = config.load_calibration()
    weights, tau = calib["weights"], calib["tau"]
    vel = _velocity_map(conn)

    candidates = [c for c in communities
                  if communities[c].size >= min_community_size and c in centroids]

    # First pass: raw per-pair signals (authority is normalized across the run,
    # so it can only be finalized once every pair's raw authority is known).
    raw: list[dict] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            sim = _cosine(centroids[a], centroids[b])
            cites = int(inter.get(frozenset((a, b)), 0))
            vbar = (_mean_velocity(communities[a].labels, vel)
                    + _mean_velocity(communities[b].labels, vel)) / 2.0
            authority_raw = (_mean_log_cited(g, communities[a].members)
                             + _mean_log_cited(g, communities[b].members)) / 2.0
            raw.append({
                "a": a, "b": b, "sim": sim, "cites": cites,
                "S": max(0.0, min(1.0, sim)),
                "D": 1.0 / (1.0 + cites),
                "V": 1.0 / (1.0 + 2.0 ** (-vbar)),   # logistic in log2 space; 0.5 at vbar=0
                "authority_raw": authority_raw,
            })

    # Min-max normalize authority across the run into [0,1]; degenerate → 0.5.
    auth_vals = [r["authority_raw"] for r in raw]
    lo, hi = (min(auth_vals), max(auth_vals)) if auth_vals else (0.0, 0.0)
    span = hi - lo

    gaps: list[GapCandidate] = []
    for r in raw:
        A = (r["authority_raw"] - lo) / span if span > 0 else 0.5
        components = {"similarity": r["S"], "disconnection": r["D"],
                      "velocity": r["V"], "authority": A}
        g_score = (weights["similarity"] * r["S"] + weights["disconnection"] * r["D"]
                   + weights["velocity"] * r["V"] + weights["authority"] * A)
        gaps.append(GapCandidate(
            a=communities[r["a"]], b=communities[r["b"]],
            semantic_similarity=r["sim"], inter_citations=r["cites"],
            gap_score=r["sim"] / (1.0 + r["cites"]),
            g_score=g_score, components=components,
            meets_threshold=g_score >= tau,
        ))
    gaps.sort(key=lambda x: x.g_score, reverse=True)
    return GapAnalysis(node2c=node2c, communities=communities, gaps=gaps[:top_k])
