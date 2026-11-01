"""Instant fields (stage 4, A4/A5): a relevance-ranked slice of the shared
base corpus, computed in milliseconds -- no download, no re-embedding, no LLM
call. A "field" becomes a saved (topic, paper_ids) pair over `corpus_v1`, not
a copy of it. See docs/DECISIONS.md OD37 for the floor/N choices.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from axiom import config


@dataclass
class SliceResult:
    paper_ids: list[str]
    total_relevant: int   # how many papers in the WHOLE corpus clear the floor
    thin: bool            # total_relevant < config.FIELD_MIN_PAPERS


def _cosine_to_all(query_vector: list[float], ids: list[str], matrix: np.ndarray) -> np.ndarray:
    """Vectorized cosine of one query against every row of `matrix` -- a single
    matrix-vector product over the whole corpus (<=20k x 768) is milliseconds;
    a per-candidate Python loop was the first draft and measured ~40x slower
    at 1k papers, which matters for the p50 < 5s target (A7)."""
    qv = np.asarray(query_vector, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    q_norm = np.linalg.norm(qv)
    if q_norm == 0.0:
        return np.zeros(len(ids))
    return (matrix @ qv) / (norms * q_norm + 1e-12)


def compute_slice(
    g: nx.DiGraph,
    vectors: dict[str, list[float]],
    query_vector: list[float],
    *,
    target_n: int,
    floor: float,
) -> SliceResult:
    """A4: the top `target_n` papers at or above `floor` cosine similarity to
    the topic query, plus their 1-hop in-corpus citation neighbours that ALSO
    clear the floor (A4 step 2) -- this is what replaces snowball-by-citation
    drift with a relevance-anchored neighborhood. `g`/`vectors` are the
    caller's cached full-corpus citation graph / dense vectors (see
    api/main.py's `_corpus_graph`/`_corpus_vectors`), not re-fetched here.
    """
    ids = list(vectors)
    if not ids:
        return SliceResult(paper_ids=[], total_relevant=0, thin=True)
    matrix = np.asarray([vectors[pid] for pid in ids], dtype=float)
    cos = _cosine_to_all(query_vector, ids, matrix)

    scored = sorted(zip(ids, cos.tolist()), key=lambda t: t[1], reverse=True)
    passing = [(pid, c) for pid, c in scored if c >= floor]
    total_relevant = len(passing)

    direct = [pid for pid, _ in passing[:target_n]]
    direct_set = set(direct)
    by_cos = dict(passing)

    neighbor_ids: set[str] = set()
    for pid in direct:
        if pid in g:
            neighbor_ids |= set(g.successors(pid)) | set(g.predecessors(pid))
    neighbor_ids -= direct_set
    neighbors = [nid for nid in neighbor_ids if by_cos.get(nid, -1.0) >= floor]

    return SliceResult(
        paper_ids=direct + neighbors,
        total_relevant=total_relevant,
        thin=total_relevant < config.FIELD_MIN_PAPERS,
    )
