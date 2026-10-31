"""Linking agent (R1): deterministic. Writes the enrichment agent's verified
output to SQLite -- `paper_concepts` (one row per distinct (canonical, kind) concept
the paper touches) and `concept_cooccurrence` (pairwise edges per paper, `concept_a <
concept_b` lexicographically, R7's input).

Writes go through `axiom.db.replace_paper_concepts` (Phase-4 architecture review: this
previously did raw SQL via the caller's connection; lifted into `db.py` so it stays the
single place SQL lives, per the repo's established convention). B4 (OD26): that helper
deletes the paper's prior rows before inserting the new set in one transaction, so
rerunning the pipeline on a paper replaces rather than duplicates. Its unique index on
(paper_id, canonical, kind) means at most one row per concept per paper -- real abstracts
routinely name the same concept more than once (e.g. "RAG" and "retrieval-augmented
generation" both resolving to the same canonical label), so link() keeps only the
highest-cosine entity mention per (canonical, kind) rather than writing one row per
mention; scripts/eval_ndcg.py and the concept graph both already only care about which
concepts a paper touches, not how many times each was named.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from axiom import db
from axiom.agents.contracts import EnrichmentResult


def link(
    conn: sqlite3.Connection,
    enrichment: EnrichmentResult,
    *,
    run_id: str = "",
    created_at: datetime | None = None,
) -> int:
    """Write `paper_concepts` + `concept_cooccurrence` rows for one paper.

    B1/B2 (OD26): every entity in an `EnrichmentResult` is considered now, with no filter.
    The old filter dropped `ontology_source == "unmatched"` entities -- step 3 of the
    enrichment cascade used to mean "no ontology match, so drop it"; it now means
    `new_concept`, "no ontology match, so keep it under its own normalized name"
    (`enrichment.py`'s `_new_concept`). There is no longer a match outcome that should be
    excluded from linking -- but see the module docstring: multiple entities can legitimately
    resolve to the same (canonical, kind), and only one row per pair survives.

    Returns the number of `paper_concepts` rows written. `db.py`'s helpers
    each commit their own write (matches the rest of `db.py`'s convention).
    """
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    entities = enrichment.enriched

    best_by_concept: dict[tuple, object] = {}
    for e in entities:
        key = (e.canonical, e.kind)
        current = best_by_concept.get(key)
        if current is None or e.cosine > current.cosine:
            best_by_concept[key] = e

    concept_rows = [
        (e.name, e.kind, e.canonical, e.cosine, e.evidence_span, e.confidence, "agent", ts)
        for e in best_by_concept.values()
    ]

    canonicals = sorted({e.canonical for e in entities})
    pairs = [
        (a, b) for i, a in enumerate(canonicals) for b in canonicals[i + 1:]
    ]
    return db.replace_paper_concepts(conn, run_id, enrichment.paper_id, concept_rows, pairs)
