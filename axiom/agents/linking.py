"""Linking agent (R1): deterministic. Writes the enrichment agent's verified
output to SQLite -- `paper_concepts` (one row per entity occurrence) and
`concept_cooccurrence` (pairwise edges per paper, `concept_a < concept_b`
lexicographically, R7's input).

Writes go through `axiom.db`'s `insert_paper_concepts`/`insert_cooccurrence`
helpers (Phase-4 architecture review: this previously did raw SQL via the
caller's connection; lifted into `db.py` so it stays the single place SQL
lives, per the repo's established convention).
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
    created_at: datetime | None = None,
) -> int:
    """Write `paper_concepts` + `concept_cooccurrence` rows for one paper.

    Only entities with a real ontology match are written -- an accepted
    `EnrichmentResult` (the only kind `axiom/agents/pipeline.py` passes here)
    cannot contain an "unmatched" entity by construction (the verifier
    rejects those), but the filter keeps this function safe to call directly
    (e.g. from a test) with unverified data too.

    Returns the number of `paper_concepts` rows written. `db.py`'s helpers
    each commit their own write (matches the rest of `db.py`'s convention).
    """
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    matched = [e for e in enrichment.enriched if e.ontology_source != "unmatched"]

    db.insert_paper_concepts(conn, [
        (enrichment.paper_id, e.name, e.kind, e.canonical, e.cosine,
         e.evidence_span, e.confidence, "agent", ts)
        for e in matched
    ])

    canonicals = sorted({e.canonical for e in matched})
    pairs = [
        (a, b) for i, a in enumerate(canonicals) for b in canonicals[i + 1:]
    ]
    db.insert_cooccurrence(conn, enrichment.paper_id, pairs)

    return len(matched)
