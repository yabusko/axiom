"""Linking agent (R1): deterministic. Writes the enrichment agent's verified
output to SQLite -- `paper_concepts` (one row per entity occurrence) and
`concept_cooccurrence` (pairwise edges per paper, `concept_a < concept_b`
lexicographically, R7's input).

Writes go through `axiom.db.replace_paper_concepts` (Phase-4 architecture review: this
previously did raw SQL via the caller's connection; lifted into `db.py` so it stays the
single place SQL lives, per the repo's established convention). B4 (OD26): that helper
deletes the paper's prior rows before inserting the new set in one transaction, so
rerunning the pipeline on a paper replaces rather than duplicates.
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

    B1/B2 (OD26): every entity in an `EnrichmentResult` is written now, with
    no filter. The old filter dropped `ontology_source == "unmatched"`
    entities -- step 3 of the enrichment cascade used to mean "no ontology
    match, so drop it"; it now means `new_concept`, "no ontology match, so
    keep it under its own normalized name" (`enrichment.py`'s `_new_concept`).
    There is no longer a match outcome that should be excluded from linking.

    Returns the number of `paper_concepts` rows written. `db.py`'s helpers
    each commit their own write (matches the rest of `db.py`'s convention).
    """
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    entities = enrichment.enriched

    concept_rows = [
        (e.name, e.kind, e.canonical, e.cosine, e.evidence_span, e.confidence, "agent", ts)
        for e in entities
    ]

    canonicals = sorted({e.canonical for e in entities})
    pairs = [
        (a, b) for i, a in enumerate(canonicals) for b in canonicals[i + 1:]
    ]
    return db.replace_paper_concepts(conn, run_id, enrichment.paper_id, concept_rows, pairs)
