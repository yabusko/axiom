"""Linking agent (R1): deterministic. Writes the enrichment agent's verified
output to SQLite -- `paper_concepts` (one row per entity occurrence) and
`concept_cooccurrence` (pairwise edges per paper, `concept_a < concept_b`
lexicographically, R7's input).

Raw SQL via the caller's connection, following `axiom/db.py`'s existing
try/finally idiom -- this module doesn't own `db.py`, so no new helper was
added there; see the agent-implementer patch note for a suggested
`db.insert_paper_concepts`/`db.insert_cooccurrence` pair if the main session
wants to consolidate this later.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

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

    Returns the number of `paper_concepts` rows written. Caller commits
    (matches `axiom/db.py`'s pattern of callers owning their connection's
    lifecycle); this function does call `conn.commit()` itself since linking
    is the terminal write for a paper and nothing else shares the transaction.
    """
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    matched = [e for e in enrichment.enriched if e.ontology_source != "unmatched"]

    if matched:
        conn.executemany(
            """
            INSERT INTO paper_concepts
                (paper_id, name, kind, canonical, cosine, evidence_span, confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (enrichment.paper_id, e.name, e.kind, e.canonical, e.cosine,
                 e.evidence_span, e.confidence, "agent", ts)
                for e in matched
            ],
        )

    canonicals = sorted({e.canonical for e in matched})
    pairs = [
        (a, b) for i, a in enumerate(canonicals) for b in canonicals[i + 1:]
    ]
    if pairs:
        conn.executemany(
            "INSERT INTO concept_cooccurrence (paper_id, concept_a, concept_b) VALUES (?, ?, ?)",
            [(enrichment.paper_id, a, b) for a, b in pairs],
        )

    conn.commit()
    return len(matched)
