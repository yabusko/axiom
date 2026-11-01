"""SQLite access layer for Axiom.

Thin helpers around the shared schema (db/schema.sql). Keeps all SQL in one
place so the data contract is enforced from a single module.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Sequence

from axiom import config


def connect(db_path=None) -> sqlite3.Connection:
    """Open a connection with foreign keys on and row access by column name."""
    path = str(db_path or config.DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, add_column_ddl: str) -> None:
    """Add `column` to `table` if it isn't already there. `CREATE TABLE IF NOT EXISTS` in
    schema.sql is a no-op against a table that already exists -- a database created before a
    column was added to that table's definition never gets it from the schema alone. SQLite
    has no `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so check first."""
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {add_column_ddl}")


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes from the shared schema (idempotent)."""
    sql = config.SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    # B4 (OD26): paper_concepts/concept_cooccurrence predate the run_id column (OD19/OD20
    # created them without it) -- a no-op on a fresh database (executescript's CREATE TABLE
    # already included it), a real backfill on one that predates this stage.
    _ensure_column(conn, "paper_concepts", "run_id", "run_id TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "concept_cooccurrence", "run_id", "run_id TEXT NOT NULL DEFAULT ''")
    # Stage-3 product redesign: Shortlist needs notes + attached papers-to-read
    # per direction, which review_queue (OD16) never had.
    _ensure_column(conn, "review_queue", "notes", "notes TEXT")
    _ensure_column(conn, "review_queue", "papers_to_read_json", "papers_to_read_json TEXT NOT NULL DEFAULT '[]'")
    conn.commit()


def reset_corpus(conn: sqlite3.Connection) -> None:
    """Clear all corpus data tables in FK-safe order."""
    _CORPUS_TABLES = (
        "review_queue",
        "paper_summaries",
        "reading_list",
        "concept_canonical",
        "concepts",
        "citation_edges",
        "paper_provenance",
        # paper_concepts/concept_cooccurrence (OD19/OD20) FK -> papers.openalex_id; must be
        # cleared before papers or a non-empty agent-pipeline run blocks every future reset.
        "paper_concepts",
        "concept_cooccurrence",
        "papers",
    )
    for table in _CORPUS_TABLES:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def text_hash(text: str | None) -> str:
    """Stable hash of abstract text for dedup / change detection."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def insert_paper(
    conn: sqlite3.Connection,
    *,
    openalex_id: str,
    title: str | None,
    abstract: str | None,
    publication_year: int | None,
    venue_id: str | None,
    venue: str | None,
    cited_by_count: int = 0,
    doi: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO papers
            (openalex_id, title, abstract, publication_year,
             venue_id, venue, cited_by_count, doi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (openalex_id, title, abstract, publication_year,
         venue_id, venue, cited_by_count, doi),
    )


def insert_concepts(
    conn: sqlite3.Connection,
    paper_id: str,
    concepts: Iterable[tuple[str, int]],
) -> None:
    """concepts: iterable of (concept_name, level)."""
    conn.executemany(
        "INSERT INTO concepts (paper_id, concept, level) VALUES (?, ?, ?)",
        [(paper_id, c, lvl) for c, lvl in concepts],
    )


def insert_citation_edges(
    conn: sqlite3.Connection,
    edges: Iterable[tuple[str, str, int | None]],
) -> None:
    """edges: iterable of (src_id, dst_id, year)."""
    conn.executemany(
        "INSERT INTO citation_edges (src_id, dst_id, year) VALUES (?, ?, ?)",
        list(edges),
    )


def insert_provenance(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    source: str,
    abstract: str | None,
    has_fulltext: bool = False,
    fetched_at: datetime | None = None,
) -> None:
    ts = (fetched_at or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO paper_provenance
            (paper_id, source, fetched_at, text_hash, has_fulltext)
        VALUES (?, ?, ?, ?, ?)
        """,
        (paper_id, source, ts, text_hash(abstract), has_fulltext),
    )


def distinct_venues(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT venue FROM papers WHERE venue IS NOT NULL ORDER BY venue"
    ).fetchall()
    return [r["venue"] for r in rows]


def upsert_canonical(conn: sqlite3.Connection, concept: str, canonical: str,
                      source: str = "auto") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO concept_canonical (concept, canonical, source) "
        "VALUES (?, ?, ?)",
        (concept, canonical, source),
    )


def manual_canonical_concepts(conn: sqlite3.Connection) -> set[str]:
    return {r["concept"] for r in
            conn.execute("SELECT concept FROM concept_canonical WHERE source = 'manual'").fetchall()}


def canonical_map(conn: sqlite3.Connection) -> dict[str, str]:
    """concept -> canonical. Concepts with no row map to themselves (identity)."""
    return {r["concept"]: r["canonical"] for r in
            conn.execute("SELECT concept, canonical FROM concept_canonical").fetchall()}


def save_summary(conn: sqlite3.Connection, paper_id: str, bullets: list[str],
                  model: str, created_at: datetime | None = None) -> None:
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO paper_summaries (paper_id, bullets_json, model, created_at) "
        "VALUES (?, ?, ?, ?)",
        (paper_id, json.dumps(bullets), model, ts),
    )
    conn.commit()


def get_summary(conn: sqlite3.Connection, paper_id: str) -> list[str] | None:
    row = conn.execute(
        "SELECT bullets_json FROM paper_summaries WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return json.loads(row["bullets_json"]) if row else None


def add_to_review_queue(conn: sqlite3.Connection, *, gap_a_label: str, gap_b_label: str,
                         title: str, claim: str, method_sketch: str,
                         datasets: list[str], supporting_paper_ids: list[str],
                         created_at: datetime | None = None) -> int:
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    cur = conn.execute(
        "INSERT INTO review_queue (gap_a_label, gap_b_label, title, claim, "
        "method_sketch, datasets_json, supporting_ids_json, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
        (gap_a_label, gap_b_label, title, claim, method_sketch,
         json.dumps(datasets), json.dumps(supporting_paper_ids), ts),
    )
    conn.commit()
    return cur.lastrowid


def list_review_queue(conn: sqlite3.Connection, status: str | None = None) -> list[sqlite3.Row]:
    if status:
        return conn.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
    return conn.execute("SELECT * FROM review_queue ORDER BY created_at DESC").fetchall()


def set_review_status(conn: sqlite3.Connection, item_id: int, status: str) -> None:
    assert status in ("pending", "approved", "rejected")
    conn.execute("UPDATE review_queue SET status = ? WHERE id = ?", (status, item_id))
    conn.commit()


def get_review_item(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()


def update_review_item(
    conn: sqlite3.Connection, item_id: int, *,
    status: str | None = None, notes: str | None = None,
    papers_to_read: list[str] | None = None,
) -> None:
    """Stage-3 Shortlist: a direction's status (draft/kept/discarded, reusing
    review_queue's status column with product-facing labels on the API
    surface -- see api/main.py's ShortlistStatus mapping), freeform notes,
    and the paper_ids a user attached as "to read." Only the fields actually
    passed get updated."""
    if status is not None:
        conn.execute("UPDATE review_queue SET status = ? WHERE id = ?", (status, item_id))
    if notes is not None:
        conn.execute("UPDATE review_queue SET notes = ? WHERE id = ?", (notes, item_id))
    if papers_to_read is not None:
        conn.execute(
            "UPDATE review_queue SET papers_to_read_json = ? WHERE id = ?",
            (json.dumps(papers_to_read), item_id),
        )
    conn.commit()


def add_bookmark(conn: sqlite3.Connection, paper_id: str,
                  added_at: datetime | None = None) -> None:
    ts = (added_at or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO reading_list (paper_id, added_at) VALUES (?, ?)",
        (paper_id, ts),
    )
    conn.commit()


def remove_bookmark(conn: sqlite3.Connection, paper_id: str) -> None:
    conn.execute("DELETE FROM reading_list WHERE paper_id = ?", (paper_id,))
    conn.commit()


def list_bookmarks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Bookmarked papers, most-recently-added first, joined with paper metadata."""
    return conn.execute(
        """
        SELECT r.paper_id, r.added_at, p.title, p.abstract, p.publication_year,
               p.venue, p.cited_by_count, p.doi
        FROM reading_list r
        JOIN papers p ON p.openalex_id = r.paper_id
        ORDER BY r.added_at DESC
        """
    ).fetchall()


def year_bounds(conn: sqlite3.Connection) -> tuple[int, int] | None:
    row = conn.execute(
        "SELECT MIN(publication_year) AS lo, MAX(publication_year) AS hi FROM papers"
    ).fetchone()
    if row is None or row["lo"] is None:
        return None
    return int(row["lo"]), int(row["hi"])


def iter_papers(conn: sqlite3.Connection) -> Sequence[sqlite3.Row]:
    """All papers, for embedding/upsert during bootstrap."""
    return conn.execute(
        "SELECT openalex_id, title, abstract, publication_year, "
        "venue, cited_by_count FROM papers"
    ).fetchall()


def papers_by_ids(
    conn: sqlite3.Connection, ids: Sequence[str]
) -> dict[str, sqlite3.Row]:
    """Fetch full paper rows (incl. abstract, doi) keyed by openalex_id.

    Used by the UI to enrich lean Qdrant hits with abstract/DOI at render time.
    """
    ids = list(ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT openalex_id, title, abstract, publication_year, venue, "
        f"cited_by_count, doi FROM papers WHERE openalex_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {r["openalex_id"]: r for r in rows}


def recent_papers(conn: sqlite3.Connection, limit: int) -> dict[str, sqlite3.Row]:
    """Fetch up to `limit` papers (no particular order beyond SQLite's default) keyed
    by openalex_id. Used by POST /agents/run when the caller doesn't supply explicit
    paper_ids -- was raw SQL inline in api/main.py, centralized here (D2)."""
    rows = conn.execute("SELECT openalex_id, title, abstract FROM papers LIMIT ?", (limit,)).fetchall()
    return {r["openalex_id"]: r for r in rows}


# --- Agentic pipeline (OD19-OD24, lifted from axiom/agents/* per the Phase-4
# architecture review -- those modules did raw SQL via the caller's
# connection, which this centralizes) ------------------------------------------

def create_agent_run(
    conn: sqlite3.Connection, run_id: str, *,
    papers_total: int, model_fast: str, model_strong: str, started_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_runs
            (run_id, status, papers_total, papers_done, model_fast, model_strong, started_at)
        VALUES (?, 'running', ?, 0, ?, ?, ?)
        """,
        (run_id, papers_total, model_fast, model_strong, started_at),
    )
    conn.commit()


def increment_agent_run_papers_done(conn: sqlite3.Connection, run_id: str) -> None:
    """D1: called once per paper as it finishes, so a run still in flight reports real
    progress via GET /agents/runs/{id} instead of papers_done only jumping to its final
    value when the whole run completes."""
    conn.execute(
        "UPDATE agent_runs SET papers_done = papers_done + 1 WHERE run_id = ?", (run_id,)
    )
    conn.commit()


def finish_agent_run(conn: sqlite3.Connection, run_id: str, *, papers_done: int, finished_at: str) -> None:
    conn.execute(
        "UPDATE agent_runs SET status = 'done', papers_done = ?, finished_at = ? WHERE run_id = ?",
        (papers_done, finished_at, run_id),
    )
    conn.commit()


def fail_agent_run(conn: sqlite3.Connection, run_id: str, *, error: str, finished_at: str) -> None:
    """D1: the other terminal state finish_agent_run doesn't cover -- a background run's
    try/finally calls this on an unhandled exception so agent_runs.status never gets stuck
    at 'running' forever. `error` isn't a column on agent_runs; callers that want it visible
    in GET /agents/runs/{id} store it per-paper via upsert_agent_run_item instead."""
    conn.execute(
        "UPDATE agent_runs SET status = 'failed', finished_at = ? WHERE run_id = ?",
        (finished_at, run_id),
    )
    conn.commit()


def get_agent_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()


def agent_run_verdict_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    """D1: per-verdict counts for GET /agents/runs/{id} (e.g. {'accepted': 12, 'rejected': 1})."""
    rows = conn.execute(
        "SELECT verdict, COUNT(*) AS n FROM agent_run_items WHERE run_id = ? GROUP BY verdict",
        (run_id,),
    ).fetchall()
    return {r["verdict"]: r["n"] for r in rows if r["verdict"] is not None}


def upsert_agent_run_item(
    conn: sqlite3.Connection, run_id: str, paper_id: str, *,
    route: str | None, verifier_attempts: int, verdict: str, error: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_run_items (run_id, paper_id, route, verifier_attempts, verdict, error)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, paper_id) DO UPDATE SET
            route = excluded.route,
            verifier_attempts = excluded.verifier_attempts,
            verdict = excluded.verdict,
            error = excluded.error
        """,
        (run_id, paper_id, route, verifier_attempts, verdict, error),
    )
    conn.commit()


def record_llm_spend(
    conn: sqlite3.Connection, run_id: str, *,
    ts: str, model: str, tokens_in: int, tokens_out: int, cost: float,
) -> None:
    """C3: one durable row per *settled* LLM call (after the response, real token
    counts/cost) -- the persistent half of the budget ledger. Call sites reserve the
    worst-case cost in-process before dispatch and settle here after; this table alone
    answers "how much has this app ever spent", survives a restart."""
    conn.execute(
        "INSERT INTO llm_spend (run_id, ts, model, tokens_in, tokens_out, cost) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, ts, model, tokens_in, tokens_out, cost),
    )
    conn.commit()


def total_llm_spend(conn: sqlite3.Connection) -> float:
    """Cumulative spend across every run -- the figure LLM_BUDGET_USD actually caps."""
    row = conn.execute("SELECT COALESCE(SUM(cost), 0) AS total FROM llm_spend").fetchone()
    return float(row["total"])


def run_llm_spend(conn: sqlite3.Connection, run_id: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) AS total FROM llm_spend WHERE run_id = ?", (run_id,)
    ).fetchone()
    return float(row["total"])


# --- Corpus manifest (stage 4, OD36/OD37) --------------------------------------
def get_corpus_meta(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM corpus_meta WHERE id = 1").fetchone()


def upsert_corpus_meta(
    conn: sqlite3.Connection, *,
    query: str | None = None, filters_json: str | None = None, size_target: int | None = None,
    papers_count: int | None = None, steps_json: str | None = None,
    spend_delta: float = 0.0, seconds_delta: float = 0.0,
) -> sqlite3.Row:
    """Insert the single corpus_meta row if absent, else update it. Spend/time
    are deltas (added to the running total) -- every other field is a plain
    overwrite, defaulting to the existing row's value when omitted (so a
    partial update, e.g. just bumping steps_json, doesn't need to repeat the
    full query/filters/size_target every call). `query`/`filters_json`/
    `size_target` are required only for the first-ever call (the INSERT)."""
    ts = datetime.now(timezone.utc).isoformat()
    existing = get_corpus_meta(conn)
    if existing is None:
        if query is None or filters_json is None or size_target is None:
            raise ValueError("query/filters_json/size_target are required to create corpus_meta")
        conn.execute(
            """
            INSERT INTO corpus_meta
                (id, version, query, filters_json, size_target, papers_count,
                 steps_json, spend_total, build_seconds, created_at, updated_at)
            VALUES (1, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (query, filters_json, size_target, papers_count or 0,
             steps_json or "{}", spend_delta, seconds_delta, ts, ts),
        )
    else:
        new_query = query if query is not None else existing["query"]
        new_filters = filters_json if filters_json is not None else existing["filters_json"]
        new_target = size_target if size_target is not None else existing["size_target"]
        new_papers = papers_count if papers_count is not None else existing["papers_count"]
        new_steps = steps_json if steps_json is not None else existing["steps_json"]
        conn.execute(
            """
            UPDATE corpus_meta SET
                query = ?, filters_json = ?, size_target = ?, papers_count = ?,
                steps_json = ?, spend_total = spend_total + ?,
                build_seconds = build_seconds + ?, updated_at = ?
            WHERE id = 1
            """,
            (new_query, new_filters, new_target, new_papers, new_steps,
             spend_delta, seconds_delta, ts),
        )
    conn.commit()
    return get_corpus_meta(conn)


def replace_paper_concepts(
    conn: sqlite3.Connection,
    run_id: str,
    paper_id: str,
    concept_rows: Iterable[tuple],
    cooccurrence_pairs: Iterable[tuple[str, str]],
) -> int:
    """B4: idempotent, atomic linking. One transaction deletes `paper_id`'s existing
    `paper_concepts`/`concept_cooccurrence` rows (from any earlier run_id) then inserts the
    new set tagged with this `run_id` -- safe to call repeatedly for the same paper (reruns
    replace instead of accumulating duplicates). Replaces the old insert_paper_concepts/
    insert_cooccurrence pair, which only ever INSERTed.

    `concept_rows`: (name, kind, canonical, cosine, evidence_span, confidence, source, created_at).
    `cooccurrence_pairs`: (concept_a, concept_b), concept_a < concept_b.
    Returns the number of `paper_concepts` rows written.
    """
    concept_rows = list(concept_rows)
    cooccurrence_pairs = list(cooccurrence_pairs)
    try:
        conn.execute("DELETE FROM paper_concepts WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM concept_cooccurrence WHERE paper_id = ?", (paper_id,))
        if concept_rows:
            conn.executemany(
                """
                INSERT INTO paper_concepts
                    (run_id, paper_id, name, kind, canonical, cosine, evidence_span, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(run_id, paper_id, *row) for row in concept_rows],
            )
        if cooccurrence_pairs:
            conn.executemany(
                "INSERT INTO concept_cooccurrence (run_id, paper_id, concept_a, concept_b) VALUES (?, ?, ?, ?)",
                [(run_id, paper_id, a, b) for a, b in cooccurrence_pairs],
            )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return len(concept_rows)


def concepts_for_papers(
    conn: sqlite3.Connection, paper_ids: Sequence[str], kinds: Sequence[str]
) -> list[str]:
    """Distinct canonical `paper_concepts` labels across a set of papers,
    restricted to `kinds` (e.g. `("method",)` or `("dataset",)`). B4: lifted
    out of api/main.py's `_entities_for_papers` -- gap detail's per-side
    methods/datasets lists."""
    if not paper_ids:
        return []
    placeholders = ",".join("?" for _ in paper_ids)
    kind_placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT DISTINCT canonical FROM paper_concepts "
        f"WHERE paper_id IN ({placeholders}) AND kind IN ({kind_placeholders}) "
        f"ORDER BY canonical",
        [*paper_ids, *kinds],
    ).fetchall()
    return [r["canonical"] for r in rows]


def paper_entities(conn: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    """All `paper_concepts` rows for one paper, most-confident first -- the
    Paper panel's "Extracted entities" list. B4: lifted out of
    api/main.py's `field_paper_detail`."""
    return conn.execute(
        "SELECT name, kind, canonical, evidence_span, confidence FROM paper_concepts "
        "WHERE paper_id = ? ORDER BY confidence DESC",
        (paper_id,),
    ).fetchall()



