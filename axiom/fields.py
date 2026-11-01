"""The "fields" registry + build job (stage-3 product redesign).

A field is one topic's corpus: its own SQLite file (`db/schema.sql`'s shape,
same as the pre-existing `config.DB_PATH`) plus its own Qdrant collection, so
more than one can exist side by side. This module owns a SEPARATE small
database, `config.FIELDS_DB_PATH`, that only tracks field metadata/status --
never corpus data itself (that stays in each field's own db, opened via
`axiom.db.connect(row["db_path"])`).

Storage layout, the "default field" migration, and the build-job step list
are all designed in `docs/product/ARCHITECTURE_NOTE.md` -- this module is
that design's implementation; see it for the "why," not restated here.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from axiom import config, corpus_slice, db, ingest
from axiom.agents import pipeline as agents_pipeline

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fields (
    id                 TEXT PRIMARY KEY,
    topic              TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'queued',  -- queued | building | ready | thin | failed
    step               TEXT,                             -- fetching | embedding | extracting | NULL
    papers_total       INTEGER,
    papers_done        INTEGER DEFAULT 0,
    db_path            TEXT NOT NULL,
    collection         TEXT NOT NULL,
    extraction_run_id  TEXT,
    cost_so_far        REAL DEFAULT 0.0,
    error              TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    last_opened_at     TEXT
);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def connect_fields(path=None) -> sqlite3.Connection:
    path = str(path or config.FIELDS_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_fields_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Stage 4 (A4): a field is now usually a SLICE of the shared base corpus
    # (source='slice'), not its own download (source='download', the
    # pre-existing path, still used by "default" and anything built via
    # scripts/build_corpus.py --extend). A slice field's db_path/collection
    # point at the shared corpus; paper_ids_json is what actually scopes it.
    _ensure_column(conn, "fields", "source", "source TEXT NOT NULL DEFAULT 'download'")
    _ensure_column(conn, "fields", "query", "query TEXT")
    _ensure_column(conn, "fields", "paper_ids_json", "paper_ids_json TEXT")
    _ensure_column(conn, "fields", "relevance_floor", "relevance_floor REAL")
    _ensure_column(conn, "fields", "total_relevant", "total_relevant INTEGER")
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_default_field(conn: sqlite3.Connection) -> None:
    """ARCHITECTURE_NOTE.md §1: the pre-existing corpus becomes field "default",
    pointing at the unchanged config.DB_PATH/COLLECTION_NAME -- no file move,
    no re-ingest. A no-op once the row exists."""
    existing = get_field(conn, config.FIELD_DEFAULT_ID)
    if existing is not None:
        return
    papers_total = None
    if Path(config.DB_PATH).exists():
        dconn = db.connect(config.DB_PATH)
        try:
            row = dconn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()
            papers_total = row["n"] if row else None
        finally:
            dconn.close()
    ts = _now()
    conn.execute(
        """
        INSERT INTO fields (id, topic, status, step, papers_total, papers_done,
                             db_path, collection, created_at, updated_at, last_opened_at)
        VALUES (?, ?, 'ready', NULL, ?, ?, ?, ?, ?, ?, ?)
        """,
        (config.FIELD_DEFAULT_ID, config.SEED_QUERY, papers_total, papers_total or 0,
         str(config.DB_PATH), config.COLLECTION_NAME, ts, ts, ts),
    )
    conn.commit()


def create_field(conn: sqlite3.Connection, *, topic: str, target: int) -> sqlite3.Row:
    field_id = uuid.uuid4().hex
    db_path = config.FIELDS_DIR / field_id / "axiom.db"
    collection = f"axiom_field_{field_id}"
    ts = _now()
    conn.execute(
        """
        INSERT INTO fields (id, topic, status, step, papers_total, papers_done,
                             db_path, collection, created_at, updated_at, last_opened_at)
        VALUES (?, ?, 'queued', NULL, ?, 0, ?, ?, ?, ?, NULL)
        """,
        (field_id, topic, target, str(db_path), collection, ts, ts),
    )
    conn.commit()
    return get_field(conn, field_id)


def create_instant_field(
    conn: sqlite3.Connection, *,
    topic: str, corpus_graph, corpus_vectors: dict[str, list[float]], query_vector: list[float],
    target_n: int = config.FIELD_SLICE_TARGET, floor: float = config.FIELD_RELEVANCE_FLOOR,
) -> sqlite3.Row:
    """A4: a field as a relevance-ranked SLICE of the shared base corpus --
    computed here (milliseconds, no network/LLM call), not downloaded. Points
    db_path/collection at the shared corpus (never copied); `paper_ids_json`
    is what actually scopes this field. Status lands directly on 'ready' or
    'thin' (A5) -- no 'queued'/'building' detour, since there's no async work
    left to do by the time this returns.
    """
    result = corpus_slice.compute_slice(
        corpus_graph, corpus_vectors, query_vector, target_n=target_n, floor=floor,
    )
    field_id = uuid.uuid4().hex
    ts = _now()
    status = "thin" if result.thin else "ready"
    conn.execute(
        """
        INSERT INTO fields (id, topic, status, step, papers_total, papers_done,
                             db_path, collection, created_at, updated_at, last_opened_at,
                             source, query, paper_ids_json, relevance_floor, total_relevant)
        VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, 'slice', ?, ?, ?, ?)
        """,
        (field_id, topic, status, len(result.paper_ids), len(result.paper_ids),
         str(config.CORPUS_DB_PATH), config.CORPUS_COLLECTION_NAME, ts, ts, ts,
         topic, json.dumps(result.paper_ids), floor, result.total_relevant),
    )
    conn.commit()
    return get_field(conn, field_id)


def get_field(conn: sqlite3.Connection, field_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM fields WHERE id = ?", (field_id,)).fetchone()


def list_fields(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Most-recently-opened first; a field never opened sorts by created_at instead."""
    return conn.execute(
        "SELECT * FROM fields ORDER BY COALESCE(last_opened_at, created_at) DESC"
    ).fetchall()


def touch_last_opened(conn: sqlite3.Connection, field_id: str) -> None:
    conn.execute("UPDATE fields SET last_opened_at = ? WHERE id = ?", (_now(), field_id))
    conn.commit()


def delete_field(conn: sqlite3.Connection, field_id: str) -> bool:
    """B4: delete a field (a saved query). `config.FIELD_DEFAULT_ID` is never
    deletable -- it's the pre-existing corpus the Streamlit "lab console"
    (OD33) still depends on, not a field a user created.

    A legacy/download field (`source='download'`) owns a private db file +
    Qdrant collection -- both are removed too (best-effort: a missing or
    unreachable Qdrant collection doesn't block deleting the field row). A
    slice field (A4, `source='slice'`) owns nothing of its own -- db_path/
    collection point at the SHARED base corpus, which this never touches;
    only its `fields` row goes away.

    Returns True if a field was found and deleted, False if it didn't exist.
    """
    if field_id == config.FIELD_DEFAULT_ID:
        raise ValueError("the default field can't be deleted")
    row = get_field(conn, field_id)
    if row is None:
        return False
    if row["source"] == "download":
        db_path = Path(row["db_path"])
        if db_path.exists():
            db_path.unlink()
        try:
            from axiom.qdrant_client import AxiomQdrant
            AxiomQdrant(collection=row["collection"]).delete_collection()
        except Exception:
            pass
    conn.execute("DELETE FROM fields WHERE id = ?", (field_id,))
    conn.commit()
    return True


def mark_interrupted_jobs_failed(conn: sqlite3.Connection) -> int:
    """B4: at API startup, any field still 'queued'/'building' belonged to a
    background build task that died with the previous process -- it will
    never finish or update again. Marks each 'failed' with a clear error
    instead of leaving it stuck forever in the UI. Returns the count fixed."""
    rows = conn.execute("SELECT id FROM fields WHERE status IN ('queued', 'building')").fetchall()
    for r in rows:
        set_status(conn, r["id"], "failed", error="Interrupted: the server restarted mid-build.")
    return len(rows)


def set_status(conn: sqlite3.Connection, field_id: str, status: str, *, error: str | None = None) -> None:
    conn.execute(
        "UPDATE fields SET status = ?, error = ?, updated_at = ? WHERE id = ?",
        (status, error, _now(), field_id),
    )
    conn.commit()


def set_step(conn: sqlite3.Connection, field_id: str, step: str | None) -> None:
    conn.execute("UPDATE fields SET step = ?, updated_at = ? WHERE id = ?", (step, _now(), field_id))
    conn.commit()


def set_papers(conn: sqlite3.Connection, field_id: str, *, done: int, total: int | None = None) -> None:
    if total is None:
        conn.execute("UPDATE fields SET papers_done = ?, updated_at = ? WHERE id = ?", (done, _now(), field_id))
    else:
        conn.execute(
            "UPDATE fields SET papers_done = ?, papers_total = ?, updated_at = ? WHERE id = ?",
            (done, total, _now(), field_id),
        )
    conn.commit()


def set_extraction_run(conn: sqlite3.Connection, field_id: str, run_id: str) -> None:
    conn.execute(
        "UPDATE fields SET extraction_run_id = ?, updated_at = ? WHERE id = ?", (run_id, _now(), field_id)
    )
    conn.commit()


def set_cost(conn: sqlite3.Connection, field_id: str, cost: float) -> None:
    conn.execute("UPDATE fields SET cost_so_far = ?, updated_at = ? WHERE id = ?", (cost, _now(), field_id))
    conn.commit()


def live_status(conn: sqlite3.Connection, field_id: str) -> dict:
    """GET /fields/{id}'s view: the registry row, plus -- while step='extracting' --
    a live read of the field's OWN db (agent_runs/llm_spend) instead of a second,
    separately-maintained counter. Avoids a second source of truth for the one
    step that already tracks per-paper progress (`db.increment_agent_run_papers_done`,
    called from inside `agents.pipeline._process_paper`)."""
    row = get_field(conn, field_id)
    if row is None:
        return None
    out = dict(row)
    if row["step"] == "extracting" and row["extraction_run_id"]:
        fdb = db.connect(row["db_path"])
        try:
            run = db.get_agent_run(fdb, row["extraction_run_id"])
            if run is not None:
                out["papers_done"] = run["papers_done"]
                out["cost_so_far"] = db.run_llm_spend(fdb, row["extraction_run_id"])
        finally:
            fdb.close()
    return out


# --- Pre-flight estimate (Start screen, before the paid extraction step) -------
# ARCHITECTURE_NOTE.md §3: grounded in the real 50-paper benchmark
# (docs/agentic/RESULTS.md) -- p50 latency 2.93s/paper, $0.0004/paper at the
# fast tier -- not invented. Scales with config.AGENT_CONCURRENCY /
# MODEL_FAST_PRICE so a future price/concurrency change (already happened
# once, OD26) flows through automatically rather than going stale here.
_BENCHMARK_P50_LATENCY_S = 2.93
_BENCHMARK_COST_PER_PAPER_USD = 0.0004


def estimate_extraction(papers_total: int) -> dict:
    est_cost = round(papers_total * _BENCHMARK_COST_PER_PAPER_USD, 2)
    est_seconds = round((papers_total / max(config.AGENT_CONCURRENCY, 1)) * _BENCHMARK_P50_LATENCY_S)
    return {"est_cost_usd": est_cost, "est_seconds": est_seconds}


# --- Build job -----------------------------------------------------------------

async def build_field(field_id: str, topic: str, target: int) -> None:
    """The 3-step build (ARCHITECTURE_NOTE.md §3): fetch -> embed -> extract.
    Each step's blocking work (network I/O, SPECTER2 encoding) runs via
    asyncio.to_thread so this background task doesn't stall the event loop
    every other request is served from. Mirrors api.main's D1 try/except
    shape: any uncaught exception marks the field 'failed' with the error
    message, instead of leaving it stuck at 'building' forever."""
    fconn = connect_fields()
    try:
        row = get_field(fconn, field_id)
        if row is None:
            return
        db_path = Path(row["db_path"])
        collection = row["collection"]
        set_status(fconn, field_id, "building")
        set_step(fconn, field_id, "fetching")
    finally:
        fconn.close()

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = db.connect(db_path)
        try:
            db.init_db(conn)
        finally:
            conn.close()

        keep = ingest.make_venue_filter(config.RELEVANT_VENUE_PATTERNS)

        def _fetch():
            with ingest.OpenAlexClient() as client:
                return ingest.snowball(client, seed_query=topic, target=target, keep=keep)

        works = await asyncio.to_thread(_fetch)

        conn = db.connect(db_path)
        try:
            n_papers = ingest.ingest_to_sqlite(conn, works.values())
        finally:
            conn.close()

        fconn = connect_fields()
        try:
            # ingest.snowball's 2-hop expansion can overshoot `target` by a
            # real amount -- once the actual count is known, it replaces the
            # requested target as papers_total so the UI never shows
            # papers_done > papers_total (looks broken, not just imprecise).
            set_papers(fconn, field_id, done=n_papers, total=n_papers)
            set_step(fconn, field_id, "embedding")
        finally:
            fconn.close()

        def _reindex():
            # sqlite3 connections are thread-affine -- must be opened INSIDE
            # the worker thread asyncio.to_thread runs this in, not before
            # (same caveat axiom/agents/pipeline.py's _build_ontology
            # documents; passing in a connection created on the event-loop
            # thread crashes with "SQLite objects created in a thread can
            # only be used in that same thread").
            from axiom.indexer import reindex_qdrant
            conn = db.connect(db_path)
            try:
                reindex_qdrant(conn, collection=collection)
            finally:
                conn.close()

        await asyncio.to_thread(_reindex)

        fconn = connect_fields()
        try:
            set_step(fconn, field_id, "extracting")
        finally:
            fconn.close()

        conn = db.connect(db_path)
        try:
            paper_rows = db.recent_papers(conn, n_papers)
        finally:
            conn.close()
        papers = [
            agents_pipeline.PaperInput(paper_id=pid, title=r["title"], abstract=r["abstract"])
            for pid, r in paper_rows.items()
        ]

        extraction_run_id = uuid.uuid4().hex
        fconn = connect_fields()
        try:
            set_extraction_run(fconn, field_id, extraction_run_id)
        finally:
            fconn.close()

        await agents_pipeline.run_pipeline(
            papers, encoder=None, db_path=db_path, run_id=extraction_run_id,
        )

        conn = db.connect(db_path)
        try:
            final_cost = db.run_llm_spend(conn, extraction_run_id)
        finally:
            conn.close()

        fconn = connect_fields()
        try:
            set_papers(fconn, field_id, done=n_papers, total=n_papers)
            set_cost(fconn, field_id, final_cost)
            set_step(fconn, field_id, None)
            set_status(fconn, field_id, "ready")
        finally:
            fconn.close()
    except Exception as exc:
        fconn = connect_fields()
        try:
            set_status(fconn, field_id, "failed", error=str(exc))
        finally:
            fconn.close()
