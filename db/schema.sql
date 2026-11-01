-- ============================================================================
-- AXIOM SHARED DATA CONTRACT — db/schema.sql
--
-- This schema is the contract the WHOLE TEAM codes against (data, graph, UI).
-- Any change to it MUST be recorded in docs/DECISIONS.md before being merged,
-- because downstream tracks (ingestion, velocity, citation graph, Streamlit)
-- all depend on these exact table and column names.
--
-- Source of truth: OpenAlex.
-- IMPORTANT: OpenAlex stores abstracts as an inverted index. They MUST be
-- reconstructed into plain text before insertion into papers.abstract here.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- One row per paper (abstract-level unit; also the unit for one Qdrant point).
CREATE TABLE IF NOT EXISTS papers (
    openalex_id      TEXT PRIMARY KEY,   -- e.g. "W2741809807"
    title            TEXT,
    abstract         TEXT,               -- reconstructed PLAIN TEXT (not inverted index)
    publication_year INTEGER,
    venue_id         TEXT,               -- OpenAlex source/venue id
    venue            TEXT,               -- human-readable venue name
    cited_by_count   INTEGER DEFAULT 0,
    doi              TEXT
);

-- Many concepts per paper (OpenAlex concepts/topics). Many-to-one to papers.
CREATE TABLE IF NOT EXISTS concepts (
    paper_id TEXT NOT NULL,              -- FK -> papers.openalex_id
    concept  TEXT NOT NULL,
    level    INTEGER,                    -- OpenAlex concept level (0 = broad)
    FOREIGN KEY (paper_id) REFERENCES papers(openalex_id)
);

-- Citation edges built from each paper's referenced_works[].
-- Present now, consumed in P2 by the NetworkX graph track (decision OD6).
CREATE TABLE IF NOT EXISTS citation_edges (
    src_id TEXT NOT NULL,                -- citing paper (owner of referenced_works)
    dst_id TEXT NOT NULL,                -- cited paper
    year   INTEGER,                      -- citing paper's publication_year
    FOREIGN KEY (src_id) REFERENCES papers(openalex_id)
    -- NOTE: NO foreign key on dst_id by design. referenced_works[] routinely
    -- points at papers outside the local corpus; the graph track needs these
    -- dangling edges, so we must not reject them.
);

-- Provenance / fetch bookkeeping. One row per paper.
CREATE TABLE IF NOT EXISTS paper_provenance (
    paper_id     TEXT PRIMARY KEY,       -- FK -> papers.openalex_id
    source       TEXT,                   -- e.g. "openalex", "synthetic"
    fetched_at   DATETIME,
    text_hash    TEXT,                   -- hash of abstract for dedup/change detection
    has_fulltext BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (paper_id) REFERENCES papers(openalex_id)
);

-- Canonical concept-label mapping (Task 3.2, OD14). LLM-merged synonyms
-- (e.g. "LoRA" / "Low-Rank Adaptation") persisted so canonicalization runs
-- once per corpus, not per query. `source='manual'` rows are a hand-set
-- override: re-running canonicalize.py never touches them.
CREATE TABLE IF NOT EXISTS concept_canonical (
    concept   TEXT PRIMARY KEY,          -- raw OpenAlex concept label
    canonical TEXT NOT NULL,             -- canonical label this concept maps to
    source    TEXT NOT NULL DEFAULT 'auto'   -- 'auto' (LLM) or 'manual' (override)
);

-- Cached 3-bullet paper summaries (Task 7.4, OD14). Computed once per paper on
-- request (LLM call is the slow part); re-summarize with force=true to refresh.
CREATE TABLE IF NOT EXISTS paper_summaries (
    paper_id     TEXT PRIMARY KEY,       -- FK -> papers.openalex_id
    bullets_json TEXT NOT NULL,          -- JSON list[str], exactly 3 bullets
    model        TEXT NOT NULL,          -- which Ollama model produced it
    created_at   DATETIME NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(openalex_id)
);

-- HITL review queue (Task 5.2, OD16). One row per generated hypothesis
-- pitch; no candidate is ever auto-promoted -- `status` only changes via an
-- explicit approve/reject action (never automatically).
CREATE TABLE IF NOT EXISTS review_queue (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    gap_a_label          TEXT NOT NULL,
    gap_b_label          TEXT NOT NULL,
    title                TEXT NOT NULL,
    claim                TEXT NOT NULL,
    method_sketch        TEXT NOT NULL,
    datasets_json        TEXT NOT NULL,   -- JSON list[str]
    supporting_ids_json  TEXT NOT NULL,   -- JSON list[str]
    status               TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    created_at           DATETIME NOT NULL
);

-- Bookmarked papers (Task 7.4, OD13). Single-user demo, no auth system exists,
-- so no user_id column — one shared reading list. Summaries are intentionally
-- NOT stored here: 3-bullet LLM summaries (axiom/summarize.py, OD14) are cached
-- per-paper in paper_summaries above, keyed by paper_id, so they survive
-- un-bookmarking and are shared with any future consumer.
CREATE TABLE IF NOT EXISTS reading_list (
    paper_id TEXT PRIMARY KEY,           -- FK -> papers.openalex_id
    added_at DATETIME NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(openalex_id)
);

-- Read-path indexes for the UI filter bar and the P2 graph track.
CREATE INDEX IF NOT EXISTS idx_papers_year    ON papers(publication_year);
CREATE INDEX IF NOT EXISTS idx_papers_venue   ON papers(venue);
CREATE INDEX IF NOT EXISTS idx_concepts_paper ON concepts(paper_id);
CREATE INDEX IF NOT EXISTS idx_edges_src      ON citation_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst      ON citation_edges(dst_id);

-- Agentic extraction pipeline (OD19-OD23, docs/agentic/ARCHITECTURE.md). Distinct from the
-- OpenAlex-sourced `concepts` table above: different provenance (LLM+embedding vs. taxonomy)
-- and confidence semantics, so rows are never merged across the two tables (OD19).
CREATE TABLE IF NOT EXISTS paper_concepts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- B4: which agent_run produced this row. DEFAULT '' is transitional -- Phase 1 (schema)
    -- lands ahead of Phase 2 (the linking code threads real run_id through
    -- db.insert_paper_concepts/linking.py); the default keeps existing call sites and tests
    -- passing in between, same additive-migration spirit as OD19/OD20.
    run_id        TEXT NOT NULL DEFAULT '',
    paper_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,            -- method | task | dataset | metric
    canonical     TEXT NOT NULL,            -- ontology-mapped label
    cosine        REAL NOT NULL,
    evidence_span TEXT NOT NULL,
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT 'agent',
    created_at    DATETIME NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(openalex_id)
);

CREATE TABLE IF NOT EXISTS concept_cooccurrence (
    run_id    TEXT NOT NULL DEFAULT '',      -- B4: transitional default, see paper_concepts above
    paper_id  TEXT NOT NULL,
    concept_a TEXT NOT NULL,                -- canonical; concept_a < concept_b
    concept_b TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(openalex_id)
);

-- B4: an existing database from before this index existed can already hold rows that
-- violate it -- the pre-cascade enrichment logic (B1) could legitimately produce more than
-- one entity resolving to the same (paper_id, canonical, kind) for one paper. CREATE UNIQUE
-- INDEX IF NOT EXISTS still fails on that pre-existing data even though the index itself
-- doesn't exist yet, breaking db.init_db()'s "safe to run against an existing database"
-- contract. Drop every duplicate but the lowest id (arbitrary but stable) before creating the
-- index, every time this script runs -- a no-op once no duplicates remain, so it's safe to
-- keep here permanently rather than as a one-off manual fix.
DELETE FROM paper_concepts WHERE id NOT IN (
    SELECT MIN(id) FROM paper_concepts GROUP BY paper_id, canonical, kind
);

-- B4: linking is idempotent per paper -- a rerun deletes that paper's prior rows (any run_id)
-- before inserting its own, in one transaction (axiom/agents/linking.py). The unique index is
-- the backstop: it makes a within-run duplicate insert (e.g. a retried attempt) fail loudly
-- rather than silently double-counting a concept in the co-occurrence graph.
CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_concepts_paper_canonical_kind
    ON paper_concepts(paper_id, canonical, kind);

-- C3: durable, cumulative LLM spend ledger -- replaces the in-memory-only, per-process
-- counter in axiom/llm_providers.py, which reset on restart and couldn't enforce a true
-- total-spend cap across multiple runs. One row per settled call (after the response comes
-- back, actual tokens/cost, not the pre-call reservation estimate).
CREATE TABLE IF NOT EXISTS llm_spend (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    ts         DATETIME NOT NULL,
    model      TEXT NOT NULL,
    tokens_in  INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | failed
    papers_total INTEGER NOT NULL DEFAULT 0,
    papers_done  INTEGER NOT NULL DEFAULT 0,
    model_fast   TEXT,
    model_strong TEXT,
    started_at   DATETIME,
    finished_at  DATETIME
);

CREATE TABLE IF NOT EXISTS agent_run_items (
    run_id             TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    route              TEXT,                -- full | title_only | skip
    verifier_attempts  INTEGER DEFAULT 0,
    verdict            TEXT,                -- accepted | rejected
    error              TEXT,
    PRIMARY KEY (run_id, paper_id),
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_concepts_paper     ON paper_concepts(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_concepts_canonical ON paper_concepts(canonical);
CREATE INDEX IF NOT EXISTS idx_cooc_a ON concept_cooccurrence(concept_a);
CREATE INDEX IF NOT EXISTS idx_cooc_b ON concept_cooccurrence(concept_b);
CREATE INDEX IF NOT EXISTS idx_llm_spend_run ON llm_spend(run_id);
