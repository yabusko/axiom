# Fields backend — architecture note (Phase E prep)

Read-only design note, written before any Phase E code. Scope: the storage layout, registry,
build-job, and endpoint surface the task brief's §4-E "Backend additions" asks for — grounded
in what `axiom/*`/`api/main.py` actually do today, not a generic REST design.

**Headline finding, stated up front because it changes the size of this work**: almost every
low-level seam a per-field backend needs *already exists* as an optional parameter on an
existing function. Nobody wired them together into a "field" concept yet, and there is no
registry, but this is new orchestration + a new table, not a rewrite of `axiom/*`:

| Seam | Already parameterized? | Evidence |
|---|---|---|
| SQLite connection | Yes — `db.connect(db_path=None)` | `axiom/db.py:17-23` |
| Qdrant collection | Yes — `AxiomQdrant.__init__(host=None, port=None, collection=None)` | `axiom/qdrant_client.py:48-54` |
| Qdrant (re)indexing | Yes — `reindex_qdrant(conn=None, *, collection=None, ...)` | `axiom/indexer.py:19-23`, built for OD11's eval corpus |
| Agent pipeline | Yes — `run_pipeline(papers, *, db_path=None, conn_factory=None, ...)` | `axiom/agents/pipeline.py:198-201` |
| Citation graph | Partial — `load_graph(conn=None)` takes a connection, not a path | `axiom/graph.py:46` |
| Concept graph | Yes — `concept_graph.analyze(conn=None, db_path=None, ...)` | `axiom/concept_graph.py:169` |
| Snowball fetch | Yes — `snowball(client, *, seed_query, target, seed_count, keep, ...)` | `axiom/ingest.py:229-237` |

What's genuinely new: a **fields registry** (nothing like it exists — `data/axiom.db` and
`axiom_v1` are the only corpus today, named by two hardcoded constants in `axiom/config.py:32,60`),
the **`/fields` orchestration endpoints**, and the **per-request plumbing** that resolves a
`field_id` into the right `db_path`/`collection` before calling the existing functions above.
`tests/conftest.py`'s `patched_db_path` fixture (`monkeypatch.setattr(config, "DB_PATH", seeded_db)`,
line 183) is exactly this pattern today, just done once per test process instead of once per request.

---

## 1. Storage layout

```
data/
  fields.db                     <- new: the registry (§2)
  axiom.db                      <- EXISTING default field, unchanged path (migration, below)
  fields/
    <field_id>/
      axiom.db                  <- per-field SQLite, schema = db/schema.sql verbatim
  eval.db                       <- untouched, OD11's separate eval corpus
```

Qdrant collections: `axiom_v1` (existing default field, unchanged name) and
`axiom_field_<field_id>` for every field created through `POST /fields`. `<field_id>` should be
a short random token (`uuid.uuid4().hex`, the same convention `agent_runs.run_id` and
`review_queue` already use via `uuid.uuid4().hex` in `api/main.py:601`), not the raw topic
string — topics contain spaces/punctuation that are awkward both as a Qdrant collection name
and as a URL path segment, and two fields can share a topic (rebuild with a bigger target,
retry after a failed run).

**Migration path — today's corpus becomes field #1, not a special case.** `data/axiom.db` /
`axiom_v1` keep their current paths/names (no file move, no re-ingest, no Qdrant re-embed —
AUDIT.md confirms this is explicitly out of scope: PRODUCT.md "What this stage is not doing").
On first startup after this lands, a migration step inserts one row into the new registry for
this pre-existing corpus, pointing at the *existing* `DB_PATH`/`COLLECTION_NAME` constants
rather than the `data/fields/<id>/` convention — i.e. the registry's row schema must tolerate
a field whose db/collection don't live at the generated path. This keeps `axiom.config.DB_PATH`
meaningful as "the default field's db" for any code that still reads it directly (Streamlit,
which per PRODUCT.md "stays as an internal lab console, not deleted," keeps working unmodified
against the real corpus) while every *new* field goes through the generated-path convention.
Concretely: the registry row for this field gets `id="default"` (or similar fixed sentinel,
not a random hex) and `db_path`/`collection` columns pointing at the literal
`config.DB_PATH`/`config.COLLECTION_NAME` values, so it is addressable via `GET /fields/default`
like any other field but Streamlit's direct `axiom.config`/`axiom.db.connect()` imports never
notice anything changed.

## 2. The fields registry

New top-level `data/fields.db` (not a table inside each field's own db, not a lightweight
index file alongside it). One table:

```sql
CREATE TABLE fields (
    id           TEXT PRIMARY KEY,     -- uuid4 hex, or "default" for the migrated corpus
    topic        TEXT NOT NULL,        -- the seed query, e.g. "retrieval-augmented generation"
    status       TEXT NOT NULL,        -- queued | building | ready | failed
    step         TEXT,                 -- current build step name (see §3), NULL once ready/failed
    papers_total INTEGER,              -- target corpus size (config.CORPUS_TARGET default)
    papers_done  INTEGER DEFAULT 0,    -- fetched-so-far count during the build step
    db_path      TEXT NOT NULL,        -- absolute path to this field's SQLite file
    collection   TEXT NOT NULL,        -- this field's Qdrant collection name
    cost_so_far  REAL DEFAULT 0.0,     -- running LLM spend for the extraction step
    error        TEXT,                 -- set on status='failed'
    created_at   DATETIME NOT NULL,
    updated_at   DATETIME NOT NULL,
    last_opened_at DATETIME            -- bumped by GET /fields/{id}; backs "recent fields"
);
```

**Why its own top-level db, not a table-per-field or a flat JSON index file:**

- The brief's own confirmed product constraint is "one active field at a time, switchable via
  a recent-fields list" (PRODUCT.md, "Primary flow" step 1; this is a *reopen*, not a
  simultaneous multi-field view). The Start screen's "recent fields" list and `GET /fields`
  (list) both need to be readable **without opening every field's full db** — a field mid-build
  or one whose db got corrupted must still show up in the list. A row-per-field registry gives
  that in one query against one small, always-present file; a table inside each field's own db
  would require opening N SQLite files just to render the Start screen's recent-fields list,
  and a field whose build crashed before `init_db()` ran wouldn't even have that table yet.
- It mirrors `agent_runs` (`db/schema.sql:175-184`) almost exactly in shape — status enum,
  papers_total/papers_done counters, started/finished timestamps — which is the right precedent
  per the task's own instruction to reuse the D1 stage-2 pattern (§3 below). The difference is
  *where* it lives: `agent_runs` lives inside the one db it's tracking (fine, since a run never
  outlives its db); a field's registry row must outlive — and be queryable independent of —
  the per-field db it describes, since "list my fields" has to work even before a given field's
  db file exists yet (status=`queued`).
- A flat JSON/index file was considered and rejected: it has no concurrent-write safety (two
  background build tasks updating `papers_done` at once), and FastAPI's existing convention for
  exactly this shape of data (status rows updated from a background task) is already a SQLite
  table (`agent_runs`) with `db.connect()` + explicit commits — reusing that gives the build job
  free crash-safety and matches house style instead of introducing a second persistence idiom.

`fields.db` is small (≤ a few hundred rows even over a long project lifetime) and local —
no indexing needs beyond the primary key; `last_opened_at` only needs a plain `ORDER BY` for
the recent-fields list, no index required at this scale.

## 3. Build job

**Step list, grounded in what the pipeline actually does** — PRODUCT.md's flow copy ("fetch →
embed → map topics → extract methods/datasets → find gaps") is aspirational marketing copy for
five steps; the real code only directly implements three of them as callable units, and gap
detection is a read-time computation, not a build step that writes anything:

| Step label (UI-facing) | Real function | File:line |
|---|---|---|
| "Finding papers" | `ingest.snowball()` then `ingest.ingest_to_sqlite()` | `axiom/ingest.py:229`, `axiom/ingest.py:326` (via `scripts/ingest_openalex.py:50-61`) |
| "Embedding papers" | `indexer.reindex_qdrant(conn, collection=...)` | `axiom/indexer.py:19` |
| "Extracting methods & datasets" (the paid step) | `agents.pipeline.run_pipeline(papers, db_path=..., ...)` | `axiom/agents/pipeline.py:198` |
| ~~"Mapping topics"~~ | not a separate step — `concept_graph.analyze()` runs lazily, read-time, cached via `lru_cache` the same way `get_gap_analysis()` is (`api/main.py:275-288`) | `axiom/concept_graph.py:169` |
| ~~"Finding gaps"~~ | not a separate step either — `gaps.analyze()` is also a lazy, cached read, not something the build writes to disk (`api/main.py:266-272`) | `axiom/gaps.py` |

So the build job has **3 real steps** (fetch, embed, extract), and the "map topics"/"find gaps"
language in the product flow should be understood as *what the Landscape/Research-gaps screens
compute on first load of a ready field*, not additional progress bar segments — don't invent two
steps that don't correspond to any code, or the progress UI will show "complete" for steps that
never actually ran anything.

Note also that `scripts/ingest_openalex.py` as it exists today **does not call the agentic
pipeline at all** — it stops after embed (`main()` returns after the `reindex_qdrant` call,
`scripts/ingest_openalex.py:74-80`). The extraction step is only ever triggered today via the
separate `POST /agents/run` endpoint. The `POST /fields` background job is therefore the first
caller that chains all three steps in one job — there is no existing "full build" function to
call; the build-job body has to invoke `snowball → ingest_to_sqlite → reindex_qdrant →
run_pipeline` itself in sequence, reusing each function's existing path/collection override
instead of their individual CLI wrappers.

**Time/cost grounding, from real numbers** (`docs/agentic/RESULTS.md`, D5 section and the
50-paper benchmark): the real 300-paper extraction run cost **$0.1095** total for 1,741
`paper_concepts` rows written (`docs/agentic/RESULTS.md` line 30-31). Per-paper latency from the
50-paper benchmark: p50=2.93s, p95=6.97s, $0.0004/paper (`docs/agentic/RESULTS.md` line 16-17).
At `AGENT_CONCURRENCY=4` (`axiom/config.py:267`) the extraction step for a 300-paper field is
I/O-bound on LLM round-trips, not CPU-bound — a reasonable **pre-flight estimate to show before
the paid step** is `papers_total * $0.0004` (≈ $0.12 for the default 300-paper target) and
`papers_total / AGENT_CONCURRENCY * p50_latency` (≈ 300/4 × 2.93s ≈ 220s, call it "~4 minutes")
as a ballpark, both computed from `config.MODEL_FAST_PRICE_IN/OUT` and the fixed concurrency —
not hardcoded, since a future model-price change (already happened once, OD26) must flow through
automatically. Fetch+embed timing has no committed benchmark in this repo to cite — don't invent
a number for it; the build job should just show "Finding papers..." / "Embedding papers..." as
indeterminate steps (paper count ticking up is enough signal, same as `agent_runs.papers_done`
already does) and reserve the concrete time/cost estimate for the one step that's actually paid
and actually measured.

**Job shape — reuse the D1 pattern verbatim**, it already fits this exactly:
`POST /fields` validates input, creates a `fields` row (`status='queued'`), calls
`background_tasks.add_task(_build_field_background, field_id, topic, target)`, returns 202 with
`{field_id}` immediately — same shape as `agents_run()` (`api/main.py:565-606`). The background
coroutine wraps the whole 3-step sequence in a `try/except/finally` that updates `fields.status`
to `building` → per-step `fields.step`/`papers_done` → `ready`, or to `failed` with `error` set
on any uncaught exception — directly mirroring `_run_agents_background`'s `fail_on_exception`
shape (`api/main.py:542-562`, which itself exists to stop `agent_runs.status` getting stuck at
`running` forever). `papers_done`/`cost_so_far` should update incrementally during the extraction
step the same way `db.increment_agent_run_papers_done()` already does per-paper
(`axiom/db.py:307-314`) — add an equivalent `db.increment_field_papers_done(fields_conn, field_id)`
in `fields.db`, called at the same call sites `run_pipeline._process_paper` already calls
`increment_agent_run_papers_done` (`axiom/agents/pipeline.py`, inside `_process_paper`), just
against the registry connection instead of the field's own db.

**`GET /fields/{id}` response shape**, modeled directly on the existing precedent
`AgentRunStatusOut` (`api/main.py:212-223`, itself backing `GET /agents/runs/{run_id}` at line
609-620) — same field names where they map 1:1:

```
{ field_id, topic, status,            # queued | building | ready | failed
  step,                                # current step label, null once ready/failed
  papers_total, papers_done,
  cost_so_far,                         # running LLM spend, sourced from fields.cost_so_far
  created_at, error }
```

## 4. Switching the active field

**Recommendation: a `field_id` query/path param on every data-reading endpoint, resolved
server-side into a `db_path`/`collection` pair per request — no server-side "current active
field" pointer.**

Concretely, a small per-request resolver:

```python
def resolve_field(field_id: str) -> FieldHandle:   # {db_path, collection}
    conn = fields_db.connect()
    row = fields_db.get_field(conn, field_id)       # 404 if missing
    ...
    return FieldHandle(db_path=row["db_path"], collection=row["collection"])
```

called at the top of every landscape/gap-detail/paper-detail/shortlist route, then threaded into
the exact same `db.connect(db_path)` / `AxiomQdrant(collection=...)` calls those routes already
make today (every existing route in `api/main.py` already does `conn = db.connect(); ...;
conn.close()` per request — e.g. lines 330-334, 358-364, 393-397 — so this is a one-line change
to each: `db.connect(handle.db_path)` instead of `db.connect()`). The `lru_cache`-wrapped
singletons (`get_graph`, `get_gap_analysis`, `get_concept_analysis`, lines 251-288) need to
become keyed by `field_id` (`lru_cache(maxsize=N)` on a function that takes `field_id` as an
argument, rather than `maxsize=1` on a zero-arg function) — still a cache, just no longer
assuming there's only one corpus in the process.

**Tradeoffs, against the rejected alternative (server-side "current field" pointer):**

- A path/query param is what the product decision already implies: "a 'recent fields' list on
  Start reopens a previous one" (PRODUCT.md) describes *navigating to* a field, which in a
  React/React-Router SPA naturally becomes a URL segment (`/fields/{id}/landscape`) — the
  browser's own back/forward and bookmarking then work for free, and a user who reopens the app
  on a field they were last looking at gets there from the URL, not from server memory.
  `web/` is a stateless SPA talking to FastAPI (PRODUCT.md, "Frontend" section) — there is no
  existing session/login concept to hang a server-side "current field" off of (PRODUCT.md
  explicitly excludes multi-user accounts/auth), so a server-side pointer would really be a
  single **global mutable** (one field "active" for the whole FastAPI process), which breaks the
  instant a second browser tab opens a different field — exactly the ambiguity the task brief
  flagged. "One active field at a time" is a *product* simplicity decision about what the UI
  shows a single user, not a claim that the backend process may only ever serve one field.
- The cost of the param approach is real but small: every reading endpoint's signature grows by
  one parameter, and the four `lru_cache(maxsize=1)` singletons need the keying change above.
  Given there are only 4 such singletons and roughly a dozen reading routes, this is a mechanical
  pass, not a redesign — worth it to avoid the cross-tab correctness bug the alternative has.

## 5. Concrete endpoint list

All new routes follow the existing conventions already in `api/main.py`: Pydantic response
models with `from_attributes=True` mirroring an `axiom/*` dataclass where one exists
(`ConfigDict(from_attributes=True)`, e.g. line 50-58), `HTTPException` for 404s, `conn =
db.connect(...); try/finally: conn.close()` per request, sync `def` for CPU-bound routes vs.
`async def` only where an LLM call is awaited (OD12's rule, e.g. `hypothesize` at line 462).

**Fields**
- `POST /fields` — body `{topic, target?}` (target defaults to `config.CORPUS_TARGET`), 202,
  returns `{field_id}`. Starts the background build (§3).
- `GET /fields` — list, each row shaped like the registry (§2) minus `db_path`/`collection`
  (internal-only, never serialized to the client) — ordered by `last_opened_at` descending for
  the "recent fields" list.
- `GET /fields/{field_id}` — status (§3's response shape). Bumps `last_opened_at`.

**Landscape** — `GET /fields/{field_id}/overview`: field summary (paper count, year span,
corpus topic), ranked topics (communities from `concept_graph.analyze()` — see the open question
below on which clustering backs this), rising/fading concepts with sparkline data (wraps
`velocity.get_top_velocity_keywords`, same shape as the existing `VelocityAnalysisOut`, line
126-134), key papers (wraps `graph.influence()`, same shape as `PaperRankOut`, line 103-112).
One endpoint, several existing dataclasses composed into one response — no new computation.

**Gap detail** — `GET /fields/{field_id}/gaps` (list, wraps `gaps.analyze()`, same shape as the
existing `GapAnalysisOut`, line 180-184) and `GET /fields/{field_id}/gaps/{gap_index}` (detail):
per-side top papers (already returned, `Community.members` sliced same as
`_render_gap_detail`'s "top-5 papers per side" in the Streamlit app per AUDIT.md), plus **per-side
methods/datasets** — new composition, not existing: join `paper_concepts` (kind IN
('method','dataset')) against each side's member paper_ids, grouped by canonical name with its
`evidence_span`, giving the "side A uses method M, side B works on dataset D" framing PRODUCT.md
calls out as what makes a gap actionable.

**Paper detail** — `GET /fields/{field_id}/papers/{paper_id}`: existing `PaperOut` fields
(line 61-69) plus `entities: list[{name, kind, canonical, evidence_span, confidence}]` — a direct
wrap of that paper's `paper_concepts` rows (`db/schema.sql:115-132`), which already carries
every field the Paper panel needs (AUDIT.md confirms this data exists and is API-exposed today,
just never read by any UI) — no schema change, this is a new read-only projection of an existing
table.

**Shortlist** — a renamed/repurposed `review_queue` surface (PRODUCT.md: "Approve/Reject becomes
kept/discarded on a *direction*"), scoped per field:
- `GET /fields/{field_id}/shortlist?status=draft|kept|discarded` — wraps `list_review_queue`.
- `POST /fields/{field_id}/shortlist` — create a draft direction (today's "Generate hypothesis
  pitch" + `add_to_review_queue`, combined into one call instead of the two independent call
  sites AUDIT.md flagged as duplicated — Streamlit's inline button and the existing
  `POST /graph/gaps/{gap_index}/hypothesize`).
- `PATCH /fields/{field_id}/shortlist/{id}` — status transition (`draft`→`kept`/`discarded`) and
  note-editing; wraps `set_review_status` plus a new `notes`/`papers_to_read` column (see open
  questions — `review_queue` has no notes/attached-papers column today).
- `GET /fields/{field_id}/shortlist/export.md` — Markdown export (PRODUCT.md: BibTeX explicitly
  deferred this stage) — new, no existing analog; a straightforward template over the shortlist
  rows already fetched by the list endpoint above.

## 6. What does NOT need to change

- **`db/schema.sql` itself** — zero changes. Nothing in it references an absolute path or a
  single global identity; every table is already scoped by its own primary key
  (`openalex_id`, `run_id`, etc.), not by any implicit "the one corpus" assumption. A per-field
  db is just `db.init_db()` run once against a fresh file at the field's `db_path` — the exact
  same call `axiom/db.py:36-45` already makes, unchanged.
- **`llm_spend`** — already scoped by `run_id` within a db (`axiom/db.py:369-394`); duplicating
  the whole table per field (one ledger per field's own db) is correct as-is and needs no
  design change. `docs/agentic/RESULTS.md` (line 39-40) already notes `data/eval.db` has "its
  own separate ledger from `data/axiom.db`" as a known, accepted gap from OD11 — per-field
  ledgers are the same pattern, not a new problem to solve.
- **`axiom/qdrant_client.AxiomQdrant`, `axiom/indexer.reindex_qdrant`, `axiom/agents/pipeline.run_pipeline`**
  — all three already take the override parameters a per-field caller needs (`collection=`,
  `collection=`, `db_path=`/`conn_factory=` respectively). No signature change.
- **`axiom/gaps.analyze`, `axiom/graph.load_graph`, `axiom/velocity.get_top_velocity_keywords`,
  `axiom/concept_graph.analyze`** — all already take a `conn` (or in `concept_graph`'s case,
  also `db_path`) as their data source rather than reaching for a global. Pointing them at a
  field's own connection is zero-change at the call site beyond passing the right `conn`.
- **The G-score calibration scaffold** (`eval/calibration.json`, `config.load_calibration()`,
  `axiom/config.py:188-206`) — PRODUCT.md explicitly keeps this global, not per-field ("the
  calibration *scaffold* stays as-is; no new screen for it"). Don't make `GAP_TAU_DEFAULT`/
  `G_SCORE_WEIGHTS` per-field; they stay process-wide config regardless of which field is open.

---

## Open questions for the build phase

1. **Concept/community labels**: `.claude/product/PROGRESS.md`'s Phase C note flags that
   `gaps.analyze()`'s community labels come from raw OpenAlex `concepts` (coarse/noisy —
   "Philosophy", "Hop (telecommunications)"), while the agent pipeline's `paper_concepts`
   canonicals are cleaner. This note's Landscape/gap-detail design above still reads `concepts`
   for topic labels (unchanged behavior) since that's what `gaps.analyze()`/`concept_graph`
   actually compute from today — switching the label *source* is a separate decision this note
   doesn't make, flagged for Phase E to pick up explicitly rather than silently inheriting.
2. **Per-field build concurrency**: can a second `POST /fields` be accepted while one is still
   `building`? This note's job shape doesn't block it (each field's build touches only its own
   `db_path`/`collection`), but the laptop constraint ("heavy jobs run only from the main
   session, one at a time," per the task brief's own rules) suggests the API should probably
   reject a second concurrent build with 409 rather than silently running two OpenAlex snowballs
   + two extraction pipelines in parallel. Not decided here.
3. **Deleting a field**: no delete endpoint is proposed above (the brief's confirmed scope is
   create/list/get-status only). If a build phase wants a "delete field" action, it needs to
   remove the per-field db file, drop the Qdrant collection, and delete the registry row — flag
   explicitly if wanted, since nothing above designs for it.
4. **`fields.db` corruption/missing-file recovery**: unlike per-field dbs (one bad field doesn't
   take down others), `fields.db` itself is a single point of failure for the whole "list/switch
   fields" UI. Worth deciding whether `GET /fields` should degrade gracefully (e.g. scan
   `data/fields/*/axiom.db` on disk as a fallback) or just 500 — not decided here, likely
   overkill for a single-user research tool but worth a conscious call.
5. **Shortlist's `notes`/`papers_to_read` columns**: PRODUCT.md's shortlist screen needs both;
   `review_queue` (`db/schema.sql:81-92`) has neither today. This note assumes new columns are
   added (additive migration, same `_ensure_column` pattern `axiom/db.py:26-33` already uses)
   but leaves the exact shape (free-text notes vs. structured, how "papers to read" references
   paper_ids) for Phase E to design against the actual wireframes in `web/design/wireframes/`.
