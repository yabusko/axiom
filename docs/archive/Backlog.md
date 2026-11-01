# LLMs for Sciences: Research Trends & Gaps Extraction Tool - Product Backlog

Axiom · full product backlog (P1-P4)

## Project Goal

**LLMs for Sciences: Research Trends & Gaps Extraction Tool** (Axiom) is a discovery platform for researchers and thesis seekers - not a paper search engine. It extracts two complementary signals from scholarly metadata: **research trends** (which concepts are accelerating via keyword velocity) and **research gaps** (under-explored conceptual regions validated by citation evidence and, for frontiers, an LLM hypothesis pitch grounded in real abstracts). Semantic search supports gap Step 1 geometry; the product centerpiece is trends + gaps extraction with mandatory HITL caveats.

---

## Implementation Status (as of 2026-07-07)

> **This backlog is the plan of record for `main` after PR merges #1–#3
> (platform integration, qdrant restructure, qdrant_ms enhancement).** Where the
> implementation diverged from the original plan, the task text has been
> rewritten to the as-built design and the superseded plan is kept inline as a
> note. Status markers throughout: ✅ Done · ⚠️ Partial / diverged · ❌ Not built.
> See `docs/DECISIONS.md` (OD6–OD17) and the `docs/21-06.md` work log for rationale.

**Built:** OpenAlex snowball ingest → SQLite → SPECTER2 → Qdrant with **hybrid
dense+sparse (RRF) retrieval** → **NetworkX** citation graph + PageRank influence
→ **OD9 research-gap detector** (community-pair, semantic-close / weakly-citing)
→ **OD10 velocity engine** (concept trend ranking, on-demand, no persisted
KEYWORD table) → Streamlit UI (Search tab + 📈 Trending tab + Citation-graph tab
with a 3D force-directed viz and a Research-gaps view + 📚 Reading-list tab) →
**OD11 nDCG@10 eval** against a real ACL/EMNLP/COLING/NAACL corpus →
**OD12 FastAPI service layer** (`api/main.py`) wrapping search, trends,
citation graph/gaps, reading-list bookmarks, summaries, hypothesis pitches, and
review queue → **OD13/OD14 reading-list** (SQLite bookmarks + cached Ollama
summaries) → **secondary toolkit** `qdrant_ms/` (PR #3, 2026-07-07): Paper
Similarity Analyzer + Gap Discovery Feed on stored Qdrant vectors.

**Diverged from plan:** corpus is **topic-snowball** (500 RAG papers, OD7), not
ACL-by-venue ≥1k · gap model is **OD9 community-pairs**, replacing the original
`s_max` + Scenario A/B + LangGraph-hypothesis pipeline · **NetworkX-only, no
Neo4j** (OD6) · the graph UX is a **3D WebGL** viz (the original "no 3D" scope
line is reversed — see Out of Scope) · velocity is **OD10 on-demand** (two-window
normalized-frequency log2-ratio), not a persisted `KEYWORD` table, since no
`PROJECT_PLAN.md` formula exists in this repo · the eval corpus (OD11) is a
separate real ACL/EMNLP/COLING/NAACL sample, not the main topic-snowball corpus
· the FastAPI layer (OD12) grew `/graph/gaps/{i}/hypothesize` and
`/review-queue` once OD16 landed, in place of the original `/gap/evaluate` ·
reading list (OD13/OD14) has both bookmarks and cited summaries · PBI 5's
hypothesis pitch (OD16) narrates an existing OD9 gap candidate rather than
evaluating free-text input, with a rule-based Verifier, not an LLM
self-report.

**Still open:** 5k corpus scale-up (T2.3) and threshold calibration +
gap-quality rating (T8.1/T8.2, both need real human input this project can't
fabricate). The React migration (PBI 7) is **deliberately deprioritized
(OD15)**, not just unbuilt — Streamlit is now the final UI, not a prototyping
shell.

**PBI 5 is now built, rescoped (OD16):** not the original per-hypothesis
free-text pipeline (that needs uncalibrated novelty-scoring logic, the same
gap T8.1 hasn't closed) but a grounded hypothesis pitch generated over an
*existing* OD9 gap candidate, with a rule-based (not LLM self-report)
Verifier and a real HITL review queue — nothing auto-promotes. nDCG@10 eval
is built (OD11) against a real ACL/EMNLP/COLING/NAACL corpus — mean nDCG@10
0.453 (hybrid), below the ≥0.65 target, with single-pass AI-drafted judgments
pending human review. FastAPI (PBI 6, OD12), reading-list bookmarks +
summaries (PBI 7 partial, OD13/OD14), keyword canonicalization (T3.2, OD14),
the hypothesis pitch + review queue (PBI 5, OD16), and the demo package
(T8.3 — README, `demo_examples.md`, mentor one-pager) are now built.

---

## Critical Path Summary

**As-built spine (2026-07-01):** `OpenAlex (snowball) → SQLite → SPECTER2 →
Qdrant (hybrid dense+sparse / RRF) → NetworkX (PageRank) → OD9 gap detector →
Streamlit`. Neo4j is **dropped** (NetworkX-first, OD6); LangGraph, FastAPI, and
React remain **future** phases. The original full-stack target is retained below
for reference.

Build greenfield from empty repo to full stack: **OpenAlex → SQLite → SPECTER2 → Qdrant → Neo4j → LangGraph → FastAPI → React**. Saleh owns the data spine and index scale-up to 5k papers. Yahor owns velocity, the 3-step gap pipeline (WHERE → WHY → WHAT), agent graph, API contracts, and calibration. Yakub owns the four-page discovery dashboard (Trending, Keyword Graph, Gap Evaluator, Reading List) via Streamlit early integration then React from Stitch. NetworkX bootstraps citation logic before Neo4j Docker migration. Gap Step 3 (LangGraph 5-node graph with Verifier) runs only for Fertile Frontier verdicts. End-state demo: mentor filters ACL 2024-2025, spots rising RAG velocity, explores a ≤50-node keyword graph, evaluates a synthetic hypothesis through all 3 gap steps with badges and `supporting_paper_ids`, bookmarks a frontier candidate to Reading List with cited summaries. PBI 8 closes with τ calibration, nDCG@10 eval, and a reproducible README runbook.

## Phase Overview

| Phase | Goal | Key deliverables | Lead | Status |
|-------|------|------------------|------|--------|
| P1 | Data foundation | OpenAlex corpus, SQLite, SPECTER2, Qdrant (+hybrid), Streamlit search | Saleh | ✅ mostly (corpus is topic-snowball, not ACL-venue ≥1k) |
| P2 | Analytics core | NetworkX graph + PageRank, OD9 gap detector, OD10 velocity engine, Streamlit graph/gaps/trending views | Yahor + Saleh | ✅ graph, gaps, velocity, canonicalization (OD14) & hypothesis pitch (OD16, no LangGraph) all built |
| P3 | UI + agents integration | FastAPI, React 4 pages, LangGraph Step 3 full | Yakub + Yahor | ⚠️ FastAPI built (OD12); React deprioritized (OD15); Step 3 built rescoped (OD16, no LangGraph dependency) |
| P4 | Eval & scale | 5k corpus, calibration notebook, nDCG + gap quality study, runbook | All | ⚠️ nDCG@10 (OD11) + runbook/demo package (T8.3) built; calibration/gap-quality/5k not built |

## Execution Order Table

| Seq | ID | Task / milestone | Depends on | Owner | Priority | Status |
|-----|-----|------------------|------------|-------|----------|--------|
| 1 | T1.1 | Repo scaffold, docker-compose skeleton, contracts + mocks | - | Yahor | High | ⚠️ scaffold ✅; `contracts/` mock layer not built; compose = Qdrant only, Neo4j dropped (OD6) |
| 2 | T1.2 | OpenAlex ingest + SQLite schema (topic snowball) | T1.1 | Saleh | High | ⚠️ ✅ but topic-snowball 500 papers (OD7), not ACL-venue ≥1k |
| 3 | T2.1 | SPECTER2 embedding pipeline | T1.2 | Saleh | High | ✅ |
| 4 | T2.2 | Qdrant collection + `search()` API | T2.1 | Saleh | High | ✅ |
| 4b | T2.5 | **Hybrid retrieval (dense+sparse BM25, RRF)** | T2.2 | Saleh | High | ✅ delivered extra (not in original plan) |
| 5 | T1.3 | Streamlit dev shell + page routing | T1.1 | Yakub | High | ✅ (Search + Trending + Citation-graph + Reading-list tabs) |
| 6 | T3.1 | Velocity engine + keyword storage | T1.2 | Yahor | High | ✅→ **OD10** on-demand engine (no persisted KEYWORD table — no PROJECT_PLAN.md formula exists in-tree) |
| 7 | T3.2 | LLM keyword canonicalization | T3.1 | Yahor | Medium | ✅→ built as **OD14** — local Ollama (`qwen2.5:7b`), no API key/cost |
| 8 | T4.1 | NetworkX citation graph builder | T1.2 | Yahor | High | ✅ + PageRank influence (OD8) |
| 9 | T4.2 | Gap detection | T2.2, T4.1 | Yahor | High | ⚠️→ superseded by **OD9** community-pair detector (not s_max + Scenario A/B) |
| 10 | T4.3 | ~~Neo4j Docker + migration~~ | T4.1 | Saleh | High | ❌→ dropped (OD6, NetworkX-first) |
| 11 | T4.4 | ~~Gap Step 2 on Neo4j Cypher~~ | T4.3, T4.2 | Saleh | High | ❌→ dropped (OD6) |
| 12 | T5.1 | LangGraph 5-node agent graph | T4.4, T3.1 | Yahor | High | ✅→ rescoped as **OD16**: single `axiom/llm.py` call over an OD9 gap candidate, no LangGraph dependency |
| 13 | T5.2 | Gap Step 3 + composite G scoring + HITL queue backend | T5.1 | Yahor | High | ✅→ Step 3 + HITL queue (OD16); composite G-score built (**OD17**: g_score over similarity/disconnection/velocity/authority, uncalibrated defaults + calibration scaffold) |
| 14 | T6.1 | FastAPI service (search, trends, gap, reading list) | T2.2, T3.1, T5.2 | Yahor | High | ✅→ built: search/trends/graph/gaps (OD12), reading-list (OD13), summaries (OD14), hypothesis+review-queue (OD16) — 17 routes |
| 15 | T7.1 | Trending Topics page (Streamlit then React) | T3.1, T6.1 | Yakub | High | ✅→ Streamlit "📈 Trending" tab built directly on `axiom.velocity` (bypasses FastAPI/PBI6, same pattern as Search/Citation-graph); React migration deprioritized (OD15) — Streamlit is the final UI |
| 16 | T7.2 | Citation/keyword graph page | T3.1, T6.1 | Yakub | High | ⚠️→ delivered as **3D citation graph + Research-gaps view** (supersedes 2D ≤50-node) |
| 17 | T7.3 | Gap Evaluator page | T5.2, T6.1 | Yakub | High | ✅→ OD9 Research-gaps view + hypothesis pitch button + Review-queue tab (OD16); no 3-step badges (no verdict model, OD9) |
| 18 | T7.4 | Reading List page (bookmarks + cited summaries) | T5.1, T6.1 | Yakub | High | ✅→ built: bookmarks (OD13) + cited summaries (OD14, local Ollama, `summarize_paper()` built independent of PBI 5's LangGraph node) |
| 19 | T2.3 | Corpus expansion to ≥5k (add NeurIPS 2024) | T2.2 | Saleh | High | ❌ at 500 |
| 20 | T2.4 | PDF extract pipeline (Marker/PyMuPDF) for intro chunks | T2.3 | Saleh | Medium | ❌ not built |
| 21 | T8.1 | Calibration notebook (τ, δ_D, δ_F) | T4.4, T5.2 | Yahor | High | ⚠️→ scaffold built (**OD17**): `export_gap_labels.py` → human labels → `calibrate_gap_thresholds.py` → `eval/calibration.json` consumed by `gaps.analyze`; a script (not .ipynb, matching repo discipline); labels still needed |
| 22 | T8.2 | nDCG@10 + gap quality evaluation | T2.3, T5.2 | All | High | ⚠️→ nDCG@10 built (OD11, single-pass labels, 0.453 hybrid); gap-quality rating still needs human raters |
| 23 | T8.3 | README runbook + mentor demo package | T7.3, T8.1 | All | High | ✅→ `README.md` rewritten, `docs/demo_examples.md` (real output, incl. a real hypothesis pitch) + a mentor one-pager built; no literal UI screenshots (can't capture a browser session in this environment) |
| 24 | T1.4 | `docs/DECISIONS.md` | - | Yahor | Medium | ✅ (logs OD6–OD17) |
| 25 | T7.5 | **qdrant_ms proximity toolkit** (secondary Streamlit app) | T2.2 | Saleh | Medium | ✅ built (PR #3); ✅ collection unified (`axiom_v1`); ✅ integrated into main UI as **🔗 Vector tools** tab (`app/vector_tools.py`); standalone `qdrant_ms/Home.py` retained for dev |

---

## PBI 1 - Corpus & Data Foundation

**Status: ⚠️ mostly built.** Schema, ingest, and DECISIONS log are done; corpus is
topic-snowball (OD7) rather than ACL-by-venue ≥1k, and the `contracts/` mock layer
was never built (search exists as concrete methods).

### User Story:
As a researcher, I want a reliable scholarly corpus with citation metadata and provenance, so that all trend and gap extraction runs on trustworthy OpenAlex data.

### PBI Acceptance Criteria:
- SQLite schema stores papers, citation edges, keywords/concepts, and provenance per PROJECT_PLAN §6.1.
- Ingest reaches ≥1,000 ACL 2024-2025 papers with idempotent re-runs and exponential backoff on API errors.
- Citation edges from `referenced_works` are complete enough for graph construction (≥5k edges on 1k corpus).
- Ingest command and env vars documented in README.

### Task 1.1 - Repo Scaffold, Contracts & Dev Environment

- **Assignee:** Yahor
- **Sequence:** Y1
- **Priority:** High
- **Status:** ⚠️ Scaffold + `requirements.txt` + `docker-compose.yml` (Qdrant only) ✅. The typed `contracts/` package with mock stubs was **not** built — `search()`/`search_hybrid()` ship as concrete methods on `AxiomQdrant`. Neo4j is not in compose (dropped, OD6).

### Description:
- Create project layout per backlog §3.3: `ingest/`, `db/`, `embeddings/`, `index/`, `graph/`, `trends/`, `gap/`, `agents/`, `api/`, `app/`, `contracts/`, `eval/`, `docs/`.
- Add `requirements.txt`, `.env.example`, `docker-compose.yml` skeleton (Qdrant + Neo4j placeholders), README with course project goal.
- Define typed contracts with mock stubs: `search()`, `get_top_velocity_keywords()`, `evaluate_gap()`, `generate_hypothesis()`, `summarize_paper()`.

### Acceptance Criteria:
- Clean clone: `pip install -r requirements.txt` succeeds.
- `contracts/` modules export documented signatures; mocks return deterministic JSON matching Pydantic models.
- `docker compose up` starts Qdrant and Neo4j containers (Neo4j may be empty until T4.3).
- README states: **LLMs for Sciences: Research Trends & Gaps Extraction Tool**.

### Task 1.2 - OpenAlex Ingest + SQLite Schema

- **Assignee:** Saleh
- **Sequence:** S1
- **Priority:** High
- **Status:** ✅ Built as **topic snowball** (OD7), not ACL-by-venue.

### Description (as-built, OD7):
- Implemented `axiom/ingest.py` (pure-Python, no torch) + `db/schema.sql`: seed a
  topic via OpenAlex `title_and_abstract.search` (relevance), then **snowball /
  2-hop** — pull papers the seeds cite *and* papers that cite the seeds (cited
  candidates ranked by co-citation frequency) — until ~`CORPUS_TARGET`. This
  densifies in-corpus citation edges (a naive topic fetch yields an all-dangling
  graph). Cursor paging, retry/back-off, abstract reconstruction from the inverted
  index, persistence to `papers`, `citation_edges` (all `referenced_works`,
  including dangling externals), `paper_provenance`, concept fields.
- CLI: `scripts/ingest_openalex.py --topic <t> --target <n> [--seeds] [--no-index]`.

> **Original plan (superseded):** venue-based ingest targeting ≥1,000 ACL 2024–2025
> papers (`--venue ACL --years 2024,2025 --limit 1000`). Replaced by topic-snowball
> because citation analysis needs a *connected* corpus, not a venue slice.

### Acceptance Criteria:
- `python scripts/ingest_openalex.py --topic "retrieval-augmented generation" --target 500` completes without error.
- SQLite: ~500 papers, in-corpus citation edges sufficient for PageRank (first run: 500 papers / 1240 in-corpus edges), provenance with `fetched_at` and `text_hash`.
- Missing abstracts flagged; re-run is idempotent (wipe + reload).
- Ingest documented in README / `docs/21-06.md`.
- ❌ *Deferred:* reaching ≥1k and multi-venue breadth (see T2.3).

### Task 1.3 - Streamlit Dev Shell

- **Assignee:** Yakub
- **Sequence:** K1
- **Priority:** High
- **Status:** ✅ Built (structure differs from the 4-page plan).

### Description (as-built):
- `app/streamlit_app.py` with two tabs: **🔍 Search** (hybrid retrieval + venue/year
  filters) and **🕸️ Citation graph** (PageRank influence, 3D force-directed viz, and
  a Research-gaps view). Wires directly to the `axiom/*` modules (no contract mocks;
  no FastAPI yet). The Trending / Reading-List pages from the original 4-page plan
  are not built — see PBI 7.

> **Original plan (superseded):** sidebar routing across four pages (Trending →
> Keyword Graph → Gap Evaluator → Reading List) against T1.1 contract mocks.

### Acceptance Criteria:
- `streamlit run app/streamlit_app.py` starts; four pages render mock data matching contract shapes.
- Sidebar order: Trending → Keyword Graph → Gap Evaluator → Reading List.
- App title references course project name.

### Task 1.4 - Architecture Decisions Log

- **Assignee:** Yahor
- **Sequence:** Y2
- **Priority:** Medium
- **Status:** ✅ Built.

### Description (as-built):
- `docs/DECISIONS.md` records the live decisions: collection/embedding contracts,
  the `search()` signature, query-expansion path, **hybrid retrieval**, **OD6**
  (NetworkX-first, no Neo4j), **OD7** (snowball ingest), **OD8** (PageRank
  influence), **OD9** (community-pair gap detection), plus a schema change-log.

> **Original plan (superseded):** entries OD1–OD6 including an OD5 gap-badge
> sequence (Geometric Void → Citation Verified → Frontier Candidate). That badge
> model belonged to the superseded s_max/Scenario-A-B gap pipeline (see PBI 4/5).

---

## PBI 2 - Semantic Index & Search API

**Status: ✅ built + exceeded.** SPECTER2 + Qdrant `search()` done, plus a
delivered **hybrid dense+sparse (RRF)** arm not in the original plan (T2.5).
Corpus is at 500 (not 5k); PDF extraction not started.

### User Story:
As a gap detection pipeline, I need a semantic index over paper abstracts, so that Step 1 can measure geometric novelty (`s_max`) and search supports optional drill-down.

### PBI Acceptance Criteria:
- SPECTER2 embeddings for all papers with abstracts; title-only fallback flagged.
- Qdrant `axiom_v1` populated with hybrid payload filters (venue, year).
- `search(query, top_k, filters)` returns ranked results in <2s p95 on 1k corpus, <2s p95 on 5k after scale-up.
- Corpus expandable to ≥5k without schema redesign.

### Task 2.1 - SPECTER2 Batch Embedding Pipeline

- **Assignee:** Saleh
- **Sequence:** S2
- **Priority:** High
- **Status:** ✅ Built (`axiom/embed.py`, `axiom/indexer.py`; GPU/MPS→CPU fallback).

### Description:
- Implement `embeddings/specter2.py`: batch-encode abstracts from SQLite using `allenai/specter2_base`, 768-dim output keyed by `openalex_id`.

### Acceptance Criteria:
- `python -m embeddings.specter2 --db data/axiom.db` embeds ≥95% of papers (remainder title-only, flagged).
- Spot-check: related RAG pair cosine >0.7; unrelated pair <0.5.
- Embeddings persist and reload without re-encoding unchanged papers (hash check).

### Task 2.2 - Qdrant Collection + `search()` API

- **Assignee:** Saleh
- **Sequence:** S3
- **Priority:** High
- **Status:** ✅ Built (`axiom/qdrant_client.py`).

### Description:
- Implement `axiom/qdrant_client.py`: collection `axiom_v1`, cosine 768-dim, payload (`paper_id`, `year`, `venue`, `title`, `cited_by_count`, `concepts`), expose `search()`.

### Acceptance Criteria:
- One point per embedded paper; year/venue filters work.
- `search("RAG for low-resource NLP", top_k=10)` returns on-topic top results with scores.
- `AxiomQdrant.search(query_vector, top_k, venues, year_range)` signature frozen (DECISIONS); importable without UI.

---

### Task 2.5 - Hybrid Retrieval (dense + sparse, RRF) — *delivered, not in original plan*

- **Assignee:** Saleh
- **Priority:** High
- **Status:** ✅ Built.

### Description (as-built):
- Each point stores **named vectors**: dense SPECTER2 (`dense`) + a sparse
  BM25-style vector (`sparse`, pure-Python hashing-trick encoder in
  `axiom/sparse.py`). `search_hybrid()` runs both arms and fuses with **Reciprocal
  Rank Fusion** in Python (pinned `qdrant-client==1.9.0` lacks server-side fusion;
  no pin bump). The frozen dense `search()` is retained; Streamlit prefers
  `search_hybrid` when the collection has sparse vectors. Query-side embedding uses
  the expansion template (SPECTER2 adapter blocked by transformers pin).

### Acceptance Criteria:
- Exact terms/acronyms return the right paper: `LoRA` → LoRA paper at rank 1,
  `RAG` → RAG paper at rank 1 (both absent from dense top-5).
- Verified by `scripts/eval_search.py --check`.

### Task 2.3 - Corpus Expansion to ≥5k Papers

- **Assignee:** Saleh
- **Sequence:** S4
- **Priority:** High
- **Status:** ❌ Not built (corpus at ~500). Note: expansion path is now
  topic-snowball breadth (more topics / larger `--target`), not a NeurIPS venue add.

### Description:
- Extend ingest filter to add NeurIPS 2024 (one-parameter change per OD2). Re-embed and re-index Qdrant collection `axiom_v2` if schema changes.

### Acceptance Criteria:
- SQLite contains ≥5000 papers across ACL 2024-2025 + NeurIPS 2024.
- Qdrant index rebuilt; `search()` p95 <2s on 5k corpus.
- README documents corpus filter parameters.

### Task 2.4 - PDF Extract for Abstract+Intro Chunks (P3+)

- **Assignee:** Saleh
- **Sequence:** S5
- **Priority:** Medium
- **Status:** ❌ Not built.

### Description:
- Implement `ingest/pdf_extract.py` using Marker or PyMuPDF on demand via DOI; store intro text hash in provenance; never redistribute PDFs.

### Acceptance Criteria:
- `python -m ingest.pdf_extract --paper-id <id>` extracts intro markdown for a paper with resolvable DOI.
- `has_fulltext=true` set in provenance; PDF deleted after extraction.
- Failures logged; paper remains abstract-only without blocking pipeline.

---

## PBI 3 - Research Trends Extraction

**Status: ✅ both tasks built (OD10, OD14).** The velocity engine and LLM
keyword canonicalization are both done — the "trends" half of the product
goal now exists alongside OD9's gaps, with synonym merging feeding into it.

### User Story:
As a thesis-seeking researcher, I want to see which concepts are **accelerating** in my subfield, so that I understand what the field is doing without reading thousands of papers.

### PBI Acceptance Criteria (as-built, OD10/OD14):
- Velocity $v_k(t)$ computed as a two-window normalized-frequency log2-ratio
  (no `PROJECT_PLAN.md` exists in this repo, so §7.2's exact formula was
  undefined — this formula was designed to satisfy the same shape: normalized
  frequency + log-ratio).
- ✅ LLM keyword canonicalization (OD14): local Ollama model, no API key/cost.
- `get_top_velocity_keywords(n, venue, year_range)` returns ranked keywords with
  a confidence warning when `recent_count < 5`.
- Trending data now exists for a future Keyword Graph, but no such graph view is
  built yet — the citation-graph track shipped a 3D **citation** graph instead
  (T7.2, superseded by OD9's design).

### Task 3.1 - Velocity Engine

- **Assignee:** Yahor
- **Sequence:** Y3
- **Priority:** High
- **Status:** ✅ Built as **OD10** (see `docs/DECISIONS.md`).

### Description (as-built, OD10):
- Implemented `axiom/velocity.py`: no persisted `KEYWORD` table (OD10 —
  mirrors OD9's on-demand pattern; no schema migration). `compute_velocity()`
  splits the corpus's available `publication_year`s into two contiguous
  windows (PRIOR / RECENT) at the midpoint, computes each OpenAlex concept's
  (`level >= 1`) normalized frequency (share of that window's papers) in each
  window, and scores `velocity = log2((recent_share + eps) / (prior_share +
  eps))`. `get_top_velocity_keywords(conn, n, venue, year_range)` is the
  backlog-contract wrapper. Wired into Streamlit as a new "📈 Trending" tab.

> **Original plan (superseded):** persist per-keyword velocity to a `KEYWORD`
> table computed "across conference-year windows... per PROJECT_PLAN §7.2" — no
> such file/table exists in this repo; OD10 computes on demand instead.

### Acceptance Criteria:
- `get_top_velocity_keywords(conn, 50)` returns ranked keywords with velocity
  and frequency for the corpus's own recent-vs-prior year windows (this
  topic-snowball corpus isn't ACL-venue-year-cadenced, so "ACL 2024 vs 2025"
  generalizes to "newer half vs older half of available years").
- Low-volume terms (`recent_count < 5`) flagged `low_confidence` in the return
  payload, not dropped.
- Computation is near-instant on the 500-paper corpus (well under the <2s/1k,
  <5s/5k budget); no SQL joins, one full `concepts` scan.

### Task 3.2 - LLM Keyword Canonicalization

- **Assignee:** Yahor
- **Sequence:** Y4
- **Priority:** Medium
- **Status:** ✅ Built as **OD14** — local Ollama (`qwen2.5:7b`), no API
  key/cost.

### Description (as-built, OD14):
- `axiom/canonicalize.py`: batches distinct `concepts` labels (50/batch),
  prompts a local Ollama model to group true synonyms only (explicit
  instruction against merging merely-related concepts), persists to a new
  `concept_canonical` table (`db/schema.sql`) rather than a `KEYWORD` table
  (consistent with OD10's on-demand design — no separate persisted keyword
  table exists to add rows to). `axiom/velocity.py` maps every concept through
  `db.canonical_map()` before counting, so synonyms merge into one trend line.
  CLI: `scripts/canonicalize_concepts.py`.

> **Original plan (superseded):** persist canonical forms in a `KEYWORD`
> table — no such table exists (OD10); `concept_canonical` is the as-built
> equivalent.

### Acceptance Criteria:
- ✅ Known synonym pairs (e.g. LoRA / Low-Rank Adaptation) map to a single
  canonical entry — verified with an ad-hoc real Ollama call before the full
  run.
- ✅ Canonicalization is idempotent; `source='manual'` rows are a supported
  override that `run()` never touches.
- ✅ No freestanding LLM claims — the model only groups labels it's given, via
  a system prompt forbidding merges of distinct-but-related concepts.
- Note: the first full run on the 30-paper demo corpus proposed **zero**
  merges — a true negative (this hand-authored corpus has no actual duplicate
  concept labels), not a broken feature.

---

## PBI 4 - Citation Graph & Research-Gap Detection (OD9)

**Status: ⚠️ built, gap model diverged.** The NetworkX graph + PageRank is done
(T4.1). The gap detector is built but as the **OD9 community-pair** model, not the
original `s_max` + Scenario A/B pipeline. Neo4j (T4.3/T4.4) is dropped (OD6).

### User Story:
As a thesis-seeking researcher, I want to see sub-topics that are **related in
meaning but disconnected by citation**, so that I can spot bridges nobody has built
yet as candidate research directions.

### PBI Acceptance Criteria (as-built, OD9):
- NetworkX digraph built from SQLite `citation_edges`; PageRank influence on the
  corpus-only subgraph (OD8).
- Gap detection = Louvain communities → TF-IDF-distinctive concept labels →
  SPECTER2 centroid per community → score community **pairs**
  `gap_score = centroid_cosine / (1 + inter_community_citations)`.
- Results are framed as **candidate** gaps, never "proven"; honest caveat that on a
  sparse single-topic corpus the signal is weak (centroid cosines ~0.95).

> **Original plan (superseded):** a hypothesis-input gap evaluator — Step 1
> geometric void (`s_max`), Step 2 Scenario A (Dead End) / B (Fertile Frontier) via
> citation-path queries returning evidence IDs, with a Neo4j Cypher backend and a
> mandatory `disclaimer: Unverified Candidate`. Replaced by the corpus-level OD9
> detector; the hypothesis-evaluation + HITL flow was rescoped and built on top
> of OD9's candidates instead (PBI 5, OD16) rather than evaluating free-text input.

### Task 4.1 - NetworkX Citation Graph Builder

- **Assignee:** Yahor
- **Sequence:** Y5
- **Priority:** High
- **Status:** ✅ Built (`axiom/graph.py`): DiGraph from `citation_edges`, corpus
  subgraph, neighbors, PageRank influence (pure-Python power iteration, no scipy),
  small drawable subgraphs. OD8.

### Description:
- Implement `graph/citation_graph.py`: load edges from SQLite, build `networkx.DiGraph`, expose neighbor queries and border-paper helpers for gap Step 2.

### Acceptance Criteria:
- Graph builds ≥1000 nodes / ≥5000 edges on 1k corpus in <60s.
- `get_neighbors(paper_id)` correct for a known hub paper.
- Rebuildable from SQLite on demand.

### Task 4.2 - Research-Gap Detector (OD9)

- **Assignee:** Yahor
- **Sequence:** Y6
- **Priority:** High
- **Status:** ✅ Built as OD9 (supersedes the s_max/Scenario-A-B design).

### Description (as-built, OD9):
- Implemented `axiom/gaps.py`: Louvain community detection on the undirected
  corpus citation graph → sub-topics; label each community by its most
  *distinctive* OpenAlex concepts (TF-IDF across communities); SPECTER2 centroid
  per community (via new `AxiomQdrant.fetch_dense_vectors()`); rank community
  **pairs** by `gap_score = centroid_cosine / (1 + inter_community_citations)`.
  NetworkX + numpy only (no scipy/torch).

### Acceptance Criteria:
- Detector returns ranked candidate community pairs with labels, centroid cosine,
  and inter-community citation count.
- First real run (500-paper RAG corpus): 12 interpretable communities; plausible
  candidates (e.g. aviation-maintenance RAG ⟷ telecom RAG, 0 inter-citations).
- Output framed as candidate gaps with the sparse-corpus caveat, never "proven".

> **Original plan (superseded):** `gap/evaluate.py` Steps 1–2 — Step 1 `s_max` via
> Qdrant, Step 2 Scenario A/B NetworkX path queries returning `GapResult`
> (verdict enum, evidence IDs, τ=0.65 "uncalibrated"). Deferred with PBI 5.

### Task 4.3 - ~~Neo4j Docker + Data Load~~ — DROPPED (OD6)

- **Assignee:** Saleh
- **Sequence:** S6
- **Priority:** ~~High~~ n/a
- **Status:** ❌ Dropped. Per **OD6** the graph store is NetworkX-first; SQLite
  `citation_edges` remains the durable edge source of truth. Revisit a dedicated
  graph DB only if NetworkX becomes a bottleneck. Original description retained
  below for reference.

### Description (superseded):
- Add Neo4j to `docker-compose.yml`; implement `graph/neo4j_loader.py` to load Paper nodes and CITES edges from SQLite per PROJECT_PLAN §6.2.

### Task 4.4 - ~~Gap Step 2 Neo4j Cypher~~ — DROPPED (OD6)

- **Assignee:** Saleh
- **Sequence:** S7
- **Priority:** ~~High~~ n/a
- **Status:** ❌ Dropped with T4.3 (NetworkX-first, OD6). Original description
  retained below for reference.

### Description (superseded):
- Port Scenario A and Scenario B to Cypher templates from PROJECT_PLAN §7.3; wire into `evaluate_gap()` behind same API contract; deprecate NetworkX path when Neo4j available.

---

## PBI 5 - Hypothesis Pitch & HITL Review Queue

**Status: ✅ Built, rescoped (OD16).** Not the original per-hypothesis
free-text pipeline (skipped — it needs new, uncalibrated novelty-scoring
logic, the same unresolved gap as T8.1) but a grounded hypothesis pitch
generated over an *existing* OD9 gap candidate, with a rule-based Verifier
(checks real community membership, not an LLM self-report) and a real HITL
review queue where nothing auto-promotes. `summarize_paper()` (OD14) and
`axiom/llm.py` (OD14) are reused rather than duplicated.

### User Story:
As a researcher with a citation-verified frontier, I want a grounded hypothesis pitch with supporting papers, so that I can evaluate a concrete thesis direction - not a hallucinated trend.

### PBI Acceptance Criteria (as-built, OD16):
- ❌ LangGraph 5-node graph: not built — a single `axiom/llm.py` call replaces
  it (no new dependency; three of the five nodes' intent folded into one
  prompt — see `docs/DECISIONS.md` OD16 for the mapping).
- ❌ "Step 2 verdict = fertile_frontier" gate: doesn't exist — Step 3 runs on
  any OD9-ranked gap candidate the user selects (no per-hypothesis verdict
  model since OD9).
- ✅ Verifier requires ≥2 `supporting_paper_ids` (rule-based: checked against
  real community membership, not trusted from the LLM); retries ≤3×;
  temperature 0.3.
- ❌ Composite gap score $G$: not built — no `PROJECT_PLAN.md` formula exists
  to calibrate against (same reasoning as OD10); OD9's `gap_score` is the
  ranking signal upstream of this.
- ✅ HITL review queue backend: `review_queue` table, list/approve/reject,
  status only changes via explicit action.

### Task 5.1 - Hypothesis Pitch Generation (was: LangGraph 5-Node Agent Graph)

- **Assignee:** Yahor
- **Sequence:** Y7
- **Priority:** High
- **Status:** ✅ Built as **OD16**.

### Description (as-built, OD16):
- `axiom/hypothesis.py`: `generate_hypothesis(gap, g, trend_context, ...)`
  takes one OD9 `GapCandidate`, grounds a prompt in the top-cited papers from
  both sides (titles/abstracts, no separate summarization step needed) plus
  OD10's top rising concepts as trend context, and asks a local Ollama model
  (OD14) for `{title, claim, method_sketch, datasets, supporting_paper_ids}`.
  A rule-based `_verify()` checks `supporting_paper_ids` actually belong to
  the two communities with ≥2 ids and at least one per side, retrying up to
  3× at temperature 0.3 with a stricter nudge each time.

> **Original plan (superseded):** `agents/graph.py` + `agents/nodes/`, a
> stateful LangGraph with 5 nodes and Pydantic boundaries, evaluating a
> user-typed `h_syn`. No LangGraph dependency was added — see OD16 for why.

### Acceptance Criteria:
- ✅ Runs end-to-end on the demo corpus — verified live: a real run against
  the top-ranked gap (PEFT/LoRA cluster ⟷ low-resource-NLP cluster) produced
  a coherent pitch on the first attempt, citing one real paper from each side.
- ✅ Returns JSON: title, claim, method_sketch, datasets, `supporting_paper_ids`.
- ✅ Verifier rejects outputs with <2 (or single-sided) supporting IDs and retries.
- ✅ `summarize_paper()` (OD14, `axiom/summarize.py`) reused for the Reading
  List; no separate `Ingest_Summarizer` node needed since abstracts are
  passed directly into the hypothesis prompt.

### Task 5.2 - Gap Step 3, HITL Queue (G Scoring dropped)

- **Assignee:** Yahor
- **Sequence:** Y8
- **Priority:** High
- **Status:** ✅ Built (Step 3 + HITL queue), **G-scoring not built** (no
  formula to calibrate — see OD16).

### Description (as-built, OD16):
- `POST /graph/gaps/{gap_index}/hypothesize` (FastAPI) generates + verifies a
  pitch and stores it in `review_queue` as `pending`. `GET /review-queue`,
  `POST /review-queue/{id}/approve`, `POST /review-queue/{id}/reject` — no
  candidate promoted without an explicit call. Same flow wired directly into
  Streamlit's Research-gaps view ("💡 Generate hypothesis pitch" button) and a
  new "🗂️ Review queue" tab (approve/reject buttons, status filter).

### Acceptance Criteria:
- ❌ Composite $G$ score: not built (no `PROJECT_PLAN.md` formula exists).
- ✅ Review queue API: list (by status), approve, reject — no candidate
  promoted without human action. Verified live: `scripts/smoke_test_api.py`
  generates a real pitch, confirms ≥2 supporting ids, approves it, and
  confirms the status change is reflected on re-fetch.
- ✅ Pipeline latency: a single hypothesis generation completes in a few
  seconds on the local 7B model against the demo corpus (LLM latency not
  hard-failed, consistent with the backlog's own carve-out).

---

## PBI 6 - FastAPI Service Layer

**Status: ✅ built for everything that exists (OD12/OD13/OD14/OD16).**
`api/main.py` wraps search, paper lookup, OD10 trends, OD9 citation
graph/gaps, OD13 reading-list bookmarks, OD14 paper summaries, and OD16
hypothesis pitches + the HITL review queue. The only thing not built is the
original `/gap/evaluate`'s per-hypothesis free-text design — rescoped away
entirely by OD16, not missing.

### User Story:
As a frontend developer, I want a stable REST API over all backend modules, so that React pages consume one contract instead of importing Python modules directly.

### PBI Acceptance Criteria:
- ✅ FastAPI app exposes search, trends, gap-analysis, reading-list
  (bookmark + summary), hypothesis-pitch, and review-queue endpoints.
- ✅ Review queue endpoint built (OD16): `GET /review-queue`,
  `POST /review-queue/{id}/{approve,reject}`.
- ✅ OpenAPI docs generated at `/docs` (verified live, 17 routes).
- ⚠️ CORS configured for local React dev (`localhost:3000`/`5173`); no
  secrets in repo (none exist yet — no LLM key is wired anywhere); "env-driven
  config" not added — host/port stay CLI flags to `uvicorn`, matching this
  repo's existing no-env-vars convention (`axiom/config.py` is plain
  constants throughout, no `.env` anywhere).

### Task 6.1 - FastAPI Core Routes

- **Assignee:** Yahor
- **Sequence:** Y9
- **Priority:** High
- **Status:** ✅ Built as OD12, extended by OD13/OD14/OD16.

### Description (as-built, OD12/OD14/OD16):
- `api/main.py`, 17 routes: `GET /health`, `/search`, `/papers/{id}`,
  `/papers/{id}/similar`, `/trends/top`, `/graph/stats`, `/graph/influence`,
  `/graph/papers/{id}/neighbors`, `/graph/gaps`, `/reading-list`
  (`GET`/`POST`/`DELETE`), `/reading-list/{id}/summarize` (`POST`, OD14),
  `/graph/gaps/{i}/hypothesize` (`POST`, OD16 — generates + verifies a
  hypothesis pitch, stores it `pending`), `/review-queue` (`GET`, filterable
  by status), `/review-queue/{id}/{approve,reject}` (`POST`). Pydantic
  response models mirror the `axiom/*` dataclasses (`SearchHit`, `PaperRank`,
  `KeywordVelocity`/`VelocityAnalysis`, `Community`/`GapCandidate`/
  `GapAnalysis`, `Neighbors`, `HypothesisPitch`) via `from_attributes=True` —
  the only place Pydantic enters the codebase; `axiom/*` stays
  dataclass-only. Lazy singletons via `functools.lru_cache` (FastAPI's
  analogue of `st.cache_resource`/`st.cache_data`). New pins:
  `fastapi==0.111.0`, `uvicorn==0.29.0`.

> **Original plan (superseded):** REST wrappers for `evaluate_gap`,
> `generate_hypothesis`, `summarize_paper`, reading-list CRUD, review-queue
> CRUD, request/response models mirroring a `contracts/` package that was
> never built (T1.1). All of these now have as-built equivalents (OD13/OD14/
> OD16) except `evaluate_gap`'s original per-hypothesis free-text design,
> which OD16 rescoped away rather than building.

### Acceptance Criteria:
- ✅ `uvicorn api.main:app` starts; `/docs` + `/openapi.json` list all 17
  routes (verified live).
- ✅ `POST /graph/gaps/{i}/hypothesize` returns a verified pitch (OD16,
  rescoped from the original per-hypothesis `/gap/evaluate` design).
- ✅ `GET /trends/top?n=50` returns the velocity table (near-instant on the
  500-paper corpus, well under 2s).
- ✅ `POST /reading-list/{paper_id}` (bookmark add/remove/list) built (OD13).
- ✅ `POST /reading-list/{paper_id}/summarize` built (OD14) — real local-Ollama
  summary verified: 3 bullets, grounded in the paper's own abstract, cached in
  SQLite (`force=true` to bypass cache).
- ✅ Integration smoke test: `scripts/smoke_test_api.py` (FastAPI
  `TestClient`, matching the existing `scripts/eval_*.py` script convention
  rather than a new pytest suite) — all 21 checks pass against the live
  Qdrant + `data/axiom.db` stack, including a real hypothesis generation +
  approve action.

---

## PBI 7 - Discovery Dashboard

**Status: ⚠️ partial, Streamlit is the final UI (OD15).** Built: **Search**
(with bookmarking), **Trending**, **Citation graph** (3D force-directed viz +
OD9 Research-gaps view + a "💡 Generate hypothesis pitch" button, OD16),
**Reading list** (bookmarks + 3-bullet local-LLM summaries, OD13/OD14), and a
**Review queue** tab (OD16 — approve/reject, nothing auto-promotes). Not
built: the original Gap Evaluator's 3-step badges (`s_max` void map, Dead
End/Frontier verdict) — rescoped away by OD9/OD16, not missing. The React
migration is **deliberately deprioritized (OD15)** — no Stitch wireframes
exist in this repo to build against, and Streamlit already delivers all of
the above end-to-end. The graph UX is **3D** by design (the original "no 3D"
scope line is reversed — see Out of Scope).

### User Story:
As a mentor or researcher, I want a polished four-page dashboard to explore trends and gaps with honest limitation badges, so that the LLMs for Sciences vision is tangible in one click-through.

### PBI Acceptance Criteria:
- ❌ All four pages in React: **deprioritized (OD15)** — Streamlit is the
  final UI, not early-dev-only.
- ✅ **Trending Topics**: velocity list, venue/year filters (sparklines not
  rendered, though per-year data exists — see Task 7.1).
- ❌ **Keyword Graph**: superseded by the 3D citation graph (T7.2).
- ❌ **Gap Evaluator**: OD9 Research-gaps view only; no 3-step badges/HITL (PBI 5).
- ✅ **Reading List**: bookmarks + 3-bullet cited summaries (OD13/OD14).
- ✅ No 3D/WebGL... — actually reversed, see Out of Scope; no hairball either way.

### Task 7.1 - Trending Topics Page

- **Assignee:** Yakub
- **Sequence:** K2
- **Priority:** High
- **Status:** ✅ Built (Streamlit). React migration deprioritized (OD15).

### Description (as-built):
- `app/streamlit_app.py`'s third tab: venue dropdown + from/to year filter bar
  (reusing `get_filter_options()`), a ranked keyword list with velocity score
  and a low-confidence flag, a bar chart of top risers, and a per-year raw
  count series per returned keyword (sparkline data, not yet rendered as a
  sparkline chart).

> **Original plan (superseded):** Streamlit page wired to contract mocks
> (T1.1), later migrated to React via FastAPI (T6.1). The React migration
> itself is now a deliberate scope decision (OD15), not a pending dependency.

### Acceptance Criteria:
- ✅ Top-50 keywords with velocity load in the Streamlit tab (FastAPI wiring
  exists, OD12, but Streamlit calls `axiom.velocity` directly by choice).
- ✅ Venue/year filters reduce results correctly.
- ❌ *Out of scope (OD15):* React page matching a Stitch wireframe — the
  React migration is deliberately deprioritized for every PBI 7 page, not
  just this one.

### Task 7.2 - Citation Graph Page (3D)

- **Assignee:** Yakub
- **Sequence:** K3
- **Priority:** High
- **Status:** ✅ Built (as a 3D citation graph, supersedes the 2D keyword graph).

### Description (as-built):
- Interactive **3D force-directed** citation graph (`3d-force-graph`, WebGL/three.js
  via CDN, embedded with `st.components.v1.html`). Two views: **Top influential**
  (top-N PageRank subgraph) and **Around a paper** (1-hop ego network). Node size =
  in-degree within the view; gold particles flow citing→cited; click to fly the
  camera. A **Research-gaps** view colours nodes by OD9 community and renders the
  two related-but-disconnected clusters for a picked candidate. Accessibility pass:
  legend, contrast, and a "Show 3D graph" toggle (scroll-trap escape hatch).

> **Original plan (superseded):** a 2D force-directed **keyword** graph, ≤50 nodes,
> node size = citation weight, colour = velocity. Replaced by the 3D citation graph;
> a velocity-coloured keyword graph is deferred with the velocity engine (T3.1).

### Acceptance Criteria:
- Renders both views on the 500-paper corpus; no black-canvas/blank failures.
- Click a node → detail (DOI, local vs global citations, top citing papers).

### Task 7.3 - Gap Evaluator Page

- **Assignee:** Yakub
- **Sequence:** K4
- **Priority:** High
- **Status:** ✅ Built, rescoped (OD9/OD16). The OD9 **Research-gaps** view
  (candidate community-pair picker + "why this is a gap" explainer + ranked
  list) plus a **hypothesis pitch** button and a **Review queue** tab
  (approve/reject) are built inside the Citation-graph tab and a dedicated
  tab respectively. The original `h_syn`-input / `s_max` void map / Dead
  End-Frontier badges are **not built** — there's no per-hypothesis verdict
  model since OD9 (see OD16 for the full reasoning).

### Description (as-built, OD9/OD16):
- Research-gaps view: candidate-pair picker, "why this is a gap" explainer,
  ranked list (OD9, pre-existing). New: a "💡 Generate hypothesis pitch"
  button generates + verifies a pitch over the selected candidate (OD16),
  displayed with title/claim/method-sketch/datasets/supporting-paper-ids and
  a non-suppressable "⚠️ Unverified Candidate" banner; a separate "🗂️ Review
  queue" tab lists pending/approved/rejected pitches with approve/reject
  buttons.

> **Original plan (superseded):** full 3-step UI — `h_syn` input, Step 1 void
> map + `s_max`, Step 2 Dead End/Fertile Frontier badge + citation evidence,
> Step 3 hypothesis pitch + $G$ breakdown. No per-hypothesis input or verdict
> model exists (OD9); Step 3's pitch + HITL queue are built (OD16), $G$ is not
> (no formula to calibrate).

### Acceptance Criteria:
- ❌ Demo Frontier/Dead-End *badge* examples: no verdict model to badge (OD9).
- ✅ Hypothesis pitch shows `supporting_paper_ids` (as inline code spans,
  citing real papers — not yet hyperlinked to a paper detail view).
- ✅ User can approve/reject candidates in the Review queue tab — verified
  live via both the UI (`AppTest`, zero exceptions) and
  `scripts/smoke_test_api.py` (generates a pitch, approves it, confirms the
  status change).
- ✅ Copy: "⚠️ Unverified Candidate" banner is shown on every pitch, in the
  UI and in the `HypothesisPitch` dataclass itself (`disclaimer` field) —
  not suppressable.

### Task 7.4 - Reading List Page

- **Assignee:** Yakub
- **Sequence:** K5
- **Priority:** High
- **Status:** ✅ Built (**OD13** bookmarks + **OD14** summaries). Both halves
  of the original acceptance criteria are now met.

### Description (as-built, OD13/OD14):
- Bookmark button on every Search result card (`app/streamlit_app.py`);
  bookmarks persist in SQLite (`axiom/db.py`: `add_bookmark`/
  `remove_bookmark`/`list_bookmarks`), not local storage — durable across
  sessions and shared with the FastAPI layer. A "🧠 Summarize" button per
  bookmarked paper calls `axiom/summarize.py`'s `summarize_paper()` (local
  Ollama, OD14): exactly 3 bullets grounded only in that paper's own abstract,
  cached in a new `paper_summaries` table so the LLM runs once per paper. Same
  capability exposed via `POST /reading-list/{paper_id}/summarize` (FastAPI,
  `force=true` to bypass the cache).

> **Original plan (superseded on the backend, not the UX):** "display 3-bullet
> LLM summaries via FastAPI" — delivered, but the LLM backend is a local
> Ollama model (OD14), not a hosted API with a key/cost decision.

### Acceptance Criteria:
- ✅ Add/remove bookmarks persists across sessions (SQLite-backed, verified via
  `scripts/smoke_test_api.py`: add → list → remove → confirmed gone).
- ✅ Summary bullets built and verified live: a real summary of the RAG seed
  paper's abstract produced 3 bullets that paraphrase, not invent, its actual
  claims; every bullet is rendered with its `paper_id` citation in the UI.
- ✅ Page loads instantly for the current bookmark counts (well under 3s;
  not yet load-tested at 20 bookmarks specifically).

---

## PBI 7b - qdrant_ms Proximity Toolkit (secondary UI)

**Status: ✅ built (PR #3, merged 2026-07-07).** A separate Streamlit app under
`qdrant_ms/` for vector-space exploration on **pre-stored** Qdrant points (no
embedding model loaded in the app). Entry: `streamlit run qdrant_ms/Home.py`.

### Delivered
- **Paper Similarity Analyzer** (`pages/1_Proximity_Analyzer.py`): pairwise
  cosine similarity, neighbourhood map, bridge papers, shared concepts.
- **Gap Discovery Feed** (`pages/2_Gap_Discovery_Feed.py`): KMeans topic
  clustering + bridgeable topic-pair scan with landscape visualization.
- Shared helpers in `qdrant_utils.py`; legacy CLI scripts moved to
  `qdrant_ms/scripts/` (reference only).

### Integration (✅ resolved, 2026-07-07)
- `qdrant_ms/qdrant_utils.py` now reads **`axiom_v1`** (was `academic_papers`).
- Render functions extracted to `app/vector_tools.py`; integrated as the
  **🔗 Vector tools** tab in `app/streamlit_app.py`. Single Streamlit process,
  single collection, single ingest pipeline.

---

## PBI 8 - Evaluation, Calibration & Scale

**Status: ⚠️ partial.** nDCG@10 retrieval eval exists (Task 8.2, first half)
against a real ACL/EMNLP/COLING/NAACL corpus (OD11) — but with single-pass,
AI-drafted judgments, not the backlog's 2-annotator criterion, and below the
≥0.65 target (hybrid 0.453). The demo package (Task 8.3) is **done** —
`README.md`, `docs/demo_examples.md`, and a mentor one-pager, all with real
numbers. Still missing: threshold calibration (τ is n/a — the s_max/verdict
model was dropped) and the gap-quality rating study — both need real human
input this package can't substitute.

### User Story:
As a team presenting to mentors, we want measured quality metrics and a reproducible runbook, so that trends and gaps claims are backed by data - not vibes.

### PBI Acceptance Criteria:
- τ, $\delta_D$, $\delta_F$ calibrated per PROJECT_PLAN §7.3 protocol (20 established + 10 dead + 10 hot topics).
- nDCG@10 ≥0.65 on 20 hand-labeled queries.
- Gap quality mean ≥3.0 on rated candidates; Dead End / Frontier accuracy ≥70% on 20 labeled voids.
- README runbook: clone → compose up → ingest → index → demo in documented steps.
- Total embedding + LLM cost for 5k papers documented as <$5.

### Task 8.1 - Threshold Calibration Notebook

- **Assignee:** Yahor
- **Sequence:** Y10
- **Priority:** High
- **Status:** ❌ Not built. τ/δ calibration was tied to the dropped s_max/Scenario
  model; if OD9 stays, calibration would instead target the `gap_score` threshold.

### Description:
- Implement `eval/calibration.ipynb`: labeled mini-set, precision/recall curves for τ, $\delta_D$, $\delta_F$; export chosen thresholds to config file consumed by `gap/evaluate.py`.

### Acceptance Criteria:
- Notebook runs end-to-end on 5k corpus with committed labeled set in `eval/labels/`.
- Selected τ documented with precision/recall at operating point.
- `evaluate_gap()` reads calibrated thresholds from config (overrides dev default 0.65).

### Task 8.2 - Retrieval & Gap Quality Evaluation

- **Assignee:** All
- **Sequence:** A1
- **Priority:** High
- **Status:** ⚠️ Partial. nDCG@10 half built (OD11): `scripts/eval_search.py`
  still compares retrieval modes with acceptance checks (LoRA/RAG top-1) on the
  small demo corpus; **new** — `eval/ndcg_queries.json` (15 queries, 86
  judgments) + `scripts/eval_ndcg.py` run real nDCG@10 against a separate,
  real ACL/EMNLP/COLING/NAACL eval corpus (`data/eval.db`, collection
  `axiom_eval_v1`, 1,500 papers sampled from `data/FULL_DATASET.jsonl`). Gap
  quality rating study still not built (needs human raters, not something to
  fabricate).

### Description (as-built, OD11):
- `scripts/ingest_eval_corpus.py`: stratified-samples ~1,500 papers evenly
  across (venue, year) from the real 23k-paper corpus into a corpus kept
  **separate** from `axiom_v1` (no citation edges/concepts, so it can't feed
  OD9/OD10 — retrieval eval only). `eval/ndcg_queries.json`'s relevance
  judgments are **single-pass, AI-drafted** (built from real sampled titles,
  not guessed blind) — explicitly not the "2 annotators" this task specifies;
  see the file's `_provenance` field and `docs/DECISIONS.md` OD11.
  `scripts/eval_ndcg.py --report eval/report.md` computes real nDCG@10 against
  live search results.

> **Original plan (unchanged for the rest):** rate 10 gap candidates 1-5
> (team + volunteers); measure Dead End/Frontier accuracy on 20 labeled voids
> — both still require real human judgment and are **not built**.

### Acceptance Criteria:
- ⚠️ nDCG@10 reported in `eval/report.md` — **hybrid mean 0.453, dense-only
  0.345** (below the ≥0.65 target; expected for a single-pass label set — see
  OD11's note that some low scores reflect judgment-coverage gaps, not
  retrieval failures).
- ❌ Gap quality mean / classification accuracy: not built (needs real human
  raters).
- ❌ Search/gap p95 timing on a 5k corpus: not logged (eval corpus is 1,500;
  main corpus is 500).

### Task 8.3 - Runbook, Demo Package & Final Report

- **Assignee:** All
- **Sequence:** A2
- **Priority:** High
- **Status:** ✅ Built, adapted to as-built scope (no Neo4j/React/HITL exist to
  document or screenshot).

### Description (as-built):
- `README.md` rewritten as the full current runbook (was frozen at the very
  first P1 stage): prerequisites (incl. optional Ollama for OD14 features),
  full run order (synthetic demo *and* real OpenAlex path), architecture
  diagram, file layout, limitations, troubleshooting.
- `docs/demo_examples.md`: **2 real gap examples** (§3, from an actual
  `gaps.analyze()` run — not illustrative), trending highlights (§2, real
  `compute_velocity()` output with an honest small-corpus caveat), plus search,
  reading-list-summary, canonicalization, and nDCG@10 examples — every number
  came from a real run on 2026-07-04.
- `docs/WALKTHROUGH.md`: added a status banner (it was silently stale, still
  claiming velocity/gaps/FastAPI as "out of scope for P1" after all three
  shipped) pointing to `README.md`/`docs/DECISIONS.md`/`docs/demo_examples.md`
  for current state, rather than a full rewrite of a 594-line historical doc.
- **Mentor one-pager** (Artifact, not a `.pptx`): status snapshot with real
  metrics (13 routes, mean nDCG@10 0.453, 8 communities, top gap-score 0.956),
  an architecture diagram, a live gap example, and the limitations section —
  screenshots of a running UI aren't something this environment can capture,
  so this substitutes a data-accurate visual summary instead.

### Acceptance Criteria:
- ✅ Fresh machine can follow `README.md` to a running demo without
  undocumented steps (both the synthetic and real-OpenAlex paths covered).
- ✅ `docs/demo_examples.md` reproducible on the demo DB — every command shown
  was actually run to produce the numbers next to it.
- ⚠️ No UI screenshots (can't capture a running browser session in this
  environment) or HITL badges (PBI 5 not built) — substituted with a
  data-driven mentor one-pager instead.
- ✅ Limitations section covers corpus scope, calibrated-vs-not rankings,
  hallucination controls (system-prompt grounding, not a formal benchmark),
  and English-only — in both `README.md` and the mentor one-pager.

---

## Out of Scope

- ~~3D WebGL / Three.js visualization~~ — **reversed (2026-06-21).** A 3D
  force-directed graph (`3d-force-graph`) is now the hero graph UX; a flat 2D
  attempt was an unreadable hairball. The *hairball* rejection still holds — 3D
  gives a rotatable/zoomable node-link view, not a cluster blob. (work log §5, T7.2)
- 2D topic-cluster hairball as primary UX (rejected per mentor feedback) — still out
- Neo4j / dedicated graph DB (NetworkX-first per OD6; revisit only if it bottlenecks)
- Reuse or migration of prior POC codebases
- LLM model training or fine-tuning
- Production SLA, multi-language support, social features

---

## Known Risks & Mitigations

- **OpenAlex abstracts missing on ~15-20% of papers.** Title-only embed with flag; exclude from gap scoring; show "limited text" in UI.
- **Neo4j is new ops for the team.** ✅ *Resolved:* staying NetworkX-first (OD6);
  Neo4j dropped. SQLite `citation_edges` is the durable edge source; revisit a graph
  DB only if NetworkX bottlenecks.
- **LangGraph Step 3 can sprawl.** ✅ *Resolved differently (OD16):* no
  LangGraph — a single `axiom/llm.py` call with a rule-based Verifier, 3
  retries, temperature 0.3. No multi-node agent graph to sprawl in the first
  place.
- **τ wrong until PBI 8 lands.** *Moot for now:* the s_max/verdict model that τ
  calibrated was dropped (OD9). If OD9 stays, calibrate the `gap_score` threshold
  instead; current gap output is labeled a **candidate** with a sparse-corpus caveat.
- **Hallucinated hypothesis pitches.** ✅ *Built and mitigated (OD16):* the
  Verifier is rule-based, not an LLM self-report — it checks
  `supporting_paper_ids` against real community membership (not just that the
  model claims they're real), requires ≥2 with both sides represented, and
  retries ≤3× on failure. HITL review queue blocks auto-promotion (every
  pitch lands `pending`). Verified live: a real pitch cited genuine papers
  from both sides of the top-ranked demo-corpus gap on the first attempt.
- **Sparse single-topic corpus weakens OD9 gaps.** Centroid cosines cluster tightly
  (~0.95) and most community pairs have 0 inter-citations. Mitigate with denser /
  multi-topic corpora (T2.3) and label candidates honestly.
- **React/Stitch migration friction.** *Resolved (OD15):* no migration —
  Streamlit is the final UI. FastAPI (PBI 6) stays built as a REST seam for
  any future consumer, but nothing currently requires it as a frontend.
- **5k embed + index runtime.** Batch embeddings with hash skip; versioned Qdrant collections (`axiom_v1` → `axiom_v2`); document GPU path in README.
- **Saleh and Yakub blocked on interface changes.** Yahor owns `contracts/`; signature changes discussed in standup before merge, not mid-sprint via silent PR.

---

## Cross-Task Dependency Hints

- T1.1 (scaffold + contracts) blocks all other tasks - mocks unblock parallel UI and ingest work.
- T1.2 (ingest) blocks T2.1, T3.1, T4.1 - no embeddings, trends, or graph without papers.
- T2.1 → T2.2 (embed before Qdrant) → T2.3 (scale after base index works).
- T3.1 (velocity) ✅ built (OD10) — the Streamlit Trending tab (T7.1) is unblocked;
  T7.2 was already superseded by the 3D citation graph, not blocked on velocity.
  T3.2 (canonicalization) ✅ built (OD14) and now feeds T7.1's label quality
  for real (synonym concepts merge before ranking).
- T4.1 (NetworkX) → T4.2 (OD9 gap detector). ~~T4.3/T4.4 (Neo4j)~~ dropped (OD6).
- T4.2 + T3.1 ✅ unblocked T5.1 (OD9 gaps + OD10 velocity both feed the
  hypothesis prompt as gap candidate + trend context) — resolved not via the
  original LangGraph design but by rescoping Step 3 to narrate an existing
  OD9 candidate (OD16), sidestepping the per-hypothesis-input design gap
  entirely rather than resolving it.
- T5.1 ✅ built, unblocked T5.2 (HITL queue) and T7.4's summaries — the
  latter was actually built independently via `summarize_paper()` (OD14)
  before T5.1 landed. T5.2 unblocked T7.3's hypothesis-pitch UI.
- T6.1 (FastAPI) ✅ built (OD12, extended by OD13/OD14/OD16 to cover
  search/trends/graph/gaps/reading-list/summaries/hypothesis/review-queue) —
  Streamlit calls `axiom/*` directly rather than through the API, by choice
  (OD12), not because the API is missing. The React pages it was meant to
  unblock (T7.1-T7.4) are **deprioritized (OD15)** — the seam stays available
  for any future consumer regardless.
- T2.3 (5k corpus) should complete before T8.1/T8.2 eval; T8.1 calibration before final threshold claims in T8.3.
- T7.3 (Gap Evaluator) and T8.2 (eval) should precede T8.3 (demo package) —
  T8.3 shipped using real gap/metric examples already in hand (§ demo package,
  OD-adapted since no screenshots/HITL badges exist to capture).
