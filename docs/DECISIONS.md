# Architecture Decisions

A running log of committed team decisions. **Any change to `db/schema.sql`
(the shared data contract) MUST be recorded here before merge.**

---

## OD6 — Graph store is NetworkX-first

NetworkX is the graph backend for P1/P2. **Do not introduce Neo4j now.** The
SQLite `citation_edges` table is the durable source of truth for edges (built
from OpenAlex `referenced_works[]`); the graph track loads it into NetworkX in
memory. Revisit a dedicated graph DB only if NetworkX becomes a bottleneck.

## Vector collection name = `axiom_v1`

The Qdrant collection is **versioned**: `axiom_v1`. Bump the suffix whenever the
embedding model or vector dimensionality changes so old and new points never mix.
One point per paper (abstract-level). Payload: `paper_id, title, year, venue,
cited_by_count, concepts[]`.

## Embedding model = `allenai/specter2_base` (768-dim)

SPECTER2 base, 768 dimensions. CPU fallback is allowed with a warning; we never
silently switch to a different model. P1 uses the base encoder with CLS pooling
over `title [SEP] abstract`. (P2 may add the SPECTER2 proximity adapter.)

## Data layer = OpenAlex → SQLite

OpenAlex is the source of truth. Abstracts arrive as an inverted index and must
be reconstructed to plain text before insertion into `papers.abstract`.

## `search()` signature (vector-store contract)

`AxiomQdrant.search(query_vector, top_k, venues=None, year_range=None)` returns
`list[SearchHit]`. Takes a **raw 768-dim vector** (the caller encodes), keeping
the vector store decoupled from the encoder. Facets AND together; `venues`
matches OR within the facet. Filters are applied natively by Qdrant. This
signature is frozen; new capability is added as new methods (e.g.
`search_hybrid`), never by changing it.

## Query-side embedding path = expansion (adapter blocked by pins)

Terse queries embed poorly when treated as documents. The preferred fix is the
SPECTER2 **ad-hoc query adapter**, but the `adapters` library pins
`transformers~=4.57`, conflicting with our committed `transformers==4.40.2`. We
do **not** bump pins, so the active path is **query expansion**: `encode_query`
wraps the query in `config.QUERY_EXPANSION_TEMPLATE` so it lands nearer the
document region. `Specter2Encoder` attempts the adapter import and logs the
active path (`[embed] query path: expansion`). Document-side encoding is
unchanged, so existing corpus points stay valid. Revisit the adapter if/when a
transformers pin bump is approved.

## Hybrid retrieval = dense + sparse, RRF-fused in Python (anticipated by D4/OD4)

Each point now stores **named vectors**: a dense SPECTER2 vector (`dense`) plus a
sparse BM25-style vector (`sparse`, pure-Python hashing-trick encoder in
`axiom/sparse.py`). `search_hybrid` runs both arms and fuses with Reciprocal
Rank Fusion. The pinned `qdrant-client==1.9.0` supports sparse vectors but **not**
the Query-API fusion primitives (`Prefetch`/`FusionQuery`, client ≥1.10), so
fusion is done in Python — no pin bump. The frozen `search` (pure dense) is
retained; the Streamlit page prefers `search_hybrid` when the collection has a
sparse vector. This is the documented win for exact terms/acronyms: query
`LoRA` now returns the *LoRA* paper at rank 1 (absent from dense top-5);
`RAG` returns the RAG paper at rank 1.

## OD7 — OpenAlex corpus is built by snowball / 2-hop expansion

Naively fetching N papers on a topic yields a graph that is almost all dangling
external edges (each paper's `referenced_works[]` mostly points outside the
fetched set), so the in-corpus citation graph is empty and every graph metric is
meaningless. Ingestion (`axiom/ingest.py`) therefore **snowballs**: seed on a
topic via the `title_and_abstract.search` filter (NOT `cited_by_count:desc`,
which returns globally-famous off-topic papers like SciPy), then expand by
pulling the papers the seeds cite *and* the papers that cite the seeds, ranking
cited candidates by co-citation frequency, until ~`CORPUS_TARGET` papers. This
densifies internal edges. OpenAlex needs no API key — only a `mailto` for the
polite pool. Abstracts are reconstructed from the inverted index before insert.
A first real run (`retrieval-augmented generation`, target 500) produced 500
papers / 1240 in-corpus edges (avg 2.5/paper), connected enough for PageRank.

## OD8 — Citation-graph influence = PageRank on the corpus-only subgraph

The graph (`axiom/graph.py`) loads `citation_edges` into a `networkx.DiGraph`
(edge `src->dst` = "src cites dst"), keeping dangling externals as nodes flagged
`in_corpus=False`. **Influence ranking runs PageRank on the corpus-only
subgraph** so metadata-less external sinks can't absorb rank; `local_in_degree`
(citations from within the corpus) is shown alongside as a transparent companion.
PageRank is a small **pure-Python power iteration** — we do NOT add `scipy` just
for `nx.pagerank` (keeps deps pinned/minimal; numpy/networkx only). This is the
NetworkX-first realisation of OD6; no Neo4j.

## OD9 — Research gaps = semantically-close, weakly-citing community pairs

The product's key value is surfacing **research gaps**, not just influence.
`axiom/gaps.py` detects them by combining the two signals we already have:
**structure** (citation graph) and **meaning** (SPECTER2 embeddings). Pipeline:
(1) Louvain community detection on the undirected corpus citation graph →
sub-topics; (2) label each community by its most *distinctive* OpenAlex concepts
(TF-IDF across communities, dropping near-universal terms like "Computer
science"); (3) an embedding centroid per community; (4) score community **pairs**
— `gap_score = centroid_cosine / (1 + inter_community_citations)`. High score =
related in meaning yet disconnected by citation = a bridge nobody has built.
NetworkX + numpy only (no scipy/torch). PageRank/influence is retained as a
companion view. Known limitation: on a sparse single-topic corpus, centroid
cosines cluster tightly (~0.95) and most cluster pairs have 0 inter-citations, so
discrimination is limited — denser/multi-topic corpora sharpen the signal.

## OD10 — Velocity computed on-demand from concepts+papers; no persisted KEYWORD table

PBI 3 / Task 3.1 asks to "persist to a `KEYWORD` table" and compute `v_k(t)`
"per PROJECT_PLAN §7.2" — **no `PROJECT_PLAN.md` file exists anywhere in this
repo** (checked 2026-07-04), so neither that table nor that formula is defined
in-tree. `axiom/velocity.py` follows the on-demand pattern OD9 already
established for community detection instead of adding a new persisted derived
table: (1) split the (optionally venue/year-filtered) corpus's available
publication years into two contiguous windows, PRIOR / RECENT, split at the
year midpoint — generalizes the backlog's "ACL 2024 vs 2025" example to a
corpus that isn't venue-year-cadenced (OD7 topic-snowball); (2) per OpenAlex
concept (`level >= 1`, dropping overly broad concepts like "Computer science",
same rationale OD9 used for community labels), compute its normalized
frequency (share of papers) in each window; (3) `velocity =
log2((recent_share + eps) / (prior_share + eps))`, epsilon-smoothed so a
brand-new or vanished concept doesn't blow up to ±inf. `recent_count < 5` is
flagged `low_confidence` (matches the backlog's `f_k < 5` acceptance
criterion) — included, not dropped. `get_top_velocity_keywords(n, venue,
year_range)` matches the backlog's function name/signature. Wired into
Streamlit as a third "📈 Trending" tab, calling `axiom.velocity` directly
(bypasses FastAPI/PBI6, same as the Search and Citation-graph tabs).

Task 3.2 (LLM keyword canonicalization — merging synonyms like LoRA / Low-Rank
Adaptation) is **deferred**: it requires introducing an LLM API call, which is
a cost/infra decision this project hasn't made yet (same reasoning PBI 5 was
deferred for).

## OD11 — Separate eval corpus (real ACL/EMNLP/COLING/NAACL) for nDCG@10, not merged into axiom_v1

A teammate's `master` branch (merged into `main`, not this branch) carries
`FULL_DATASET.jsonl`: 23,002 real papers — ACL (8,784) / EMNLP (7,124) / COLING
(3,677) / NAACL (3,417), years 2020–2025, 100% abstract coverage, plus
`gs_citation` (a Google Scholar citation *count*, not `referenced_works`) and
no concept/keyword tags. Pulled into `data/FULL_DATASET.jsonl` (gitignored,
same as `data/axiom.db`) for **Task 8.2's nDCG@10 retrieval eval only**.

Kept in a **separate** SQLite db (`data/eval.db`) + Qdrant collection
(`axiom_eval_v1`), never merged into `data/axiom.db` / `axiom_v1`: without
citation edges it can't feed OD9 gap detection or PageRank, and without
concepts it can't feed the OD10 velocity engine — it only has what retrieval
eval needs (title, abstract, venue, year, citation count).
`scripts/ingest_eval_corpus.py` stratified-samples ~1,500 papers evenly across
(venue, year) so no single slice dominates, reusing `db.insert_paper()` /
`reindex_qdrant()` unchanged (the latter gained an optional `collection=`
param to target `axiom_eval_v1` instead of `axiom_v1`).

`eval/ndcg_queries.json` (15 queries, 86 judgments) is a **single-pass,
AI-drafted** label set — built by reading real sampled titles (not guessed
blind) and grading relevance 0–3, but **not** the backlog's "2 annotators"
acceptance criterion. `scripts/eval_ndcg.py` computes real nDCG@10 against
live search results: **hybrid mean nDCG@10 = 0.453**, dense-only = 0.345
(below the backlog's ≥0.65 target). Two queries scored 0.0 not because
retrieval failed — inspection showed genuinely relevant hits the judgment set
simply didn't cover (built via keyword-pattern search over the sample, not an
exhaustive pass). Treat these numbers as a working demo signal, not a
validated quality claim, until a second human annotator reviews
`eval/ndcg_queries.json`.

## OD12 — FastAPI scoped to what's built; hypothesis/reading-list/review-queue endpoints deferred

Task 6.1 originally specified REST wrappers for `search`, `get_top_velocity_keywords`,
`evaluate_gap`, `generate_hypothesis`, `summarize_paper`, reading-list CRUD, and
review-queue CRUD. Only the first two categories — plus paper lookup and the
OD9 citation graph/gaps — actually exist in this codebase, so `api/main.py`
wraps exactly those: `/health`, `/search`, `/papers/{id}`,
`/papers/{id}/similar`, `/trends/top`, `/graph/stats`, `/graph/influence`,
`/graph/papers/{id}/neighbors`, `/graph/gaps`. `/gap/evaluate` (3-step
hypothesis), `/reading-list/*` summaries, and a review-queue endpoint are
**not built** — they depend on the LLM hypothesis pipeline (PBI 5, deferred —
needs an API-key/cost decision) and reading-list bookmark storage (PBI 7 /
T7.4, not built), neither of which exists to wrap.

Pydantic response models mirror the `axiom/*` dataclasses (`SearchHit`,
`PaperRank`, `KeywordVelocity`/`VelocityAnalysis`, `Community`/`GapCandidate`/
`GapAnalysis`, `Neighbors`) via `from_attributes=True` — this is the one place
Pydantic enters the codebase; the `axiom/*` modules keep their existing
dataclass convention (no pydantic internally, per the established repo
style) and the API layer is a thin JSON-serializable boundary on top.
Lazy singletons (`get_encoder`/`get_store`/`get_graph`/`get_gap_analysis`) use
`functools.lru_cache` — FastAPI's equivalent of `app/streamlit_app.py`'s
`st.cache_resource`/`st.cache_data` pattern, same idea, different framework
primitive. New pinned deps: `fastapi==0.111.0`, `uvicorn==0.29.0` (same
mid-2024 pin era as the rest of `requirements.txt`).

Verified for real (not just imported): `uvicorn api.main:app` starts, `/docs`
+ `/openapi.json` list all 9 routes, and `scripts/smoke_test_api.py` (a
FastAPI `TestClient` script, not a new pytest suite — matches the existing
`scripts/eval_*.py` convention) passes all 10 checks against the live
Qdrant + `data/axiom.db` backing data.

## OD13 — Reading-list bookmarks only; no LLM summaries

Task 7.4 ("Reading List page") bundles two separable things: bookmark
persistence and 3-bullet LLM-generated summaries. Only the first is buildable
without an LLM — `summarize_paper()` belongs to the deferred PBI 5 pipeline.
Split them: **schema change** — new `reading_list` table (`db/schema.sql`),
`paper_id` primary key + `added_at`, no `user_id` (single-user demo, no auth
system exists). `axiom/db.py` gained `add_bookmark`/`remove_bookmark`/
`list_bookmarks` (same "all SQL in `db.py`" convention as every other table).
Wired into both the Streamlit app (a fourth "📚 Reading list" tab, plus an
add/remove button on every Search result card — calling `axiom/db.py`
directly, same as the other tabs) and the FastAPI layer (OD12) as
`GET/POST/DELETE /reading-list[/{paper_id}]`. Verified live: the Streamlit app
runs all four tabs with zero exceptions (`AppTest`), and
`scripts/smoke_test_api.py` exercises add → list → duplicate-add-on-missing-
paper (404) → remove → confirms removal, all passing against the live stack.

Explicitly not built (at the time): cited summary bullets, since that needed
`summarize_paper()`. Resolved by OD14, below.

## OD14 — Local LLM via Ollama unblocks T3.2 and reading-list summaries

PBI 5, T3.2, and the reading-list summary half of T7.4 were all deferred for
the same reason: "needs an LLM API-key/cost decision." That decision is now:
**use a local Ollama model, not a hosted API** — no key, no per-call cost,
no data leaves the machine. `axiom/llm.py` is a thin `httpx` wrapper around
Ollama's REST API (`http://localhost:11434/api/chat`), default model
`qwen2.5:7b`, `format: "json"` for structured output with a defensive
brace-extraction fallback (small local models occasionally wrap valid JSON in
prose). No new pinned dependency — httpx is already in `requirements.txt`;
the `ollama` PyPI client is skipped on purpose, matching this repo's
"minimal deps" discipline.

Two things built on top of it, both verified against a live local Ollama
instance (`qwen2.5:7b`, models also available: `llama3.1:latest`,
`qwen2.5-coder:14b`):

- **T3.2 (`axiom/canonicalize.py`)**: batches distinct `concepts` labels
  (50/batch), asks the model to group true synonyms only (explicit
  instruction: never merge merely-related-but-distinct concepts), persists to
  a new `concept_canonical` table (`source='auto'` vs `'manual'` — manual
  rows are never overwritten by a re-run). `axiom/velocity.py`'s
  `compute_velocity()` now maps every concept through `db.canonical_map()`
  before accumulating counts, so a synonym pair merges into one trend line.
  Verified live on the demo corpus: correctly proposed **zero** merges,
  because this hand-authored 30-paper corpus has no true duplicate concept
  labels to begin with (confirmed by a separate ad-hoc test: "LoRA" +
  "Low-Rank Adaptation" *do* merge correctly when both are present in a
  batch) — a true negative, not a broken feature.
- **T7.4 summaries (`axiom/summarize.py`)**: `summarize_paper()` produces
  exactly 3 bullets grounded only in the given abstract text (system prompt
  forbids outside knowledge/invented numbers), cached in a new
  `paper_summaries` table (`paper_id`, `bullets_json`, `model`,
  `created_at`) so the LLM call runs once per paper, not on every page
  view. Wired into both the Streamlit Reading-list tab ("🧠 Summarize"
  button) and `POST /reading-list/{paper_id}/summarize` (FastAPI, `force=true`
  to bypass the cache). Verified live: a real summary of the RAG seed paper's
  abstract produced 3 bullets that paraphrase, not invent, its actual claims.

PBI 5's LangGraph hypothesis pipeline can now use the same `axiom/llm.py`
client — but it is **not built in this pass**: the backlog's own OD9 note
already flags that the hypothesis-evaluation flow needs a design pass first
(the current gap detector is corpus-level community-pairs, not OD5's original
per-hypothesis `s_max`/Scenario-A-B input), and that redesign is independent
of which LLM backend is used. Resolving the LLM question does not by itself
resolve that design gap.

## OD15 — React migration (PBI 7) deprioritized; Streamlit is the final UI

The original plan treated Streamlit as an early-dev shell only, to be replaced
by a polished React frontend (from Stitch wireframes) for the mentor-facing
demo, with FastAPI (OD12) built specifically as the seam React would consume.
**Decision: skip the migration.** Streamlit already delivers all four working
tabs (Search, Trending, Citation graph/gaps, Reading list) end-to-end, and the
Stitch wireframes the original plan was meant to implement were never part of
this repo — building React now would mean designing the UI blind rather than
following a spec, for a polish gain that isn't worth that risk given the
project's remaining priorities. FastAPI (OD12) stays built and verified (it
has value beyond React — a stable REST seam for any future consumer, and
already used by `scripts/smoke_test_api.py`), but nothing currently consumes
it as a frontend, and that's fine. Revisit only if a concrete need for a
non-Streamlit UI shows up.

## OD16 — PBI 5 scoped to narrating existing OD9 gap candidates, not per-hypothesis evaluation

The original Step 3 assumed a per-hypothesis pipeline: user types a free-text
`h_syn`, the system scores geometric novelty (`s_max`) and a citation-scenario
verdict (Dead End / Fertile Frontier), then — only for Fertile Frontier — a
5-node LangGraph (Ingest_Summarizer, Trend_Aggregator, Gap_Narrator,
Hypothesis_Generator, Verifier) pitches a grounded hypothesis. OD9 replaced
Steps 1-2 with a corpus-level community-pair detector that takes no
per-hypothesis input, so that pipeline has nothing to plug into — the
mismatch flagged back in OD9 and repeated in OD14.

**Decision:** skip the free-text hypothesis/novelty-scoring problem entirely
(it needs new, uncalibrated threshold logic — the same T8.1 gap that's
already unresolved) and scope Step 3 to what OD9 already provides: given one
of its ranked gap candidates, generate a grounded pitch citing real papers
from both sides. `axiom/hypothesis.py`'s `generate_hypothesis()` is a single
`axiom/llm.py` call (OD14) — no LangGraph dependency added, consistent with
this project's "minimal deps" discipline (same reasoning as OD10 skipping a
`KEYWORD` table). It folds in three of the original five nodes' intent
without separate agent objects: paper grounding replaces Ingest_Summarizer
(abstracts are already short enough to pass directly), the OD10 velocity
engine's top risers are passed as prompt context in place of a
Trend_Aggregator node, and Gap_Narrator + Hypothesis_Generator collapse into
one prompt since OD9 already supplies the "why this is a gap" framing
(`_render_gap_detail` in the UI).

**The Verifier is rule-based, not an LLM self-report** — a real difference
from the original design, and a stronger guarantee: it checks
`supporting_paper_ids` actually belong to the two communities (not just that
the model claims they do) and requires >=2 ids with at least one from each
side, retrying up to 3× at temperature 0.3 per the backlog's acceptance
criteria. Verified live: a real run against the top-ranked demo-corpus gap
(PEFT/LoRA cluster vs. low-resource-NLP cluster) produced a coherent,
genuinely grounded pitch on the first attempt, correctly citing one paper
from each side (`S0005` the LoRA paper, `S0010` a back-translation paper).

**No composite G-score.** The backlog wants "$G$ with a weight breakdown per
PROJECT_PLAN §7.3 defaults" — that file doesn't exist (established in OD10),
so there are no defaults to implement; inventing arbitrary weights would
manufacture false calibration-sounding precision. OD9's `gap_score` already
serves as the ranking signal upstream of hypothesis generation.

**HITL review queue** (Task 5.2's other half) is a real, minimal addition:
new `review_queue` table (`db/schema.sql`) — every generated pitch lands as
`pending`; only an explicit approve/reject action (Streamlit "🗂️ Review
queue" tab or `POST /review-queue/{id}/{approve,reject}`) changes status.
Nothing is ever auto-promoted, matching the backlog's non-negotiable
requirement. A non-suppressable "⚠️ Unverified Candidate" banner accompanies
every pitch in both the UI and the dataclass itself
(`HypothesisPitch.disclaimer`).

Verified live end-to-end: `scripts/smoke_test_api.py` (21 checks) generates a
real pitch against the live corpus, confirms >=2 supporting ids, and confirms
an approve action actually changes status; the Streamlit app renders all 5
tabs with zero exceptions (`AppTest`).

---

## OD17 — Composite gap G-score + τ calibration scaffold (supersedes OD16's "No composite G-score")

OD16 declined to build a composite G-score: with no `PROJECT_PLAN.md` §7.3 to
supply weights (established in OD10), inventing them would "manufacture false
calibration-sounding precision," so OD9's raw `gap_score = similarity /
(1 + inter_citations)` stayed the only ranking signal.

**What changed:** the missing half was never the formula — it was a way to make
the numbers empirical instead of arbitrary. OD17 adds that: a calibration
scaffold (`scripts/export_gap_labels.py` → human labels → `scripts/
calibrate_gap_thresholds.py` → `eval/calibration.json`) that turns a labeled
mini-set into a chosen threshold τ (and, optionally, weights) via a
precision/recall sweep. With that consumer in place, a composite score stops
being false precision and becomes a *calibratable* one.

**The G-score.** `axiom/gaps.GapCandidate` now carries `g_score`, its four
normalized `components`, and `meets_threshold`. All four signals already exist
on the OD9 candidates — no new data, no new deps:

```
G = w_sim·S + w_disc·D + w_vel·V + w_auth·A
  S = clamp(centroid cosine, 0, 1)                     relatedness
  D = 1 / (1 + inter_citations)                        disconnection (OD9's factor, now additive)
  V = 1 / (1 + 2^(−v̄))                                 trend heat: logistic of mean concept velocity (OD10); 0.5 when the corpus spans <2 years
  A = min-max over the run of mean log1p(cited_by)     authority/substance of the two clusters
defaults (UNCALIBRATED): w_sim=0.4, w_disc=0.3, w_vel=0.2, w_auth=0.1; τ=0.65
```

Authority uses each paper's global `cited_by_count` (already on the graph nodes)
rather than PageRank — zero extra passes; PageRank is the noted alternative if a
future calibration wants intra-corpus centrality instead. Candidates now rank by
`g_score`; with the defaults S+D dominate, so ordering stays close to OD9's.

**Consumption.** `axiom/config.load_calibration()` reads `eval/calibration.json`
when present (merged over the defaults, ignoring a malformed file) and
`gaps.analyze` calls it each run — so dropping in a calibrated file overrides
τ/weights with no code change, closing Task 8.1's acceptance criterion
("`evaluate_gap()` reads calibrated thresholds from config, overrides dev
default 0.65").

**Honest caveat / still deferred.** The weights and τ ship UNCALIBRATED and are
labeled so in code, the API (`components`/`meets_threshold` are transparent), the
Streamlit gap detail, and this doc. Producing the human labels
(`eval/labels/gap_labels.csv`) is out of scope — it needs real annotator input,
the same PBI 8 gap OD11/OD16 already flag. A "notebook" (`calibration.ipynb`)
was specified; this repo keeps eval tooling as scripts that emit markdown
reports (`scripts/eval_ndcg.py`), so the scaffold follows that discipline
instead — pure-stdlib, importable, unit-tested (`tests/test_calibrate.py`).

---

## OD18 — Unified Qdrant collection + dual gap views

**Context.** PR #3 (`qdrant_ms/`) landed with its own Streamlit entry point
reading a separate `academic_papers` collection. This split the corpus: the main
platform wrote `axiom_v1`; the toolkit expected `academic_papers`. Users running
only the main ingest would see an empty qdrant_ms UI.

**Decision (2026-07-07).** Unify on `axiom_v1`:
- `qdrant_ms/qdrant_utils.py`: `COLLECTION_NAME = "axiom_v1"` (was `"academic_papers"`).
- `app/vector_tools.py` (new): wraps qdrant_ms page logic as render functions
  `render_proximity_analyzer` / `render_vector_gap_discovery`, reading from the
  same `AxiomQdrant.client` used by the rest of the app.
- `app/streamlit_app.py`: new **🔗 Vector tools** tab (6th tab) calls these
  render functions, giving one entry point for the whole UI.
- `qdrant_ms/Home.py` retained as a standalone dev entry point for the qdrant_ms
  team.

**Dual gap views are both kept, clearly labelled:**
- 🕸️ *Research gaps* (Citation graph tab) — OD9: Louvain community detection +
  citation structure + semantic centroids. Citation evidence is the discriminator.
- 🔗 *Vector Gap Discovery* (Vector tools tab) — KMeans clustering + bridge-paper
  scan on raw embedding geometry only. Complementary signal, no citation data.

**No schema change.** No new Qdrant collection or SQLite table. The only code
change to the data path is the collection name constant in `qdrant_utils.py`.

---

## OD19 — Agentic pipeline: `graph._pagerank` promoted to public `graph.pagerank`

**Context.** The new concept co-occurrence graph (R7, `axiom/concept_graph.py`) needs an
"analogous" PageRank to the one `gaps.py`/`graph.py` already run on the citation graph.

**Decision.** Promote the existing private `_pagerank` power-iteration routine in `graph.py` to a
public `pagerank(g, alpha, max_iter, tol)` and have `concept_graph.py` call it, instead of writing
a second copy that could silently drift from the first. `gaps.py`'s own use of it is unaffected.

## OD20 — Agent-extracted concepts never merge into the OpenAlex `concepts` table

**Context.** The agentic pipeline (`axiom/agents/*`) extracts method/task/dataset/metric entities
per paper via LLM + embedding matching — a different provenance and confidence model than the
OpenAlex taxonomy already stored in `concepts`.

**Decision.** New tables `paper_concepts` and `concept_cooccurrence` hold agent output; rows never
merge into `concepts`. Same separation discipline OD11 used for the eval corpus, applied here to
provenance instead of corpus identity. See docs/agentic/ARCHITECTURE.md §4 for the schema.

## OD21 — R8 structural retrieval arm is eval-only, not wired into production search

**Context.** R8 asks for an optional concept-overlap retrieval arm fused into the existing
dense+sparse RRF. Doing this on the interactive search path would mean an LLM call (to extract
the query's concepts) per user query — too slow/costly for Streamlit/API, and OD4 already froze
`AxiomQdrant.search`/`search_hybrid`'s signatures.

**Decision.** The structural arm lives only in `scripts/eval_ndcg.py --mode hybrid_struct`, scored
against the eval corpus's `paper_concepts`, fused in Python the same way OD4 already does. `search`
and `search_hybrid` stay untouched.

## OD22 — Pydantic v2 is a second, narrower contract boundary (agent calls only)

**Context.** OD12 established dataclasses as the internal convention, with Pydantic only mirroring
them at the FastAPI boundary. The agentic pipeline needs `model_json_schema()` to build the
Anthropic `output_config.format` payload — Pydantic is structurally required there, not a style
choice.

**Decision.** `axiom/agents/contracts.py` uses Pydantic v2 for every LLM/agent call boundary
(coordinator, extraction, enrichment, verifier). The rest of `axiom/*` keeps OD12's dataclass
convention; this isn't a reversal, just a second boundary where the tooling demands it.

## OD23 — `VERIFIER_COSINE_MIN` set by manual inspection, not swept to a target

**Context.** The deterministic verifier (R3) rejects entity→ontology matches below a cosine
threshold. Honesty rule (AXIOM_AGENTIC_TASK.md §1) forbids picking a threshold to hit a target
number.

**Decision.** `axiom.config.VERIFIER_COSINE_MIN` is set by inspecting (entity, ontology-match)
pairs sorted by cosine and picking the value where true/false matches visibly separate, recorded
with the date and sample size in a `config.py` comment.

**Calibrated (2026-10-31).** A real 30-paper pipeline run (Phase 3 testing) produced 109 real
(entity, ontology-match, cosine) triples — more than the ~30 minimum — against the main corpus's
40-label ontology (`concept_canonical` ∪ `concepts`). Inspecting all 109 sorted by cosine: matches
are clean down to ~0.95 (e.g. `0.9748 'translation'->'Machine translation'`); below that, clear
false positives appear (`0.9477 'Classification'->'Quantization'`, `0.9447 'summarization'->
'Text segmentation'`). Set `VERIFIER_COSINE_MIN = 0.95`. **Honest caveat:** this isn't a perfect
cutoff on a ~40-label ontology — one false positive still clears the bar (`0.9531 'Dual Encoder'
->'Decoding'`), and a few correct matches right at the boundary may get rejected and retried. A
single global threshold on this small ontology trades some precision for simplicity; revisit with
per-kind thresholds (method/task/dataset/metric) if the false-positive rate matters more later.

## OD24 — `truststore` added to trust the OS cert store for outbound HTTPS

**Context.** Running the agentic pipeline for real against the Anthropic API (and, separately,
loading SPECTER2 from HuggingFace) failed on this dev machine's network with
`CERTIFICATE_VERIFY_FAILED` on every `httpx`/`requests` call — `requests`/`httpx` both pin
verification to `certifi`'s bundled CA list, and that list doesn't include whatever root CA is
terminating TLS on this network (a corporate/proxy inspection signature). A raw
`ssl.create_default_context()` handshake to the same host succeeds, because Windows' own
certificate store already trusts that CA — only Python's `certifi`-pinned HTTP libraries don't.

**Decision.** Added `truststore==0.10.5` (requirements.txt) and call
`truststore.inject_into_ssl()` once, at the top of `axiom/config.py` (imported by everything
before any network call), so every `ssl.SSLContext` — and therefore every `httpx`/`requests`
call in the app — uses the OS trust store instead of `certifi`'s bundled list. This is a
judgment call specific to a network environment that terminates/inspects outbound TLS: it makes
the app trust whatever the OS already trusts, which is the correct behavior on a corporate-managed
machine but is worth knowing about on any other network this code runs on.

## OD25 — Phase 4 review: fixed High/Medium findings, logging the rest (Low/Informational)

**Context.** `software-architect` and `security-reviewer` reviewed the full `feat/agentic-pipeline`
diff (Phase 4). One High finding (`LLM_BUDGET_USD` enforced per-run instead of as a process-wide
total) and one Medium (raw SQL outside `axiom/db.py` in `linking.py`/`pipeline.py`) were fixed
immediately, along with two Low/cheap wins the architect flagged as high-payoff (stale concept
graph cache after a run; a stale comment in `concept_graph.py`). Per the task's own rule, the
remaining Low/Informational findings are logged here rather than fixed, since none change behavior
a user would notice on a local single-user laptop tool:

- **Budget-check race (Low, security-reviewer).** `_ensure_budget`/`_record_spend` still check-then-
  record without a single atomic reservation, so up to `AGENT_CONCURRENCY` (4) concurrently-running
  papers could each pass the check before any records spend. Quantified worst case: ~3 extra calls
  × ~$0.04 (max single-call cost at `_DEFAULT_MAX_TOKENS=4096`, strong tier) ≈ $0.12 over the $3.00
  cap (~4%). Not worth the complexity of a reservation scheme at this $ scale; revisit if
  `LLM_BUDGET_USD` or `AGENT_CONCURRENCY` grow materially.
- **`httpx.AsyncClient` not pooled (Low, software-architect).** `axiom/llm_providers.py` builds and
  tears down a client per Anthropic call instead of reusing one across a run. Correct (no leak),
  just a latency nit under `AGENT_CONCURRENCY` fan-out. Defer unless bench numbers show it matters.
- **`synthesis.py` imports underscore-prefixed names from `axiom/hypothesis.py`** (`_SYSTEM`,
  `_prompt`, `_verify`) to reuse OD16's verifier rules without duplicating them (deliberate, per
  OD22) — but it's a cross-module encapsulation break worth promoting to public names later, the
  same way OD19 promoted `graph._pagerank`, now that there are two call sites.
- **`POST /agents/run`'s paper-fetch is sync DB I/O inside an `async def`** (Low, software-architect)
  — blocks the event loop briefly; harmless for a single local user with no concurrent requests.
- **`/agents/run` costs real money per call, no auth/rate-limit** (Low, security-reviewer,
  explicitly theoretical under the existing localhost-only assumption, which this diff doesn't
  change — `CORS_ORIGINS` untouched, no `0.0.0.0` bind introduced). Worth a README note (Phase 5)
  that this port must never be exposed beyond localhost, unlike the rest of the app's $0 endpoints.
- **Prompt injection / key handling / dependency hygiene**: security-reviewer found no issues —
  `stop_reason` is checked on every path before parsing, `max_tokens` bounds cost regardless of
  abstract content, `evidence_span` never reaches a shell/eval/file path, `.env` is confirmed
  gitignored and never logged, `truststore` is a legitimate package. No action needed.

---

## OD26 — Stage-2 fixes, Phase 1: design note (B1/C3/D1) + schema/config deltas

**Context.** `AXIOM_FIXES_TASK.md` Phase 0 audit (branch `fix/agentic-review`, off
`feat/agentic-pipeline`) confirmed every listed finding against the code as of `0db07f2` (full
checklist in `.claude/fixes/PROGRESS.md`). This decision covers the three findings the task
calls out for an explicit design note before coding, plus the schema/config changes Phase 1
front-loads so the two Phase 2 implementer worktrees build against settled contracts.

**B1 — matching cascade (design, not yet built).** Enrichment's problem isn't the threshold
(that part, `VERIFIER_COSINE_MIN`/OD23, is fine) -- it's that there's only one step: embed the
bare entity name with SPECTER2 (a document-level, title+abstract-concat encoder) and take the
arg-max, with no exact-match short-circuit and no separation between "a good match" and "the
least-bad match in a list of bad ones." Replacing it with a 3-step cascade:
1. exact/normalized match (casefold, NFKC, collapsed whitespace) against `concept_canonical`
   and `concepts.concept` labels -- cheap, exact, no embedding call needed for the common case
   of an entity already named the way the ontology names it.
2. embedding match against the same pool using a short-phrase model
   (`ENRICHMENT_PHRASE_MODEL = BAAI/bge-small-en-v1.5`, not SPECTER2 -- a document encoder is
   the wrong tool for a 2-4 word entity name), accepted only if `cosine >= VERIFIER_COSINE_MIN`
   **and** `top1 - top2 >= ENRICHMENT_MARGIN_DELTA` (a close second place is a sign the match is
   arbitrary, not confident).
3. otherwise `new_concept`: **kept**, not dropped -- `canonical` = the normalized entity name
   itself. This is the real behavior change: today's `unmatched` entities are filtered out of
   linking entirely (`linking.py:38`); a paper using a method/dataset not yet in the ontology
   currently contributes nothing to the concept graph for that entity. `new_concept` fixes that
   at the cost of the graph slowly absorbing one-off names -- acceptable since `concept_graph`
   already treats low-degree nodes as uninteresting for PageRank/communities.

`ontology_source` is renamed `openalex_concept|concept_canonical|unmatched` ->
`lexical|embedding|new_concept` (contracts.py enum + every call site in enrichment.py,
linking.py, verifier.py, and the tests that assert on it) -- same three-way shape, clearer
names that match what actually produced the match rather than which table it came from.
`ENRICHMENT_PHRASE_MODEL` and `ENRICHMENT_MARGIN_DELTA` are added to config.py now as
placeholders (the margin is a guess pending the B3 gate); pipeline-implementer builds the
cascade against them in Phase 2, and the real delta value lands in a follow-up OD after the
user labels `eval/enrichment_pairs.csv`.

**C3 — persistent budget ledger (design, not yet built).** Two separate problems, same area:
(a) spend lived only in `axiom/llm_providers.py` module globals, so it reset on every process
restart and a `GET /agents/runs` history can't explain its own cost; (b) `_ensure_budget`
checks-then-records with no reservation, so `AGENT_CONCURRENCY` (4) concurrent calls can all
pass the pre-call check before any of them commits spend (already logged as an accepted Low
risk in OD25, ~4% overshoot at current concurrency -- this fix removes the gap rather than
re-accepting it). Design: a new `llm_spend` table (added to schema.sql now) holds one row per
*settled* call (after the response, real token counts); an in-process `asyncio.Lock`-guarded
reservation ledger (dict of `run_id -> reserved_usd`, implementer's call) tracks worst-case
cost (`prompt_token_estimate * price_in + max_tokens * price_out`) from the moment a call is
dispatched, released and replaced by the real `llm_spend` row once it settles. The budget check
sums `committed (from llm_spend, cross-run total) + reserved (in-memory, this process)` against
`LLM_BUDGET_USD` -- no race window, and the total survives a restart because the committed half
is durable. `_spend_total`'s existing process-wide (not per-run) scoping was already correct
(OD25) and carries over unchanged; this only adds durability and closes the reservation gap.

**D1 — background runs (design, not yet built).** `POST /agents/run` moves from `await`ing the
full pipeline inline to: insert the `agent_runs` row (`status=queued`), kick off the pipeline as
a FastAPI `BackgroundTasks` job, return `202 {run_id}` immediately. The background job wraps the
whole pipeline in `try/finally`: sets `status=running` on start, `status=done` or
`status=failed` (with the error) in `finally`, and increments `agent_runs.papers_done` after
each paper rather than only at the end, so `GET /agents/runs/{id}` reflects real progress on
a run still in flight. That endpoint's response also gains per-verdict counts (group
`agent_run_items` by `verdict`) and cost-so-far (sum `llm_spend` filtered by `run_id`, once C3
lands) -- both currently absent. This is `api/main.py`-owned (main session), not
pipeline-implementer's worktree, since main owns that file per the task's role split; D3/D4's
`axiom/agents/*` side (real `verifier_attempts` count, promoting `hypothesis._SYSTEM` etc. to
public names) is pipeline-implementer's, the endpoint wiring is main's.

**Other schema/config deltas applied now (not separately noted above):**
- `paper_concepts`/`concept_cooccurrence` gain a `run_id` column (B4) and a new unique index
  `(paper_id, canonical, kind)` on `paper_concepts` -- the backstop for idempotent linking;
  the actual delete-then-insert-in-one-transaction logic is pipeline-implementer's.
- `MODEL_FAST`/`MODEL_STRONG` updated to `claude-haiku-5-5` / `claude-sonnet-5-5` with prices
  re-verified live against platform.claude.com on 2026-10-31 (not from memory, since the task
  flagged these as liable to have drifted): haiku-5-5 is $0.10/$0.50 per MTok for prompts
  <=100k tokens (the only regime this app hits) vs haiku-4-5's $1/$5 -- a ~10x cost cut for the
  fast tier; sonnet-5-5 is $2/$10, identical to the sonnet-5 it replaces, so this is a free
  quality upgrade, not a cost tradeoff. `AGENT_MAX_TOKENS` replaces the flat 4096 with
  per-agent-type values (route=200, extraction=1500, synthesis keeps 4096) -- wiring into
  `complete_json` call sites is infra-implementer's (C4).
- `AXIOM_OS_TRUSTSTORE` env gate added around OD24's `truststore.inject_into_ssl()` call (C5)
  -- opt-in now instead of unconditional, since rewiring process-wide TLS trust is the right
  call on this inspected-TLS laptop but a surprising one anywhere else; README troubleshooting
  update is infra-implementer's (F3 overlap).
- `ROUTE_TITLE_ONLY_MIN_WORDS = 40` added for A5's rule-based routing threshold; the actual
  rule-before-LLM logic in `coordinator.py` is pipeline-implementer's.

## OD27 — Stage-2 fixes, Phase 2: pipeline-core findings landed (A1-A5, B1/B2/B5, D3/D4)

**A1/A3 (verifier.py).** `verify_extraction` now checks `evidence_span` against the title OR
the abstract (previously abstract only), with both sides run through a new `normalize_text`
(NFKC, casefold, collapsed whitespace, unified dash/quote variants) before comparing. A new
check, `name_relates_to_span`, fails an entity whose claimed `name` doesn't actually relate to
its own cited span -- substring, a multi-word name's own first-letter acronym, or token
overlap `|tokens(name) & tokens(span)| / |tokens(name)|` >= 0.5 (denominator is the name's own
token count, not the span's, so a long span doesn't dilute the score). New `VerifierError`
code `name_span_mismatch`.

**A2 (extraction.py/pipeline.py).** `extraction.extract` now accumulates PASSING entities
across retry attempts (keyed by name+kind+evidence_span) instead of one pass/fail verdict for
the whole response; a response that mixes a good entity with a bad one no longer drops the good
one. The whole call only raises `VerificationFailed` (paper rejected) if every entity ever
proposed, across every attempt, failed verification -- distinguished from a paper that
genuinely has nothing to extract (zero entities proposed, not an error) via a new
`empty_after_retry` `VerifierError` code attached to the exception. `extract()`'s return value
changed from the bare `ExtractionResult` to `ExtractionOutcome(paper_id, entities, attempts)` so
D3's real attempt count travels with it; `enrichment.enrich` got the matching
`EnrichmentOutcome(result, attempts)`. `agent_run_items.verdict`'s own vocabulary
(accepted/rejected/skipped/error) is unchanged -- `bench_agents.py`'s rejection-rate counting
(by the per-attempt `*.verifier` trace line's `verdict` field, "accepted"/"rejected" only) is
unaffected; `empty_after_retry` only ever appears inside the recorded error TEXT.

**A4 (contracts.py).** `paper_id` removed from `RouteDecision` and `ExtractionResult` --
`extra="forbid"` means a model (or a stub mimicking one) that tries to echo a `paper_id` back
now fails Pydantic validation outright, it can't silently succeed with a wrong value.
`axiom/agents/pipeline.py` sources every write's paper id from its own `PaperInput`, never from
anything a model returned.

**A5 (coordinator.py).** `route()` is now three deterministic rules, no LLM call: no title/no
abstract -> skip; abstract missing or under `ROUTE_TITLE_ONLY_MIN_WORDS` words -> title_only;
else -> full. These are exhaustive, so the LLM coordinator call (`route_via_llm`, kept for a
future genuinely-ambiguous case) is unreachable from `route()` today -- a real cost reduction,
not a gap, since routing used to cost one fast-tier call per paper regardless. New
`RouteDecision.route_source: Literal["rule", "llm"]` records which path decided. `title_only`
gets its own extraction prompt (`extraction._SYSTEM_TITLE_ONLY`) and verifies spans against the
title only, ignoring whatever (already-judged-too-thin) abstract text the input still carries.

**B1/B2 (enrichment.py/linking.py).** Built per OD26's design: exact/normalized match (no
embedding call) -> `PhraseEncoder` (new, mean-pooled `BAAI/bge-small-en-v1.5` via plain
`transformers.AutoModel`/`AutoTokenizer`, same pattern as `axiom/embed.py`'s `Specter2Encoder`
but the bge mean-pooling recipe, not SPECTER2's CLS-token recipe; no `sentence-transformers`
dependency added) gated on cosine >= `VERIFIER_COSINE_MIN` AND top1-top2 margin >=
`ENRICHMENT_MARGIN_DELTA` -> `new_concept` (kept). `linking.link` no longer filters anything --
every enriched entity is written now that nothing is dropped as "unmatched". `axiom/agents/
pipeline.py::run_pipeline` gained a `phrase_encoder` parameter (defaults to a lazily-built
process-wide singleton, `enrichment.get_default_phrase_encoder()`); the existing `encoder`
parameter (still accepted so `api/main.py`'s `POST /agents/run` call keeps working unchanged)
is no longer used by enrichment -- SPECTER2 was always the wrong tool for a short entity name,
and now nothing routes entity-matching through it.

**B5 (graph.py).** `_pagerank`'s power iteration now reads each edge's `weight` attribute
(default 1 when absent) for both `out_deg` and the per-successor rank share, instead of a plain
edge count. `tests/test_graph.py` adds a regression proving this is numerically identical to
the old unweighted behavior when every edge has the default weight of 1 (cross-checked against
`networkx.pagerank(..., weight=None)`), plus a weighted case and a hand-checked symmetric-cycle
fixed point. `tests/test_concept_graph.py`'s `autouse` fixture that monkeypatched `graph.
pagerank` with `nx.pagerank` (under a docstring claiming the real function "does not exist
yet" -- it already did) is deleted; those tests now exercise axiom's own implementation.

**D3 (pipeline.py).** The accepted path's `agent_run_items.verifier_attempts` is now
`max(extraction_outcome.attempts, enrichment_outcome.attempts)` -- the real number of
self-correction rounds this paper needed -- instead of a hardcoded `1`.

**D4 (hypothesis.py/synthesis.py).** `axiom.hypothesis`'s shared pieces (`_SYSTEM`, `_prompt`,
`_side_papers`, `_verify`) are promoted to public names (`SYSTEM`, `build_prompt`,
`side_papers`, `verify`) and a new `explain()` gives the structured reasons behind a failed
verification; `axiom.agents.synthesis.generate_hypothesis` (strong-tier, via the provider
layer) is the one real implementation going forward and now nudges its retries with `explain()`'s
actual failure reasons instead of a fixed generic reminder string, the same self-correction
pattern `verifier.format_errors_for_retry` already uses. `axiom.hypothesis.generate_hypothesis`
(the original, synchronous, Ollama-only call) is NOT deleted: `api/main.py`'s `POST /graph/
gaps/{i}/hypothesize` and `app/streamlit_app.py`'s hypothesize button both still call it
directly by that name, and this track doesn't own either file. It got the same `explain()`-based
nudge for consistency but is otherwise unchanged, and is flagged in its own docstring as the
path to delete once those two call sites are repointed at `axiom.agents.synthesis.
generate_hypothesis(gap, g, *, run_id, trend_context=None, max_retries=...)`.

## OD28 — B3 AI-labeling: no human review; a budget-estimator bug and a model-generation break found running it

**Context.** The user explicitly asked to replace stage-2's remaining human-labeling gates
(B3, E3) with independent AI labeling and run to the end of the stage without stopping for
manual review. `scripts/ai_label_enrichment_pairs.py` runs two independent, blind judges
(`strong` = claude-sonnet-5-5, a new `opus` tier = claude-opus-5-5) over
`eval/enrichment_pairs.csv`: same rubric, temperature-controlled (see below), each seeing
only entity/kind/evidence_span/candidate, never cosine/margin. A label counts for calibration
only when both judges agree on "1" or "0"; disagreements and "unsure" are reported but
excluded. A deterministic (non-LLM) acronym-definition detector flags, but never overrides,
disagreements where the paper's own abstract defines a relevant acronym. **Every label this
produces is AI-labeled with no human review — call it that everywhere (RESULTS.md,
DECISIONS.md, README), never "human-labeled" or "hand-labeled".** `scripts/label_pairs.py`
(the blind human CLI built for the original B3 gate) and `scripts/label_cli.py` are kept,
unused for now, for an optional future human pass.

**Bug found and fixed while running this for real: `temperature` is rejected outright.**
The first real run came back 100% "unsure" at $0.0000 spend -- every one of 200 calls failed
with `HTTP 400: "temperature is deprecated for this model"`. Every model this app now targets
(the 5.5 generation: `MODEL_FAST`/`MODEL_STRONG`/the new `MODEL_OPUS`) rejects the parameter
outright; `axiom.agents.synthesis.generate_hypothesis` (D4) has been passing a real
temperature value since it was written and would have hit the exact same failure the first
time it actually ran against the live API with a key set -- this had never been exercised for
real before now. Fixed in `AnthropicProvider.complete_json`: the Anthropic payload never
includes `temperature` regardless of what's passed; the parameter stays on `complete_json`'s
signature only so `OllamaProvider`'s local models (which still honor it) don't need a
different call shape.

**Bug found and fixed: the budget-reservation cost estimate undercounts real cost by ~2x.**
The real run's actual spend ($0.5381) came in well over its own pre-run worst-case estimate
($0.3901) -- which should be impossible for a true upper bound. Measured directly from the
real run's durable `llm_spend` rows: ~1250 input chars (system + prompt + the structured-
output JSON schema) against ~616 real `tokens_in` -- a ~2.0 chars/token ratio for this kind of
short, technical text, not the ~4 `_estimate_worst_case_cost` assumed, and that function also
never counted the schema's own serialized size as billed input at all (it's sent via
`output_config.format.json_schema`, and real content). Both fixed: `_estimate_worst_case_cost`
now divides by 2, not 4, and `AnthropicProvider.complete_json` includes the schema's JSON size
in the estimate's input length. This is a C3 budget-safety fix, not just a B3 one -- the old
estimate's claimed "always >= the eventual real cost" guarantee did not actually hold, which
matters for the one thing C3 exists to prevent (overshooting `LLM_BUDGET_USD`).

**B3 result** (after both fixes, real run, $0.5381 spent this run / $0.5647 cumulative total):
Cohen's kappa between the two judges was 0.7983 over all 100 pairs (substantial agreement).
92/100 pairs got a usable agreed label (15 agreed "1", 77 agreed "0"); 8 were "unsure" on at
least one side; zero outright A-vs-B disagreements.

**Calibration (`scripts/calibrate_enrichment_threshold.py`, report at
`eval/enrichment_threshold_calibration.md`):** highest recall at precision >= 0.9 picks
tau=0.93 (barely below the inherited 0.95), delta=0.0 (no margin requirement) --
precision=0.929, recall=0.867 on the 92 usable pairs. Applied to the reference run's full set
of 208 `new_concept` entities (not just the 100-pair sample), only 6 would flip to a real
ontology match under this looser threshold.

**Decision: ontology gap, not threshold miscalibration.** 77/92 agreed labels were "0" --
including among the HIGHEST-cosine candidates in the sample (it was built from the top 100 by
cosine) -- and loosening tau/delta barely moves the `new_concept` rate. The embedding model
produces moderately high cosine similarity for genuinely different concepts (broader/
narrower, related-but-distinct, version differences) at a rate that makes a looser threshold
a precision trade, not a real fix: the ontology (572 real candidate labels, not a sparse pool)
simply doesn't contain most of these entities. `VERIFIER_COSINE_MIN`/`ENRICHMENT_MARGIN_DELTA`
are updated to `0.93`/`0.0` (replacing OD26's placeholder `0.95`/`0.05`) -- a small, real,
labeled-data-backed correction, not a loosening aimed at reducing the `new_concept` rate.

**Follow-up: merge synonymous `new_concept` labels** (`scripts/merge_new_concept_synonyms.py`,
applied for real). Since the ontology genuinely lacks these entities, the real risk is the
SAME entity getting a different surface-form canonical in different papers, fragmenting the
concept graph (e.g. "RAG" vs "retrieval-augmented generation" vs "retrieval augmented
generation (RAG)" as three separate nodes instead of one). Ran the same exact/embedding
matching logic pairwise over the `new_concept` canonicals themselves (same tau=0.93, no
margin -- there's no runner-up to compare against when clustering a set against itself).
Result: 11 clusters, 21 canonicals absorbed into one representative each (the most-occurring
raw name in the cluster), 20 `paper_concepts` rows relabeled, 9 exact-duplicate rows this
created cleaned up afterward. The biggest cluster: "retrieval-augmented generation" absorbed
10 variants including both acronym forms ("...(RAG)") and compound-method variants
("graph-based...", "naive...", "self-knowledge guided..."). Known limitation, accepted: this
pass uses plain cosine similarity with no margin and no broader/narrower nuance -- unlike
B3's own AI rubric, it cannot distinguish a true synonym from a narrower/broader near-
neighbor (one cluster, "graph reasoning" absorbing "knowledge graph reasoning", is arguably
too aggressive under that rubric). Net positive for graph quality; a human or AI review pass
over the merge list is a reasonable future refinement, not required now.

---

## Schema change log

| Date       | Change                                              | By |
|------------|-----------------------------------------------------|----|
| 2026-06-04 | Initial P1 schema: papers, concepts, citation_edges, paper_provenance + read-path indexes | UI track |
| 2026-07-04 | Added `reading_list` (paper_id, added_at) — bookmarks only, OD13 | UI track |
| 2026-07-04 | Added `concept_canonical` (concept, canonical, source) and `paper_summaries` (paper_id, bullets_json, model, created_at) — OD14 | UI track |
| 2026-07-04 | Added `review_queue` (id, gap_a_label, gap_b_label, title, claim, method_sketch, datasets_json, supporting_ids_json, status, created_at) — OD16 | UI track |
| 2026-10-30 | Added `paper_concepts`, `concept_cooccurrence`, `agent_runs`, `agent_run_items` + indexes — agentic pipeline, OD19/OD20 | agentic track |
| 2026-10-31 | Added `run_id` to `paper_concepts`/`concept_cooccurrence`, unique index `(paper_id, canonical, kind)`, new `llm_spend` table + index — stage-2 fixes B4/C3, OD26 | fixes track |
