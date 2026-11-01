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
threshold. Honesty rule (the agentic design spec) forbids picking a threshold to hit a target
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

**Context.** an architecture and security review pass covered the full `feat/agentic-pipeline`
diff (Phase 4). One High finding (`LLM_BUDGET_USD` enforced per-run instead of as a process-wide
total) and one Medium (raw SQL outside `axiom/db.py` in `linking.py`/`pipeline.py`) were fixed
immediately, along with two Low/cheap wins the architect flagged as high-payoff (stale concept
graph cache after a run; a stale comment in `concept_graph.py`). Per the task's own rule, the
remaining Low/Informational findings are logged here rather than fixed, since none change behavior
a user would notice on a local single-user laptop tool:

- **Budget-check race (Low).** `_ensure_budget`/`_record_spend` still check-then-
  record without a single atomic reservation, so up to `AGENT_CONCURRENCY` (4) concurrently-running
  papers could each pass the check before any records spend. Quantified worst case: ~3 extra calls
  × ~$0.04 (max single-call cost at `_DEFAULT_MAX_TOKENS=4096`, strong tier) ≈ $0.12 over the $3.00
  cap (~4%). Not worth the complexity of a reservation scheme at this $ scale; revisit if
  `LLM_BUDGET_USD` or `AGENT_CONCURRENCY` grow materially.
- **`httpx.AsyncClient` not pooled (Low).** `axiom/llm_providers.py` builds and
  tears down a client per Anthropic call instead of reusing one across a run. Correct (no leak),
  just a latency nit under `AGENT_CONCURRENCY` fan-out. Defer unless bench numbers show it matters.
- **`synthesis.py` imports underscore-prefixed names from `axiom/hypothesis.py`** (`_SYSTEM`,
  `_prompt`, `_verify`) to reuse OD16's verifier rules without duplicating them (deliberate, per
  OD22) — but it's a cross-module encapsulation break worth promoting to public names later, the
  same way OD19 promoted `graph._pagerank`, now that there are two call sites.
- **`POST /agents/run`'s paper-fetch is sync DB I/O inside an `async def`** (Low)
  — blocks the event loop briefly; harmless for a single local user with no concurrent requests.
- **`/agents/run` costs real money per call, no auth/rate-limit** (Low, the review,
  explicitly theoretical under the existing localhost-only assumption, which this diff doesn't
  change — `CORS_ORIGINS` untouched, no `0.0.0.0` bind introduced). Worth a README note (Phase 5)
  that this port must never be exposed beyond localhost, unlike the rest of the app's $0 endpoints.
- **Prompt injection / key handling / dependency hygiene**: the review found no issues —
  `stop_reason` is checked on every path before parsing, `max_tokens` bounds cost regardless of
  abstract content, `evidence_span` never reaches a shell/eval/file path, `.env` is confirmed
  gitignored and never logged, `truststore` is a legitimate package. No action needed.

---

## OD26 — Stage-2 fixes, Phase 1: design note (B1/C3/D1) + schema/config deltas

**Context.** `the fixes design spec` Phase 0 audit (branch `fix/agentic-review`, off
`feat/agentic-pipeline`) confirmed every listed finding against the code as of `0db07f2` (full
checklist in `.claude/fixes/PROGRESS.md`). This decision covers the three findings the task
calls out for an explicit design note before coding, plus the schema/config changes Phase 1
front-loads so the two Phase 2 build branches build against settled contracts.

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
placeholders (the margin is a guess pending the B3 gate); the agents/concept-graph work builds the
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
reservation ledger (dict of `run_id -> reserved_usd`) tracks worst-case
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
the agents/concept-graph work's branch, since main owns that file; D3/D4's
`axiom/agents/*` side (real `verifier_attempts` count, promoting `hypothesis._SYSTEM` etc. to
public names) is the agents/concept-graph work's, the endpoint wiring is main's.

**Other schema/config deltas applied now (not separately noted above):**
- `paper_concepts`/`concept_cooccurrence` gain a `run_id` column (B4) and a new unique index
  `(paper_id, canonical, kind)` on `paper_concepts` -- the backstop for idempotent linking;
  the actual delete-then-insert-in-one-transaction logic is the agents/concept-graph work's.
- `MODEL_FAST`/`MODEL_STRONG` updated to `claude-haiku-5-5` / `claude-sonnet-5-5` with prices
  re-verified live against platform.claude.com on 2026-10-31 (not from memory, since the task
  flagged these as liable to have drifted): haiku-5-5 is $0.10/$0.50 per MTok for prompts
  <=100k tokens (the only regime this app hits) vs haiku-4-5's $1/$5 -- a ~10x cost cut for the
  fast tier; sonnet-5-5 is $2/$10, identical to the sonnet-5 it replaces, so this is a free
  quality upgrade, not a cost tradeoff. `AGENT_MAX_TOKENS` replaces the flat 4096 with
  per-agent-type values (route=200, extraction=1500, synthesis keeps 4096) -- wiring into
  `complete_json` call sites is the provider/eval-tooling work's (C4).
- `AXIOM_OS_TRUSTSTORE` env gate added around OD24's `truststore.inject_into_ssl()` call (C5)
  -- opt-in now instead of unconditional, since rewiring process-wide TLS trust is the right
  call on this inspected-TLS laptop but a surprising one anywhere else; README troubleshooting
  update is the provider/eval-tooling work's (F3 overlap).
- `ROUTE_TITLE_ONLY_MIN_WORDS = 40` added for A5's rule-based routing threshold; the actual
  rule-before-LLM logic in `coordinator.py` is the agents/concept-graph work's.

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

## OD29 — E3 AI-vs-AI second-grader review: a real bug found running it, then a real result

**Context.** Same user instruction as OD28: replace E3's human-review gate with an
independent AI second opinion. `scripts/ai_second_grader_sample.py` picks a fixed-seed random
10-query sample of `eval/ndcg_queries_v2.json` and has a second, independent judge re-grade
the same pooled candidates from scratch, blind to the original grader's judgments and to the
retrieval ranks.

**Bug found and fixed: the "second grader" wasn't actually a different model.**
`scripts/build_eval_v2_queries.py::grade_query` hardcoded `tier="fast"` -- reusing it
unmodified for the "second, independent, stronger" grader the task called for silently
re-ran the SAME fast-tier model that produced the original labels, confirmed by the first
real run's trace file (`claude-haiku-5-5` on both sides). That run's kappa (0.6827) measured
same-model inference noise, not real cross-model agreement, and was discarded. Fixed:
`grade_query` gained a `tier` parameter (default `"fast"`, so `build_eval_v2_queries.py`'s own
usage is unchanged) and the second-grader script now actually passes `tier="strong"`.

**Also found: `data/eval.db` predates the agentic schema entirely.** It's kept separate from
the main corpus db by design (OD11), so nothing had ever run `db.init_db()` against it --
the first attempt crashed outright on a missing `llm_spend` table. Fixed by calling
`db.init_db()` on it before use (verified safe: `CREATE TABLE IF NOT EXISTS` left its 1500
existing papers untouched).

**Also found: the real spend from this script is invisible to the durable ledger.**
`grade_query` doesn't thread a `conn` through to `complete_json`, so nothing lands in
`llm_spend` for this call path -- `db.total_llm_spend` would report $0 after a run that
genuinely spent money. Reports `llm_providers.spend_so_far(run_id)` (the in-process total)
instead, same as `build_eval_v2_queries.py`'s own cost line already does.

**Result** (real run, strong tier, $0.1129 spent): Cohen's kappa between the original
fast-tier grader and this independent strong-tier re-grade was **0.6319** (moderate-to-
substantial agreement) over 129 judgments sampled from 10 queries, agreement rate 77.5%.
Lower than B3's 0.7983 same-same-task kappa, which is expected: a 3-point ordinal relevance
judgment has more room for two different models to disagree than a binary same-concept
judgment does. Written to `eval/ndcg_v2_second_grader_sample.json` -- the original
`ndcg_queries_v2.json` and its `_provenance` (AI-drafted, fast tier, no human review) are
untouched; this is a second opinion recorded alongside it, not a replacement.

## OD30 — D5 query-aware structural arm: real data, honest negative result

**Context.** D5 asked for a genuinely query-aware structural retrieval arm (map the query
itself to ontology concepts via the B1 cascade, score candidates by weighted overlap with
their own `paper_concepts`) -- distinct from the existing `hybrid_struct` arm, which only
scores how well a candidate's concepts overlap with OTHER candidates in its own pool and
never looks at the query. Needed real `paper_concepts` data on the eval corpus first.

**Ran the real pipeline on 300 real eval-corpus papers** (`logs/run_agent_benchmark_eval_corpus.py`,
not committed): 300/300 accepted, $0.1095 spent, 188s wall time, 1741 `paper_concepts` rows
written. (Separate budget pool from the main corpus's `llm_spend` -- `data/eval.db` and
`data/axiom.db` are different files with their own `llm_spend` tables; worth noting as a gap
in C3's "total spend" guarantee across the two, not worth fixing now at this $ scale.)

**Bug found and fixed while building the comparison script
(`scripts/eval_structural_query_aware.py`): the eval corpus has no ontology at all.**
`enrichment.build_ontology()` draws its pool from `concept_canonical`/`concepts` -- both
empty on `data/eval.db`, which was never ingested with real OpenAlex concept metadata (the
pre-existing OD11/OD20 gap the original audit already flagged). Every query matched zero
concepts on the first real run. Fixed by falling back to the agent-extracted
`paper_concepts.canonical` set as the query-matching pool when the normal source is empty --
the only real concepts this corpus actually has. 44/51 queries matched at least one concept
after the fix (ontology pool size 1512).

**Result, honestly reported either way as instructed:** mean nDCG@10 over the 51 v2 queries,
same candidate pool size for all three arms -- hybrid alone **0.8904**, hybrid + the existing
pool-cohesion structural arm **0.6168** (notably worse), hybrid + this query-aware arm
**0.8727** (roughly neutral, slightly worse). **Neither structural arm helps on this corpus.**
The query-aware arm is a much smaller regression than the pool-cohesion one, consistent with
it actually using query-relevant signal rather than pure pool self-similarity, but it's still
a regression, not an improvement. Per the task's own instruction, this is NOT exposed as an
optional search mode -- `scripts/eval_structural_query_aware.py` stays an eval-only analysis
script, same as the existing `hybrid_struct` arm. Report at
`eval/report_hybrid_struct_query_aware.md`.

## OD31 — keep the eval corpus built from the uncleaned `FULL_DATASET.jsonl`, not `cleaned_dataset.jsonl`

**Context.** Stage 3 (product) repo cleanup, item A.1: `FULL_DATASET.jsonl` and
`cleaned_dataset.jsonl` were committed at the repo root (2 x 78 MB, 23,002 rows each),
although `data_prep/README.md` claimed they were git-ignored and `axiom.config.
EVAL_DATASET_PATH`/`scripts/ingest_eval_corpus.py` expect `data/FULL_DATASET.jsonl` --
a path that didn't exist (the real file was at the repo root). `diff` of the two files
confirms exactly 806/23,002 rows (3.5%) differ -- `clean_dataset.py`'s dedupe/
admin-row-drop/text-normalization rules changed those rows; the other 22,196 are
byte-identical.

**Decision.** Leave `EVAL_DATASET_PATH` pointing at `FULL_DATASET.jsonl` (the uncleaned
file), now relocated to `data/FULL_DATASET.jsonl` so the path in code finally matches
reality. Do not switch the eval corpus to `cleaned_dataset.jsonl` this stage, and do not
re-ingest `data/eval.db`.

**Why.** The entire stage-2 eval history (nDCG@10 results, OD29/OD30's AI-grading
and structural-arm comparisons, `eval/ndcg_queries_v2.json`) was built against the
corpus `ingest_eval_corpus.py` already produced from the uncleaned file. Switching the
source now would silently invalidate every one of those numbers without first
auditing what the 806 differing rows actually fix (dedup? garbled characters? dropped
admin/TOC rows mixed in as fake papers?) -- that audit is real work this stage's
product-redesign scope doesn't need and the fast-mode laptop rules (no re-ingest,
heavy jobs one at a time) argue against doing speculatively. Both files stay
available in `data/` (git-ignored) so a future stage can diff the 806 rows and
re-ingest deliberately if the cleaning turns out to matter for eval quality.

**Action taken.** Moved both files from the repo root into `data/` (matching
`config.EVAL_DATASET_PATH` and the existing blanket `data/` gitignore rule),
untracked them from git (`git rm --cached`, history keeps the old committed blobs --
no history rewrite this stage), and documented the regeneration path (including
pulling the old committed copies back out of `origin/main` history) in
`data_prep/README.md`.

## OD32 — Start flow always builds the full ~300-paper corpus, no small-first option

**Context.** Stage 3 (product) Phase B product definition needed a default for the
Start screen's corpus-size behavior; the task brief's own laptop rule already caps
every field at <=300 papers, but didn't say whether Start should default to that
ceiling immediately or offer a faster small build first.

**Decision.** Start always targets the full ~300-paper corpus (no separate
small/fast tier). No UI control to pick a smaller size for the MVP.

**Why.** The real corpus already built in stage 2 (300 OpenAlex papers, topic
"retrieval-augmented generation") is the one the demo scenario (see
`docs/product/PRODUCT.md`) will actually use, and stage 2's own pipeline runs
on it are the system's only real performance/cost data point so far. Offering
a second, smaller default size multiplies the states the build-progress UI and
its time/cost estimate need to handle correctly for no validated benefit this
stage -- nothing in the audit or the user's answers asked for a faster-but-smaller
tier. If a real need for incremental/expandable fields shows up later, it's a
straightforward additive UI change, not a backend redesign (the per-field
SQLite+Qdrant storage plan in §4E already supports re-running a build against
an existing field).

## OD33 — React migration un-deprioritized; reverses OD15

OD15 (2026-07) deprioritized the React migration: no spec existed to build against
("the Stitch wireframes... were never part of this repo"), and Streamlit already
covered the four tabs that existed then end-to-end, for a "polish gain that isn't
worth that risk."

**Decision: build it now, in `web/` (React + Vite + TypeScript).** Both conditions
OD15 cited no longer hold. A real spec exists this time: `docs/product/PRODUCT.md`
(personas/flow/IA), `web/design/wireframes/` (26 low-fi screens with real data),
and `docs/product/DESIGN_SYSTEM.md` (tokens + 8 component specs, WCAG-verified) --
stage-3 Phases B-D built exactly the design-to-implement OD15 said was missing.
And this isn't a polish pass on working functionality: `AUDIT.md` found the
current Streamlit app's actual defects are structural (no in-app corpus switcher,
the agent pipeline's output never surfaces in the UI, `api/main.py` has zero
real callers) -- `web/` finally uses the FastAPI seam OD12 built and OD15 left
unconsumed, against the new `/fields/*` endpoints (`ARCHITECTURE_NOTE.md`).

Streamlit is NOT deleted -- it stays as an internal "lab console" against the
pre-existing non-field-scoped endpoints/modules, per `PRODUCT.md`.

## OD34 — Gap/community labels prefer agent `paper_concepts`, fall back to OpenAlex `concepts`

**Context.** Flagged as an open question during stage-3 Phase C wireframing (PROGRESS.md):
`gaps.py`'s `community_labels()` labeled every community from the raw OpenAlex `concepts`
table, which is coarse/noisy as user-facing copy ("Philosophy", "Hop (telecommunications)").
OD20 already keeps agent-extracted concepts in a separate `paper_concepts` table (never
merged into `concepts`), with their own synonym-merge step (`link()` in
`axiom/agents/linking.py` keeps one row per `(canonical, kind)`, so "RAG" and
"retrieval-augmented generation" mentions in the same paper collapse to one canonical label)
-- a materially cleaner label source, but only populated for papers the agent pipeline has
actually run on (50/300 in the real corpus as of this decision).

**Decision.** `community_labels()` now runs its existing TF-IDF-across-communities scoring
twice -- once over `paper_concepts.canonical`, once over `concepts` (OpenAlex, canonicalized
via OD14's `concept_canonical` table as before) -- and picks per community: agent concepts
when a community has at least `GAP_LABEL_MIN_AGENT_CONCEPTS` (3, `axiom/config.py`) distinct
canonicals, OpenAlex otherwise. A community's label source is never mixed mid-community. 3
was picked to match `top=4` labels per community (roughly "enough distinct concepts to fill
the label list"), not swept against a target -- same spirit as OD23's manual
`VERIFIER_COSINE_MIN`, revisit if real usage shows it's too strict/loose once agent-pipeline
coverage grows past the current 50-paper pilot subset.

Verified against the real 300-paper corpus (`docs/product/gap-labeling/gaps_before.png` vs
`gaps_after.png`): before, every visible gap headline read "Natural language processing
cluster" or "Philosophy cluster" (near-useless, the TF-IDF generic-filter already does the
best it can with only 5-ish distinct OpenAlex labels across most communities); after, the
same gaps read "Language model cluster", "chatgpt cluster", "bridging operation cluster",
"natural questions cluster" -- distinctive, agent-sourced, for every community that had
enough agent coverage. No schema change (`paper_concepts` already existed, OD20); no API
contract change (`GapAnalysisOut`/`TopicOut.labels` unchanged shape, just better content).

## OD35 — Map screen: PCA (not UMAP) for the embedding projection, PageRank cap at 500 nodes

**Context.** The Map screen (stage-3 product redesign) needs a 2D/3D layout for paper
embeddings, computed server-side and cached, plus a citation-graph 3D view capped for laptop
GPUs. The task allowed UMAP "if it installs without new heavy deps."

**Decision.** PCA via `sklearn.decomposition.PCA` (`axiom/embedding_map.py`) -- `scikit-learn`
is already pinned in `requirements.txt` and unused elsewhere, so this is zero new backend
dependencies. UMAP (`umap-learn`) needs `numba` + `pynndescent`, breaking the repo's
established "no new heavy deps" bar (`gaps.py`: "Pure-Python: networkx + numpy only").
Deferred, not decided against -- revisit if a real need for UMAP's better global structure
shows up once fields grow past a few hundred papers. PCA is deterministic (fixed
`random_state=0`), which is what makes caching the projection per field_id correct.

Both Map views (citation graph, embedding map) cap at `config.MAP_NODE_CAP` (500) nodes via
the same `graphmod.top_influential_subgraph` (PageRank) already used for `/graph/subgraph` --
no new ranking logic, same notion of "most important papers" everywhere in the app. At the
real 300-paper corpus this never triggers (300 < 500); verified the capping path itself
(`top_influential_subgraph` + `embedding_map.project` together) against the real corpus with
a smaller n=50 cap, confirming it returns exactly 50 ranked nodes with projected coordinates
-- the same two calls `_field_map_data` chains, just with a cap small enough to actually bite.

`/fields/{field_id}/map` is the first field-scoped endpoint to actually use the
per-field-`lru_cache` pattern the stage-3 backend-additions note flagged as a reasonable
future follow-up (PROGRESS.md Session 2) -- `_field_map_data` is `@lru_cache(maxsize=8)`
keyed by `field_id`, safe because the PCA projection is deterministic and a field's corpus
doesn't change shape between builds.

## OD36 — Stage 4: a single prebuilt base corpus, narrow niche, resumable at paper/budget granularity

**Context.** Every prior stage downloaded a fresh per-field corpus on demand (snowball + embed +
extract, minutes, real $). Stage 4 wants fields to be instant -- the slow work moves ahead of time
into one explicit, resumable script (`scripts/build_corpus.py`) building ONE shared corpus.

**Decision (A1 scope).** Domain: retrieval / RAG / QA / LLM knowledge use, 2019-2026 -- narrow by
design so a modest size (1000, `--size`, grows to 20k with no code change) stays dense rather than
diffuse. Source: OpenAlex works with an abstract, `search=` (relevance ranking) + `filter=` (date
range, `has_abstract:true`, venue allow-list resolved to source ids -- see OD38), cursor-paginated
past the ~10k offset-pagination ceiling. Storage: `data/corpus/corpus.db` (same `db/schema.sql`
shape as every other db this app writes) + Qdrant collection `corpus_v1`, fully separate from both
the eval corpus (OD11) and any legacy per-field db.

**Resumability semantics, decided explicitly rather than left implicit:** the 6 steps (fetch/store/
embed/edges/extract/canonicalize) checkpoint in a new `corpus_meta` table (`steps_json`), mirrored
to `data/corpus/manifest.json`. Fetch+store+edges share one pass (`ingest.write_works`, one paper ->
one SQLite transaction covering all three) and are resumable at PAPER granularity: a paper already
in `papers` is skipped via `OpenAlexClient.search_relevant`'s `skip_ids`, so an interrupted fetch
just re-walks pages faster on rerun, never redoes real work or re-spends budget. Embed is resumable
at STEP granularity only (`axiom.indexer.reindex_qdrant` already recreates the whole collection,
not incremental) -- accepted because it's bounded local compute, not metered spend or unbounded
network; a kill mid-embed just re-embeds the whole corpus next run. Extract is resumable at PAPER
granularity via existing state, not a counter: "needs extraction" = no `paper_concepts` row yet,
regardless of which `run_id` a prior successful extraction used, so a `--budget`-limited partial run
always continues exactly where it left off next invocation, and the global `LLM_BUDGET_USD` cap
(B4's "fix" target: that cap is cross-run/global, not per-run) is sidestepped entirely by giving
`build_corpus.py` its own invocation-scoped budget check (`spent_already` read straight from the
corpus db's own `llm_spend` table, not the global ledger).

**Real bug found and fixed running this for real: `canonicalize`'s 1500-token budget overflows on a
real batch.** `AGENT_MAX_TOKENS["canonicalize"]` (OD14) was sized for "a 50-label batch," but
`canonicalize.py`'s own schema requires every ungrouped label to appear as its own singleton group
in the JSON output (`{"canonical": "X", "members": ["X"]}`) -- a real 56-label batch with few true
synonyms repeats most labels twice, overflowing 1500 tokens and hitting `LLMMaxTokensError` on the
very first real run. Bumped to 4096 (matches `_DEFAULT_MAX_TOKENS`); same "model/schema got more
verbose than the original estimate assumed" pattern OD28 already hit twice for other agents.

## OD37 — A4 instant fields: relevance-floor slicing, calibrated against the real corpus (and its real limits)

**Context.** A4 wants `POST /fields` to return in seconds: a relevance-ranked slice of the shared
base corpus (`axiom/corpus_slice.py`) instead of a download. A field becomes a saved
`(topic, paper_ids)` pair -- `fields.db_path`/`collection` point at the SHARED corpus unchanged
(never copied); a new `paper_ids_json` column is what actually scopes it. `api/main.py`'s
`_field_graph`/`_field_vectors` branch on its presence: a slice field gets
`get_corpus_graph().subgraph(paper_ids)` (a citation to a paper outside the slice drops out
entirely, same effect "outside the corpus" already has elsewhere) and the cached
`get_corpus_vectors()` dict (no live Qdrant scroll per field -- the actual mechanism behind
"instant"); a legacy/download field (`source='download'`, e.g. "default") is untouched, same
behavior as every prior stage.

**The floor itself, decided from REAL measurement, not a guess.** First pick was 0.80 dense SPECTER2
cosine (`config.QUERY_EXPANSION_TEMPLATE`-wrapped query vs. every corpus paper's stored vector,
`corpus_slice._cosine_to_all` — one vectorized matrix-vector product, not a per-candidate Python
loop: ~40x faster at 1000 papers, which matters for A7's p50 target). Run for real against the live
1000-paper corpus, 0.80 was USELESS: "protein structure prediction" and "medieval european history"
both cleared it for 300+ papers each — the exact flat-score compression `QUERY_EXPANSION_TEMPLATE`'s
own comment already warns about ("terse queries land in a mushy region... flat ~0.82-0.88"), here
biting a NEW feature the original note never anticipated. Swept 0.85/0.88/0.90/0.92 against 3
in-niche and 4 off-niche probe topics (paper counts clearing each floor, logged in this stage's
session notes): 0.90 is the first floor with real separation -- in-niche topics land 36-634 papers,
3 of 4 off-niche probes land under `FIELD_MIN_PAPERS` (20) ("medieval european history" at 40 is a
known false positive, not eliminated). **Accepted, not solved**: SPECTER2 cosine doesn't cleanly
separate every off-niche query from a narrow-by-design corpus where most papers already share
substantial NLP/LLM vocabulary -- same root cause Part B's planned percentile-based gap-similarity
fix exists for for a different feature. A5's "thin coverage" is honest about what it catches, not
a guarantee it catches everything.

**Real measurement (A7, `scripts/measure_field_latency.py`, `eval/field_latency.json`):** 20 topics
(16 in-niche, 4 off-niche) against the real 1000-paper corpus, vectors+graph loaded once (process
singletons, same pattern `get_graph()`/`get_store()` already use) and reused across all 20 --
p50=0.164s, p95=0.389s, far under the 5s target. "protein structure prediction" (8 papers) correctly
landed 'thin'; every in-niche topic landed 'ready'.

## OD38 — OpenAlex requires an API key at any real volume; venue filtering moved server-side

**Context.** The original ingest-layer assumption (OD7-era, `config.py`'s own comment: "No API key
needed; OpenAlex asks only for a mailto") no longer holds: OpenAlex now caps KEYLESS traffic to a
shared, per-IP $0.10/day budget. The first real 1000-paper build attempt hit `401 Unauthorized`
after only 21 papers -- not a bug, a real policy change undocumented anywhere in this repo.

**Decision.** `OPENALEX_API_KEY` (env var, free key from openalex.org/settings/api, its own $1/day)
replaces `OPENALEX_MAILTO` everywhere (`OpenAlexClient.__init__`/`_get`) -- `OPENALEX_MAILTO` is
deleted outright, not deprecated-and-kept, since it pointed at a teammate's address left over from
the original team-project stages and keyless traffic is no longer viable at any real volume anyway.
`build_corpus.py` fails fast with a clear message if no key is set, rather than silently fetching a
fraction of what was asked.

**Also fixed while at it: venue filtering moved server-side.** The old `make_venue_filter`/`keep()`
pattern over-fetched whole /works pages and discarded most results client-side (the actual cause of
burning the $0.10 keyless budget on only 21 admitted papers) -- `OpenAlexClient.resolve_source_ids`
resolves `config.RELEVANT_VENUE_PATTERNS` to concrete source ids ONCE (6 small `/sources` lookups),
baked into `filter=primary_location.source.id:S1|S2|...` so every `/works` page fetched counts fully
toward `--size`. Real effect on the actual 1000-paper build: fetched exactly 1000/1000 in 5 pages
(vs. the first attempt's 21 papers across several pages before exhausting budget). Also bumped
`OPENALEX_BATCH_SIZE` 50 -> 100 (OpenAlex's own OR-filter cap) for `works_by_ids` (snowball/`--extend`
path) -- fewer requests for the same referenced-works lookups.

## OD39 — A4 slice fields: "rising concepts" wasn't scoped to the slice at all

**Context.** Found running a post-build sanity check on the real corpus (per-user request, three
topics' fields): `GET /fields/{id}/overview`'s `rising` concepts were IDENTICAL across three
completely different topic slices. Root cause: `axiom.velocity.compute_velocity` has no paper-id
filter, and `field_overview`/`field_shortlist_create` both call it with `conn = db.connect(row
["db_path"])` -- for a slice field that `db_path` is the SHARED base corpus (OD37's whole point:
no copy), so velocity always scanned the entire 1000-paper corpus regardless of which field asked.
`key_papers`/`topics`/gaps were all correctly scoped already (they go through `_field_graph`'s
sliced subgraph); velocity was the one path that read straight off `db_path` with no slice
awareness at all.

**Decision.** `compute_velocity`/`get_top_velocity_keywords` gained an optional `paper_ids`
filter (restricts `paper_year` construction; downstream concept-counting already skips any
paper not in that dict, so no second filter point needed). `field_overview` and
`field_shortlist_create` (the "Draft direction" trend-context grounding) both now pass
`_slice_paper_ids(row)` -- `None` for a legacy/download field, same behavior as before. Verified
against the real corpus: the three probe topics now return genuinely different rising-concept
lists. **Known, separate, pre-existing issue -- not fixed here:** the lists are still mostly noisy
raw OpenAlex concept labels ("Feature (linguistics)", "Field (mathematics)") rather than the
agent-extracted canonical concepts OD34 already prefers for gap labels specifically -- `velocity.py`
still reads only the raw `concepts` table. Flagged for Part B, not expanded into here (this OD is
about the scoping bug, not a second pass at label quality).

## OD40 — Stage 4 B4: the LLM budget cap moves from process-lifetime to per-run

**Context.** `config.LLM_BUDGET_USD` was a single, cross-run, process-lifetime total (a
Phase-4 architecture review decision, with its own regression test asserting the design
was intentional: "two run_ids, each individually under budget, must still trip the cap
once their combined spend reaches it"). That's defensible for a short-lived eval/bench
script invoked once and exited. It's wrong for a long-lived API process: the real base
corpus build this stage already ran spent $0.41 of the $3 default into that SAME shared,
cumulative pool -- every later call sharing that process (a live "Draft direction" request,
a future `--update`) would draw against what's left of ONE $3 lifetime allowance, with an
early, unrelated run able to fully starve a later, independent one of budget it never
touched.

**Decision.** `axiom/llm_providers.py`'s budget check (`_committed_spend`/`_reserve_budget`/
`_release_budget`) is now scoped to `run_id`: `db.run_llm_spend(conn, run_id)` (durable,
per-run) replaces `db.total_llm_spend(conn)` (durable, cross-run) as the committed-spend
figure; the in-process reservation ledger becomes `_reserved_by_run: dict[str, float]`
instead of one process-wide float. `config.LLM_BUDGET_USD` (unchanged default, $3.00) now
means "per run_id," matching what `scripts/build_corpus.py --budget` (A2, OD36) already did
at the script level -- this closes the same gap one layer down, inside every individual
LLM call the agent pipeline/API make. The four tests that encoded the OLD cross-run
contract (`test_budget_is_global_not_per_run`, the reservation-ledger tests) are rewritten
to assert the opposite -- different run_ids are isolated; the SAME run_id still correctly
accumulates and trips its own cap, durably, concurrently-safely.

**Shown in the UI.** `corpus_meta` gained a `budget_usd` column (the actual `--budget` the
last build ran under, not a guessed constant) and `GET /corpus` gained `budget_usd`/
`extraction_complete`; the app footer shows "extraction stopped early ($X of $Y budget)"
whenever a corpus build's extract step landed `'partial'` instead of `'done'`.

**Trade-off, stated plainly:** the old design's real safety property -- "this dev box can
never spend more than $3 total, ever, no matter how many runs happen" -- is gone. What
replaces it is "each individual run (a corpus build, a hypothesis draft, an agent-pipeline
invocation) can't exceed its own $3" -- the number of runs in a day is now unbounded in
aggregate. Accepted: `scripts/build_corpus.py` already has its own explicit `--budget`
pre-flight confirmation (A2) for the one genuinely expensive, deliberately-triggered
operation; the live API's own per-request calls (a single hypothesis draft, ~$0.01) are
cheap enough that per-run $3 ceilings on each are not a meaningful aggregate risk.

## OD41 — Review fix: "barely cites" only when the cross-citation share is actually low

**Context.** The gap headline ("X cluster barely cites the Y cluster") used that phrase for
EVERY candidate gap unconditionally. Real field data (measured against an actual field built
this stage, 86 in-corpus citations total): the 6 candidate gaps' `inter_citations` were
1/2/3/7/8/14, i.e. cross-citation shares of 1.2%/2.3%/3.5%/8.1%/9.3%/16.3%. Calling a pair
that shares 14 of a field's 86 total in-corpus citations (16.3%) "barely" connected is not
accurate -- it's a real, substantial citation link, just not a DENSE one; "barely" only
describes the low end of that real spread.

**Decision.** `share = inter_citations / total_in_corpus_citations`; `share < 0.05` keeps
"barely cites" (covers 1.2%/2.3%/3.5% from the real measurement above -- genuinely sparse),
`share >= 0.05` switches to neutral, still-specific phrasing: "shares N citations with"
(covers 8.1%/9.3%/16.3% -- a real connection, honestly stated as one). 5% isn't swept
against a labeled target (same spirit as OD23's `VERIFIER_COSINE_MIN`, OD37's relevance
floor) -- it's picked to cleanly separate the one real field's measured spread into two
groups that read right in plain English, and is cheap to revisit once more fields exist to
check it against. `GapAnalysisOut` gained `total_in_corpus_citations` (computed once per
field, not per gap) so `GapCard`'s list view can apply the same rule as `GapDetail`'s single-
gap view, which already had the figure.

## OD42 — External OpenAlex corpus-artifact check per gap (B1)

**Context.** A gap candidate can be numerically well-formed -- two sizable, semantically
close, weakly-cited communities -- purely because this field's corpus is a few hundred
papers, not because the two areas are actually disconnected in the wider literature. The
in-corpus citation count alone can't tell the two cases apart.

**Decision.** On-demand (a button, not computed for every gap on page load -- a live
OpenAlex request per gap would slow the gap list down for a signal most users won't ask
for every time), `GET /fields/{id}/gaps/{gap_id}/external-check` counts OpenAlex works
whose title+abstract mention BOTH sides' top label (`title_and_abstract.search`'s
multi-word filter value requires every word present, unlike the top-level `search=`
param's OR-ranked relevance match -- so a single filter call is literally "works that
combine both concepts," not an approximation). `>= 50` flags `is_likely_artifact`.

Calibrated against 5 real label pairs from the live corpus's actual gap candidates:
"GPT-3"+"retrieve-and-read" -> 26, "dense passage retrieval"+"retrieve-and-read" -> 9,
"2WikiMultihopQA"+"ScienceQA" -> 0 (genuinely narrow pairings, real works are scarce) vs.
"hallucination"+"retrieval-augmented generation" -> 4655, "chain of thought prompting"+
"reasoning" -> 3882 (both sides are themselves broad, independently well-established
areas -- not disconnected in the wider literature at all, just in this corpus's small
citation subgraph). 50 sits cleanly in the gap between those two real clusters; same
"accepted, not fully solved" spirit as OD37's relevance floor -- revisit once more real
gaps exist to calibrate against.

## OD43 — Distributed agent workers: Redis Streams, single-writer scope, and what didn't move

**Context (AXIOM_DISTRIBUTED_TASK.md).** Make each agent role (coordinator/extraction/
enrichment/linking) an independent process so the pipeline scales horizontally
(`docker compose up --scale worker-extraction=3`), keeping today's in-process
`asyncio.gather` path (`axiom/agents/pipeline.py`) as the default.

**Decision: plain Redis Streams + consumer groups, not a framework (Celery, etc.).**
The task's own acceptance bar -- acks, consumer groups, `XAUTOCLAIM` reclaim after a
visibility timeout, a dead-letter stream after N failed deliveries -- is exactly
Streams' native primitive set, not something a task-queue framework adds on top. A
framework would add a dependency, its own retry/serialization conventions (fighting
the existing Pydantic contracts, which the task explicitly says to reuse verbatim),
and a second thing to learn, for zero capability this corpus's scale needs. Redis is
already the one new service added (`docker-compose.yml`); Streams is a few commands
on it, not a new system.

**Retry/dead-letter design: `attempt` lives in the message, not Redis's internal PEL
delivery count.** On a *handled* failure (the agent call raised), the worker
re-`XADD`s the SAME message onto its own input stream with `attempt+1`, then `XACK`s
the original -- simple, portable across redis-py versions, and exactly what the task's
own message schema implies by listing `attempt` as a field. `XAUTOCLAIM` is for the
other failure mode entirely: a worker that goes dark *mid-message* (SIGKILLed, not a
handled exception) leaves its message stuck in the consumer group's PEL, un-acked,
un-retried by the logic above. A periodic `XAUTOCLAIM` pass (`STREAM_CLAIM_IDLE_MS`,
default 30s) reclaims anything idle that long onto a live consumer, which then runs
it through the exact same handler/retry/dead-letter path as a fresh message. Verified
for real (`tests/test_workers_integration.py` + `scripts/bench_distributed.py`'s
fault-injection scenario): kill an extraction worker mid-run, every paper still
finishes.

**Single-writer SQLite -- scoped to the corpus-write tables, not literally "linking
touches a connection and nothing else does."** `coordinator` and `enrichment` never
call an LLM at all (`coordinator.route()` is rule-based; B1 already moved entity
matching off an LLM onto a local phrase-model cascade) and never open a DB connection
-- fully stateless, matching the task's "can run anywhere" framing exactly. `linking`
is the only role that writes `agent_runs`/`agent_run_items`/`paper_concepts`/
`concept_cooccurrence` -- every terminal outcome from every stage (skip, rejected,
error, accepted) is forwarded to it as a message, not written locally by whichever
stage produced it, so those tables really do have exactly one writer.

`extraction` is the one deviation: it's the sole LLM-calling role, and
`llm_providers.complete_json`'s existing durable budget check needs a live sqlite
connection to `llm_spend` to enforce the cap *before* spending, not after. Funneling
that through linking (the task's "which the linking side records" phrasing) would mean
extraction could spend unboundedly before any check ever ran -- defeating the budget
system's entire purpose. Extraction keeps its own connection, scoped only to
`llm_spend` (an append-only ledger table, not touched by the single-writer concern the
task is actually protecting against -- concurrent *upserts* to `paper_concepts`).
Documented here as a deliberate, narrow divergence rather than silently reinterpreted.

**What didn't move: synthesis.** Cluster labeling and hypothesis pitches
(`axiom/agents/synthesis.py`) need every paper in a run to have finished linking
first -- data that doesn't exist yet at the moment distributed mode's `run_pipeline()`
call returns (it only enqueues). Out of this task's explicit scope (coordinator/
extraction/enrichment/linking only); distributed mode silently skips `clusters`/`gaps`
rather than pretending to support them. A follow-up would need either a 5th
"synthesis" role triggered once `agent_runs.papers_done == papers_total`, or a
separate polling script -- not attempted here.

**Real measurement:** `eval/distributed_bench.json` (`scripts/bench_distributed.py`),
real Redis, real subprocesses (not Docker -- same code path, no image-build time
spent to measure it), real LLM calls on 40 real papers from `data/axiom.db` (scaled
down from the task's suggested 200 after this environment's background-process
supervision proved unreliable for one long multi-scenario run -- see
`docs/distributed/RESULTS.md`'s own note), cost capped and estimated before
spending. Real 3-worker run ~2.07x the throughput of 1 worker, not 3x (enrichment/
linking stay single-instance and become the shared bottleneck); fault injection
(kill an extraction worker mid-run) -- every paper still finished.

**`STREAM_CLAIM_IDLE_MS` real bug.** The first real 3-worker run (not the numbers
kept) came back with `papers_done` exceeding `papers_total` -- a message still
being legitimately processed got `XAUTOCLAIM`'d by a second consumer and double-
processed, because the original 30s guess was shorter than real per-paper
processing latency under contention (p95 84-164s, real LLM calls). Raised to 180s
with the real data in `axiom/config.py`'s comment; re-run came back clean
(`reclaim_count=0` for the non-fault-injected runs, `1` for the one that
deliberately killed a worker). A reminder that this value has to be tuned against
real observed processing latency, not guessed.

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
| 2026-11-01 | Added `corpus_meta` (single-row build manifest) to the shared schema.sql — stage 4, OD36 | stage4 track |
| 2026-11-01 | `fields.db` (a separate db, not schema.sql): added `source`, `query`, `paper_ids_json`, `relevance_floor`, `total_relevant` columns to `fields` — stage 4 A4, OD37 | stage4 track |
| 2026-11-01 | Added `budget_usd` to `corpus_meta` — stage 4 B4, OD40 | stage4 track |
