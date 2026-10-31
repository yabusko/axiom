# Axiom P1 — Complete Walkthrough

> **⚠️ Historical, P1-only.** Written 2026-06-04 for the very first
> data-foundation seam. Everything in its own §13 "P2 seams" table has since
> been built — velocity (OD10), citation graph + gaps (OD9), FastAPI (OD12),
> keyword canonicalization + reading-list summaries (OD14) — see
> `docs/DECISIONS.md` (OD6–OD14) for what actually shipped, `README.md` for
> the current run order, and `docs/demo_examples.md` for real output of every
> built feature. This file is kept as-is below for its still-accurate detail
> on the original SQLite/Qdrant/SPECTER2 seam and the search-quality upgrades
> — just don't read its "out of scope" claims as current.

A full, self-contained explanation of the **P1 data-foundation seam** and the
**search-quality upgrades** built on top of it: what each piece is, why it
exists, how the data flows, how to run it, and how every claim was verified with
real runs. Read it top-to-bottom and you should understand every file in the repo
and be able to reproduce the demo.

- **Audience:** anyone on the Axiom team (UI, data, graph) plus future-you.
- **Status:** P1 built and verified 2026-06-04; search-quality upgrades
  (sane defaults → query expansion → hybrid dense+sparse) built and verified the
  same day.
- **Scope:** P1 only. Velocity engine, citation graph, gap pipeline, React, and
  FastAPI are explicitly *not* here — see §13 for where they plug in later.

---

## 1. What Axiom is (and what it is not)

**Axiom is a thesis-discovery tool for researchers.** Its purpose is to surface
*what is trending* in a subfield and *what conceptual space is unexplored* — i.e.
to help a researcher find a thesis-worthy gap.

It is **not** a paper search engine. Semantic search over papers (what P1 builds)
is the *foundation* the later trend/gap features stand on, not the product itself.

**P1 — the "data-foundation seam"** — is the minimum that lets the UI track work
end-to-end before real ingestion and the production encoder exist:

1. A SQLite schema that is the **shared data contract** for the whole team.
2. A local **Qdrant** vector store (via Docker) + a Python client wrapper.
3. A **SPECTER2** abstract encoder (768-dim) with a CPU fallback.
4. A **bootstrap** that loads a synthetic 30-paper corpus so the UI runs today.
5. One **Streamlit** page: semantic search box + venue/year filter bar.

On top of that foundation, three **search-quality upgrades** (§6) make retrieval
actually good for the kinds of queries a researcher types — including terse ones
("LoRA", "RAG") that pure dense search handles badly.

---

## 2. Architecture & data flow

```
                          ┌──────────────────────────────────────┐
   OpenAlex  (P2, not yet) │   db/schema.sql                       │
        │                  │   THE SHARED DATA CONTRACT            │
        │ ingestion (P2)   │   papers · concepts · citation_edges  │
        ▼                  │          · paper_provenance           │
  ┌───────────────┐        └──────────────────────────────────────┘
  │ bootstrap_     │                    │ defines tables
  │ synthetic.py   │  writes rows       ▼
  │ (30 fake       ├───────────────►  SQLite  (data/axiom.db)
  │  NLP papers)   │                    │
  └──────┬─────────┘                    │ reads title+abstract, concepts
         │                              ▼
         │   per paper, TWO vectors:
         │   ┌─────────────────────────────────────────────┐
         │   │ DENSE: axiom/embed.py (SPECTER2, 768-dim,     │
         │   │        CLS over "title [SEP] abstract")       │
         │   │ SPARSE: axiom/sparse.py (BM25-style,          │
         │   │        hashing trick, pure Python)            │
         │   └───────────────────────┬─────────────────────┘
         │                           │ named vectors + payload
         └──────────────────────────▼─────────────────────┐
                          ┌─────────────────────────────────┐
                          │ axiom/qdrant_client.py          │ collection "axiom_v1"
                          │ AxiomQdrant                     │ 1 point / paper, 2 named
                          │  search()         (dense only)  │ vectors: dense + sparse
                          │  search_hybrid()  (dense+sparse │ payload: paper_id,title,
                          │                    RRF-fused)    │ year,venue,cited_by,concepts
                          └───────────────┬─────────────────┘
                                          │
   query text ──► axiom/embed.py          ▼
   encode_query() (expansion) ──► app/streamlit_app.py
                                  query box + venue multiselect + From/To year
                                  (prefers search_hybrid when index supports it)
```

**The two "contracts" everyone codes against** (frozen first, before anything
else was built):

1. `db/schema.sql` — the table/column names.
2. `AxiomQdrant.search(query_vector, top_k, venues, year_range) -> list[SearchHit]`.

Changing either ripples across tracks, so changes are logged in
`docs/DECISIONS.md`. New capability (like hybrid search) is added as **new
methods**, never by changing the frozen `search` signature.

---

## 3. The SQLite contract (`db/schema.sql`)

SQLite is the **durable source of truth** for paper metadata, concepts, and
citation edges. Qdrant holds only vectors + a denormalized payload for fast
filtering; if Qdrant is wiped, it can be fully rebuilt from SQLite.

### Tables

| Table | Purpose | Key columns |
|---|---|---|
| `papers` | one row per paper | `openalex_id` (PK), `title`, `abstract`, `publication_year`, `venue_id`, `venue`, `cited_by_count`, `doi` |
| `concepts` | many concepts per paper | `paper_id`, `concept`, `level` |
| `citation_edges` | citation graph edges (P2) | `src_id`, `dst_id`, `year` |
| `paper_provenance` | fetch bookkeeping | `paper_id` (PK), `source`, `fetched_at`, `text_hash`, `has_fulltext` |

### Design decisions baked into the schema

- **Abstracts are plain text here.** OpenAlex stores abstracts as an *inverted
  index* (`{word: [positions]}`). Ingestion (P2) must reconstruct them to plain
  text *before* inserting into `papers.abstract`. The encoder assumes plain text.
- **`citation_edges.dst_id` has no foreign key — on purpose.** A paper's
  `referenced_works[]` routinely cites papers outside the local corpus. Those
  "dangling" edges are exactly what the graph track needs, so the schema must not
  reject them. `src_id` *does* have an FK (the citing paper is always in-corpus).
- **Five read-path indexes** (`publication_year`, `venue`, `concepts.paper_id`,
  `citation_edges.src_id`, `citation_edges.dst_id`). These make the UI filter bar
  and the P2 graph reads index lookups instead of full scans.
- **`text_hash`** (SHA-256 of the abstract) supports dedup / change-detection
  during real ingestion.

The schema was **not touched** by the search-quality upgrades — those live
entirely in the embedding/vector layers.

---

## 4. The search contracts (`AxiomQdrant`)

### 4a. `search` — frozen, dense-only

```python
def search(
    query_vector: list[float],            # 768-dim SPECTER2 embedding of the query
    top_k: int = 10,
    venues: list[str] | None = None,      # payload filter: match ANY of these (OR)
    year_range: tuple[int, int] | None = None,  # inclusive (min, max) on payload.year
) -> list[SearchHit]:
    ...
```

```python
@dataclass
class SearchHit:
    paper_id: str
    title: str
    year: int
    venue: str
    cited_by_count: int
    score: float
    concepts: list[str]
```

- **Takes a raw vector, not text.** The vector store stays decoupled from the
  encoder; the Streamlit page owns the `encode → search` wiring.
- **Filter semantics:** facets combine with **AND**; within `venues`, matches are
  **OR**. Filters are applied **natively by Qdrant**, so `top_k` is honored
  *after* filtering.
- **Typed `SearchHit`:** the UI never touches raw Qdrant objects.
- This signature is **frozen**. The upgrades kept it byte-for-byte; the only
  internal change is that it now targets the **named** dense vector (§6c).

### 4b. `search_hybrid` — new, dense + sparse

```python
def search_hybrid(
    query_vector: list[float],   # dense SPECTER2 query embedding
    query_text: str,             # raw query string, used to build the sparse vector
    top_k: int = 10,
    venues: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
) -> list[SearchHit]:
    ...
```

Runs a dense search and a sparse (BM25-style) search, then fuses them with
**Reciprocal Rank Fusion** in Python. Added as a *new* method so the frozen
`search` contract is untouched (§6c). The score on returned hits is the RRF fused
score, **not** a cosine similarity (see the note in §6c).

### 4c. Helpers

- `supports_hybrid()` — True if the collection has a sparse vector configured.
  The Streamlit page calls this to decide between `search_hybrid` and `search`.
- `recreate_collection()` — drops & recreates `axiom_v1` with two **named**
  vectors (`dense` + `sparse`). Used by the idempotent bootstrap.
- `upsert_papers(points, vectors, sparse_vectors=None)` — one point per paper;
  stores the named dense vector and, when provided, the named sparse vector.

---

## 5. File-by-file

```
db/schema.sql                    Shared SQLite contract (§3). Unchanged by upgrades.

axiom/config.py                  Single source of truth for constants:
                                 - DB_PATH, SCHEMA_PATH
                                 - QDRANT_HOST/PORT, COLLECTION_NAME = "axiom_v1"
                                 - MODEL_ID = "allenai/specter2_base", VECTOR_SIZE = 768
                                 - DEFAULT_TOP_K = 10
                                 - QUERY_EXPANSION_TEMPLATE  (Step 2)
                                 - DENSE/SPARSE_VECTOR_NAME, SPARSE_VOCAB_SIZE,
                                   BM25_K1/B, RRF_K, HYBRID_CANDIDATES,
                                   SPARSE_STOPWORDS  (Step 3)

axiom/db.py                      SQLite access layer. connect/init/insert helpers,
                                 distinct_venues(), year_bounds(), iter_papers().
                                 TODO(P2): OpenAlex ingestion plugs in here.

axiom/embed.py                   Specter2Encoder. Loads allenai/specter2_base via
                                 transformers; encode() = [CLS] over
                                 "title [SEP] abstract" (the SPECTER recipe).
                                 Device order CUDA > MPS > CPU (CPU warns).
                                 encode_query() = QUERY-SIDE path (Step 2): wraps
                                 terse queries in an abstract-shaped sentence.
                                 query_path is logged at init
                                 ("[embed] query path: expansion").

axiom/sparse.py                  NEW (Step 3). BM25-style sparse encoder, pure
                                 Python, zero new deps. tokenize(), term_index()
                                 (stable BLAKE2 hashing trick), BM25SparseEncoder
                                 (fits IDF + avgdl, encodes documents),
                                 encode_query_sparse() (presence weights).

axiom/qdrant_client.py           AxiomQdrant wrapper (§4). Named-vector collection
                                 lifecycle, upsert (dense + optional sparse),
                                 count(), supports_hybrid(), search() [frozen,
                                 dense], search_hybrid() [dense+sparse, RRF].
                                 Defines SearchHit.

scripts/bootstrap_synthetic.py   The 30-paper synthetic corpus (inline data) plus
                                 load_sqlite() and load_qdrant(). Builds BOTH the
                                 dense and sparse vectors. IDEMPOTENT (§7).

scripts/eval_search.py           NEW. Search-quality eval harness: runs a fixed
                                 10-query set across modes (raw / expansion /
                                 hybrid) and prints top-5 per query (§6, §11).

app/streamlit_app.py             The UI (§8). Search box, venue multiselect,
                                 From/To year selectboxes, Top-K. Prefers
                                 search_hybrid when supported.

docker-compose.yml               Qdrant service only (qdrant/qdrant:v1.9.0).
requirements.txt                 Pinned, minimal: torch==2.2.2,
                                 transformers==4.40.2, qdrant-client==1.9.0,
                                 streamlit==1.34.0, pandas, numpy, tqdm.

docs/DECISIONS.md                Committed decisions + the schema change log.
docs/WALKTHROUGH.md              This file.
README.md                        Quick run order + the 5 demo queries.
```

> Note: `qdrant/populate_qdrant.py` is a **separate earlier experiment** from
> another teammate (different model `all-MiniLM-L6-v2`, 384-dim, collection
> `academic_papers`). It is **not** part of the P1 seam and is left untouched.

---

## 6. Search-quality upgrades (the three steps)

**The symptom.** Short/vague queries ("TOOLS for AI", "LoRA") returned a flat,
near-random ranking, while full research-phrased queries ranked cleanly. **Root
cause:** the query vector was embedded as if it were a *document*, so terse
queries landed in a mushy region of the space. Acronyms that pure dense search
can't see (LoRA, RAG) made it worse.

The fix was three changes, applied **in order**, verified and committed after
each so a later step can't mask an earlier regression.

### 6a. Step 1 — sane demo defaults

`app/streamlit_app.py`:
- **Top-K** defaults to `DEFAULT_TOP_K = 10` (not the corpus size), and its cap is
  driven by the actual point count (no magic `30`).
- The **year filter** defaults to the full `[min, max]` span — nothing silently
  filtered.
- The **venue** multiselect defaults to empty = all venues.

Cheap, can't-fail, and it stops the UI from dumping a flat 29-row list.

### 6b. Step 2 — fix the query-side embedding

Goal: query vectors should live in the same region as document vectors.

**Preferred approach (not used) — SPECTER2 ad-hoc query adapter.** The proper fix
is the adapter trained for query→document matching. But the `adapters` library
pins `transformers~=4.57`, which conflicts with our committed
`transformers==4.40.2`. Installing it would force a pin bump (a committed-decision
change), so **we do not install it.** `Specter2Encoder` *attempts* the import so
the code is adapter-ready, then falls back and logs the active path.

**Active approach — query expansion.** `encode_query()` wraps a terse query in an
abstract-shaped sentence before encoding:

```
config.QUERY_EXPANSION_TEMPLATE
  = "This paper presents {query} for natural language processing."
```

This nudges the query vector toward the document region. **Document-side encoding
is unchanged**, so existing corpus points stay valid — no re-encode needed for
this step.

**Result (raw vs expansion, top-1):** all 5 full-phrased queries keep clean
top-1 (no regression; the PEFT query even promotes the original *LoRA* paper to
#1). Terse queries mostly improve — `RAG` goes from a wrong top-1 to the
*Retrieval-Augmented Generation* paper at #1. The one case expansion **can't**
fix is the bare acronym `LoRA` (the template dilutes the token) — which is
exactly what Step 3 is for.

### 6c. Step 3 — hybrid retrieval (dense + sparse)

Add a sparse keyword arm so exact terms/acronyms dense misses (LoRA, RAG, dataset
names) still match.

- **`axiom/sparse.py`** builds BM25-style sparse vectors with the *hashing trick*
  (a stable BLAKE2 hash maps tokens → indices, so no vocabulary file has to be
  persisted or shared). Document side carries full BM25 weight (IDF ×
  tf-saturation × length norm); query side carries presence weights (1.0).
  Because IDF is baked into the doc weights, `query · doc == BM25 score`.
- **Named vectors.** The collection now stores two named vectors per point:
  `dense` (SPECTER2) and `sparse` (BM25). `recreate_collection` and
  `upsert_papers` were updated; the bootstrap populates both.
- **`search_hybrid`** runs a native dense search and a native sparse search, then
  fuses them with **Reciprocal Rank Fusion** (`score = Σ 1/(RRF_K + rank)`).
- **Why RRF in Python (not server-side):** the pinned `qdrant-client==1.9.0`
  supports sparse vectors but **not** the Query-API fusion primitives
  (`Prefetch`/`FusionQuery`), which arrived in client ≥1.10. Fusing in Python
  avoids a pin bump. (Decision logged; revisit if the client pin is raised.)
- **Frozen contract preserved.** `search` (pure dense) is unchanged; hybrid is a
  new method. The Streamlit page calls `supports_hybrid()` and prefers
  `search_hybrid`, falling back to `search` otherwise.
- **Zero-score sparse hits are dropped before fusion.** When a payload filter is
  applied (the app always sends at least the default full-span year filter),
  Qdrant returns *all* filtered points for a sparse query — the genuine keyword
  match plus many zero-overlap points scored `0.0`. Left in, those zero hits earn
  RRF rank credit and flatten the ranking (e.g. `LoRA` drops to #3). `search_hybrid`
  filters the sparse arm to `score > 0` so only real keyword matches contribute.

**Result — the hybrid win (top-3):**

| Query | Dense+expansion | Hybrid |
|---|---|---|
| `LoRA` | LoRA paper **absent** from top-5 | **LoRA: Low-Rank Adaptation #1** |
| `RAG` | RAG paper #1 (dense already nailed it) | RAG paper **#1** (kept) |
| 5 full-phrased queries | clean top-1 | **clean top-1 (no regression)** |

> **Reading hybrid scores:** RRF fused scores are small (~0.03) and are *not*
> cosine similarities — they encode rank agreement between the two arms, not
> absolute similarity. What matters is the *ordering*. For `LoRA`, the correct
> paper now stands clearly apart (≈0.030 vs ≈0.016 for the rest), where dense
> alone produced a flat band of wrong papers.

---

## 7. The synthetic corpus & why it's idempotent

`scripts/bootstrap_synthetic.py` contains 30 hand-written, plausible ACL-style
NLP papers (`S0001`–`S0030`) spanning **2021–2025** across venues
`ACL / EMNLP / NeurIPS / ICLR / NAACL`, grouped into coherent topics so that
nearest-neighbor results are visibly sensible:

| Topic | Papers |
|---|---|
| Retrieval-Augmented Generation | S0001–S0004 |
| LoRA / PEFT | S0005–S0008 |
| Low-resource & multilingual | S0009–S0012 |
| Evaluation | S0013–S0014 |
| Instruction tuning / alignment | S0015–S0017 |
| Hallucination / factuality | S0018–S0019 |
| Long context | S0020–S0021 |
| Prompting / reasoning | S0022–S0024 |
| Distillation / efficiency | S0025–S0026 |
| Embeddings / retrieval models | S0027–S0030 |

Each paper carries a realistic abstract, concept list (with levels), a
`cited_by_count`, and a synthetic `references` list that becomes `citation_edges`
rows (so the P2 graph track has real-shaped data to build on).

**Two stages, both idempotent:**

1. `load_sqlite()` — `init_db()`, then `DELETE FROM` each table, then re-insert
   all 30 papers + concepts + edges + provenance.
2. `load_qdrant()` — reads papers + concepts back from SQLite, encodes the
   **dense** vectors (SPECTER2 over `title [SEP] abstract`) and the **sparse**
   BM25 vectors (over `title + abstract + concepts`), then `recreate_collection()`
   (drops & recreates `axiom_v1` with named dense+sparse vectors) and upserts 30
   points.

Because both stages wipe-then-write, you can run the script any number of times
and always land in the same clean state.

---

## 8. How the Streamlit page works

`app/streamlit_app.py`, top to bottom:

1. **Cached singletons** (`@st.cache_resource`) for the SPECTER2 encoder and the
   `AxiomQdrant` client; `@st.cache_data` for the filter options (venue list +
   year bounds from SQLite).
2. **Guard rails (no stack traces):**
   - Qdrant unreachable → clear in-UI error + `st.stop()`.
   - Empty index (`count() == 0`) → warning to run the bootstrap + `st.stop()`.
3. **Filter bar:**
   - **Venue** multiselect (empty = all venues).
   - **From year / To year** selectboxes. *Why two selectboxes and not a range
     slider:* a range slider draws both value labels on the same spot when both
     handles sit on one year, so a single-year pick showed two overlapping
     numbers. From/To makes a single year explicit (From == To) with no overlap,
     and a reversed range is normalized via `(min, max)`. Output is still
     `year_range = (from, to)` — unchanged downstream.
   - **Top-K** number input (default 10, capped at corpus size).
   - **Retrieval toggle** — `Hybrid (dense + sparse)` (default) vs `Dense only`,
     shown only when `store.supports_hybrid()`. Lets you demo the hybrid win live
     (`LoRA` finds little under dense-only, jumps to #1 under hybrid).
4. **Query flow:** non-empty query → `encoder.encode_query(query)` →
   `search_hybrid` or `search` depending on the toggle → `render_hits`.
5. **Result cards** (`render_hits`): each hit is a `st.expander` — header is the
   score · title · (year · venue · cites); the body shows concepts as code-span
   chips, the **abstract** (fetched from SQLite by `paper_id` via
   `db.papers_by_ids`, cached), a **DOI link** when present, and a **"🔎 Similar
   papers"** button.
6. **Similar-papers view:** the button sets `st.session_state["similar_to"]` and
   reruns; the page then renders `store.similar_papers(paper_id, …)` neighbors
   under a "Papers similar to: …" header, with a "← Back to search" button.
   Clicking "Similar" inside this view chains exploration to a new paper.

The page never imports raw Qdrant objects — it only sees `SearchHit` (enriched
with abstract/DOI from SQLite at render time).

---

## 9. How to run it

### Prerequisites
- **Docker** (for Qdrant).
- **Python 3.10–3.12.** ⚠️ Not 3.13/3.14 — the pinned `torch`/`transformers`
  have no wheels there (see §12).

### Steps (from the repo root)

```bash
# 1. Create a venv on a supported Python and install the pinned stack
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2. Start Qdrant
docker compose up -d

# 3. Load the synthetic corpus into SQLite + Qdrant (idempotent;
#    first run downloads SPECTER2 weights, ~440 MB)
python scripts/bootstrap_synthetic.py

# 4. Launch the search UI
streamlit run app/streamlit_app.py
```

Expected bootstrap output:

```
[sqlite] wrote 30 papers to .../data/axiom.db
[embed] loading allenai/specter2_base (first run downloads weights)...
[embed] query path: expansion
[embed] encoding 30 abstracts on device=mps        # or cuda / cpu
[sparse] built BM25 vectors (avgdl=50.8, vocab=689 terms)
[qdrant] recreating collection 'axiom_v1'...
[qdrant] upserted 30 points into 'axiom_v1' (dense + sparse)
Bootstrap complete. Run:  streamlit run app/streamlit_app.py
```

### Demo queries

Five full-phrased (clean top-1) plus terse ones that exercise the upgrades:

1. `reducing hallucination in retrieval-augmented generation`
2. `parameter efficient fine-tuning with low-rank adapters`
3. `low-resource machine translation with limited parallel data`
4. `evaluating language models beyond accuracy`
5. `chain-of-thought reasoning and self-consistency`
6. `LoRA`  → returns the *LoRA* paper at #1 (the hybrid win)
7. `RAG`   → returns the *Retrieval-Augmented Generation* paper at #1
8. `cheap finetuning` → QLoRA / LoRA at the top

Then exercise the filter bar: set **Venue = ICLR**, or set **From year = To year
= 2024**, and watch results change.

---

## 10. The eval harness

`scripts/eval_search.py` makes quality **demonstrable, not eyeballed**. It runs a
fixed 10-query set (the 5 demo queries + 5 terse/awkward ones) across one or more
modes and prints top-5 (score, title, year, venue) plus the score spread.

```bash
python scripts/eval_search.py                       # raw vs expansion vs hybrid
python scripts/eval_search.py --modes raw expansion # Step 2 comparison
python scripts/eval_search.py --modes expansion hybrid  # Step 3 comparison
```

Modes:
- **raw** — query embedded as a *document* (pre-upgrade behavior).
- **expansion** — `encode_query()`'s abstract-shaped wrapping (Step 2).
- **hybrid** — dense + sparse, RRF-fused (Step 3); auto-skipped with a clear
  message if the index has no sparse vectors yet.

---

## 11. Verification — real runs (2026-06-04)

Everything below was executed, not assumed.

- **P1 bootstrap:** 30 papers → SQLite (90 concept rows, 29 citation edges);
  SPECTER2 encoded all 30 abstracts on **Apple MPS**; 30 points upserted.
- **Hybrid rebuild:** dense + sparse vectors upserted; `supports_hybrid()` → True.
- **Step 2 eval (raw → expansion):** 5 full queries keep top-1; `RAG` fixed to
  top-1; bare `LoRA` still missing (expected → Step 3).
- **Step 3 eval (expansion → hybrid):** `LoRA` now returns the *LoRA* paper at
  **#1** (was absent from the dense top-5); `RAG` stays #1; all 5 full queries
  keep top-1 (no regression).
- **Streamlit:** boots headless and serves — `GET /` → HTTP 200,
  `/_stcore/health` → ok. Year filter verified via Streamlit `AppTest`:
  single-year `From=To=2023` returns only 2023 papers with no exception; reversed
  `From=2025,To=2022` normalizes to 2022–2025.

**Acceptance criteria — met:**
- ✅ 5 full-phrased queries keep clean top-1 relevance (no regression).
- ✅ Terse queries show better top-1/top-3 relevance.
- ✅ `LoRA` and `RAG` surface the LoRA / RAG papers in the top-3 (the hybrid win).

---

## 12. Environment gotchas (read before you run)

1. **Python version.** This dev machine's default `python3` is **3.14**, and the
   pinned `torch==2.2.2` / `transformers==4.40.2` have **no wheels for 3.14**
   (torch only ships ≥2.9 there). Fix applied: `brew install python@3.12` and
   build `.venv` from `/opt/homebrew/bin/python3.12`. Always activate that venv —
   bare `python3` is still 3.14 and won't work. `requirements.txt` stays pinned
   at the team-committed versions (correct for any 3.10–3.12).

2. **The `adapters` library is intentionally NOT installed.** It would force
   `transformers` up to ~4.57 (a pin bump). The query-side fix uses expansion
   instead (§6b). Enabling the true SPECTER2 query adapter is a P2 follow-up that
   requires an approved pin bump.

3. **`qdrant-client==1.9.0` has sparse vectors but not server-side fusion.**
   Hence RRF is done in Python (§6c). A client ≥1.10 would allow native
   `query_points` fusion — also a pin-bump decision, deferred.

---

## 13. P2 seams (where the next tracks plug in)

Marked in code with `# TODO(P2)` — interfaces only, no implementation:

| Seam | Location | What plugs in |
|---|---|---|
| **Ingestion** | `axiom/db.py` (bottom) | Replace synthetic loading with OpenAlex fetch + inverted-index abstract reconstruction, writing through the same `insert_*` helpers. |
| **Better query embedding** | `axiom/embed.py` (`_resolve_query_path`) | With an approved transformers pin bump, load `allenai/specter2_adhoc_query` for queries + `allenai/specter2` (proximity) for documents, and re-encode the corpus so both sides align. |
| **Native hybrid fusion** | `axiom/qdrant_client.py` (`search_hybrid`) | With `qdrant-client` ≥1.10, swap Python RRF for server-side `query_points` fusion. |
| **Velocity + citation graph** | `axiom/qdrant_client.py` (bottom) | Read the same payload fields (`year`, `concepts`, `cited_by_count`) to rank trending work and detected gaps. `citation_edges` is already populated for a NetworkX-first graph (decision **OD6** — no Neo4j in P1/P2). |

**Explicitly out of scope for P1:** velocity engine, citation graph, LangGraph,
gap pipeline, React, FastAPI.

---

## 14. Committed decisions (see `docs/DECISIONS.md` for full text)

- **OD6** — graph store is NetworkX-first; no Neo4j now.
- **Collection** = `axiom_v1`, versioned; one point per paper; named vectors
  `dense` + `sparse`.
- **Embedding model** = `allenai/specter2_base`, 768-dim, CLS pooling; CPU
  fallback warns, never silently swaps models.
- **Data layer** = OpenAlex → SQLite; abstracts reconstructed from the inverted
  index.
- **`search` signature** is frozen; new capability is added as new methods.
- **Query path = expansion** (adapter blocked by the transformers pin).
- **Hybrid retrieval = dense + sparse, RRF-fused in Python** (client predates
  server-side fusion); anticipated by D4/OD4.

---

## 15. One-line summary

P1 gives the team a frozen SQLite contract and a Qdrant-backed SPECTER2 semantic
search over a synthetic 30-paper corpus; the search-quality upgrades add sane
demo defaults, query expansion, and hybrid dense+sparse retrieval — so terse and
acronym queries (LoRA, RAG) now rank correctly while full queries don't regress,
all without changing the data contract or bumping a single pinned dependency.
```
