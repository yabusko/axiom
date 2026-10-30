# Axiom

**LLMs for Sciences: Research Trends & Gaps Extraction Tool.** Axiom is *not* a
paper search engine — it surfaces two complementary signals from scholarly
metadata: **research trends** (which concepts are accelerating) and **research
gaps** (conceptual regions that are semantically close but barely connected by
citation). Semantic search is the foundation those signals stand on, not the
product itself.

> This README reflects the **as-built** system (2026-07-04), which has
> diverged from the original plan in several places — see
> `docs/DECISIONS.md` (OD6–OD14) for the full rationale on every divergence,
> and `Backlog.md` for the task-by-task status.

---

## What's built

| Capability | Status |
|---|---|
| OpenAlex ingest (topic-snowball) → SQLite → SPECTER2 → Qdrant | ✅ |
| Hybrid dense+sparse (RRF) retrieval | ✅ |
| NetworkX citation graph + PageRank influence | ✅ |
| Research-gap detector (community-pairs, OD9) | ✅ |
| Velocity/trends engine (OD10) | ✅ |
| Keyword canonicalization (local Ollama, OD14) | ✅ |
| Reading list — bookmarks + local-LLM summaries (OD13/OD14) | ✅ |
| FastAPI service layer (OD12) | ✅ |
| nDCG@10 retrieval eval, real corpus (OD11) | ✅ (mean 0.453, single-pass labels) |
| Streamlit UI: Trending, Research gaps, Search, Reading list, Review queue, **Advanced: vector geometry** | ✅ |
| LLM hypothesis pitch + HITL review queue (PBI 5) | ✅ (OD16 — grounded pitch + rule-based verifier + review queue) |
| React frontend | ❌ not started |
| Composite gap G-score (OD17) | ✅ (uncalibrated defaults) |
| Threshold (τ) calibration scaffold (PBI 8) | ⚠️ scaffold built (OD17); τ/weights still need human labels |
| Agentic extraction pipeline (router→extraction→enrichment→linking, R1-R6) | ✅ real 30-paper run, $0.09 — synthesis (strong tier) built but not yet wired into a run, see `docs/agentic/RESULTS.md` |
| Concept co-occurrence graph (Louvain + PageRank, R7) | ✅ real run: 5 communities, 109 concept rows |
| Eval v2: ~50-query pooled retrieval judging (R9) | ✅ AI-drafted (51 real queries); not yet human-reviewed |
| Agent tracing + benchmark (R10) | ✅ real numbers: $0.003/paper, p50 6.8s, see `eval/agent_bench.json` |

See `docs/demo_examples.md` for real, reproducible output of every ✅ row, and
`docs/agentic/RESULTS.md` for the agentic pipeline's real numbers and known gaps. **Note:**
`POST /agents/run` makes real, billed calls to the Anthropic API — never expose this app's API
port beyond localhost.

---

## Prerequisites

- **Docker** (for Qdrant).
- **Python 3.10–3.12.** ⚠️ Not 3.13/3.14 — the pinned `torch`/`transformers`
  have no wheels there.
- **[Ollama](https://ollama.com)**, running locally (`ollama serve`), with a
  model pulled (default `qwen2.5:7b`: `ollama pull qwen2.5:7b`) — only needed
  for keyword canonicalization and reading-list summaries (OD14). Everything
  else runs without it.

---

## Run order (from the repo root)

```bash
# 1. venv + pinned deps
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2. Start Qdrant
docker compose up -d

# 3a. Quick demo: 30-paper synthetic corpus (idempotent)
python scripts/bootstrap_synthetic.py

# 3b. OR a real corpus via OpenAlex topic-snowball (needs internet; no API key)
python scripts/ingest_openalex.py --topic "retrieval-augmented generation" --target 500

# 4. (optional) Keyword canonicalization — needs `ollama serve` running
python scripts/canonicalize_concepts.py

# 5. Launch the UI
streamlit run app/streamlit_app.py

# 6. (optional) FastAPI service layer, for API/React consumption
uvicorn api.main:app --reload   # docs at http://localhost:8000/docs
```

### Sharing the corpus with teammates

The `axiom_v1` Qdrant collection and `data/axiom.db` SQLite database are not
committed to git (they are gitignored). The reproducible way to share the
corpus is to re-run the ingest on each machine:

```bash
docker compose up -d
python scripts/ingest_openalex.py --topic "natural language processing" --target 500
streamlit run app/streamlit_app.py
```

All tabs — including the **Advanced: vector geometry** tab — then read from the same
`axiom_v1` collection, so there is no need to run a separate Streamlit process.

### Verify it end-to-end

```bash
python scripts/smoke_test_api.py     # FastAPI, 17 checks against live data
python scripts/eval_search.py --check   # search acceptance checks (LoRA/RAG top-1)
```

### Unit tests

```bash
pip install -r requirements-dev.txt
pytest                               # unit tests for velocity/gaps/graph/db/api/…
```

The suite is self-contained (no Qdrant/Ollama/torch needed at runtime): heavy
deps are stubbed in `tests/conftest.py`, so it runs both in the project `.venv`
and on a bare Python 3.13/3.14 env where the ML stack can't install.

---

## Retrieval evaluation (nDCG@10, OD11)

A separate, real 1,500-paper ACL/EMNLP/COLING/NAACL sample (kept apart from
the main corpus — see OD11) drives an nDCG@10 harness:

```bash
python scripts/ingest_eval_corpus.py     # one-time: samples + embeds the eval corpus
python scripts/eval_ndcg.py --report eval/report.md
```

Current result: **mean nDCG@10 = 0.453** (hybrid). See `eval/report.md` and
its caveats (relevance judgments are single-pass/AI-drafted, not yet
human-reviewed).

---

## Architecture

```
OpenAlex (snowball, OD7)                     db/schema.sql  ← shared data contract
        │                                          │
        ▼                                          ▼
scripts/ingest_openalex.py ──────────────►  SQLite (data/axiom.db)
        │                                          │
        │ SPECTER2 (axiom/embed.py)                │ papers · concepts · citation_edges
        │ + BM25 sparse (axiom/sparse.py)          │ paper_provenance · reading_list
        ▼                                          │ concept_canonical · paper_summaries
  Qdrant `axiom_v1` (axiom/qdrant_client.py)        │
   dense + sparse named vectors                     │
        │                                          │
        │                        axiom/graph.py ◄──┤ (NetworkX, PageRank, OD6/OD8)
        │                        axiom/gaps.py  ◄──┤ (Louvain communities + gap pairs, OD9)
        │                        axiom/velocity.py ◄┤ (trend ranking, OD10)
        │                        axiom/canonicalize.py, axiom/summarize.py ◄─ axiom/llm.py (Ollama, OD14)
        ▼
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ app/streamlit_app.py         │        │ api/main.py (FastAPI, OD12)   │
│ Trending · Research gaps ·   │        │ 21 REST routes over the same  │
│ Search · Reading list ·      │        │ axiom/* modules — no React    │
│ Review queue · Advanced:     │        │ consumer yet                  │
│ vector geometry              │        │                               │
│ (imports axiom/* + vector_tools)│      │                               │
└─────────────────────────────┘        └──────────────────────────────┘
```

**Agentic extraction pipeline** (router-worker: coordinator → extraction → enrichment → linking
→ synthesis; R1-R11): additive on top of the above, async, calling the real Anthropic API with a
local-Ollama fallback. Full module map, contracts, and schema delta in
`docs/agentic/ARCHITECTURE.md`; real results in `docs/agentic/RESULTS.md`.

---

## Layout

```
db/schema.sql                    shared SQLite contract (papers, concepts, citation_edges,
                                  paper_provenance, reading_list, concept_canonical, paper_summaries)
axiom/config.py                  paths, Qdrant/Ollama config, all tunable constants
axiom/db.py                      SQLite connect/init/insert/query helpers (all SQL lives here)
axiom/ingest.py                  OpenAlex snowball fetch (pure-Python, no torch)
axiom/embed.py                   SPECTER2 encoder (GPU→CPU fallback)
axiom/sparse.py                  BM25-style sparse encoder (hashing trick)
axiom/qdrant_client.py           AxiomQdrant: collection lifecycle, search, search_hybrid
axiom/indexer.py                 shared embed+upsert path (bootstrap, ingest, eval corpus)
axiom/graph.py                   NetworkX citation graph, PageRank influence
axiom/gaps.py                    OD9 research-gap detector (Louvain + centroid gap-score)
axiom/velocity.py                OD10 trend/velocity engine
axiom/canonicalize.py            T3.2 LLM keyword canonicalization (OD14)
axiom/summarize.py               T7.4 paper summarization (OD14)
axiom/hypothesis.py              OD16 hypothesis pitch + rule-based verifier (PBI 5)
axiom/llm.py                     shared local-Ollama client (OD14)

app/streamlit_app.py             6-tab UI: Trending, Research gaps, Search, Reading list, Review queue, Advanced: vector geometry
app/vector_tools.py              Vector tools render functions (Proximity Analyzer + Vector Gap Discovery)
api/main.py                      FastAPI service layer (OD12)

scripts/bootstrap_synthetic.py   30-paper synthetic demo corpus
scripts/ingest_openalex.py       real OpenAlex ingest CLI
scripts/canonicalize_concepts.py runs T3.2 over the corpus
scripts/ingest_eval_corpus.py    samples the real eval corpus (OD11)
scripts/eval_search.py           search-mode comparison + acceptance checks
scripts/eval_ndcg.py             nDCG@10 harness (OD11)
scripts/smoke_test_api.py        FastAPI integration smoke test

eval/ndcg_queries.json           15 queries, 86 single-pass AI-drafted relevance judgments
eval/report.md                   generated nDCG@10 report

tests/                           pytest unit suite (conftest stubs torch/qdrant so it
                                  runs without the ML stack); requirements-dev.txt

docs/DECISIONS.md                every committed decision (OD1–OD14) + schema change log
docs/demo_examples.md            real, reproducible output of every built feature
docs/WALKTHROUGH.md              deep-dive on the original P1 seam (historical — see its banner)
Backlog.md                       full task-by-task status (as-built vs. original plan)
docker-compose.yml               Qdrant only
```

---

## Limitations

- **Corpus scope.** The default demo corpus is a 30-paper synthetic set; the
  real-data path (`ingest_openalex.py`) snowballs from one seed topic to
  ~500 papers (OD7) — not the ≥1k/≥5k multi-venue corpus the original plan
  targeted (T2.3, not built by choice this round).
- **Uncalibrated rankings.** `gap_score`, the composite `g_score` (OD17), and
  `velocity` are ranking heuristics. The G-score's weights and threshold τ ship
  as uncalibrated defaults: the calibration pipeline exists
  (`scripts/export_gap_labels.py` → `scripts/calibrate_gap_thresholds.py` →
  `eval/calibration.json`, consumed by `gaps.analyze`), but the human labels it
  needs (`eval/labels/gap_labels.csv`) are not yet produced (T8.1).
- **Hallucination controls.** Local-LLM outputs (canonicalization, summaries)
  are constrained by system prompts (grounding-only instructions) and
  span at most 3 bullets / small synonym groups — not independently verified
  against a hallucination benchmark.
- **English-only.** OpenAlex ingestion, SPECTER2, and the Ollama prompts all
  assume English abstracts; no multilingual handling anywhere in the pipeline.
- **Single-user.** No auth; the reading list is one shared list, not per-user.

---

## Troubleshooting

- **"Cannot reach Qdrant" in the UI** → `docker compose up -d`, wait a few
  seconds, then bootstrap/ingest.
- **"The index is empty"** → run `scripts/bootstrap_synthetic.py` or
  `scripts/ingest_openalex.py`.
- **Canonicalization/summaries fail with "Ollama unreachable"** → `ollama
  serve` isn't running, or the model in `axiom.config.OLLAMA_MODEL` isn't
  pulled (`ollama pull qwen2.5:7b`).
- **Slow first encode** → expected on CPU; SPECTER2 weights download once and
  are cached afterward.
