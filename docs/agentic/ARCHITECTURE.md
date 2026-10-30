# Agentic pipeline — architecture

Contracts for the router-worker agent pipeline (see `AXIOM_AGENTIC_TASK.md`). Additive to the existing `axiom/*` stack (OD1–OD18); nothing here changes a frozen signature.

## 1. Module map

| Path | Purpose |
|---|---|
| `axiom/llm_providers.py` | `complete_json(tier, system, prompt, model_cls)` async facade; `anthropic` (httpx, no SDK) + `ollama` (wraps `axiom/llm.py`) providers; retry/backoff, cost+token accounting, trace emission |
| `axiom/agents/contracts.py` | All Pydantic v2 models for agent boundaries (§3) |
| `axiom/agents/coordinator.py` | `route(paper) -> RouteDecision`, fast tier |
| `axiom/agents/extraction.py` | `extract(paper) -> ExtractionResult`, fast tier |
| `axiom/agents/enrichment.py` | `enrich(entities) -> EnrichmentResult`; deterministic ontology match + SPECTER2 cosine, no LLM call |
| `axiom/agents/linking.py` | Deterministic; writes `paper_concepts` + `concept_cooccurrence` via `axiom/db.py` |
| `axiom/agents/synthesis.py` | Strong tier; concept-cluster labels + relocated `generate_hypothesis` (OD16 verifier/HITL rules unchanged) |
| `axiom/agents/verifier.py` | Deterministic checks (schema/span/ontology/cosine); builds `VerifierError[]`; drives the ≤3-attempt retry |
| `axiom/agents/pipeline.py` | Per-paper state machine (coordinator→extraction→enrichment→linking) + run-level `asyncio.gather` under `asyncio.Semaphore(config.AGENT_CONCURRENCY)`; writes `agent_runs`/`agent_run_items` |
| `axiom/concept_graph.py` | Co-occurrence graph from `concept_cooccurrence`; Louvain clusters + PageRank (R7) |
| `axiom/tracing.py` | Appends one JSON line per LLM call to `eval/runs/<run_id>.jsonl` |
| `api/main.py` (additions) | `async def POST /agents/run`, `GET /agents/runs/{id}`; sync `GET /concepts/communities`, `GET /concepts/central` — existing sync routes untouched |
| `scripts/bench_agents.py` | `--n 50` → `eval/agent_bench.json`: rejection rate (first-pass vs final), p50/p95, $/paper, $/run |
| `scripts/label_cli.py` | Human review/override of `eval/ndcg_queries_v2.json` labels, records `annotator` |
| `eval/ndcg_queries_v2.json` | ~50 queries, 0–2 scale, pooled top-10 across modes, `_provenance: "AI-drafted"` |
| `scripts/eval_ndcg.py` (extended) | Adds `dense`/`hybrid`/`hybrid_struct` (+ optional `dense_general`) modes; v1 file/report untouched |

## 2. Data flow

```
paper_id ── coordinator (fast) ──RouteDecision──┐
                                                  │ skip → trace, stop
                                                  │ full | title_only
                                                  ▼
                               ┌── extraction (fast) ──Entity[]──┐
                 attempt<=3    │        ┌──────────────┐          │
            ┌──────────────────┘        │   VERIFIER   │◄─────────┘
            │  retry: prompt + ValidationError.errors()  │ schema / evidence_span
            │  + verifier reasons → same agent            │ substring / ontology / cosine
            └──────────────────▶ extraction ─────────────┘
                                                  │ ok
                               ┌── enrichment (det.+SPECTER2) ──EnrichedEntity[]──┐
                 attempt<=3    │                                                  │
            ┌──────────────────┘            VERIFIER (same 4 checks) ◄───────────┘
            └──────────────────▶ enrichment ──────────────┘
                                                  │ ok
                               linking (det.) → SQLite: paper_concepts, concept_cooccurrence
                        (after all papers in run) ▼
                               synthesis (strong) → cluster labels + hypothesis pitch (OD16 rules)
                     every LLM call above ─────────▶ eval/runs/<run_id>.jsonl
```

Per-paper state machines run concurrently via `asyncio.gather` under the semaphore; `synthesis` runs once per run, after linking completes for its input papers.

## 3. Pydantic v2 contracts (`axiom/agents/contracts.py`)

```python
class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str; action: Literal["full", "title_only", "skip"]; reason: str
class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str; kind: Literal["method", "task", "dataset", "metric"]
    evidence_span: str; confidence: float = Field(ge=0.0, le=1.0)
class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str; entities: list[Entity]
class EnrichedEntity(Entity):
    canonical: str; cosine: float = Field(ge=-1.0, le=1.0)
    ontology_source: Literal["openalex_concept", "concept_canonical", "unmatched"]
class EnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str; enriched: list[EnrichedEntity]
class VerifierError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str; code: Literal["schema", "evidence_span", "ontology_membership", "cosine_threshold"]; message: str
class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool; attempt: int; errors: list[VerifierError] = []
class RunStatusOut(BaseModel):              # api/main.py response, from_attributes=True (OD12 pattern)
    model_config = ConfigDict(from_attributes=True)
    run_id: str; status: Literal["queued", "running", "done", "failed"]
    papers_total: int; papers_done: int; model_fast: str; model_strong: str
    started_at: str | None; finished_at: str | None
```

Constraints (`ge`/`le`) are enforced by Pydantic; the provider layer strips them from the JSON Schema sent to the Anthropic API (unsupported keywords) — the verifier, not the API, is the backstop for those checks.

## 4. Schema delta (`db/schema.sql`, additive) — log in `docs/DECISIONS.md`'s schema-change log

New tables: `paper_concepts` (agent-extracted entity rows: name, kind, canonical, cosine, evidence_span, confidence, source, created_at; FK → `papers.openalex_id`), `concept_cooccurrence` (paper_id, concept_a < concept_b; FK → `papers.openalex_id`), `agent_runs` (run_id PK, status, papers_total, papers_done, model_fast, model_strong, started_at, finished_at), `agent_run_items` (run_id+paper_id PK, route, verifier_attempts, verdict, error). Indexes on `paper_concepts(paper_id)`, `paper_concepts(canonical)`, `concept_cooccurrence(concept_a/b)`.

## 5. New `axiom/config.py` keys

`MODEL_FAST`/`MODEL_STRONG` (Anthropic model IDs) + per-1M-token input/output prices + `MODEL_PRICE_CHECKED_DATE` (verified against the live Claude API pricing reference). `LLM_PROVIDER_PRIORITY = ("anthropic", "ollama")`, anthropic selected iff `ANTHROPIC_API_KEY` is set (mirrors OD14's no-key philosophy). `ANTHROPIC_BASE_URL`, `ANTHROPIC_TIMEOUT`, `ANTHROPIC_MAX_RETRIES`, `ANTHROPIC_BACKOFF_CAP`. `AGENT_MAX_RETRIES = 3` (R4). `AGENT_CONCURRENCY` (R5 semaphore limit). `ENTITY_KINDS`, `ROUTE_ACTIONS` (single source for the Literal values). `VERIFIER_COSINE_MIN` (§7.5). `TRACE_DIR`, `EVAL_V2_QUERIES_PATH`, `EVAL_V2_SCALE_MAX = 2`, `AGENT_BENCH_PATH`. `STRUCT_RETRIEVAL_CANDIDATES` (R8 budget cap).

## 6. Integration with existing graph/gaps/velocity/retrieval (OD11-safe)

`axiom/concept_graph.py` is new and independent of `graph.py`/`gaps.py` — builds a co-occurrence graph from `concept_cooccurrence`, runs Louvain + PageRank. `gaps.py`'s citation-graph community detection and `g_score` are untouched. `graph.py`'s private `_pagerank` is promoted to public `pagerank(g, alpha, max_iter, tol)` so `concept_graph.py` reuses the same power-iteration implementation (§7.1). The pipeline is DB-path-parametrized exactly like `db.connect` already supports: runs against the main corpus write `data/axiom.db`; runs against the eval corpus (R8) write `data/eval.db` — rows never cross, preserving OD11. R8's structural arm is **not** added to `qdrant_client.py`'s frozen `search`/`search_hybrid` (OD4); it lives only in `scripts/eval_ndcg.py --mode hybrid_struct` as a third RRF arm, eval-only.

## 7. Decisions to log in `docs/DECISIONS.md` (OD19–OD23)

1. Promote `graph._pagerank` to public `graph.pagerank` — one shared implementation for the citation graph and the new concept co-occurrence graph, instead of a silently drifting copy.
2. Agent-extracted concepts live in new `paper_concepts`/`concept_cooccurrence` tables, never merged into the OpenAlex-sourced `concepts` table — different provenance/confidence semantics, same separation discipline OD11 used for the eval corpus.
3. R8's structural retrieval arm is eval-only (`scripts/eval_ndcg.py`), not wired into production `search_hybrid` — an LLM call per interactive query is too slow/costly for Streamlit/API; OD4's frozen signatures stay frozen.
4. Pydantic v2 is the contract only at LLM/agent call boundaries (`axiom/agents/contracts.py`); the rest of `axiom/*` keeps the OD12 dataclass convention — `model_json_schema()` is structurally required for `output_config.format`, not a style reversal of OD12.
5. `VERIFIER_COSINE_MIN` is set by manual inspection of ~30 (entity, ontology-match) pairs sorted by cosine, picking the value where true/false matches visibly separate — recorded with date and sample size in a `config.py` comment, not swept to a target.

## 8. R11 check

No LangChain/LangGraph/agent framework introduced. Plain `asyncio` + `httpx` (already pinned) + Pydantic v2 (already a transitive FastAPI dep, now also used directly) + existing `networkx`/`numpy`. Hand-rolled coordinator/worker functions, `asyncio.gather`/`Semaphore` — matches OD10/OD14's minimal-deps discipline.
