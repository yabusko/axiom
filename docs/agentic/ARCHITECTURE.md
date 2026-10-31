# Agentic pipeline — architecture

Contracts for the router-worker agent pipeline (see the agentic design spec). Additive to the existing `axiom/*` stack (OD1–OD18); nothing here changes a frozen signature.

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
    action: Literal["full", "title_only", "skip"]; reason: str
    route_source: Literal["rule", "llm"] = "rule"  # A5: no paper_id -- see A4 below
class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str; kind: Literal["method", "task", "dataset", "metric"]
    evidence_span: str; confidence: float = Field(ge=0.0, le=1.0)
class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[Entity]  # A4: no paper_id -- the model is never asked for or
    # trusted with one; axiom/agents/pipeline.py always attaches its own input's
    # paper_id downstream (EnrichmentResult.paper_id, the linking.py write), never
    # anything parsed out of a model response. RouteDecision above is the same way.
class EnrichedEntity(Entity):
    canonical: str; cosine: float = Field(ge=-1.0, le=1.0)
    ontology_source: Literal["lexical", "embedding", "new_concept"]  # OD26/B1, was openalex_concept|concept_canonical|unmatched
class EnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str; enriched: list[EnrichedEntity]
class VerifierError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    code: Literal["schema", "evidence_span", "ontology_membership", "cosine_threshold",
                  "name_span_mismatch", "empty_after_retry"]  # A2/A3 additions
    message: str
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

New tables: `paper_concepts` (agent-extracted entity rows: run_id, name, kind, canonical, cosine, evidence_span, confidence, source, created_at; FK → `papers.openalex_id`), `concept_cooccurrence` (run_id, paper_id, concept_a < concept_b; FK → `papers.openalex_id`), `agent_runs` (run_id PK, status, papers_total, papers_done, model_fast, model_strong, started_at, finished_at), `agent_run_items` (run_id+paper_id PK, route, verifier_attempts, verdict, error). Indexes on `paper_concepts(paper_id)`, `paper_concepts(canonical)`, `concept_cooccurrence(concept_a/b)`.

OD26/B4/C3 additions: `run_id` column on `paper_concepts`/`concept_cooccurrence` (which run produced the row); unique index `paper_concepts(paper_id, canonical, kind)` (idempotent-linking backstop — `linking.py` deletes a paper's prior rows before reinserting, in one transaction, so this should never actually trip in normal operation); new `llm_spend` table (id, run_id, ts, model, tokens_in, tokens_out, cost) + `idx_llm_spend_run` — durable, cross-run-cumulative spend ledger replacing the in-memory-only counter in `llm_providers.py`.

## 5. New `axiom/config.py` keys

`MODEL_FAST`/`MODEL_STRONG` (Anthropic model IDs) + per-1M-token input/output prices + `MODEL_PRICE_CHECKED_DATE` (verified against the live Claude API pricing reference — re-verified 2026-10-31, OD26: `claude-haiku-5-5`/`claude-sonnet-5-5`). `AGENT_MAX_TOKENS` dict (OD26/C4: per-agent-type max_tokens — route/extraction/synthesis — instead of one flat constant everywhere). `LLM_PROVIDER_PRIORITY = ("anthropic", "ollama")`, anthropic selected iff `ANTHROPIC_API_KEY` is set (mirrors OD14's no-key philosophy). `ANTHROPIC_BASE_URL`, `ANTHROPIC_TIMEOUT`, `ANTHROPIC_MAX_RETRIES`, `ANTHROPIC_BACKOFF_CAP`. `AGENT_MAX_RETRIES = 3` (R4). `AGENT_CONCURRENCY` (R5 semaphore limit). `ENTITY_KINDS`, `ROUTE_ACTIONS` (single source for the Literal values). `ROUTE_TITLE_ONLY_MIN_WORDS = 40` (OD26/A5: rule-based routing threshold). `VERIFIER_COSINE_MIN` (§7.5). `ENRICHMENT_PHRASE_MODEL`, `ENRICHMENT_MARGIN_DELTA` (OD26/B1: short-phrase model + top1-top2 margin for the matching cascade; margin is a placeholder pending the B3 label-and-calibrate gate). `TRACE_DIR`, `EVAL_V2_QUERIES_PATH`, `EVAL_V2_SCALE_MAX = 2`, `AGENT_BENCH_PATH`. `STRUCT_RETRIEVAL_CANDIDATES` (R8 budget cap). `AXIOM_OS_TRUSTSTORE` env var (OD26/C5: gates OD24's `truststore.inject_into_ssl()`, opt-in not unconditional).

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

## 9. Stage-2 fixes, Phase 1 design (full rationale in `docs/DECISIONS.md` OD26)

Three findings from the fixes design spec got a design pass before Phase 2 build:

- **B1 (enrichment cascade).** Replace the single embed-and-argmax step with: exact/normalized
  match → short-phrase-model (`ENRICHMENT_PHRASE_MODEL`) match gated on cosine **and**
  top1−top2 margin (`ENRICHMENT_MARGIN_DELTA`) → `new_concept` (kept, canonical = normalized
  name — today's `unmatched` entities are silently dropped from linking; this stops that).
  `ontology_source` renamed `lexical|embedding|new_concept` (§3).
- **C3 (budget ledger).** New durable `llm_spend` table (§4) + an in-process reservation
  (worst-case cost reserved before dispatch, settled after) closes the check-then-record race
  flagged as an accepted risk in OD25, and survives a process restart.
- **D1 (background runs).** `POST /agents/run` returns `202 {run_id}` immediately
  (`BackgroundTasks`, `try/finally` → `done`/`failed`); `GET /agents/runs/{id}` gains
  per-verdict counts and cost-so-far. `api/main.py`-owned (main session), not a separate branch.

None of these are built yet — Phase 2 implementation code against the schema/config deltas
already applied (§4/§5).

## 10. Stage-2 fixes, Phase 2: pipeline-core landed changes

B1/B2 above are built (`axiom/agents/enrichment.py`'s `_match_one` cascade + `PhraseEncoder`,
mean-pooled `BAAI/bge-small-en-v1.5`, not SPECTER2). Also landed, beyond Phase 1's three
design-noted findings: A1 (`verifier.verify_extraction` checks the title too, both sides
normalized via `verifier.normalize_text`); A2 (per-entity verdicts, accumulated across
extraction's retry attempts -- a paper is rejected only if every entity it ever proposed
failed every attempt; `empty_after_retry` on `VerificationFailed.errors` distinguishes that
from a paper that genuinely has nothing to extract); A3 (`name_span_mismatch` -- the claimed
entity name must relate to its own evidence_span); A4 (`paper_id` removed from `RouteDecision`/
`ExtractionResult`, §3); A5 (`coordinator.route` is rule-based, no LLM call in the common
path -- `route_via_llm` is kept but unreachable, the three rules are exhaustive); D3 (real
verifier-attempt counts recorded on the accepted path, not a hardcoded 1); D4 (`axiom.
hypothesis`'s shared prompt/verify/side-paper-selection helpers promoted to public names;
`axiom.agents.synthesis.generate_hypothesis` is the one implementation to call going
forward -- see `docs/DECISIONS.md` for why the Ollama-only `axiom.hypothesis.
generate_hypothesis` is still present). B5 (`graph._pagerank` now reads the edge `weight`
attribute; `tests/test_concept_graph.py` exercises the real implementation, no more
monkeypatched stand-in).
