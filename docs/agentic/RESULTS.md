# Agentic pipeline — results

Every number below traces to a committed file or a script that produced it; none are estimated.

## Stage-2 fixes update (2026-10-31) — supersedes the stage-1 numbers below

The main corpus (`data/axiom.db`) was replaced with a **real OpenAlex corpus** (300 papers,
topic "retrieval-augmented generation") -- the 30-paper corpus the section below measured was
entirely synthetic test data, not real papers. Model IDs refreshed:
`MODEL_FAST=claude-haiku-5-5` ($0.10/$0.50 per 1M in/out, <=100k-token tier),
`MODEL_STRONG=claude-sonnet-5-5` ($2/$10) -- `axiom/config.py`, checked live against
platform.claude.com 2026-10-31 (`docs/DECISIONS.md` OD26).

- **Real 50-paper pipeline run** (real OpenAlex papers, calibrated config): 50/50 accepted,
  `eval/agent_bench.json` (regenerated from run `0e3c0fd630b9466b89de6ac8f2a5f741`) —
  rejection rate first_pass=10.0% final=0.0%, latency p50=2.93s p95=6.97s, $0.0004/paper,
  $0.0217/run.
- **B1 enrichment cascade recalibrated from real AI-labeled data** (`docs/DECISIONS.md` OD28):
  `VERIFIER_COSINE_MIN=0.93`, `ENRICHMENT_MARGIN_DELTA=0.0` (was 0.95/0.05) — two independent
  blind AI judges (sonnet-5-5 + opus-5-5, no human review) labeled 100 real (entity, candidate)
  pairs, Cohen's kappa 0.7983, 92/100 usable agreed labels. Calibration
  (`eval/enrichment_threshold_calibration.md`): precision=0.929, recall=0.867; only 6/208
  `new_concept` entities would flip under the new threshold — **the high new_concept rate is a
  real ontology gap, not a miscalibrated threshold**. 21 synonymous `new_concept` labels merged
  into 11 representatives (`scripts/merge_new_concept_synonyms.py`, applied for real).
- **E3 AI second-grader review** (`docs/DECISIONS.md` OD29): an independent strong-tier
  re-grade of a 10-query sample of `eval/ndcg_queries_v2.json`'s pooled candidates, blind to
  the original fast-tier grader's labels and ranks. Cohen's kappa **0.6319** over 129 judgments
  (77.5% agreement) — `eval/ndcg_v2_second_grader_sample.json`. AI-vs-AI, not human agreement.
- **D5 query-aware structural retrieval arm** (`docs/DECISIONS.md` OD30): ran the real pipeline
  on 300 real eval-corpus papers (1741 `paper_concepts` rows, $0.1095). Mean nDCG@10 over the
  51 v2 queries: hybrid alone **0.8904**, hybrid + existing pool-cohesion structural arm
  **0.6168** (worse), hybrid + new query-aware structural arm **0.8727** (roughly neutral,
  still a small regression) — `eval/report_hybrid_struct_query_aware.md`. **Neither structural
  arm helps; not exposed as a search mode**, per the honest-reporting instruction this was
  built under.
- **Total LLM spend this stage: ~$0.72** (50-paper run $0.0217 + B3 labeling $0.5381 + E3
  $0.1129 + D5's 300-paper run $0.1095 ≈ against the $3.00 `LLM_BUDGET_USD` cap, tracked
  durably in `llm_spend`; `data/eval.db` has its own separate ledger from `data/axiom.db`, a
  known gap noted in OD30, not fixed).
- **Real bugs found and fixed running this stage for real** (full detail in `docs/DECISIONS.md`
  OD26-OD30): a migration that broke on pre-existing duplicate rows; `reset_corpus` missing the
  new FK-dependent tables; a `run_id` column that silently never reached an existing table;
  every current model rejecting the `temperature` parameter outright; the budget-reservation
  estimator undercounting real cost by ~2x; a within-paper unique-index collision when two
  entity names resolve to the same concept; a "second grader" that was silently the same model
  as the first; the eval corpus never having been migrated to the agentic schema; the eval
  corpus having no ontology data at all.
- **Deferred, not done this stage** (see "Deferred" below for the full list with reasons):
  architecture/security review pass, fact-checker pass, `dense_general` eval mode.

---

## Stage-1 baseline (2026-10-30, historical — numbers below are from the synthetic 30-paper run)

Real run against `data/axiom.db` (30 papers, synthetic test data) and the real eval corpus
(`data/eval.db`, 1500 papers), real Anthropic API calls (`claude-haiku-4-5`), 2026-10-30/31.

## Model IDs, endpoints, spend

- `MODEL_FAST = claude-haiku-4-5` ($1.00/$5.00 per 1M in/out tokens), `MODEL_STRONG =
  claude-sonnet-5` ($2.00/$10.00 per 1M) — `axiom/config.py`, prices checked 2026-10-30.
  `MODEL_STRONG` is wired but not yet exercised by a real run (no synthesis call made; see R6).
- 4 new API endpoints (`api/main.py`): `POST /agents/run`, `GET /agents/runs/{run_id}`,
  `GET /concepts/communities`, `GET /concepts/central` — 21 total routes in the app.
- **Total real spend: $0.3226** of the $3.00 `LLM_BUDGET_USD` cap — $0.0901 (30-paper agent
  benchmark, `eval/agent_bench.json`) + $0.2325 (51-query v2 grading, `eval/runs/eval_v2_build.jsonl`,
  which sums to this exactly). `build_eval_v2_queries.py` hardcodes `run_id="eval_v2_build"`
  (not a fresh uuid per invocation), so an earlier 2-query dry run of the same script appended 2
  extra lines to this trace file before the real 51-query run; those 2 duplicate lines were
  removed from the committed file as test pollution, not part of the real result.

## Per-requirement status

| R | Status | Evidence | Measured value |
|---|---|---|---|
| R1 Router-worker pipeline | done | `eval/runs/f9db14ede8d94f04965d37d83c4d5f97.jsonl`, `eval/agent_bench.json` | 30/30 papers routed; coordinator->extraction->enrichment->linking, real API |
| R2 Pydantic v2 contracts | done | `axiom/agents/contracts.py` | 8 models at every agent boundary |
| R3 Deterministic verifier | done | `docs/DECISIONS.md` OD23 | `VERIFIER_COSINE_MIN=0.95`, calibrated from 109 real (entity, match, cosine) triples |
| R4 Self-correction (<=3 attempts) | done | `data/axiom.db.agent_run_items` (run f9db14ede8d94f04965d37d83c4d5f97) | 1/30 papers hit 3 attempts, correctly rejected (hallucinated `evidence_span`) |
| R5 Async (`asyncio.gather`+semaphore) | done | `axiom/agents/pipeline.py` | `AGENT_CONCURRENCY=4`; 30-paper run wall time 61.1s |
| R6 Model tiers fast/strong | partial | `eval/agent_bench.json.model_ids` | only `claude-haiku-4-5` (fast) exercised; strong tier not yet called (no synthesis run — see "Known gaps" below) |
| R7 Concept graph (co-occurrence) | done | live query against `data/axiom.db`, re-verified 2026-10-31 | 109 `paper_concepts` / 104 `concept_cooccurrence` rows -> 5 Louvain communities, real PageRank (top: "Bias" 0.051). **This row is a live measurement of local, gitignored DB state, not a frozen file — it will drift if the pipeline runs again; re-run `axiom.concept_graph.analyze()` against `data/axiom.db` for the current numbers.** |
| R8 Structural retrieval arm | done, documented no-op | `eval/report_hybrid_struct.md` | nDCG@10=0.890 vs hybrid's 0.932; eval corpus has no `paper_concepts` by design (OD11/OD20), so the arm is inert; delta is a top-200-vs-top-10 pool-size artifact, not structural signal |
| R9 Eval v2 (~50 queries) | done (AI-drafted) | `eval/ndcg_queries_v2.json`, `scripts/build_eval_v2_queries.py` | 51 real queries, pooled dense+hybrid, graded by `claude-haiku-4-5`; NOT yet reviewed by `scripts/label_cli.py` |
| R10 Tracing + benchmark | done | `eval/agent_bench.json` | rejection rate first_pass=10.9% final=0.8%; latency p50=6.80s p95=10.69s; $0.003/paper, $0.0901/run |
| R11 Framework-free | confirmed true | `requirements.txt` (no diff needed — nothing added fits that category); `truststore` (OD24) is a TLS-trust shim, not an agent framework | plain `asyncio`+`httpx`+Pydantic v2 throughout |

## Retrieval eval (v2, real eval corpus, 51 queries)

| Mode | Mean nDCG@10 | Report |
|---|---|---|
| dense | 0.828 | `eval/report_dense_v2.md` |
| hybrid | 0.932 | `eval/report_hybrid_v2.md` |
| hybrid_struct | 0.890 | `eval/report_hybrid_struct.md` |
| dense_general | not run | needs a separate one-time embedding cache (`BAAI/bge-small-en-v1.5`), deferred as optional per spec |

v1 (unchanged for comparison): hybrid 0.453 (`eval/report.md`, still on disk, re-verified
2026-10-31), dense 0.345 (no standalone v1 dense report file currently on disk; this figure is
pre-existing project history recorded in `docs/DECISIONS.md` and `docs/archive/Backlog.md` before this task
started, not reproduced here). Not directly comparable to v2 (different queries,
different scale, different corpus coverage after the real 1500-paper eval ingest vs v1's
original sample).

## Tests

223 collected, 222 passed, 1 pre-existing failure unrelated to this work
(`tests/test_calibrate.py::test_end_to_end_writes_json_and_report`, failing on an assertion that
the generated report contains "Chosen τ" — `scripts/calibrate_gap_thresholds.py` does have a
genuine Windows-console `UnicodeEncodeError` on a `→` character when run directly outside
pytest's output capture, but the pytest-level failure itself is a report-content mismatch under
pytest's capture machinery, not a fully pinned-down root cause. Reproduces identically with none
of this feature's changes present, and no commit on this branch touches that script or its test).

## Integration bugs found and fixed during Build/Test/Review

Code-level fixes, all committed (see `git log` on this branch): `graph._pagerank` directed-only
vs the new undirected concept graph; `axiom/embed.py`'s blocking HuggingFace freshness check on
a network with a TLS-inspecting proxy; `.env` never actually loaded; the same TLS issue blocking
the real Anthropic API (fixed with `truststore`, OD24, a user-approved new dependency); a
float64 cosine landing a hair past 1.0 and crashing Pydantic validation instead of reaching the
verifier; `paper_id` missing from every trace line, breaking `bench_agents.py`'s per-paper math;
`LLM_BUDGET_USD` enforced per-run instead of globally; raw SQL outside `axiom/db.py`.

Local-environment fixes with no commit (touch only gitignored `data/` paths, so there's nothing
to trace in `git log` — expected, not a gap): `data/axiom.db` predated the new schema tables,
migrated additively via `db.init_db()`; `data/FULL_DATASET.jsonl` was at the repo root instead of
where `scripts/ingest_eval_corpus.py` expects it, copied into place.

## Known gaps / not done

- Synthesis (cluster labels + hypothesis pitch, strong tier) wiring landed in stage-2 fixes
  (`POST /agents/run {"synthesize": true}`) but hasn't been exercised with a real run yet.
- `dense_general` eval mode not run (needs a separate heavy embedding-cache build).
- `scripts/label_cli.py review` human pass on the v2 labels not run (needs an actual annotator).
- `_ensure_budget`/`_record_spend` has a bounded check-then-record race under concurrency
  (worst case ≈$0.12 over the $3 cap at `AGENT_CONCURRENCY=4`) — logged as OD25 (stage 1);
  stage-2's C3 fix supersedes this for the Anthropic path (atomic reserve/settle).

## Deferred (stage-2 fixes, 2026-10-31)

D5 (structural arm + 300-paper eval run) and E3 (second grader + kappa) were **completed**
this stage — see the stage-2 update above, not deferred. Genuinely deferred:

- **`dense_general` eval mode** — needs a separate one-time embedding-cache build
  (`BAAI/bge-small-en-v1.5` over the full eval corpus), a heavier one-off job not run this
  stage; optional per the original spec.
- **Architecture/security review pass** — stage-1 had a dedicated review phase (OD25); this
  stage's real-run-driven bug-fixing (temperature, budget estimator, migrations) covers a lot
  of the same ground a review would have flagged, but a dedicated second pass wasn't run.
- **Fact-checker pass** — stage-1 had a dedicated pass verifying every claim in RESULTS/README
  against code and `eval/` files; this update was written directly against the same evidence
  (file paths, real numbers) but wasn't independently re-verified by a fresh pass.
- **E3's full pool / deeper review** — only a fixed 10-query sample was AI-re-graded, not the
  full 51-query pool; a deeper pass (more queries, or a third grader) would strengthen the
  kappa measurement's statistical weight but wasn't run this stage.
