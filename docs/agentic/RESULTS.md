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
  architecture/security review pass, an automated accuracy check on generated hypotheses,
  `dense_general` eval mode.

---

## Stage-1 history (2026-10-30, superseded by the stage-2 update above)

Initial 30-paper synthetic-corpus pipeline run + v1 retrieval eval, 2026-10-30 -- every number
and status row has since been superseded (some corrected, not just updated); see `docs/DECISIONS.md` OD19-OD25 for the full decision record, not restated here.
