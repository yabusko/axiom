# Query-aware structural retrieval arm (D5, stage-2 fixes)

Mean nDCG@10 over 51 v2 queries, same candidate pool size for all three arms:

| Arm | Mean nDCG@10 |
|---|---|
| hybrid alone | 0.8904 |
| hybrid + pool-cohesion structural (existing `hybrid_struct`) | 0.6168 |
| hybrid + query-aware structural (D5, this script) | 0.8727 |

44/51 queries matched at least one ontology concept via the B1 cascade (exact/normalized, then embedding gated on VERIFIER_COSINE_MIN=0.93/ENRICHMENT_MARGIN_DELTA=0.0, OD28's calibrated values).

> Judgments in `eval/ndcg_queries_v2.json` are AI-labeled (fast tier originally; see OD29 for an independent AI second opinion on a 10-query sample), no human review.
