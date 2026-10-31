# Enrichment embedding-match threshold calibration (B3, stage-2 fixes)

- Source labels: `eval/enrichment_pairs.csv`, AI-labeled (sonnet-5-5 + opus-5-5 agreement; disagreements and unsure excluded) -- no human review.
- Labeled pairs used: 92
- Rule: highest recall at precision >= 0.9
- Chosen: tau=0.93, delta=0.0 (precision=0.929, recall=0.867, tp=13 fp=1 fn=2 tn=76)
- Of 208 `new_concept` entities in the reference run, 6 would flip to a real embedding match under this threshold.
