"""Generate the real eval v2 query/judgment file (R9, AXIOM_AGENTIC_TASK.md §4).

For each of ~50 queries: pool top-k candidates from `dense` + `hybrid` against
the eval corpus (config.EVAL_COLLECTION_NAME / config.EVAL_DB_PATH), grade the
pooled set 0..config.EVAL_V2_SCALE_MAX with the `fast` tier (one LLM call per
query, all its candidates at once), and write config.EVAL_V2_QUERIES_PATH in
the schema scripts/eval_ndcg.py and scripts/label_cli.py already expect
(`judgments` + a parallel `judgments_meta` carrying provenance/annotator per
label -- see eval/ndcg_queries_v2.json's existing skeleton/docstring).

Every judgment is `_provenance: "AI-drafted"` until a human runs
`scripts/label_cli.py review` to override individual labels.

Usage (from repo root):
    python scripts/build_eval_v2_queries.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from axiom import config, db, llm_providers
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant

# Reuses v1's 15 queries verbatim (same topics, still plausible in the eval
# sample) plus 35 new topics spanning the venues/years the eval corpus draws
# from (ACL/EMNLP/COLING/NAACL 2020-2025) -- a reasonable query-topic spread,
# not sourced from any single paper.
QUERIES = [
    "retrieval-augmented generation for question answering",
    "parameter-efficient fine-tuning with low-rank adapters",
    "detecting and mitigating hallucination in large language models",
    "chain-of-thought prompting for reasoning tasks",
    "contrastive learning for sentence embeddings",
    "cross-lingual transfer for low-resource languages",
    "instruction tuning for large language models",
    "knowledge distillation for compact language models",
    "bias and fairness evaluation in NLP models",
    "dense passage retrieval for open-domain QA",
    "named entity recognition with pretrained transformers",
    "abstractive text summarization with pretrained models",
    "dialogue state tracking for task-oriented dialogue",
    "data augmentation techniques for text classification",
    "few-shot learning for natural language understanding",
    "prompt engineering for zero-shot classification",
    "speech recognition with self-supervised pretraining",
    "machine translation for low-resource language pairs",
    "sentiment analysis on social media text",
    "coreference resolution in long documents",
    "graph neural networks for knowledge base completion",
    "adversarial robustness of text classifiers",
    "multi-hop reasoning over knowledge graphs",
    "topic modeling for large text corpora",
    "aspect-based sentiment analysis",
    "neural machine translation with transformer architectures",
    "text-to-SQL semantic parsing",
    "event extraction from news text",
    "relation extraction with distant supervision",
    "question generation from text passages",
    "evaluation metrics for text generation quality",
    "self-supervised pretraining objectives for language models",
    "code generation with large language models",
    "commonsense reasoning benchmarks for language models",
    "long-document summarization with hierarchical attention",
    "entity linking to knowledge bases",
    "paraphrase detection and generation",
    "hate speech and toxic language detection",
    "active learning for text annotation",
    "multimodal learning combining vision and language",
    "constituency and dependency parsing with neural models",
    "word sense disambiguation with contextual embeddings",
    "stance detection in argumentative text",
    "numerical and temporal reasoning in text",
    "model compression and quantization for deployment",
    "calibration and uncertainty estimation in NLP models",
    "span-based extractive question answering",
    "text style transfer and controllable generation",
    "legal and biomedical domain adaptation for NLP",
    "benchmark dataset construction for NLP evaluation",
    "interpretability and attention analysis in transformers",
]


class QueryJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    relevance: int = Field(ge=0, le=config.EVAL_V2_SCALE_MAX)
    reason: Literal["off_topic", "tangential", "on_topic"]


class GradingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judgments: list[QueryJudgment]


_SYSTEM = (
    "You are grading retrieval relevance for an IR evaluation. Given a search "
    "query and a list of candidate papers (id, title, abstract), score each "
    "candidate's relevance to the query on a 0-2 scale: 0 = not relevant/off "
    "topic, 1 = tangentially/marginally relevant, 2 = clearly and directly "
    "relevant. Judge the paper's actual topic against the query, not surface "
    "keyword overlap. Return a judgment for every candidate id given, in any "
    "order."
)


def _prompt(query: str, candidates: list[dict]) -> str:
    lines = [f"Query: {query}", "", "Candidates:"]
    for c in candidates:
        abstract = (c["abstract"] or "")[:500]
        lines.append(f"- id={c['paper_id']} | title: {c['title']} | abstract: {abstract}")
    return "\n".join(lines)


class GradingValidationError(RuntimeError):
    """The grader's judgment set didn't exactly cover the candidate pool --
    duplicate id(s), id(s) outside the pool, or candidate id(s) left
    unjudged -- after every retry was exhausted."""


def validate_judgment_coverage(
    candidate_ids: list[str], judgments: list[QueryJudgment],
) -> list[str]:
    """Every candidate id must get EXACTLY one judgment: no duplicates, no
    missing ids, no judgments for ids outside the pool. Returns a list of
    human-readable problems (empty == valid)."""
    candidate_set = set(candidate_ids)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for j in judgments:
        if j.paper_id in seen:
            duplicates.add(j.paper_id)
        seen.add(j.paper_id)

    problems = []
    if duplicates:
        problems.append(f"duplicate judgment(s) for id(s): {', '.join(sorted(duplicates))}")
    extra = seen - candidate_set
    if extra:
        problems.append(f"judgment(s) for id(s) not in the candidate pool: {', '.join(sorted(extra))}")
    missing = candidate_set - seen
    if missing:
        problems.append(f"missing judgment(s) for id(s): {', '.join(sorted(missing))}")
    return problems


async def grade_query(
    query: str, candidates: list[dict], run_id: str, *, max_retries: int | None = None,
    tier: str = "fast",
) -> GradingResult:
    """Grade one query's pooled candidates with `tier` (defaults to `fast`, this script's
    own usage), retrying (same agent, errors folded back into the prompt) until the judgment
    set covers the candidate pool exactly once each -- a grader that drops an id, duplicates
    one, or invents one outside the pool silently corrupts the nDCG labels downstream, so
    that's checked here rather than trusted. `tier="strong"` (or `"opus"`) is how
    scripts/ai_second_grader_sample.py gets a genuinely different, stronger model for its
    independent second opinion -- it is NOT a re-run of the same `fast` model."""
    max_retries = max_retries if max_retries is not None else config.AGENT_MAX_RETRIES
    candidate_ids = [c["paper_id"] for c in candidates]
    base_prompt = _prompt(query, candidates)

    last_exc: Exception | None = None
    prompt = base_prompt
    for attempt in range(1, max_retries + 1):
        try:
            result = await llm_providers.complete_json(
                tier, _SYSTEM, prompt, GradingResult,
                run_id=run_id, agent="eval_v2_grading", attempt=attempt,
            )
        except (ValidationError, llm_providers.LLMError) as exc:
            last_exc = exc
            prompt = base_prompt
            continue

        problems = validate_judgment_coverage(candidate_ids, result.judgments)
        if not problems:
            return result
        last_exc = GradingValidationError(
            f"{len(result.judgments)} judgment(s) for {len(candidate_ids)} candidate(s): "
            + "; ".join(problems)
        )
        prompt = (
            f"{base_prompt}\n\nYour previous response did not match the candidate set "
            f"exactly: {'; '.join(problems)}. Return EXACTLY one judgment per candidate id "
            "listed above -- no duplicates, no missing ids, no extra ids."
        )

    raise last_exc  # type: ignore[misc]  # loop always either returns or sets last_exc


async def main() -> None:
    encoder = Specter2Encoder()
    store = AxiomQdrant(collection=config.EVAL_COLLECTION_NAME)
    if store.count() == 0:
        raise SystemExit(
            f"'{config.EVAL_COLLECTION_NAME}' is empty -- run scripts/ingest_eval_corpus.py first."
        )
    conn = db.connect(config.EVAL_DB_PATH)

    run_id = "eval_v2_build"
    out_queries = []
    for i, query in enumerate(QUERIES, start=1):
        vec = encoder.encode_query(query)
        dense_hits = store.search(vec, top_k=10)
        hybrid_hits = store.search_hybrid(query_vector=vec, query_text=query, top_k=10)

        pooled_from: dict[str, set[str]] = {}
        for mode, hits in (("dense", dense_hits), ("hybrid", hybrid_hits)):
            for h in hits:
                pooled_from.setdefault(h.paper_id, set()).add(mode)

        paper_ids = list(pooled_from.keys())
        if not paper_ids:
            print(f"[{i}/{len(QUERIES)}] '{query}': no candidates, skipping")
            continue
        placeholders = ",".join("?" for _ in paper_ids)
        rows = conn.execute(
            f"SELECT openalex_id, title, abstract FROM papers WHERE openalex_id IN ({placeholders})",
            paper_ids,
        ).fetchall()
        candidates = [{"paper_id": r["openalex_id"], "title": r["title"], "abstract": r["abstract"]}
                      for r in rows]

        result = await grade_query(query, candidates, run_id)
        judgments = {j.paper_id: j.relevance for j in result.judgments}
        judgments_meta = {
            j.paper_id: {"provenance": "AI-drafted", "annotator": None, "note": j.reason}
            for j in result.judgments
        }
        out_queries.append({
            "query": query,
            "pooled_from": sorted({m for modes in pooled_from.values() for m in modes}),
            "judgments": judgments,
            "judgments_meta": judgments_meta,
        })
        print(f"[{i}/{len(QUERIES)}] '{query}': {len(candidates)} candidates graded")

    conn.close()

    payload = {
        "_provenance": "AI-drafted (graded by config.MODEL_FAST via scripts/build_eval_v2_queries.py); "
                        "not yet reviewed by scripts/label_cli.py.",
        "_scale": "0-2 (0=not relevant, 1=marginally relevant, 2=clearly relevant)",
        "_scale_max": config.EVAL_V2_SCALE_MAX,
        "queries": out_queries,
    }
    config.EVAL_V2_QUERIES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[write] {len(out_queries)} queries -> {config.EVAL_V2_QUERIES_PATH}")
    print(f"[cost] run {run_id}: ${llm_providers.spend_so_far(run_id):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
