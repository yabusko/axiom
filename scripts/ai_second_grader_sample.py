"""AI second-grader review for a sample of eval v2 queries (E3, stage-2 fixes).

The original `scripts/build_eval_v2_queries.py` graded every query's pooled
candidates once, with the `fast` tier. This script picks a fixed-seed random
sample of `N_QUERIES` of those queries and has a SECOND, independent judge
(the `strong` tier) re-grade the SAME pooled candidate pool from scratch --
same rubric, same 0-2 scale -- blind to the first grader's judgments and to
the original retrieval ranks (it never sees them; it only sees query +
candidate id/title/abstract, exactly like the first grader did).

Writes `eval/ndcg_v2_second_grader_sample.json` (the original
`ndcg_queries_v2.json` is never modified) and reports Cohen's kappa between
the two graders on the sampled queries' judgments -- this is AI-vs-AI
agreement, not a human-vs-AI comparison; label it that way everywhere.

Usage (from repo root):
    python scripts/ai_second_grader_sample.py --estimate-only
    python scripts/ai_second_grader_sample.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from axiom import config, db, llm_providers
from scripts.build_eval_v2_queries import GradingResult, grade_query
from scripts.label_agreement import compute_agreement

N_QUERIES = 10
SAMPLE_SEED = 9012
SOURCE_PATH = config.EVAL_V2_QUERIES_PATH
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "ndcg_v2_second_grader_sample.json"
PROVENANCE = "AI-labeled second-grader review (sonnet-5-5, independent of the original fast-tier grader)"


def sample_query_indices(n_total: int, n_sample: int, seed: int = SAMPLE_SEED) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), min(n_sample, n_total)))


def load_source() -> dict:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def candidates_for(conn, paper_ids: list[str]) -> list[dict]:
    if not paper_ids:
        return []
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"SELECT openalex_id, title, abstract FROM papers WHERE openalex_id IN ({placeholders})",
        paper_ids,
    ).fetchall()
    return [{"paper_id": r["openalex_id"], "title": r["title"], "abstract": r["abstract"]}
            for r in rows]


def estimate_cost(data: dict, indices: list[int]) -> float:
    """A realistic expected-cost estimate (not a worst-case bound -- grade_query's own
    max_tokens is the flat 4096 default, assuming that every call would be wildly
    pessimistic). chars/2, not chars/4: a real B3 calibration run measured a ~2.0
    chars/token ratio for this kind of short technical text (see
    llm_providers._estimate_worst_case_cost's docstring); output is estimated at ~25 tokens
    per candidate judgment plus a small fixed JSON overhead, not the full max_tokens cap."""
    total = 0.0
    for i in indices:
        q = data["queries"][i]
        n_candidates = len(q["judgments"])
        # grade_query's prompt includes each candidate's title + up to 500 chars of abstract,
        # plus ~400 chars of system/boilerplate text.
        chars_in = 400 + n_candidates * 550
        tokens_in = chars_in / 2
        tokens_out = 50 + n_candidates * 25
        total += ((tokens_in / 1_000_000) * config.MODEL_STRONG_PRICE_IN
                  + (tokens_out / 1_000_000) * config.MODEL_STRONG_PRICE_OUT)
    return total


async def run(estimate_only: bool) -> None:
    data = load_source()
    indices = sample_query_indices(len(data["queries"]), N_QUERIES)
    print(f"[second_grader] sampled {len(indices)}/{len(data['queries'])} queries "
          f"(seed={SAMPLE_SEED}): {indices}")

    estimate = estimate_cost(data, indices)
    print(f"[second_grader] worst-case cost estimate: ${estimate:.4f}")
    if estimate_only:
        return

    conn = db.connect(config.EVAL_DB_PATH)
    try:
        # data/eval.db predates the agentic schema (OD19+) -- it's kept separate/frozen from
        # the main corpus db per OD11's isolation discipline, so nothing has ever run
        # db.init_db() against it; the llm_spend table (C3) doesn't exist there yet.
        db.init_db(conn)
        before = db.total_llm_spend(conn)
        run_id = uuid.uuid4().hex
        async with httpx.AsyncClient(
            base_url=config.ANTHROPIC_BASE_URL, timeout=config.ANTHROPIC_TIMEOUT,
        ) as client:
            results = {}
            for i in indices:
                q = data["queries"][i]
                paper_ids = list(q["judgments"].keys())
                candidates = candidates_for(conn, paper_ids)
                # grade_query doesn't accept client/conn (build_eval_v2_queries.py predates
                # that seam) -- it still works, just opens its own client/uses the in-process
                # budget total for this one query's single call rather than the pooled one.
                result: GradingResult = await grade_query(q["query"], candidates, run_id)
                results[i] = {j.paper_id: j.relevance for j in result.judgments}
                print(f"[second_grader] [{i}] '{q['query']}': {len(candidates)} candidates graded")
        after = db.total_llm_spend(conn)
    finally:
        conn.close()

    out_queries = []
    for i in indices:
        q = data["queries"][i]
        out_queries.append({
            "query": q["query"],
            "query_index_in_source": i,
            "original_judgments": q["judgments"],
            "second_grader_judgments": results[i],
        })

    payload = {
        "_provenance": PROVENANCE,
        "_source_file": str(SOURCE_PATH.name),
        "_sample_seed": SAMPLE_SEED,
        "queries": out_queries,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # AI-vs-AI agreement: flatten both graders' judgments across the sampled queries into
    # (query_index, paper_id) -> relevance maps and compute kappa, same helper E1/E3's
    # scripts/label_agreement.py already provides.
    original_flat, second_flat = {}, {}
    for i in indices:
        q = data["queries"][i]
        for pid, rel in q["judgments"].items():
            original_flat[f"{i}::{pid}"] = rel
        for pid, rel in results[i].items():
            second_flat[f"{i}::{pid}"] = rel
    agreement = compute_agreement(original_flat, second_flat)

    print(f"\n[second_grader] wrote {len(out_queries)} queries -> {OUT_PATH}")
    print(f"[second_grader] actual spend this run: ${after - before:.4f} "
          f"(cumulative total: ${after:.4f} / ${config.LLM_BUDGET_USD:.2f} cap)")
    print(f"[second_grader] AI-vs-AI agreement over {agreement['n_common']} judgments: "
          f"rate={agreement['agreement_rate']:.3f} kappa={agreement['kappa']:.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args(argv)
    asyncio.run(run(args.estimate_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
