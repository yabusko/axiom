"""Stage 4 B1: an AI judge (strong tier) rates the top-10 gap candidates for
5 real topics against the real prebuilt base corpus -- a product-validity
check, not a retrieval-quality eval (that's scripts/eval_ndcg.py). A gap
candidate can be numerically well-formed (two sizable, semantically close,
weakly-cited communities -- OD9/OD17's g_score) and still be a bad pitch to a
researcher: two communities that are close in embedding space only because
SPECTER2 centroids compress everything into a narrow band (OD37's known
failure mode), or a pair that's "weakly cited" only because the field is
small, not because the connection is genuinely unexplored.

Rubric (1-5 each, independent of g_score, which the judge is NOT shown --
otherwise it would just be grading g_score's own arithmetic back to itself):
  - plausibility: is this a real, substantive connection between two actual
    sub-areas, not noise from small-sample community detection?
  - novelty: does the pairing look genuinely underexplored, not an
    already-well-trodden combination that just happens to undercite?
  - actionability: could a researcher turn this into a concrete next paper?

Writes eval/gap_judgments.json (raw) + eval/gap_judgments_report.md
(per-topic and overall means). Real LLM spend, tracked durably via a single
run_id through axiom.llm_providers same as every other real-spend script
this stage -- `--budget` stops before exceeding it, default $0.50.

Usage (from repo root):
    python scripts/judge_gaps.py --estimate-only
    python scripts/judge_gaps.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field, ValidationError

from axiom import config, db, graph as graphmod, llm_providers
from axiom.corpus_slice import compute_slice
from axiom.embed import Specter2Encoder
from axiom.gaps import analyze
from axiom.qdrant_client import AxiomQdrant

TOPICS = [
    "retrieval-augmented generation",
    "dense passage retrieval",
    "hallucination in question answering",
    "in-context learning",
    "chain of thought prompting",
]
TOP_N_GAPS = 10
TOP_N_PAPERS_PER_SIDE = 3
OUT_JSON = config.PROJECT_ROOT / "eval" / "gap_judgments.json"
OUT_REPORT = config.PROJECT_ROOT / "eval" / "gap_judgments_report.md"

SYSTEM = (
    "You are a skeptical research-gap reviewer. You are shown two paper "
    "clusters from a citation graph that a tool flagged as a candidate "
    "research gap: close in meaning but weakly connected by citations. "
    "Judge whether this is a real, worthwhile gap for a researcher to "
    "pursue, not whether the tool's math is internally consistent. Be "
    "skeptical of pairs that look like an artifact of a small corpus or "
    "noisy community detection rather than a genuine blind spot."
)


class GapJudgment(BaseModel):
    plausibility: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    worth_pursuing: bool
    rationale: str


def _side_block(label: str, size: int, papers: list) -> str:
    lines = [f"Cluster: {label} ({size} papers)"]
    for p in papers:
        lines.append(f"  - {p.title or p.paper_id} ({p.year or '?'})")
    return "\n".join(lines)


def build_prompt(topic: str, gap, g) -> str:
    top_a = _top_papers(g, gap.a.members, TOP_N_PAPERS_PER_SIDE)
    top_b = _top_papers(g, gap.b.members, TOP_N_PAPERS_PER_SIDE)
    return (
        f"Field topic: {topic}\n\n"
        f"{_side_block(gap.a.labels[0] if gap.a.labels else 'cluster A', gap.a.size, top_a)}\n\n"
        f"{_side_block(gap.b.labels[0] if gap.b.labels else 'cluster B', gap.b.size, top_b)}\n\n"
        f"These two clusters cite each other only {gap.inter_citations} time(s).\n\n"
        "Rate this candidate gap on the rubric. plausibility/novelty/actionability "
        "are each a single integer from 1 to 5 (never a fraction, never out of 5 "
        "again -- just the digit 1, 2, 3, 4, or 5)."
    )


def _top_papers(g, member_ids: list[str], top_k: int) -> list:
    ranked = sorted(
        member_ids, key=lambda pid: g.nodes[pid].get("cited_by_count", 0) if pid in g.nodes else 0,
        reverse=True,
    )[:top_k]
    return [graphmod.PaperRank(
        paper_id=pid, title=g.nodes.get(pid, {}).get("title"), year=g.nodes.get(pid, {}).get("year"),
        venue=g.nodes.get(pid, {}).get("venue"), cited_by_count=g.nodes.get(pid, {}).get("cited_by_count", 0),
        local_in_degree=g.in_degree(pid) if pid in g.nodes else 0, pagerank=0.0,
    ) for pid in ranked]


def estimate_cost(n_gaps: int) -> float:
    """~900 chars prompt (3 papers/side titles) + ~400 chars system; output is
    a short rubric response (~120 tokens). chars/2 ratio per the same real
    B3 calibration scripts/ai_second_grader_sample.py already measured and
    documents for this kind of short technical text."""
    tokens_in = (len(SYSTEM) + 900) / 2
    tokens_out = 120
    per_call = ((tokens_in / 1_000_000) * config.MODEL_STRONG_PRICE_IN
                + (tokens_out / 1_000_000) * config.MODEL_STRONG_PRICE_OUT)
    return per_call * n_gaps


async def judge_one(gap, g, topic: str, run_id: str, max_attempts: int = 3) -> GapJudgment | None:
    """A single out-of-range field (seen for real: the model once returned
    `55`/`35`/`45` instead of `5`/`3`/`4`) shouldn't abort a run that's
    already spent real money on earlier gaps -- retry a few times, then
    return None (recorded as "unjudged" in the output) rather than crash."""
    prompt = build_prompt(topic, gap, g)
    for attempt in range(1, max_attempts + 1):
        try:
            return await llm_providers.complete_json(
                "strong", SYSTEM, prompt, GapJudgment, run_id=run_id, agent="judge_gaps",
                attempt=attempt,
            )
        except ValidationError as exc:
            print(f"    [judge_gaps] attempt {attempt}/{max_attempts} failed validation "
                  f"for gap {gap.gap_id[:8]}: {exc.errors()[0]['msg'] if exc.errors() else exc}")
    return None


async def run(budget: float, estimate_only: bool) -> None:
    meta_conn = db.connect(config.CORPUS_DB_PATH)
    meta = db.get_corpus_meta(meta_conn)
    meta_conn.close()
    if meta is None or meta["papers_count"] == 0:
        print("Base corpus not built yet -- run scripts/build_corpus.py first.")
        return

    print(f"Loading base corpus (graph + vectors, {meta['papers_count']} papers)...")
    full_graph = graphmod.load_graph(db.connect(config.CORPUS_DB_PATH))
    vectors = AxiomQdrant(collection=config.CORPUS_COLLECTION_NAME).fetch_dense_vectors()
    encoder = Specter2Encoder()

    per_topic_gaps: dict[str, list] = {}
    per_topic_graph: dict[str, object] = {}
    for topic in TOPICS:
        query_vector = encoder.encode_query(topic)
        slice_result = compute_slice(
            full_graph, vectors, query_vector,
            target_n=config.FIELD_SLICE_TARGET, floor=config.FIELD_RELEVANCE_FLOOR,
        )
        sub_g = full_graph.subgraph(slice_result.paper_ids).copy()
        conn = db.connect(config.CORPUS_DB_PATH)
        try:
            gap_analysis = analyze(sub_g, conn, vectors, top_k=TOP_N_GAPS)
        finally:
            conn.close()
        gaps = sorted(gap_analysis.gaps, key=lambda c: c.g_score, reverse=True)[:TOP_N_GAPS]
        per_topic_gaps[topic] = gaps
        per_topic_graph[topic] = sub_g
        print(f"  {topic}: {len(slice_result.paper_ids)} papers, {len(gaps)} candidate gaps")

    n_total_gaps = sum(len(v) for v in per_topic_gaps.values())
    estimate = estimate_cost(n_total_gaps)
    print(f"\n[judge_gaps] {n_total_gaps} gaps to judge, worst-case cost estimate: ${estimate:.4f}")
    if estimate_only:
        return
    if estimate > budget:
        print(f"[judge_gaps] estimate ${estimate:.4f} exceeds --budget ${budget:.2f} -- aborting, "
              "no LLM calls made. Re-run with a higher --budget to proceed.")
        return

    run_id = uuid.uuid4().hex
    results: dict[str, list[dict]] = {}
    for topic, gaps in per_topic_gaps.items():
        g = per_topic_graph[topic]
        results[topic] = []
        for gap in gaps:
            judgment = await judge_one(gap, g, topic, run_id)
            if judgment is None:
                print(f"    [{topic}] {gap.gap_id[:8]} -> SKIPPED (no valid judgment after retries)")
                continue
            results[topic].append({
                "gap_id": gap.gap_id,
                "a_label": gap.a.labels[0] if gap.a.labels else None,
                "b_label": gap.b.labels[0] if gap.b.labels else None,
                "inter_citations": gap.inter_citations,
                "g_score": gap.g_score,
                "judgment": judgment.model_dump(),
            })
            print(f"    [{topic}] {gap.gap_id[:8]} -> plausibility={judgment.plausibility} "
                  f"novelty={judgment.novelty} actionability={judgment.actionability} "
                  f"worth_pursuing={judgment.worth_pursuing}")

    spend = llm_providers.spend_so_far(run_id)
    payload = {
        "_provenance": f"AI-judged ({llm_providers.model_for_tier('strong')}), stage 4 B1 -- see "
                        "scripts/judge_gaps.py's module docstring for the rubric and what it "
                        "deliberately withholds (g_score) from the judge.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "spend_usd": round(spend, 4),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(payload)
    print(f"\n[judge_gaps] wrote {OUT_JSON} and {OUT_REPORT}")
    print(f"[judge_gaps] actual spend this run: ${spend:.4f}")


def write_report(payload: dict) -> None:
    lines = [
        "# Gap judge report (stage 4 B1)",
        "",
        f"Generated {payload['generated_at']} · {len(payload['results'])} topics · "
        f"spend ${payload['spend_usd']:.4f}",
        "",
        payload["_provenance"],
        "",
    ]
    all_scores = {"plausibility": [], "novelty": [], "actionability": []}
    all_worth = []
    for topic, rows in payload["results"].items():
        lines.append(f"## {topic} ({len(rows)} gaps judged)")
        lines.append("")
        lines.append("| Gap | A | B | citations | plausibility | novelty | actionability | worth pursuing |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            j = r["judgment"]
            lines.append(
                f"| {r['gap_id'][:8]} | {r['a_label']} | {r['b_label']} | {r['inter_citations']} | "
                f"{j['plausibility']} | {j['novelty']} | {j['actionability']} | "
                f"{'yes' if j['worth_pursuing'] else 'no'} |"
            )
            for k in all_scores:
                all_scores[k].append(j[k])
            all_worth.append(j["worth_pursuing"])
        lines.append("")

    n = len(all_worth)
    if n:
        lines.append("## Overall")
        lines.append("")
        for k, vals in all_scores.items():
            lines.append(f"- Mean {k}: {sum(vals) / n:.2f} / 5")
        lines.append(f"- Worth pursuing: {sum(all_worth)}/{n} "
                      f"({100 * sum(all_worth) / n:.0f}%)")
        lines.append("")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=0.50)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args(argv)
    asyncio.run(run(args.budget, args.estimate_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
