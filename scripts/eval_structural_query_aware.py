"""Query-aware structural retrieval arm (D5, stage-2 fixes).

The existing `scripts/eval_ndcg.py --mode hybrid_struct` structural arm scores a candidate
by how well its concepts overlap with the OTHER candidates already in its own retrieved
pool (pool cohesion) -- it never looks at the query itself. D5 asks for a genuinely
query-aware arm instead: map the query text to ontology concepts using the same B1 matching
cascade (no LLM call), then score each candidate by how many of ITS OWN paper_concepts match
those query concepts (weighted by how many concepts overlap, not just whether any do).

Needs real `paper_concepts` data on the eval corpus first -- see
logs/run_agent_benchmark_eval_corpus.py (a real pipeline run on a paper subset; not
committed, lives in logs/). Compares three arms at the same pool size (top_k, with the same
candidate pool pulled via `--mode hybrid`'s own pool): hybrid alone, hybrid + the existing
pool-cohesion structural arm, hybrid + this query-aware arm. Reports nDCG@10 for each,
honestly, whichever comes out ahead.

Usage (from repo root):
    python scripts/eval_structural_query_aware.py --k 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from axiom import config, db
from axiom.agents import enrichment, verifier
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant
from scripts.eval_ndcg import ndcg_at_k, rrf_fuse, structural_scores

REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "report_hybrid_struct_query_aware.md"
MAX_SPAN_WORDS = 4


def query_spans(query: str, max_n: int = MAX_SPAN_WORDS) -> list[str]:
    """Every contiguous word n-gram up to `max_n` words, plus the whole query -- the
    candidate phrases checked against the ontology. Longest first (not that it changes
    which concepts get matched, just makes the matching loop's early results the more
    specific ones, easier to read if ever printed for debugging)."""
    words = query.split()
    spans = {query}
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            spans.add(" ".join(words[i:i + n]))
    return sorted(spans, key=len, reverse=True)


async def match_against_ontology(
    text: str, *, encoder, ontology: enrichment.Ontology, tau: float, delta: float,
) -> str | None:
    """Same two-step cascade B1 uses for entities (exact/normalized, then embedding gated on
    cosine+margin) -- but returns None instead of falling through to new_concept, since a
    query span that isn't a real ontology concept shouldn't score anything."""
    name_norm = verifier.normalize_text(text)
    idx = ontology.normalized_labels.get(name_norm)
    if idx is not None:
        return ontology.labels[idx]
    if ontology.vectors.shape[0] == 0:
        return None
    raw = await asyncio.to_thread(encoder.encode, [text])
    vec = enrichment._normalize_rows(np.array(raw, dtype=float))
    sims = (vec @ ontology.vectors.T)[0]
    order = np.argsort(sims)[::-1]
    top1_idx = int(order[0])
    top1 = max(-1.0, min(1.0, float(sims[top1_idx])))
    top2 = max(-1.0, min(1.0, float(sims[int(order[1])]))) if len(order) > 1 else -1.0
    if top1 >= tau and (top1 - top2) >= delta:
        return ontology.labels[top1_idx]
    return None


async def query_concepts(
    query: str, *, encoder, ontology: enrichment.Ontology,
    tau: float = config.VERIFIER_COSINE_MIN, delta: float = config.ENRICHMENT_MARGIN_DELTA,
) -> set[str]:
    """Every ontology concept any word-span of `query` maps to -- the query's own concept
    set, built with exactly the thresholds B1's entity matching was calibrated with (OD28),
    not a separate tuning knob."""
    matched = set()
    for span in query_spans(query):
        concept = await match_against_ontology(span, encoder=encoder, ontology=ontology,
                                                tau=tau, delta=delta)
        if concept:
            matched.add(concept)
    return matched


def query_aware_structural_scores(
    conn, candidate_ids: list[str], concepts: set[str],
) -> dict[str, float]:
    """Weighted overlap: each candidate's score is how many of the query's matched
    concepts it shares -- not binary presence/absence, so a candidate sharing 3 query
    concepts outranks one sharing 1, same "weighted" meaning scripts/eval_ndcg.py's own
    pool-cohesion arm uses (counts, not booleans)."""
    if not concepts or not candidate_ids:
        return {}
    placeholders = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT paper_id, canonical FROM paper_concepts WHERE paper_id IN ({placeholders})",
        list(candidate_ids),
    ).fetchall()
    by_paper: dict[str, set[str]] = {}
    for r in rows:
        by_paper.setdefault(r["paper_id"], set()).add(r["canonical"])
    return {pid: float(len(by_paper.get(pid, set()) & concepts)) for pid in candidate_ids}


async def run(k: int) -> None:
    queries_data = json.loads(config.EVAL_V2_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = queries_data["queries"]

    encoder = Specter2Encoder()
    store = AxiomQdrant(collection=config.EVAL_COLLECTION_NAME)
    if store.count() == 0:
        raise SystemExit(f"'{config.EVAL_COLLECTION_NAME}' is empty -- run "
                          "scripts/ingest_eval_corpus.py first.")

    conn = db.connect(config.EVAL_DB_PATH)
    n_concepts = conn.execute("SELECT COUNT(*) FROM paper_concepts").fetchone()[0]
    if n_concepts == 0:
        conn.close()
        raise SystemExit(
            "paper_concepts is empty on the eval corpus -- run the agent pipeline over a "
            "real subset first (logs/run_agent_benchmark_eval_corpus.py) so this arm has "
            "data to score against."
        )
    print(f"[d5] eval corpus has {n_concepts} real paper_concepts rows")

    phrase_encoder = enrichment.PhraseEncoder()
    ontology = enrichment.build_ontology(conn, phrase_encoder)
    print(f"[d5] ontology pool size: {len(ontology.labels)}")

    pool_k = max(k, config.STRUCT_RETRIEVAL_CANDIDATES)
    rows_hybrid, rows_struct, rows_query_aware = [], [], []
    for q in queries:
        judgments: dict[str, int] = q["judgments"]
        vec = encoder.encode_query(q["query"])
        hybrid_hits = store.search_hybrid(query_vector=vec, query_text=q["query"], top_k=pool_k)
        if not hybrid_hits:
            continue
        hybrid_ranking = [h.paper_id for h in hybrid_hits]
        by_id = {h.paper_id: h for h in hybrid_hits}

        def score_mode(ranking: list[str]) -> float:
            top = ranking[:k]
            hits = [by_id[pid] for pid in top if pid in by_id]
            ranked_rel = [judgments.get(h.paper_id, 0) for h in hits]
            return ndcg_at_k(ranked_rel, list(judgments.values()), k=k)

        rows_hybrid.append((q["query"], score_mode(hybrid_ranking[:k])))

        pool_struct_scores = structural_scores(conn, hybrid_ranking)
        struct_ranking = sorted(hybrid_ranking, key=lambda pid: pool_struct_scores.get(pid, 0.0), reverse=True)
        fused_struct = rrf_fuse([hybrid_ranking, struct_ranking], top_k=k)
        rows_struct.append((q["query"], score_mode(fused_struct)))

        concepts = await query_concepts(q["query"], encoder=phrase_encoder, ontology=ontology)
        qa_scores = query_aware_structural_scores(conn, hybrid_ranking, concepts)
        qa_ranking = sorted(hybrid_ranking, key=lambda pid: qa_scores.get(pid, 0.0), reverse=True)
        fused_qa = rrf_fuse([hybrid_ranking, qa_ranking], top_k=k)
        rows_query_aware.append((q["query"], score_mode(fused_qa), len(concepts)))

        print(f"  {q['query']!r}: {len(concepts)} query concept(s) matched")

    conn.close()

    mean_hybrid = sum(s for _, s in rows_hybrid) / len(rows_hybrid) if rows_hybrid else 0.0
    mean_struct = sum(s for _, s in rows_struct) / len(rows_struct) if rows_struct else 0.0
    mean_qa = sum(s for _, s, _ in rows_query_aware) / len(rows_query_aware) if rows_query_aware else 0.0
    queries_with_any_concept = sum(1 for _, _, n in rows_query_aware if n > 0)

    print(f"\n[d5] mean nDCG@{k} over {len(rows_hybrid)} queries:")
    print(f"[d5]   hybrid alone:                {mean_hybrid:.4f}")
    print(f"[d5]   hybrid + pool-cohesion struct: {mean_struct:.4f}")
    print(f"[d5]   hybrid + query-aware struct:   {mean_qa:.4f}")
    print(f"[d5]   queries with >=1 matched concept: {queries_with_any_concept}/{len(rows_query_aware)}")

    lines = [
        "# Query-aware structural retrieval arm (D5, stage-2 fixes)",
        "",
        f"Mean nDCG@{k} over {len(rows_hybrid)} v2 queries, same candidate pool size for all three arms:",
        "",
        "| Arm | Mean nDCG@10 |",
        "|---|---|",
        f"| hybrid alone | {mean_hybrid:.4f} |",
        f"| hybrid + pool-cohesion structural (existing `hybrid_struct`) | {mean_struct:.4f} |",
        f"| hybrid + query-aware structural (D5, this script) | {mean_qa:.4f} |",
        "",
        f"{queries_with_any_concept}/{len(rows_query_aware)} queries matched at least one "
        "ontology concept via the B1 cascade (exact/normalized, then embedding gated on "
        f"VERIFIER_COSINE_MIN={config.VERIFIER_COSINE_MIN}/"
        f"ENRICHMENT_MARGIN_DELTA={config.ENRICHMENT_MARGIN_DELTA}, OD28's calibrated values).",
        "",
        "> Judgments in `eval/ndcg_queries_v2.json` are AI-labeled (fast tier originally; see "
        "OD29 for an independent AI second opinion on a 10-query sample), no human review.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[d5] wrote {REPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)
    asyncio.run(run(args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
