"""nDCG@10 retrieval-quality eval against the real ACL/EMNLP/COLING/NAACL eval
corpus (Backlog PBI 8 / Task 8.2), run against `data/eval.db` + Qdrant
collection `axiom_eval_v1` (OD11) -- built by `scripts/ingest_eval_corpus.py`.

Relevance judgments in eval/ndcg_queries.json are SINGLE-PASS, AI-DRAFTED
(see that file's "_provenance" field and docs/DECISIONS.md OD11) -- the
backlog's acceptance criterion ("2 annotators") is not yet met, so the numbers
here are a working demo signal, not a validated quality claim.

Usage (from repo root) -- v1, UNCHANGED:
    python scripts/eval_ndcg.py                 # hybrid retrieval (default)
    python scripts/eval_ndcg.py --mode dense
    python scripts/eval_ndcg.py --report eval/report.md

--- Eval v2 (R9/R8, docs/agentic/ARCHITECTURE.md) -------------------------
Two new modes read `eval/ndcg_queries_v2.json` (config.EVAL_V2_QUERIES_PATH,
0..config.EVAL_V2_SCALE_MAX scale) instead of the v1 file -- v1's file/report
are never touched by any v2 invocation:

    python scripts/eval_ndcg.py --mode hybrid_struct   # + concept-overlap RRF arm
    python scripts/eval_ndcg.py --mode dense_general   # SPECTER2 vs a general embedder
    python scripts/eval_ndcg.py --mode dense --v2      # v1 modes can also run against
    python scripts/eval_ndcg.py --mode hybrid --v2     # the v2 query/judgment pool

`--mode hybrid_struct` adds a third, eval-only RRF arm (OD21): it calls
`AxiomQdrant.search_hybrid` exactly as-is (no new parameter -- OD4's
signature stays frozen) to get the dense+sparse fused ranking, computes a
concept-overlap "structural" ranking from the eval DB's `paper_concepts`/
`concept_cooccurrence` tables (candidate pool capped at
config.STRUCT_RETRIEVAL_CANDIDATES), and RRF-fuses the two rankings in
Python -- the same pattern OD4 already uses for dense+sparse, just one level
up. `--db` picks which SQLite DB backs that structural arm (defaults to
`config.EVAL_DB_PATH`; never hardcoded, OD11-safe).

`--mode dense_general` compares SPECTER2 against a small general-purpose
local embedder (BAAI/bge-small-en-v1.5) with NO Qdrant collection of its own.
It needs a one-time cache of that embedder's vectors for the eval corpus
(`eval/general_embeddings_cache.json`, `{paper_id: [floats...]}`) built
ahead of time -- this script never silently triggers that embedding job (a
real model download + corpus pass belongs to the main session's heavy-job
budget, not an eval-script side effect). Missing cache -> a clear SystemExit,
not a crash or a silent skip.

Every v2 report is written to `eval/report_<mode>.md` by default (distinct
from v1's `eval/report.md`, matching docs/agentic/ARCHITECTURE.md §1); pass
`--report` to override.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from math import log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from axiom import config, db
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant, SearchHit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = PROJECT_ROOT / "eval" / "ndcg_queries.json"          # v1, unchanged
GENERAL_CACHE_PATH = PROJECT_ROOT / "eval" / "general_embeddings_cache.json"  # v2 dense_general

V2_MODES = ("hybrid_struct", "dense_general")


def dcg(relevances: list[int]) -> float:
    return sum(rel / log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances: list[int], all_judged_relevances: list[int], k: int = 10) -> float:
    """ranked_relevances: judged relevance (0 if unjudged) of the top-k retrieved hits.

    all_judged_relevances: every judgment for this query, used to build the
    ideal ranking (ties broken by insertion order, ideal DCG uses top-k of the
    sorted-descending judgments same as standard nDCG@k).
    """
    actual = dcg(ranked_relevances[:k])
    ideal = dcg(sorted(all_judged_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# R8: hybrid_struct -- concept-overlap structural RRF arm (eval-only, OD21)
# ---------------------------------------------------------------------------
def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K, top_k: int | None = None) -> list[str]:
    """Reciprocal Rank Fusion over any number of ranked id lists.

    Same formula `AxiomQdrant.search_hybrid` already uses for dense+sparse
    (score = sum 1/(RRF_K + rank), rank 1-based) -- reused here, in Python,
    one level up, to fuse the hybrid_struct arm into the dense+sparse result
    without touching `qdrant_client.py`'s frozen signatures (OD4/OD21).
    """
    fused: dict[str, float] = {}
    for arm in rankings:
        for rank, item_id in enumerate(arm, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    ids = [item_id for item_id, _ in ranked]
    return ids[:top_k] if top_k is not None else ids


def structural_scores(conn: sqlite3.Connection, candidate_ids: list[str]) -> dict[str, float]:
    """Concept-overlap cohesion score for each candidate within its own pool.

    `paper_concepts` gives each candidate's canonical concepts; `concept_cooccurrence`
    gives how often each concept pair co-occurs corpus-wide. A candidate scores
    highly when (a) its concepts are shared by other candidates already in the
    same retrieved pool (structural agreement with the rest of the result set)
    and (b) its own concept pairs are well-established co-occurrences rather
    than a noisy one-off extraction. Pure SQL + arithmetic -- no LLM/embedder
    call, so this can run inline in the eval loop without a budget concern.

    Returns {} immediately (every candidate scores 0) if `paper_concepts` has
    no rows for this corpus yet -- e.g. the agentic pipeline hasn't been run
    over the eval corpus. That degrades hybrid_struct to a no-op re-ranking
    (RRF of [hybrid_ranking, empty] just returns the hybrid ranking), not a
    crash.
    """
    if not candidate_ids:
        return {}
    placeholders = ",".join("?" for _ in candidate_ids)
    concept_rows = conn.execute(
        f"SELECT paper_id, canonical FROM paper_concepts WHERE paper_id IN ({placeholders})",
        list(candidate_ids),
    ).fetchall()
    by_paper: dict[str, set[str]] = {}
    for r in concept_rows:
        by_paper.setdefault(r["paper_id"], set()).add(r["canonical"])
    if not by_paper:
        return {}

    pool_concept_count: dict[str, int] = {}
    for concepts in by_paper.values():
        for c in concepts:
            pool_concept_count[c] = pool_concept_count.get(c, 0) + 1

    all_concepts = sorted({c for concepts in by_paper.values() for c in concepts})
    cooc_weight: dict[tuple[str, str], int] = {}
    if all_concepts:
        c_placeholders = ",".join("?" for _ in all_concepts)
        cooc_rows = conn.execute(
            f"SELECT concept_a, concept_b, COUNT(*) AS n FROM concept_cooccurrence "
            f"WHERE concept_a IN ({c_placeholders}) AND concept_b IN ({c_placeholders}) "
            f"GROUP BY concept_a, concept_b",
            all_concepts + all_concepts,
        ).fetchall()
        for r in cooc_rows:
            cooc_weight[(r["concept_a"], r["concept_b"])] = r["n"]

    scores: dict[str, float] = {}
    for pid in candidate_ids:
        concepts = sorted(by_paper.get(pid, ()))
        overlap = sum(pool_concept_count[c] - 1 for c in concepts)
        pair_strength = sum(
            cooc_weight.get((a, b), 0)
            for i, a in enumerate(concepts) for b in concepts[i + 1:]
        )
        scores[pid] = float(overlap + pair_strength)
    return scores


# ---------------------------------------------------------------------------
# dense_general -- SPECTER2 vs a small general-purpose embedder (optional)
# ---------------------------------------------------------------------------
class GeneralEncoder:
    """Small general-purpose local embedder (BAAI/bge-small-en-v1.5) for the
    optional `dense_general` arm -- a non-domain-specific baseline next to
    SPECTER2. Lazy-loaded exactly like `axiom.embed.Specter2Encoder`; built
    from `transformers.AutoModel`/`AutoTokenizer` (already-pinned deps, no new
    dependency) rather than `sentence-transformers`. Only constructed when
    `--mode dense_general` is actually passed -- never imported/instantiated
    by any test or by another mode.
    """

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode_query(self, query: str) -> list[float]:
        # BGE's documented query recipe: a fixed instruction prefix + mean
        # pooling + L2 normalization (no [CLS]-token trick, unlike SPECTER2).
        torch = self._torch
        prefixed = "Represent this sentence for searching relevant passages: " + query
        with torch.no_grad():
            enc = self.tokenizer(prefixed, return_tensors="pt", truncation=True,
                                  max_length=512).to(self.device)
            out = self.model(**enc)
            emb = out.last_hidden_state.mean(dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb[0].cpu().tolist()


def load_general_vectors(cache_path: Path) -> dict[str, list[float]]:
    """{paper_id: vector} cache for the general embedder, built ahead of time.

    Never triggers the embedding job itself -- that is a heavy, one-time
    main-session task (download + encode the whole eval corpus), out of
    scope for an eval script invocation. Missing cache is a clear SystemExit.
    """
    if not cache_path.exists():
        raise SystemExit(
            f"'{cache_path}' not found -- `--mode dense_general` needs a one-time "
            "cache of BAAI/bge-small-en-v1.5 embeddings for the eval corpus "
            "({paper_id: [floats...]}). Build it once (a heavy job -- see the "
            "eval-implementer patch note for the exact format) before running "
            "this mode."
        )
    return json.loads(cache_path.read_text(encoding="utf-8"))


def brute_force_rank(query_vector: list[float], vectors: dict[str, list[float]],
                      top_k: int) -> list[tuple[str, float]]:
    """Cosine-rank every cached vector against the query -- no Qdrant collection
    exists for the general embedder (different dimensionality than the pinned
    768-dim SPECTER2 collection), so this is a plain brute-force search instead."""
    if not vectors:
        return []
    ids = list(vectors.keys())
    mat = np.asarray([vectors[i] for i in ids], dtype=float)
    q = np.asarray(query_vector, dtype=float)
    qn = float(np.linalg.norm(q)) or 1.0
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    sims = (mat @ q) / (norms * qn)
    order = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in order]


def fetch_paper_rows(conn: sqlite3.Connection, paper_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"SELECT openalex_id, title, publication_year, venue, cited_by_count "
        f"FROM papers WHERE openalex_id IN ({placeholders})",
        list(paper_ids),
    ).fetchall()
    return {r["openalex_id"]: r for r in rows}


# ---------------------------------------------------------------------------
# Query dispatch
# ---------------------------------------------------------------------------
def run_query(mode: str, query: str, encoder: Specter2Encoder,
              store: AxiomQdrant, top_k: int, *,
              struct_conn: sqlite3.Connection | None = None,
              struct_candidates: int = config.STRUCT_RETRIEVAL_CANDIDATES,
              general_encoder: "GeneralEncoder | None" = None,
              general_vectors: dict[str, list[float]] | None = None,
              paper_rows: dict[str, sqlite3.Row] | None = None) -> list[SearchHit]:
    if mode == "dense":
        vec = encoder.encode_query(query)
        return store.search(vec, top_k=top_k)
    if mode == "hybrid":
        vec = encoder.encode_query(query)
        return store.search_hybrid(query_vector=vec, query_text=query, top_k=top_k)
    if mode == "hybrid_struct":
        vec = encoder.encode_query(query)
        pool_k = max(top_k, struct_candidates)
        hybrid_hits = store.search_hybrid(query_vector=vec, query_text=query, top_k=pool_k)
        if not hybrid_hits:
            return []
        hybrid_ranking = [h.paper_id for h in hybrid_hits]
        scores = structural_scores(struct_conn, hybrid_ranking) if struct_conn is not None else {}
        struct_ranking = sorted(hybrid_ranking, key=lambda pid: scores.get(pid, 0.0), reverse=True)
        fused_ids = rrf_fuse([hybrid_ranking, struct_ranking], top_k=top_k)
        by_id = {h.paper_id: h for h in hybrid_hits}
        return [by_id[pid] for pid in fused_ids if pid in by_id]
    if mode == "dense_general":
        vec = general_encoder.encode_query(query)
        ranked = brute_force_rank(vec, general_vectors or {}, top_k)
        rows = paper_rows or {}
        hits = []
        for pid, score in ranked:
            r = rows.get(pid)
            hits.append(SearchHit(
                paper_id=pid,
                title=r["title"] if r else "",
                year=r["publication_year"] if r else 0,
                venue=r["venue"] if r else "",
                cited_by_count=r["cited_by_count"] if r else 0,
                score=score,
            ))
        return hits
    raise ValueError(f"unknown mode: {mode}")


def write_report(report_path: Path, *, mode: str, collection: str, corpus_size: int,
                  rows: list[tuple[str, float, list[SearchHit]]], mean_ndcg: float,
                  k: int, is_v2: bool, scale_max: int) -> None:
    lines = [
        "# nDCG@10 retrieval eval report" + (" (v2)" if is_v2 else ""),
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} · mode=`{mode}` · "
        f"collection=`{collection}` · corpus={corpus_size} papers · {len(rows)} queries",
        "",
    ]
    if is_v2:
        lines += [
            f"> **Caveat:** relevance judgments in `{config.EVAL_V2_QUERIES_PATH.name}` are "
            f"pooled, AI-drafted on a 0-{scale_max} scale (see its `_provenance` field) -- "
            "not yet reviewed by `scripts/label_cli.py`. Treat this as a working demo "
            "signal, not a validated quality claim, until a human pass records overrides.",
            "",
        ]
    else:
        lines += [
            "> **Caveat:** relevance judgments in `eval/ndcg_queries.json` are "
            "single-pass, AI-drafted (see its `_provenance` field and "
            "`docs/DECISIONS.md` OD11) -- not the backlog's '2 annotators' "
            "acceptance criterion. Treat this as a working demo signal, not a "
            "validated quality claim. A low score can also mean the judgment "
            "set didn't happen to cover a paper the retriever correctly "
            "surfaced (judgments were built via keyword-pattern search over "
            "the sample, not an exhaustive relevance pass) -- inspect low "
            "scorers before treating them as retrieval failures.",
            "",
        ]
    lines += [
        f"**Mean nDCG@{k}: {mean_ndcg:.3f}**",
        "",
        "| Query | nDCG@10 |",
        "|---|---|",
    ]
    for query, score, _ in rows:
        lines.append(f"| {query} | {score:.3f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[report] wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "hybrid", "hybrid_struct", "dense_general"],
                     default="hybrid")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--report", type=str, default=None,
                    help="write a markdown report to this path (e.g. eval/report.md)")
    ap.add_argument("--v2", action="store_true",
                    help="use eval v2 queries (config.EVAL_V2_QUERIES_PATH, "
                         "0..config.EVAL_V2_SCALE_MAX scale) instead of v1 -- "
                         "implied automatically for --mode hybrid_struct/dense_general")
    ap.add_argument("--queries", type=str, default=None,
                    help="override the queries json path")
    ap.add_argument("--db", type=str, default=None,
                    help="eval SQLite db path for hybrid_struct's structural arm / "
                         "dense_general's metadata lookup (default: config.EVAL_DB_PATH)")
    args = ap.parse_args()

    is_v2 = args.v2 or args.mode in V2_MODES
    queries_path = Path(args.queries) if args.queries else (
        config.EVAL_V2_QUERIES_PATH if is_v2 else QUERIES_PATH
    )

    data = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = data["queries"]

    struct_conn = None
    encoder = None
    general_encoder = None
    general_vectors = None
    paper_rows = None
    store = None

    if args.mode == "dense_general":
        general_vectors = load_general_vectors(GENERAL_CACHE_PATH)
        general_encoder = GeneralEncoder()
        db_path = args.db or str(config.EVAL_DB_PATH)
        struct_conn = db.connect(db_path)
        paper_rows = fetch_paper_rows(struct_conn, list(general_vectors.keys()))
        corpus_size = len(general_vectors)
        collection_label = "n/a (brute-force, no Qdrant collection)"
    else:
        encoder = Specter2Encoder()
        store = AxiomQdrant(collection=config.EVAL_COLLECTION_NAME)
        if store.count() == 0:
            raise SystemExit(
                f"'{config.EVAL_COLLECTION_NAME}' is empty -- run "
                f"scripts/ingest_eval_corpus.py first."
            )
        if args.mode == "hybrid_struct":
            db_path = args.db or str(config.EVAL_DB_PATH)
            struct_conn = db.connect(db_path)
        corpus_size = store.count()
        collection_label = config.EVAL_COLLECTION_NAME

    try:
        rows = []
        for q in queries:
            judgments: dict[str, int] = q["judgments"]
            hits = run_query(
                args.mode, q["query"], encoder, store, args.k,
                struct_conn=struct_conn, struct_candidates=config.STRUCT_RETRIEVAL_CANDIDATES,
                general_encoder=general_encoder, general_vectors=general_vectors,
                paper_rows=paper_rows,
            )
            ranked_rel = [judgments.get(h.paper_id, 0) for h in hits]
            score = ndcg_at_k(ranked_rel, list(judgments.values()), k=args.k)
            rows.append((q["query"], score, hits))
            print(f"nDCG@{args.k} = {score:.3f}   {q['query']!r}")
    finally:
        if struct_conn is not None:
            struct_conn.close()

    mean_ndcg = sum(r[1] for r in rows) / len(rows) if rows else 0.0
    print(f"\nMean nDCG@{args.k} over {len(rows)} queries "
          f"(mode={args.mode}, collection={collection_label}): {mean_ndcg:.3f}")

    report_arg = args.report
    if report_arg is None and is_v2:
        report_arg = str(PROJECT_ROOT / "eval" / f"report_{args.mode}.md")

    if report_arg:
        write_report(
            Path(report_arg), mode=args.mode, collection=collection_label,
            corpus_size=corpus_size, rows=rows, mean_ndcg=mean_ndcg, k=args.k,
            is_v2=is_v2, scale_max=config.EVAL_V2_SCALE_MAX,
        )


if __name__ == "__main__":
    main()
