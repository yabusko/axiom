"""Build/update the shared, prebuilt base corpus (stage 4, Part A).

A single narrow-niche corpus (retrieval / RAG / QA / LLM knowledge use,
2019-2026, see docs/DECISIONS.md OD36) that fields slice instantly from
(axiom/corpus_slice.py) instead of each field downloading its own. This
script is the ONLY writer to data/corpus/corpus.db + the corpus_v1 Qdrant
collection.

    # first build, ~1000 papers (prints its own time/cost estimate first):
    python scripts/build_corpus.py

    # grow to 20k -- same code path, just a bigger --size:
    python scripts/build_corpus.py --size 20000

    # fetch/embed/extract only what's new since the last build:
    python scripts/build_corpus.py --update

    # thin coverage on a topic? grow the base corpus around it:
    python scripts/build_corpus.py --extend "multi-hop question answering"

    # non-interactive (for a background job):
    python scripts/build_corpus.py --yes

Every step is checkpointed in `corpus_meta.steps_json` -- re-running this
script skips whatever's already marked "done" and continues from there
(resumability semantics: OD37). `--budget` (default config.CORPUS_BUDGET_
DEFAULT_USD) caps the paid extraction step; if the remaining budget can't
afford every paper still needing extraction, this run does as many as it can
afford and leaves the step "partial" -- the NEXT run (same command) picks up
exactly where this one left off, since "needs extraction" is defined by an
absence of `paper_concepts` rows, not by a counter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import canonicalize, config, db, ingest
from axiom.agents import pipeline as agents_pipeline
from axiom.fields import _BENCHMARK_COST_PER_PAPER_USD, _BENCHMARK_P50_LATENCY_S

LOG_PATH = config.PROJECT_ROOT / "logs" / "build_corpus.log"


def log(msg: str) -> None:
    line = f"[build_corpus] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _steps(conn) -> dict:
    meta = db.get_corpus_meta(conn)
    return json.loads(meta["steps_json"]) if meta else {}


def _mark_step(conn, name: str, status: str, **meta_kwargs) -> None:
    steps = _steps(conn)
    steps[name] = status
    db.upsert_corpus_meta(conn, steps_json=json.dumps(steps), **meta_kwargs)


def _write_manifest(conn) -> None:
    meta = db.get_corpus_meta(conn)
    if meta is None:
        return
    config.CORPUS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_MANIFEST_PATH.write_text(
        json.dumps({
            "version": meta["version"], "query": meta["query"],
            "filters": json.loads(meta["filters_json"]), "size_target": meta["size_target"],
            "papers_count": meta["papers_count"], "steps": json.loads(meta["steps_json"]),
            "spend_total_usd": round(meta["spend_total"], 4),
            "build_seconds": round(meta["build_seconds"], 1),
            "created_at": meta["created_at"], "updated_at": meta["updated_at"],
        }, indent=2),
        encoding="utf-8",
    )


def _sync_spend(conn) -> None:
    """corpus_meta.spend_total tracks the DB's own cumulative llm_spend table
    (extraction + canonicalize both write there) -- re-synced as an absolute
    total, not accumulated deltas, so a partial/interrupted step never
    double-counts on the next run."""
    total = conn.execute("SELECT COALESCE(SUM(cost), 0) AS t FROM llm_spend").fetchone()["t"]
    meta = db.get_corpus_meta(conn)
    delta = float(total) - (meta["spend_total"] if meta else 0.0)
    if abs(delta) > 1e-9:
        db.upsert_corpus_meta(conn, spend_delta=delta)


def _filters(from_year: int, to_year: int) -> dict:
    return {
        "from_publication_date": f"{from_year}-01-01",
        "to_publication_date": f"{to_year}-12-31",
        "venue_patterns": config.RELEVANT_VENUE_PATTERNS,
        "has_abstract": True,
    }


def step_fetch_store(conn, *, query: str, size: int, filters: dict, source_ids: list[str], log=log) -> int:
    """Steps 1+2+4 (A2): fetch relevant works, write to SQLite (papers +
    concepts + citation edges together -- `ingest.write_works` does all
    three per paper in one pass, so "store" and "edges" complete atomically
    with "fetch" rather than as separate passes over the same data).

    OD38: the venue allow-list is applied server-side via `source_ids`
    (resolved once in main(), via `OpenAlexClient.resolve_source_ids`) baked
    into `filter=`, not a client-side `keep()` post-filter -- every /works
    page fetched counts fully toward `size`, which matters now that keyless
    OpenAlex traffic has a real $ budget per day (OD38).
    """
    existing_ids = {r["openalex_id"] for r in conn.execute("SELECT openalex_id FROM papers").fetchall()}
    n_before = len(existing_ids)
    filter_parts = [
        f"from_publication_date:{filters['from_publication_date']}",
        f"to_publication_date:{filters['to_publication_date']}",
        "has_abstract:true",
    ]
    if source_ids:
        filter_parts.append("primary_location.source.id:" + "|".join(source_ids))
    filter_str = ",".join(filter_parts)
    n_written = 0
    with ingest.OpenAlexClient() as client:
        batch: list[dict] = []
        for w in client.search_relevant(query, filter_str, size, skip_ids=existing_ids, log=log):
            batch.append(w)
            if len(batch) >= 200:
                n_written += ingest.write_works(conn, batch, log=log)
                db.upsert_corpus_meta(conn, query=query, filters_json=json.dumps(filters),
                                       size_target=size, papers_count=n_before + n_written)
                batch = []
        if batch:
            n_written += ingest.write_works(conn, batch, log=log)
    total = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    db.upsert_corpus_meta(conn, query=query, filters_json=json.dumps(filters),
                           size_target=size, papers_count=total)
    edge_count = conn.execute("SELECT COUNT(*) AS n FROM citation_edges").fetchone()["n"]
    log(f"fetch+store: {n_written} new papers this run, {total} total, {edge_count} citation edges")
    return n_written


def step_embed(conn, *, collection: str, log=log) -> None:
    """Step 3: SPECTER2 dense + BM25 sparse into Qdrant. Full rebuild each
    time (axiom.indexer.reindex_qdrant recreates the collection) -- see
    OD37 for why this step is resumable at step-granularity only, not
    per-paper: it's bounded local compute, not metered spend or unbounded
    network, so redoing it whole on an interrupted rerun is cheap enough to
    not be worth incremental-upsert complexity. `conn` must point at the
    corpus DB explicitly -- reindex_qdrant() defaults to config.DB_PATH
    (the legacy default-field DB) when given none."""
    from axiom.indexer import reindex_qdrant
    reindex_qdrant(conn, collection=collection, log=log)


def step_extract(conn, *, run_id: str, budget: float, db_path, log=log) -> tuple[bool, bool]:
    """Step 5: agent extraction, only on papers with no `paper_concepts` row
    yet (resumable: a prior run's successfully-extracted papers are never
    resubmitted, regardless of which run_id produced them). Returns
    (done, did_work): `done` iff every outstanding paper was submitted this
    call (step fully done); `did_work` False when there was nothing to do
    (a no-op resume) -- callers use it to avoid recording a real step
    duration for a call that did no real work."""
    todo_rows = conn.execute(
        "SELECT openalex_id, title, abstract FROM papers "
        "WHERE openalex_id NOT IN (SELECT DISTINCT paper_id FROM paper_concepts)"
    ).fetchall()
    if not todo_rows:
        log("extract: nothing outstanding")
        return True, False

    spent_already = conn.execute("SELECT COALESCE(SUM(cost),0) AS t FROM llm_spend").fetchone()["t"]
    remaining_budget = max(0.0, budget - float(spent_already))
    n_afford = int(remaining_budget / _BENCHMARK_COST_PER_PAPER_USD) if _BENCHMARK_COST_PER_PAPER_USD > 0 else len(todo_rows)
    batch = todo_rows[:max(n_afford, 0)]
    if not batch:
        log(f"extract: budget exhausted (${spent_already:.2f} spent of ${budget:.2f}), "
            f"{len(todo_rows)} papers still outstanding -- re-run with --budget to continue")
        return False, False

    est_cost = len(batch) * _BENCHMARK_COST_PER_PAPER_USD
    est_seconds = (len(batch) / max(config.AGENT_CONCURRENCY, 1)) * _BENCHMARK_P50_LATENCY_S
    log(f"extract: submitting {len(batch)}/{len(todo_rows)} outstanding papers "
        f"(est. ${est_cost:.2f}, ~{est_seconds:.0f}s)")

    papers = [agents_pipeline.PaperInput(paper_id=r["openalex_id"], title=r["title"], abstract=r["abstract"])
              for r in batch]
    asyncio.run(agents_pipeline.run_pipeline(papers, encoder=None, db_path=db_path, run_id=run_id))
    _sync_spend(conn)
    log(f"extract: batch done, {len(todo_rows) - len(batch)} papers remain outstanding")
    return len(batch) == len(todo_rows), True


def step_canonicalize(conn, *, log=log) -> None:
    """Step 6: synonym merge over OpenAlex concept labels (OD14's existing
    canonicalize.run, unchanged) -- skips concepts already canonicalized or
    manually overridden, so a rerun only processes genuinely new labels."""
    canonicalize.run(conn, log=log)
    _sync_spend(conn)


def run_extend(conn, *, topic: str, log=log) -> None:
    """A3: grow the shared base corpus around a thin-coverage topic via the
    existing citation-snowball (axiom.ingest.snowball) -- the "old download
    path," reused here (not via the live API) as this stage's only caller.
    Non-destructive: writes through ingest.write_works, never resets."""
    keep = ingest.make_venue_filter(config.RELEVANT_VENUE_PATTERNS)
    log(f"--extend {topic!r}: snowballing into the base corpus")
    with ingest.OpenAlexClient() as client:
        works = ingest.snowball(client, seed_query=topic, target=config.FIELD_SLICE_TARGET, keep=keep, log=log)
    n = ingest.write_works(conn, works.values(), log=log)
    total = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    db.upsert_corpus_meta(conn, query=config.CORPUS_QUERY,
                           filters_json=json.dumps(_filters(config.CORPUS_FROM_YEAR, config.CORPUS_TO_YEAR)),
                           size_target=total, papers_count=total)
    log(f"--extend: +{n} papers ({total} total) -- run build_corpus.py again to embed/extract them")
    # Extend only adds papers; it deliberately does NOT re-mark embed/extract
    # "done" as stale here -- the next normal run sees the new papers (no
    # paper_concepts row, not yet in the embedded set) and processes them
    # through the same checkpointed steps as a fresh build.
    steps = _steps(conn)
    steps["embed"] = "partial"
    steps["extract"] = "partial"
    db.upsert_corpus_meta(conn, steps_json=json.dumps(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the shared base corpus.")
    parser.add_argument("--size", type=int, default=config.CORPUS_SIZE_DEFAULT,
                        help="target corpus size (default: %(default)s)")
    parser.add_argument("--query", default=config.CORPUS_QUERY,
                        help="relevance search query (default: %(default)r)")
    parser.add_argument("--budget", type=float, default=config.CORPUS_BUDGET_DEFAULT_USD,
                        help="hard cap on this run's cumulative LLM spend, USD (default: %(default)s)")
    parser.add_argument("--yes", action="store_true", help="skip the y/N confirmation")
    parser.add_argument("--update", action="store_true",
                        help="fetch only papers published since the last build, then embed/extract just those")
    parser.add_argument("--extend", metavar="TOPIC",
                        help="snowball-expand the base corpus around TOPIC (for a thin-coverage field)")
    parser.add_argument("--no-extract", action="store_true", help="skip the paid agent-extraction step")
    parser.add_argument("--no-canonicalize", action="store_true", help="skip the synonym-merge step")
    args = parser.parse_args()

    if not config.OPENALEX_API_KEY:
        print("No OPENALEX_API_KEY set (axiom/config.py / .env). Keyless OpenAlex traffic "
              "shares a tiny ($0.10/day, per-IP) budget -- get a free key with its own "
              "$1/day at https://openalex.org/settings/api and set OPENALEX_API_KEY. (OD38)")
        return

    config.CORPUS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(config.CORPUS_DB_PATH)
    db.init_db(conn)

    if args.extend:
        est_papers = config.FIELD_SLICE_TARGET
        print(f"This will snowball-expand the base corpus around {args.extend!r} "
              f"by up to ~{est_papers} papers, then embed + extract them "
              f"(est. ${est_papers * _BENCHMARK_COST_PER_PAPER_USD:.2f}).")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return
        run_extend(conn, topic=args.extend, log=log)
        conn.close()
        return

    from_year = config.CORPUS_FROM_YEAR
    if args.update:
        meta = db.get_corpus_meta(conn)
        if meta is None:
            print("No existing corpus to --update; run without --update first.")
            conn.close()
            return
        from_year = int(meta["updated_at"][:4])
        print(f"--update: fetching papers published since {meta['updated_at'][:10]} "
              f"(from_publication_date >= {from_year}-01-01), then embedding/extracting only the new ones.")
    with ingest.OpenAlexClient() as client:
        source_ids = client.resolve_source_ids(config.RELEVANT_VENUE_PATTERNS, log=log)

    if not args.update:
        already = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
        remaining = max(0, args.size - already)
        est_requests = len(config.RELEVANT_VENUE_PATTERNS) + -(-remaining // 200)  # sources lookups + ceil(remaining/200)
        est_cost = args.size * _BENCHMARK_COST_PER_PAPER_USD if not args.no_extract else 0.0
        est_seconds = (args.size / max(config.AGENT_CONCURRENCY, 1)) * _BENCHMARK_P50_LATENCY_S if not args.no_extract else 0.0
        print(f"This will build/resume the base corpus: query={args.query!r}, "
              f"size<={args.size}, years {from_year}-{config.CORPUS_TO_YEAR}, "
              f"venue-filtered server-side to {len(source_ids)} source ids "
              f"({len(config.RELEVANT_VENUE_PATTERNS)} patterns).")
        print(f"Steps: fetch -> store -> embed -> edges -> "
              f"{'extract -> ' if not args.no_extract else ''}"
              f"{'canonicalize' if not args.no_canonicalize else ''}")
        print(f"Estimated OpenAlex requests this run: ~{est_requests} "
              f"({already} papers already fetched, {remaining} remaining at 200/page).")
        print(f"Estimated extraction cost: ${est_cost:.2f} (budget cap ${args.budget:.2f}), "
              f"~{est_seconds / 60:.0f} min for extraction alone, plus fetch+embed time "
              f"(network + CPU-bound, not estimated here).")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            conn.close()
            return

    filters = _filters(from_year, config.CORPUS_TO_YEAR)
    t0 = time.time()
    step_seconds: dict[str, float] = {}

    # --update always re-fetches (looking for papers published since the last
    # build) regardless of the "done" checkpoint -- that checkpoint exists to
    # make a normal/resumed run skip repeat work, not to make --update a no-op.
    # Review fix: step_seconds only gets an entry for a step that ACTUALLY RAN
    # this invocation -- a skipped (already 'done') step contributes nothing
    # here, so a later no-op resume can never overwrite its real historical
    # duration with ~0 in the durable step_seconds_json (see db.upsert_corpus_meta).
    ts = time.time()
    if args.update or _steps(conn).get("fetch_store") != "done":
        n_new = step_fetch_store(conn, query=args.query, size=args.size, filters=filters, source_ids=source_ids, log=log)
        _mark_step(conn, "fetch_store", "done")
        if args.update and n_new > 0:
            # New papers mean embed/extract have outstanding work again, even
            # though both were already "done" for the pre-update corpus.
            steps = _steps(conn)
            steps["embed"] = "partial"
            db.upsert_corpus_meta(conn, steps_json=json.dumps(steps))
        step_seconds["fetch_store"] = round(time.time() - ts, 1)
    else:
        log("fetch_store: already done, skipping")

    ts = time.time()
    if _steps(conn).get("embed") != "done":
        try:
            step_embed(conn, collection=config.CORPUS_COLLECTION_NAME, log=log)
            _mark_step(conn, "embed", "done")
        except ImportError as exc:
            log(f"embed: skipped, ML stack unavailable ({exc}); re-run on Python 3.10-3.12")
        step_seconds["embed"] = round(time.time() - ts, 1)
    else:
        log("embed: already done, skipping")

    ts = time.time()
    if not args.no_extract:
        run_id = f"corpus-{int(t0)}"
        done, did_work = step_extract(conn, run_id=run_id, budget=args.budget, db_path=config.CORPUS_DB_PATH, log=log)
        _mark_step(conn, "extract", "done" if done else "partial")
        if did_work:
            step_seconds["extract"] = round(time.time() - ts, 1)
    else:
        log("extract: skipped (--no-extract)")

    ts = time.time()
    if not args.no_canonicalize and _steps(conn).get("canonicalize") != "done":
        step_canonicalize(conn, log=log)
        _mark_step(conn, "canonicalize", "done")
        step_seconds["canonicalize"] = round(time.time() - ts, 1)
    else:
        log("canonicalize: already done or skipped")

    elapsed = time.time() - t0
    db.upsert_corpus_meta(conn, seconds_delta=elapsed, budget_usd=args.budget, step_seconds_delta=step_seconds)
    _write_manifest(conn)
    meta = db.get_corpus_meta(conn)
    log(f"done in {elapsed:.0f}s this run -- corpus: {meta['papers_count']} papers, "
        f"${meta['spend_total']:.4f} total spend, steps={meta['steps_json']}")

    # A7: corpus build time per step (cumulative, durable across resumes) + total spend.
    eval_path = config.PROJECT_ROOT / "eval" / "corpus_build.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps({
        "papers_count": meta["papers_count"], "this_run_seconds": round(elapsed, 1),
        "this_run_step_seconds": step_seconds,
        "cumulative_step_seconds": json.loads(meta["step_seconds_json"]),
        "cumulative_build_seconds": round(meta["build_seconds"], 1),
        "cumulative_spend_usd": round(meta["spend_total"], 4), "steps": json.loads(meta["steps_json"]),
    }, indent=2), encoding="utf-8")
    log(f"wrote {eval_path}")
    conn.close()


if __name__ == "__main__":
    main()
