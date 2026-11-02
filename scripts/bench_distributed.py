"""Real throughput (1 vs 3 extraction workers), p95 per-paper latency, and a
fault-injection check (kill an extraction worker mid-run, confirm every
paper still finishes via XAUTOCLAIM reclaim) -> eval/distributed_bench.json
(or eval/distributed_bench<suffix>.json with --out-suffix).

Spawns real `python -m axiom.workers <role>` subprocesses (not Docker --
same code path, much faster to iterate; the compose/Dockerfile setup in
docker-compose.yml is the deployment shape, this benchmark just doesn't pay
the image-build cost to measure it) against a real Redis (`docker compose
up -d redis`) and real LLM calls (extraction only -- coordinator/enrichment/
linking never call an LLM, see axiom/workers.py's module docstring).

Usage (from repo root, Redis already running):
    python scripts/bench_distributed.py --estimate-only
    python scripts/bench_distributed.py --scenario 1   # one scenario per invocation --
    python scripts/bench_distributed.py --scenario 2   # run each one as its own bounded
    python scripts/bench_distributed.py --scenario 3   # foreground call, verify, move on
    python scripts/bench_distributed.py --scenario all --merge   # combine the 3 JSON files
    python scripts/bench_distributed.py --scenario local   # same papers, same pipeline code,
                                                             # today's default in-process
                                                             # asyncio.gather path (no Redis,
                                                             # no subprocesses) -- a same-
                                                             # machine baseline for scenarios
                                                             # 1-3, not a replacement for them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom.agents.pipeline import PaperInput

N_PAPERS = 40
POLL_INTERVAL_S = 1.0
RUN_TIMEOUT_S = 300          # hard deadline per scenario -- never block forever
WORKER_STARTUP_GRACE_S = 12  # time for enrichment's ontology/phrase-model load
OUT_PATH = config.PROJECT_ROOT / "eval" / "distributed_bench.json"
LOG_DIR = config.PROJECT_ROOT / "eval" / "runs" / "distributed_bench_logs"
SCENARIO_OUT = {
    1: config.PROJECT_ROOT / "eval" / "distributed_bench_scenario1.json",
    2: config.PROJECT_ROOT / "eval" / "distributed_bench_scenario2.json",
    3: config.PROJECT_ROOT / "eval" / "distributed_bench_scenario3.json",
}
LOCAL_OUT = config.PROJECT_ROOT / "eval" / "distributed_bench_local.json"


def load_papers(n: int = N_PAPERS) -> list[PaperInput]:
    conn = db.connect(config.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT openalex_id, title, abstract FROM papers "
            "WHERE abstract IS NOT NULL AND length(abstract) > 100 LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        conn.close()
    return [PaperInput(paper_id=r["openalex_id"], title=r["title"], abstract=r["abstract"]) for r in rows]


def estimate_cost(n_papers: int) -> float:
    """~700 tokens in (title+abstract+system prompt), ~150 tokens out per
    paper, haiku (MODEL_FAST) pricing -- the one LLM-calling stage."""
    tokens_in = n_papers * 700
    tokens_out = n_papers * 150
    return ((tokens_in / 1_000_000) * config.MODEL_FAST_PRICE_IN
            + (tokens_out / 1_000_000) * config.MODEL_FAST_PRICE_OUT)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def spawn_worker(role: str, log_path: Path) -> subprocess.Popen:
    log_f = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "REDIS_URL": config.REDIS_URL}
    return subprocess.Popen(
        [sys.executable, "-m", "axiom.workers", role],
        stdout=log_f, stderr=subprocess.STDOUT, env=env,
        cwd=str(config.PROJECT_ROOT),
    )


def stop_workers(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    deadline = time.time() + 10
    for p in procs:
        remaining = max(0.1, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()


_RECLAIM_LINE = re.compile(r"reclaimed (\d+) stuck message")


def count_reclaims(log_dir: Path) -> int:
    """Sum the real message count from each log line (axiom/workers.py's
    reclaim_stuck() logs "reclaimed N stuck message(s)" -- one XREADGROUP
    batch can read up to 10 messages at once, so a single dead worker can
    leave several pending, all reported on one log line, not one per
    message; counting LINES containing "reclaimed" undercounts)."""
    total = 0
    for f in log_dir.glob("*.log"):
        for m in _RECLAIM_LINE.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            total += int(m.group(1))
    return total


async def run_scenario(
    *, name: str, n_extraction: int, papers: list[PaperInput],
    kill_extraction_at_fraction: float | None = None,
) -> dict:
    from axiom import workers as workers_mod

    run_id = f"bench-{name}-{uuid.uuid4().hex[:8]}"
    scenario_log_dir = LOG_DIR / run_id
    scenario_log_dir.mkdir(parents=True, exist_ok=True)

    client = workers_mod.get_redis_client()
    for stream in (workers_mod.STREAM_ROUTE, workers_mod.STREAM_EXTRACT, workers_mod.STREAM_ENRICH,
                   workers_mod.STREAM_LINK, workers_mod.STREAM_DEAD):
        await client.delete(stream)

    conn = db.connect(config.DB_PATH)
    try:
        db.create_agent_run(conn, run_id, papers_total=len(papers),
                             model_fast=config.MODEL_FAST, model_strong=config.MODEL_STRONG,
                             started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    finally:
        conn.close()

    print(f"[{name}] starting {n_extraction} extraction worker(s) + "
          f"1 coordinator/enrichment/linking each ...")
    procs = {
        "coordinator": [spawn_worker("coordinator", scenario_log_dir / "coordinator.log")],
        "enrichment": [spawn_worker("enrichment", scenario_log_dir / "enrichment.log")],
        "linking": [spawn_worker("linking", scenario_log_dir / "linking.log")],
        "extraction": [spawn_worker("extraction", scenario_log_dir / f"extraction-{i}.log")
                        for i in range(n_extraction)],
    }
    await asyncio.sleep(WORKER_STARTUP_GRACE_S)

    t_start = time.monotonic()
    await workers_mod.enqueue_run(client, run_id, papers)
    print(f"[{name}] enqueued {len(papers)} papers at t=0")

    seen_at: dict[str, float] = {}
    killed_one = kill_extraction_at_fraction is None
    deadline = t_start + RUN_TIMEOUT_S
    conn = db.connect(config.DB_PATH)
    try:
        while time.monotonic() < deadline:
            rows = conn.execute(
                "SELECT paper_id FROM agent_run_items WHERE run_id = ?", (run_id,)
            ).fetchall()
            now = time.monotonic()
            for r in rows:
                seen_at.setdefault(r["paper_id"], now - t_start)
            done = len(seen_at)
            if not killed_one and kill_extraction_at_fraction is not None and \
                    done >= len(papers) * kill_extraction_at_fraction:
                victim = procs["extraction"][0]
                print(f"[{name}] FAULT INJECTION: killing extraction worker pid={victim.pid} "
                      f"at {done}/{len(papers)} done")
                victim.kill()
                victim.wait(timeout=10)
                killed_one = True
            if done >= len(papers):
                break
            await asyncio.sleep(POLL_INTERVAL_S)
    finally:
        conn.close()
    t_end = time.monotonic()

    all_procs = [p for group in procs.values() for p in group]
    stop_workers(all_procs)
    await client.aclose()

    conn = db.connect(config.DB_PATH)
    try:
        run_row = db.get_agent_run(conn, run_id)
        spend = db.run_llm_spend(conn, run_id)
    finally:
        conn.close()

    elapsed_s = t_end - t_start
    latencies = list(seen_at.values())
    result = {
        "scenario": name,
        "n_extraction_workers": n_extraction,
        "papers_total": len(papers),
        "papers_done": run_row["papers_done"],
        "elapsed_s": round(elapsed_s, 1),
        "throughput_papers_per_min": round(len(papers) / (elapsed_s / 60), 2) if elapsed_s > 0 else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.50), 2),
        "p95_latency_s": round(percentile(latencies, 0.95), 2),
        "spend_usd": round(spend, 4),
        "reclaim_count": count_reclaims(scenario_log_dir),
        "fault_injected": kill_extraction_at_fraction is not None,
        "all_papers_finished": run_row["papers_done"] == len(papers),
    }
    print(f"[{name}] done: {json.dumps(result, indent=2)}")
    return result


async def run_local_scenario(*, name: str, papers: list[PaperInput]) -> dict:
    """Same papers, same agent code, today's default `AXIOM_EXECUTION=local`
    in-process `asyncio.gather` path (`axiom/agents/pipeline.py`) -- no
    Redis, no subprocesses, one process. A same-machine baseline for how
    much the distributed scenarios' throughput/latency actually costs or
    buys versus just running the pipeline directly."""
    from axiom.agents import enrichment as enrichment_mod
    from axiom.agents import pipeline as pipeline_mod

    run_id = f"bench-{name}-{uuid.uuid4().hex[:8]}"
    t_start = time.monotonic()
    seen_at: dict[str, float] = {}
    done_event = asyncio.Event()

    async def poll() -> None:
        poll_conn = db.connect(config.DB_PATH)
        try:
            while not done_event.is_set():
                rows = poll_conn.execute(
                    "SELECT paper_id FROM agent_run_items WHERE run_id = ?", (run_id,)
                ).fetchall()
                now = time.monotonic()
                for r in rows:
                    seen_at.setdefault(r["paper_id"], now - t_start)
                await asyncio.sleep(POLL_INTERVAL_S)
        finally:
            poll_conn.close()

    poll_task = asyncio.create_task(poll())
    print(f"[{name}] starting in-process pipeline (AGENT_CONCURRENCY={config.AGENT_CONCURRENCY}) ...")
    try:
        phrase_encoder = enrichment_mod.get_default_phrase_encoder()
        await pipeline_mod.run_pipeline(
            papers, encoder=None, db_path=config.DB_PATH, run_id=run_id, phrase_encoder=phrase_encoder,
        )
    finally:
        done_event.set()
        await poll_task
    t_end = time.monotonic()

    conn = db.connect(config.DB_PATH)
    try:
        run_row = db.get_agent_run(conn, run_id)
        spend = db.run_llm_spend(conn, run_id)
    finally:
        conn.close()

    elapsed_s = t_end - t_start
    latencies = list(seen_at.values())
    result = {
        "scenario": name,
        "n_extraction_workers": None,
        "papers_total": len(papers),
        "papers_done": run_row["papers_done"],
        "elapsed_s": round(elapsed_s, 1),
        "throughput_papers_per_min": round(len(papers) / (elapsed_s / 60), 2) if elapsed_s > 0 else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.50), 2),
        "p95_latency_s": round(percentile(latencies, 0.95), 2),
        "spend_usd": round(spend, 4),
        "reclaim_count": 0,
        "fault_injected": False,
        "all_papers_finished": run_row["papers_done"] == len(papers),
    }
    print(f"[{name}] done: {json.dumps(result, indent=2)}")
    return result


SCENARIO_SPECS = {
    1: dict(name="1-extraction-worker", n_extraction=1),
    2: dict(name="3-extraction-workers", n_extraction=3),
    3: dict(name="3-extraction-workers-fault-injected", n_extraction=3, kill_extraction_at_fraction=0.3),
}
LOCAL_SCENARIO_SPEC = dict(name="local-async")


async def run_one_scenario_to_file(scenario_num: int, papers: list[PaperInput]) -> dict:
    result = await run_scenario(papers=papers, **SCENARIO_SPECS[scenario_num])
    SCENARIO_OUT[scenario_num].write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[bench] wrote {SCENARIO_OUT[scenario_num]}")
    return result


async def run_local_scenario_to_file(papers: list[PaperInput]) -> dict:
    result = await run_local_scenario(papers=papers, **LOCAL_SCENARIO_SPEC)
    out_path = LOCAL_OUT
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[bench] wrote {out_path}")
    return result


def merge_scenarios(n_papers: int) -> dict:
    missing = [n for n, p in SCENARIO_OUT.items() if not p.exists()]
    if missing:
        raise SystemExit(f"[bench] missing scenario result file(s) for {missing} -- "
                          f"run --scenario {{1,2,3}} first")
    scenarios = [json.loads(SCENARIO_OUT[n].read_text(encoding="utf-8")) for n in (1, 2, 3)]
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_papers": n_papers,
        "scenarios": scenarios,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] merged -> {OUT_PATH}")
    return payload


def main(argv: list[str] | None = None) -> int:
    global RUN_TIMEOUT_S, OUT_PATH, LOCAL_OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=N_PAPERS)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--scenario", choices=["1", "2", "3", "all", "local"], default="all",
                     help="Run one scenario (writes eval/distributed_bench_scenarioN.json), "
                          "'all' to run all three in one process (background-unsafe in "
                          "this environment -- prefer running 1/2/3 separately), or 'local' "
                          "for the in-process asyncio.gather baseline (no Redis/subprocesses).")
    ap.add_argument("--merge", action="store_true",
                     help="Combine the 3 per-scenario JSON files into eval/distributed_bench.json "
                          "(use after running --scenario 1, 2, and 3 separately).")
    ap.add_argument("--timeout", type=int, default=RUN_TIMEOUT_S,
                     help=f"Hard per-scenario deadline in seconds (default {RUN_TIMEOUT_S}) -- "
                          "raise for a larger --n, e.g. 200 papers at 1 extraction worker.")
    ap.add_argument("--out-suffix", default="",
                     help="Suffix added to every output filename (e.g. '_n200') so a second "
                          "--n run doesn't overwrite the first's scenario files.")
    args = ap.parse_args(argv)

    RUN_TIMEOUT_S = args.timeout
    if args.out_suffix:
        for n in SCENARIO_OUT:
            SCENARIO_OUT[n] = SCENARIO_OUT[n].with_name(f"{SCENARIO_OUT[n].stem}{args.out_suffix}.json")
        OUT_PATH = OUT_PATH.with_name(f"{OUT_PATH.stem}{args.out_suffix}.json")
        LOCAL_OUT = LOCAL_OUT.with_name(f"{LOCAL_OUT.stem}{args.out_suffix}.json")

    if args.merge:
        merge_scenarios(args.n)
        return 0

    papers = load_papers(args.n)
    estimate = estimate_cost(len(papers))
    n_scenarios = 1 if args.scenario != "all" else 3
    print(f"[bench] {len(papers)} real papers loaded from {config.DB_PATH}")
    print(f"[bench] worst-case cost estimate ({n_scenarios} scenario(s), extraction-only LLM calls): "
          f"${estimate * n_scenarios:.4f}")
    if args.estimate_only:
        return 0
    if estimate * n_scenarios > 0.20:
        print(f"[bench] estimate exceeds the $0.20 cap -- aborting, no LLM calls made. "
              f"Re-run with a smaller --n.")
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.scenario == "all":
        async def run_all() -> dict:
            return {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_papers": len(papers),
                "scenarios": [await run_one_scenario_to_file(n, papers) for n in (1, 2, 3)],
            }
        payload = asyncio.run(run_all())
        OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[bench] wrote {OUT_PATH}")
    elif args.scenario == "local":
        asyncio.run(run_local_scenario_to_file(papers))
    else:
        asyncio.run(run_one_scenario_to_file(int(args.scenario), papers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
