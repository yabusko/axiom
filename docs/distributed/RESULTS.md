# Distributed workers — results

Every number below traces to `eval/distributed_bench.json` (`scripts/bench_distributed.py`),
a real run on 2026-11-01: real Redis (`docker compose up -d redis`), real
`python -m axiom.workers <role>` subprocesses (not Docker containers — same code
path, no image-build time spent to measure it), real LLM calls (extraction only —
coordinator/enrichment/linking never call an LLM, see `docs/DECISIONS.md` OD43)
against 40 real papers from `data/axiom.db`. Cost estimated ($0.0058/scenario) and
capped at $0.20 before any call was made; actual total spend across all three
scenarios was **$0.0507**.

`n=40`, not the task's suggested 200: this environment's background-process
supervision proved unreliable for a single long-running multi-scenario script (see
"A real bug found running this" below) — running each scenario as its own short,
directly-observed foreground invocation was the reliable path, and 40 papers is
enough to show a clean 1-vs-3 throughput signal without each scenario running
long enough to make that impractical. `scripts/bench_distributed.py --scenario N`
supports re-running at `--n 200` the same way, scenario by scenario.

## Throughput: 1 vs 3 extraction workers

| Scenario | Workers | Papers | Elapsed | Throughput | p50 latency | p95 latency | Spend |
|---|---|---|---|---|---|---|---|
| 1-extraction-worker | 1 | 40/40 | 171.6s | 13.98 papers/min | 91.0s | 163.8s | $0.0159 |
| 3-extraction-workers | 3 | 40/40 | 83.0s | 28.92 papers/min | 42.0s | 73.1s | $0.0176 |

**~2.07x throughput from 3x the extraction workers**, not 3x — coordinator,
enrichment, and linking each stay at exactly 1 worker in every scenario (only
`extraction` scales, per the task), so they become the shared bottleneck once
extraction itself is no longer the slowest stage. Latency (p50/p95) is "time from
enqueue to this paper's last agent_run_items write," not a single LLM call's
latency — it's dominated by queueing behind the 40-paper batch, which is exactly
why it drops by roughly the same ~2x the throughput improved.

## Fault injection

| Scenario | Workers | Papers | Elapsed | Reclaimed messages | All finished? |
|---|---|---|---|---|---|
| 3-extraction-workers-fault-injected | 3 (1 killed at 12/40 done) | 40/40 | 218.2s | 8 | **yes** |

One of the three extraction workers was `kill()`ed mid-run (12/40 papers already
done). `XREADGROUP` reads up to 10 messages per batch, and this worker's `_process_
one` loop handles a batch sequentially — it had pulled a fresh batch and was
partway through it when killed, so 8 messages (not just the one it was actively
on) sat unacked in the consumer group's PEL. Once `STREAM_CLAIM_IDLE_MS` elapsed,
`XAUTOCLAIM` reclaimed all 8 onto the two surviving workers, which finished them
like any other message. Elapsed time is higher than the non-fault-injected
3-worker scenario specifically because of that wait (see below) — the real cost
of the chosen idle threshold, not overhead from the reclaim mechanism itself
(each of the 8 was processed exactly once). **Every paper still finished** — the
task's actual claim to verify.

## A real bug found running this for real

The first real 3-worker run (not in the numbers above, logs discarded) finished
with `papers_done=44` for 40 enqueued papers and multiple reclaimed messages -- a
correctness bug, not a benchmark artifact. `STREAM_CLAIM_IDLE_MS`'s original guess
(30s) was **shorter than real per-paper processing latency under contention** (p95
84-164s across the runs here): a message still being legitimately processed by its
original consumer looked "stuck" to `XAUTOCLAIM` and got reclaimed and processed a
*second* time by another consumer, racing the first. Fixed by raising the default
to 180s (`axiom/config.py`, comment has the full real-data justification) and
re-running -- `reclaim_count=0` for the clean non-fault-injected runs above, `8`
for the real fault injection (the only scenario that should reclaim anything at
this setting, and exactly as many messages as the killed worker's unfinished
batch, not more). Documented in `docs/DECISIONS.md` OD43 rather than silently
patched. (A second, smaller bug caught while writing this up: the benchmark's own
`count_reclaims()` originally counted *log lines* containing "reclaimed" rather
than the message count each line reports -- undercounted 8 as 1. Fixed in
`scripts/bench_distributed.py`, numbers above are the corrected count.)

## Environment note: background-process reliability

Two earlier attempts at running the full 3-scenario benchmark as one
`run_in_background` invocation failed: the first was corrupted by an unrelated
`pytest -m redis` run executed concurrently against the *same* Redis instance and
stream names (both flushed/wrote the same keys); the second surfaced two
duplicate orchestrator processes under two different Python interpreters running
simultaneously against the same Redis DB and SQLite file, for reasons not fully
root-caused. Both were caught by comparing `agent_runs.papers_done` against
`papers_total` rather than trusted blindly. The numbers in this file are from
three separate, directly-observed foreground runs (`--scenario 1`, `--scenario 2`,
`--scenario 3`, then `--merge`) with an explicit process-list check between each
one confirming a clean shutdown before the next started.
