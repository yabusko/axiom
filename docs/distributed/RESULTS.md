# Distributed workers — results

Every number below traces to `eval/distributed_bench_n40.json` / `eval/distributed_bench_n200.json`
/ `eval/distributed_bench_local_n40.json` / `eval/distributed_bench_local_n200.json`
(`scripts/bench_distributed.py`), real runs on 2026-11-01/02: real Redis (`docker
compose up -d redis`), real `python -m axiom.workers <role>` subprocesses (not
Docker containers — same code path, no image-build time spent to measure it),
real LLM calls (extraction only — coordinator/enrichment/linking never call an
LLM, see `docs/DECISIONS.md` OD43). Two corpus sizes, both from the task's own
suggestion: `n=40` and `n=200` real papers from `data/axiom.db`, each at 1 vs 3
extraction workers plus a fault-injection scenario, plus an in-process
`AXIOM_EXECUTION=local` baseline (`--scenario local`) on the same papers for
comparison. Cost estimated and capped before every run; actual total spend
across all 9 runs (n=40 and n=200 x {1 worker, 3 workers, fault-injected,
local}, plus one extra fault-injection re-verification at a lower reclaim
timeout — see "The real root cause" below) was **$0.3778** — over the $0.30
target for this work by about $0.08. Each individual run was estimated and
capped before spending ($0.0058–$0.0735/scenario); the overrun was from not
summing the full planned set against the target before starting, not from any
one run exceeding its own estimate.

## Throughput: 1 vs 3 extraction workers

**n=40**

| Scenario | Workers | Papers | Elapsed | Throughput | p50 latency | p95 latency | Spend |
|---|---|---|---|---|---|---|---|
| 1-extraction-worker | 1 | 40/40 | 72.1s | 33.30 papers/min | 32.47s | 68.08s | $0.0172 |
| 3-extraction-workers | 3 | 40/40 | 28.4s | 84.63 papers/min | 14.69s | 23.38s | $0.0166 |
| local-async (in-process) | — | 40/40 | 42.6s | 56.39 papers/min | 24.38s | 38.63s | $0.0174 |

**n=200**

| Scenario | Workers | Papers | Elapsed | Throughput | p50 latency | p95 latency | Spend |
|---|---|---|---|---|---|---|---|
| 1-extraction-worker | 1 | 200/200 | 301.3s | 39.83 papers/min | 158.33s | 277.07s | $0.0735 |
| 3-extraction-workers | 3 | 200/200 | 108.3s | 110.82 papers/min | 57.38s | 99.20s | $0.0733 |
| local-async (in-process) | — | 200/200 | 246.5s | 48.68 papers/min | 129.50s | 232.54s | $0.0718 |

**3 workers reach ~2.5x (n=40) to ~2.8x (n=200) the throughput of 1**, not the
full 3x — `coordinator`/`enrichment`/`linking` each stay at exactly 1 instance
in every scenario (only `extraction` scales, per the task), so every message
still drains through exactly one of each regardless of how many extraction
workers run; that's a real, plausible reason the ceiling sits below 3x, though
it was never isolated and measured directly (see OD44's own caveat on this).
This *replaces* an earlier, incorrect version of this claim: the first
pre-fix measurement here showed only ~2.07x and attributed essentially all of
the shortfall to that same explanation, stated as settled fact. The real
dominant cause at the time was a different bug entirely (below) — once that
was fixed, throughput moved from ~2.07x to ~2.5–2.8x, which is what's actually
consistent with "other single-instance stages share some of the ceiling," not
proof that they're the whole story.

**Distributed vs. local-async.** 3 distributed extraction workers beat the
in-process baseline outright (84.6 vs 56.4 papers/min at n=40; 110.8 vs 48.7 at
n=200) — real horizontal scaling of the one LLM-calling stage, not just
overhead moved around. A *single* distributed extraction worker is slower than
the local baseline at both sizes (33.3 vs 56.4 at n=40; 39.8 vs 48.7 at
n=200): the local path runs all 4 roles as one `asyncio.gather` fan-out in one
process with no network hop between stages, while 1 distributed worker pays
the Redis round-trip between every stage for the same per-paper work, with
none of the horizontal-scaling payoff yet. The crossover point (distributed
wins) is exactly where this task's premise expects it: once `extraction`
itself scales past 1 worker.

Latency (p50/p95) is "time from enqueue to this paper's last
`agent_run_items` write," not a single LLM call's latency — it's dominated by
queueing depth (how many papers are ahead of this one given the active
concurrency), which is why it drops by roughly the same factor throughput
improves by, at both worker counts and both corpus sizes.

## Fault injection

| n | Workers | Papers | Elapsed | Reclaimed messages | STREAM_CLAIM_IDLE_MS | All finished? |
|---|---|---|---|---|---|---|
| 40 | 3 (1 killed at 12/40 done) | 40/40 | 80.8s | 4 | 60s (shipped value) | **yes** |
| 200 | 3 (1 killed at 61/200 done) | 200/200 | 229.7s | 2 | 180s (pre-fix value, not re-run at 200) | **yes** |

One of the three extraction workers was `kill()`ed mid-run in both cases.
**Every paper still finished** in both — the task's actual claim to verify,
true at both corpus sizes and (for n=40) at both reclaim-timeout values. The
reclaimed-message count is now consistently small (2–4, matching
`AGENT_CONCURRENCY`'s batch size) instead of the old bug's up-to-10 per dead
worker — direct evidence of the root-cause fix below, not just the timeout
change. The n=40 row at 60s was re-run specifically to confirm the lower
timeout still recovers correctly; n=200's fault scenario was measured once,
before the timeout was retuned, and wasn't repeated at 60s to keep spend down
(it would cost another ~$0.07 to re-confirm at the same scale — the n=40
re-run already demonstrates the lower timeout is safe).

## The real root cause (OD44) — and why OD43's 180s fix was a symptom patch

The imbalance the task asked to investigate (one worker making far more calls
than the others) and the double-processing bug OD43 patched by raising
`STREAM_CLAIM_IDLE_MS` to 180s were **the same bug**: `run_consumer_loop()`
called `XREADGROUP` with `count=10` but then handled messages **one at a
time** in a plain `for` loop. With one slow LLM call per extraction message, a
worker that checked out a batch of 10 left up to 9 of them sitting idle in its
own PEL while it worked through the 10th — long enough that `XAUTOCLAIM`
mistook "still working through my own backlog" for "dead," reclaiming work
onto a less-loaded consumer. That's directly why load never balanced evenly:
whichever worker happened to read first kept getting handed other workers'
reclaimed backlog too.

**Fix:** `run_consumer_loop()` now reads `count=config.AGENT_CONCURRENCY` (4 —
the same concurrency knob the in-process pipeline already uses) and processes
the batch concurrently via `asyncio.Semaphore(concurrency)` instead of
sequentially. Direct evidence it worked — real Anthropic API call counts per
extraction worker, from the actual benchmark logs:

| Run | Worker 0 | Worker 1 | Worker 2 |
|---|---|---|---|
| n=40, 3 workers (clean) | 17 | 15 | 15 |
| n=200, 3 workers (clean) | 77 | 73 | 76 |

Balanced within a few calls of each other at both scales — compare to the
pre-fix shape the task described (one worker at 22 calls, two others at 14
each, out of 40 papers).

**`STREAM_CLAIM_IDLE_MS` lowered 180s → 60s.** With occupancy no longer
multiplied by prefetch depth, the question is just "how long can one real
extraction call (plus a self-correction retry) legitimately take," not "how
deep is the queue." Every non-fault-injected scenario above came back
`reclaim_count=0` even while 180s was still in effect — itself evidence real
occupancy is nowhere near that high. The cleanest (least-queued) real run,
3 workers/n=40, measured p95=23.4s; 60s keeps >2x headroom over that. Fault
injection re-verified clean at the new value (table above): elapsed time for
recovery dropped from 197.0s (at 180s) to 80.8s (at 60s), with every paper
still finishing — the real cost of whichever threshold is chosen, not
overhead from the reclaim mechanism itself.

Full writeup, including the second (tracing-file) bug this fix's own
re-benchmarking surfaced: `docs/DECISIONS.md` OD44.

## Environment note: background-process reliability

Earlier attempts at running the full 3-scenario benchmark as one
`run_in_background` invocation were unreliable in this environment (unrelated
concurrent test runs against the same Redis instance/stream names, duplicate
orchestrator processes) — running each scenario as its own short, directly-
observed foreground invocation (`--scenario N`, then `--merge`) was the
reliable path, with an explicit process-list check between each one
confirming a clean shutdown before the next started. `--out-suffix` keeps the
n=40 and n=200 result files separate (`eval/distributed_bench_n40.json` /
`_n200.json`), and `--timeout` raises the per-scenario deadline for the
larger run (200 papers at 1 worker takes longer than the default 300s cap).
