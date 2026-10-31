"""Agent-pipeline benchmark (R10): `--n 50` -> `eval/agent_bench.json`.

Reads the LLM-call trace files `axiom/tracing.py` writes to `eval/runs/<run_id>.jsonl`
(config.TRACE_DIR) -- one JSON object per LLM call. Computes:
  - verifier rejection rate, first-pass (attempt==1) vs final (the last retry
    attempt in each agent-call sequence) -- R4's self-correction loop.
  - run latency p50/p95 -- wall-clock span per (run_id, paper_id): the time
    from its first trace line to its last trace line's end (ts + latency).
  - $/paper, $/run -- summed call cost, averaged over distinct papers/runs.
  - the distinct model IDs actually used.

NOTE (gap flagged for the main session -- see
.claude/agentic/patch_notes_eval_implementer.md): `axiom/tracing.py` does not
exist in this worktree (it is owned by the parallel agent-implementer
workstream, a different git worktree) -- AXIOM_AGENTIC_TASK.md §4 and
docs/agentic/ARCHITECTURE.md §1/§2 only describe its trace line in prose
("agent, model, tokens in/out, latency, cost, attempt, verdict + reasons"),
not as a field-by-field schema. The key names below are inferred from that
prose plus `db/schema.sql`'s `agent_run_items.verdict` vocabulary
("accepted"/"rejected", the one place the real schema is unambiguous).
`_get()` tolerates a few alternate spellings per field so this script
degrades gracefully instead of crashing if the real tracing.py differs
slightly; confirm/align via the patch note. Primary assumed shape, one JSON
object per line:

    {"run_id": str, "paper_id": str, "agent": str, "model": str, "tier": str,
     "attempt": int, "tokens_in": int, "tokens_out": int, "latency_s": float,
     "cost_usd": float, "verdict": "accepted" | "rejected", "reasons": [str],
     "ts": "<ISO-8601>"}

Usage (from repo root):
    python scripts/bench_agents.py --n 50
    python scripts/bench_agents.py --n 50 --trace-dir eval/runs --out eval/agent_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config

# Alternate key spellings this script tolerates, in priority order, in case
# the real axiom/tracing.py (sibling-owned, not in this worktree) differs
# slightly from the inferred schema above.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "run_id": ("run_id",),
    "paper_id": ("paper_id",),
    "agent": ("agent", "agent_name", "worker"),
    "model": ("model", "model_id"),
    "attempt": ("attempt", "attempt_no"),
    "tokens_in": ("tokens_in", "input_tokens", "prompt_tokens"),
    "tokens_out": ("tokens_out", "output_tokens", "completion_tokens"),
    "latency_s": ("latency_s", "latency", "latency_seconds", "duration_s"),
    "cost_usd": ("cost_usd", "cost", "cost_usd_total"),
    "verdict": ("verdict",),
    "ts": ("ts", "timestamp", "time"),
}


def _get(line: dict, field: str, default=None):
    for key in _KEY_ALIASES[field]:
        if key in line:
            return line[key]
    return default


def _parse_ts(value) -> float | None:
    """ISO-8601 string or epoch number -> epoch seconds; None if unparsable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy.percentile's default method),
    pure Python so this module has no new numeric dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def iter_trace_lines(trace_dir: Path, run_id: str | None = None):
    """Yield every parsed JSON object across trace files in `trace_dir`.

    With `run_id` given, reads only that run's file (`trace_dir/<run_id>.jsonl`
    -- the exact name `axiom/tracing.py.trace_path` writes), so a bench run
    never mixes lines from unrelated runs sitting in the same directory.
    Without it, every `*.jsonl` file is read, as before.

    Malformed lines are skipped (benchmarking must never crash on a partially
    written trace file from a run that's still in progress or was killed).
    """
    if not trace_dir.exists():
        return
    if run_id is not None:
        paths = [trace_dir / f"{run_id}.jsonl"]
    else:
        paths = sorted(trace_dir.glob("*.jsonl"))
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue


def select_papers(lines: list[dict], n: int) -> set[str]:
    """The first `n` distinct paper_ids encountered, in first-seen order --
    the laptop rule's "agent benchmark <= 50 papers" cap (AXIOM_AGENTIC_TASK.md
    §1), applied here rather than left to whatever is on disk."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in lines:
        pid = _get(line, "paper_id")
        if pid and pid not in seen_set:
            seen_set.add(pid)
            seen.append(pid)
        if len(seen) >= n:
            break
    return seen_set


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rejection_rates(lines: list[dict]) -> dict:
    """First-pass vs final verifier rejection rate (R4).

    Only verifier-layer lines count here -- `axiom/agents/verifier.py.
    log_attempt` writes one trace line per verification attempt with
    `agent="<agent>.verifier"` and `verdict` in {"accepted", "rejected"}.
    The SAME trace file also has provider-layer lines (`agent="extraction"`,
    verdicts "ok"/"refusal"/"schema_invalid"/... from `axiom/llm_providers.py`)
    for the underlying LLM calls -- mixing those into this count would both
    double-count a paper's attempts and misread "schema_invalid" or "refusal"
    as if they were verifier rejections. Filter to lines whose `agent` ends
    in ".verifier" before grouping.

    Groups the remaining lines into (run_id, paper_id, agent) call sequences
    (one sequence = one agent's retries for one paper within one run).
    First-pass rejection looks only at attempt==1; final rejection looks at
    the sequence's highest-attempt line (what the pipeline ultimately did
    after <=3 tries).
    """
    verifier_lines = [l for l in lines if str(_get(l, "agent") or "").endswith(".verifier")]
    sequences: dict[tuple, list[dict]] = {}
    for line in verifier_lines:
        key = (_get(line, "run_id"), _get(line, "paper_id"), _get(line, "agent"))
        sequences.setdefault(key, []).append(line)

    first_pass_total = first_pass_rejected = 0
    final_total = final_rejected = 0
    for seq in sequences.values():
        first = min(seq, key=lambda l: _get(l, "attempt", 1))
        last = max(seq, key=lambda l: _get(l, "attempt", 1))
        first_pass_total += 1
        if _get(first, "verdict") == "rejected":
            first_pass_rejected += 1
        final_total += 1
        if _get(last, "verdict") == "rejected":
            final_rejected += 1

    return {
        "first_pass": (first_pass_rejected / first_pass_total) if first_pass_total else 0.0,
        "final": (final_rejected / final_total) if final_total else 0.0,
        "n_call_sequences": first_pass_total,
    }


def run_latency_stats(lines: list[dict]) -> dict:
    """p50/p95 wall-clock latency per (run_id, paper_id), in seconds.

    `ts` is logged at call COMPLETION (`axiom/tracing.py.log_call` stamps it
    after the call returns), so a call's span is `[ts - latency_s, ts]`, not
    `[ts, ts + latency_s]` -- treating `ts` as the start would exclude the
    call's own duration from the measured span and understate elapsed time.
    A paper's "run latency" is then the earliest start to the latest end
    across every agent call that paper went through in that run -- an
    approximation of the per-paper pipeline wall time from call-level traces
    (no dedicated per-paper start/stop line exists).
    """
    groups: dict[tuple, list[dict]] = {}
    for line in lines:
        key = (_get(line, "run_id"), _get(line, "paper_id"))
        groups.setdefault(key, []).append(line)

    spans: list[float] = []
    for group in groups.values():
        starts, ends = [], []
        for line in group:
            ts = _parse_ts(_get(line, "ts"))
            latency = _get(line, "latency_s", 0.0) or 0.0
            if ts is None:
                continue
            starts.append(ts - float(latency))
            ends.append(ts)
        if starts and ends:
            spans.append(max(ends) - min(starts))

    return {
        "p50": _percentile(spans, 50),
        "p95": _percentile(spans, 95),
        "n_papers_with_timing": len(spans),
    }


def cost_stats(lines: list[dict]) -> dict:
    total = sum(float(_get(line, "cost_usd", 0.0) or 0.0) for line in lines)
    n_papers = len({_get(l, "paper_id") for l in lines if _get(l, "paper_id")})
    n_runs = len({_get(l, "run_id") for l in lines if _get(l, "run_id")})
    return {
        "total_usd": total,
        "per_paper_usd": (total / n_papers) if n_papers else 0.0,
        "per_run_usd": (total / n_runs) if n_runs else 0.0,
    }


def model_ids(lines: list[dict]) -> list[str]:
    return sorted({_get(l, "model") for l in lines if _get(l, "model")})


def build_report(lines: list[dict]) -> dict:
    papers = sorted({_get(l, "paper_id") for l in lines if _get(l, "paper_id")})
    runs = sorted({_get(l, "run_id") for l in lines if _get(l, "run_id")})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trace_lines": len(lines),
        "n_papers": len(papers),
        "n_runs": len(runs),
        "model_ids": model_ids(lines),
        "rejection_rate": rejection_rates(lines),
        "run_latency_seconds": run_latency_stats(lines),
        "cost_usd": cost_stats(lines),
        "papers_included": papers,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                     help="max distinct papers to include (laptop cap, "
                          "AXIOM_AGENTIC_TASK.md §1)")
    ap.add_argument("--trace-dir", type=str, default=None,
                     help="override config.TRACE_DIR")
    ap.add_argument("--out", type=str, default=None,
                     help="override config.AGENT_BENCH_PATH")
    ap.add_argument("--run-id", type=str, default=None,
                     help="restrict to one run's trace file (trace_dir/<run_id>.jsonl) "
                          "instead of every *.jsonl file in trace_dir")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir) if args.trace_dir else config.TRACE_DIR
    all_lines = list(iter_trace_lines(trace_dir, run_id=args.run_id))
    if not all_lines:
        source = f"run '{args.run_id}' under" if args.run_id else "under"
        raise SystemExit(
            f"no trace lines found for {source} '{trace_dir}' -- run the agent "
            f"pipeline first (eval/runs/<run_id>.jsonl)."
        )

    keep_papers = select_papers(all_lines, args.n)
    lines = [l for l in all_lines if _get(l, "paper_id") in keep_papers]

    report = build_report(lines)
    out_path = Path(args.out) if args.out else config.AGENT_BENCH_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"papers={report['n_papers']} runs={report['n_runs']} "
          f"trace_lines={report['n_trace_lines']}")
    print(f"rejection rate: first_pass={report['rejection_rate']['first_pass']:.3f} "
          f"final={report['rejection_rate']['final']:.3f}")
    print(f"run latency p50={report['run_latency_seconds']['p50']:.2f}s "
          f"p95={report['run_latency_seconds']['p95']:.2f}s")
    print(f"cost: ${report['cost_usd']['per_paper_usd']:.4f}/paper "
          f"${report['cost_usd']['per_run_usd']:.4f}/run "
          f"(${report['cost_usd']['total_usd']:.4f} total)")
    print(f"models: {', '.join(report['model_ids']) or '(none)'}")
    print(f"\n[report] wrote {out_path}")


if __name__ == "__main__":
    main()
