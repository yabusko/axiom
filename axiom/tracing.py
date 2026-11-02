"""Agent-pipeline tracing (R10, docs/agentic/ARCHITECTURE.md §1).

One JSON line per LLM call (and per verifier attempt) appended to
`config.TRACE_DIR / f"{run_id}.jsonl"`. Plain `json.dumps` + file append --
no logging framework, matching R11's framework-free discipline. Used by
`axiom/llm_providers.py` (every provider call) and `axiom/agents/verifier.py`
(every verification attempt, including ones that cost no tokens).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import filelock

from axiom import config


def trace_path(run_id: str) -> Path:
    return config.TRACE_DIR / f"{run_id}.jsonl"


def log_call(
    run_id: str,
    *,
    agent: str,
    model: str | None,
    verdict: str,
    paper_id: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_s: float = 0.0,
    cost_usd: float = 0.0,
    attempt: int = 1,
    reasons: list[str] | None = None,
) -> dict:
    """Append one trace line and return the record that was written.

    `agent` names the calling agent (e.g. "coordinator", "extraction",
    "verifier"). `verdict` is a short free-text outcome tag (e.g. "ok",
    "refusal", "max_tokens", "error", "accepted", "rejected"). `reasons` is a
    list of human-readable strings (e.g. verifier error codes/messages) --
    empty when the call needed no explanation. `paper_id` is the paper this
    call was for (R10's $/paper and per-paper latency need it); `None` for
    calls that aren't scoped to one paper (e.g. synthesis's per-cluster/
    per-gap calls).
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "paper_id": paper_id,
        "agent": agent,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_s": latency_s,
        "cost_usd": cost_usd,
        "attempt": attempt,
        "verdict": verdict,
        "reasons": reasons or [],
    }
    path = trace_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # OD44: distributed mode runs several extraction-worker *processes*
    # against the same run_id -- a plain threading.Lock only ever serializes
    # writers inside one process, so two workers' concurrent appends to the
    # shared jsonl file could still interleave mid-write (found for real
    # running scripts/bench_distributed.py's 3-worker scenarios: a torn
    # line like `easons": []}` spliced into the middle of another).
    # filelock.FileLock is an OS-level advisory lock, built fresh against
    # the current (test-patchable) TRACE_DIR each call -- so it serializes
    # across processes too, and two separate FileLock objects pointed at
    # the same path still mutually exclude correctly.
    with filelock.FileLock(str(path.parent / ".trace.lock")):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return record


def read_trace(run_id: str) -> list[dict]:
    """Read back a run's trace lines (used by tests and bench_agents.py)."""
    path = trace_path(run_id)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
