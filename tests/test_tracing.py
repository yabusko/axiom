"""Tests for axiom.tracing -- one JSON line per LLM call (R10)."""
from __future__ import annotations

import json
import multiprocessing

from axiom import config, tracing


def test_log_call_creates_trace_dir(tmp_path, monkeypatch):
    missing_dir = tmp_path / "does" / "not" / "exist" / "yet"
    monkeypatch.setattr(config, "TRACE_DIR", missing_dir)

    assert not missing_dir.exists()
    tracing.log_call("run1", agent="extraction", model="claude-haiku-4-5", verdict="ok")
    assert missing_dir.exists()
    assert (missing_dir / "run1.jsonl").exists()


def test_log_call_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)

    record = tracing.log_call(
        "run1", agent="extraction", model="claude-haiku-4-5", verdict="ok",
        tokens_in=100, tokens_out=50, latency_s=1.23, cost_usd=0.0007,
        attempt=2, reasons=["looked fine"],
    )
    lines = (tmp_path / "run1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])

    for key in ("agent", "model", "tokens_in", "tokens_out", "latency_s",
                "cost_usd", "attempt", "verdict", "reasons", "run_id", "ts"):
        assert key in parsed

    assert parsed["agent"] == "extraction"
    assert parsed["model"] == "claude-haiku-4-5"
    assert parsed["tokens_in"] == 100
    assert parsed["tokens_out"] == 50
    assert parsed["latency_s"] == 1.23
    assert parsed["cost_usd"] == 0.0007
    assert parsed["attempt"] == 2
    assert parsed["verdict"] == "ok"
    assert parsed["reasons"] == ["looked fine"]
    assert parsed["run_id"] == "run1"
    assert parsed == record


def test_log_call_defaults_reasons_to_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    record = tracing.log_call("run1", agent="coordinator", model=None, verdict="ok")
    assert record["reasons"] == []
    assert record["tokens_in"] == 0
    assert record["cost_usd"] == 0.0


def test_log_call_appends_multiple_lines_same_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    tracing.log_call("run1", agent="coordinator", model="m", verdict="ok", attempt=1)
    tracing.log_call("run1", agent="extraction", model="m", verdict="rejected", attempt=1)
    tracing.log_call("run1", agent="extraction", model="m", verdict="ok", attempt=2)

    records = tracing.read_trace("run1")
    assert len(records) == 3
    assert [r["agent"] for r in records] == ["coordinator", "extraction", "extraction"]


def test_log_call_separates_runs_into_separate_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    tracing.log_call("run1", agent="coordinator", model="m", verdict="ok")
    tracing.log_call("run2", agent="coordinator", model="m", verdict="ok")

    assert len(tracing.read_trace("run1")) == 1
    assert len(tracing.read_trace("run2")) == 1


def test_read_trace_missing_file_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    assert tracing.read_trace("does-not-exist") == []


def _write_many(trace_dir: str, run_id: str, n: int) -> None:
    """multiprocessing.Process target (module-level so it's picklable on
    Windows' spawn start method) -- a real separate OS process, not just
    another thread/task in this one, so it needs its own TRACE_DIR set
    directly (pytest's monkeypatch is parent-process-local)."""
    from pathlib import Path

    from axiom import config as config_mod
    from axiom import tracing as tracing_mod

    config_mod.TRACE_DIR = Path(trace_dir)
    for i in range(n):
        tracing_mod.log_call(run_id, agent="extraction", model="m", verdict="ok", paper_id=str(i))


def test_log_call_survives_concurrent_processes_without_corrupting_lines(tmp_path, monkeypatch):
    """OD44: distributed mode runs several extraction-worker *processes*
    against the same run_id -- a torn/interleaved line (one process'
    partial write spliced into another's) is exactly the real corruption
    found running scripts/bench_distributed.py's 3-worker scenarios before
    this fix. Real multiprocessing.Process, not threads/mocks -- a plain
    threading.Lock would pass this test too (only ever enforced inside one
    process), so it has to be real separate processes to catch the bug."""
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    run_id = "concurrent-run"
    n_per_proc = 60
    n_procs = 3
    procs = [
        multiprocessing.Process(target=_write_many, args=(str(tmp_path), run_id, n_per_proc))
        for _ in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_per_proc * n_procs
    for line in lines:
        json.loads(line)  # every line parses whole -- no torn/interleaved write
