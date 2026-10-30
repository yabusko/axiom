"""Tests for axiom.tracing -- one JSON line per LLM call (R10)."""
from __future__ import annotations

import json

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
