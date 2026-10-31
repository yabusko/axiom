"""Offline tests for scripts/bench_agents.py (R10).

Fake `eval/runs/*.jsonl` trace fixtures only -- no real agent pipeline, no
axiom.tracing (doesn't exist in this worktree; see the module's NOTE and the
patch note for the schema gap this infers around).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import bench_agents


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")


@pytest.fixture
def trace_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# iter_trace_lines
# ---------------------------------------------------------------------------
def test_iter_trace_lines_reads_every_file(trace_dir):
    _write_jsonl(trace_dir / "run1.jsonl", [{"a": 1}, {"a": 2}])
    _write_jsonl(trace_dir / "run2.jsonl", [{"a": 3}])
    lines = list(bench_agents.iter_trace_lines(trace_dir))
    assert len(lines) == 3


def test_iter_trace_lines_skips_malformed_lines(trace_dir):
    path = trace_dir / "run1.jsonl"
    path.write_text('{"a": 1}\nNOT JSON\n{"a": 2}\n\n', encoding="utf-8")
    lines = list(bench_agents.iter_trace_lines(trace_dir))
    assert lines == [{"a": 1}, {"a": 2}]


def test_iter_trace_lines_missing_dir_returns_empty(tmp_path):
    assert list(bench_agents.iter_trace_lines(tmp_path / "nope")) == []


def test_iter_trace_lines_run_id_restricts_to_one_file(trace_dir):
    _write_jsonl(trace_dir / "run1.jsonl", [{"a": 1}, {"a": 2}])
    _write_jsonl(trace_dir / "run2.jsonl", [{"a": 3}])
    lines = list(bench_agents.iter_trace_lines(trace_dir, run_id="run1"))
    assert lines == [{"a": 1}, {"a": 2}]


def test_iter_trace_lines_run_id_missing_file_returns_empty(trace_dir):
    _write_jsonl(trace_dir / "run1.jsonl", [{"a": 1}])
    assert list(bench_agents.iter_trace_lines(trace_dir, run_id="nope")) == []


# ---------------------------------------------------------------------------
# select_papers (the --n cap)
# ---------------------------------------------------------------------------
def test_select_papers_caps_at_n_first_seen():
    lines = [{"paper_id": "P1"}, {"paper_id": "P2"}, {"paper_id": "P1"}, {"paper_id": "P3"}]
    assert bench_agents.select_papers(lines, 2) == {"P1", "P2"}


def test_select_papers_n_larger_than_available():
    lines = [{"paper_id": "P1"}, {"paper_id": "P2"}]
    assert bench_agents.select_papers(lines, 50) == {"P1", "P2"}


# ---------------------------------------------------------------------------
# rejection_rates
# ---------------------------------------------------------------------------
def _call(run_id, paper_id, agent, attempt, verdict, **extra):
    return {"run_id": run_id, "paper_id": paper_id, "agent": agent,
            "attempt": attempt, "verdict": verdict, **extra}


def test_rejection_rate_first_pass_vs_final_self_correction():
    # extraction.verifier on P1: rejected at attempt 1, accepted at attempt 2 (final ok).
    # extraction.verifier on P2: rejected at attempt 1, still rejected at attempt 3 (final fail).
    lines = [
        _call("r1", "P1", "extraction.verifier", 1, "rejected"),
        _call("r1", "P1", "extraction.verifier", 2, "accepted"),
        _call("r1", "P2", "extraction.verifier", 1, "rejected"),
        _call("r1", "P2", "extraction.verifier", 2, "rejected"),
        _call("r1", "P2", "extraction.verifier", 3, "rejected"),
    ]
    rates = bench_agents.rejection_rates(lines)
    assert rates["n_call_sequences"] == 2
    assert rates["first_pass"] == pytest.approx(1.0)   # both sequences rejected at attempt 1
    assert rates["final"] == pytest.approx(0.5)        # only P2's sequence still rejected


def test_rejection_rate_ignores_provider_layer_lines():
    """Only `*.verifier` lines count -- provider-layer call outcomes (ok/
    refusal/schema_invalid from axiom/llm_providers.py) share the same
    (run_id, paper_id, agent) shape but must not be mistaken for verifier
    verdicts, and must not be double-counted alongside the real verifier
    line for the same attempt."""
    lines = [
        _call("r1", "P1", "extraction", 1, "schema_invalid"),   # provider-layer, not a verdict
        _call("r1", "P1", "extraction", 2, "ok"),               # provider-layer, not a verdict
        _call("r1", "P1", "extraction.verifier", 1, "rejected"),
        _call("r1", "P1", "extraction.verifier", 2, "accepted"),
    ]
    rates = bench_agents.rejection_rates(lines)
    assert rates["n_call_sequences"] == 1
    assert rates["first_pass"] == pytest.approx(1.0)
    assert rates["final"] == pytest.approx(0.0)


def test_rejection_rate_all_accepted_first_try():
    lines = [_call("r1", "P1", "extraction.verifier", 1, "accepted")]
    rates = bench_agents.rejection_rates(lines)
    assert rates["first_pass"] == 0.0
    assert rates["final"] == 0.0


def test_rejection_rate_empty_input():
    rates = bench_agents.rejection_rates([])
    assert rates["first_pass"] == 0.0
    assert rates["final"] == 0.0
    assert rates["n_call_sequences"] == 0


# ---------------------------------------------------------------------------
# run_latency_stats
# ---------------------------------------------------------------------------
def test_run_latency_stats_computes_span_per_paper():
    # ts is logged at call COMPLETION, so each call's span is [ts - latency_s, ts]:
    #   call 1: [00:00:00 - 1.0, 00:00:00] = [-1.0, 0.0]
    #   call 2: [00:00:05 - 2.0, 00:00:05] = [3.0, 5.0]
    # span = max(ends) - min(starts) = 5.0 - (-1.0) = 6.0s
    lines = [
        {"run_id": "r1", "paper_id": "P1", "ts": "2026-01-01T00:00:00Z", "latency_s": 1.0},
        {"run_id": "r1", "paper_id": "P1", "ts": "2026-01-01T00:00:05Z", "latency_s": 2.0},
    ]
    stats = bench_agents.run_latency_stats(lines)
    assert stats["p50"] == pytest.approx(6.0)
    assert stats["n_papers_with_timing"] == 1


def test_run_latency_stats_single_call_span_is_its_own_latency():
    """A paper with exactly one trace line: span == that call's latency_s,
    not 0 -- regression check for the ts-is-end-of-call fix (C/E1)."""
    lines = [{"run_id": "r1", "paper_id": "P1", "ts": "2026-01-01T00:00:10Z", "latency_s": 4.0}]
    stats = bench_agents.run_latency_stats(lines)
    assert stats["p50"] == pytest.approx(4.0)


def test_run_latency_stats_p50_p95_over_multiple_papers():
    def mk(pid, start_sec, latency):
        return {"run_id": "r1", "paper_id": pid,
                "ts": f"2026-01-01T00:00:{start_sec:02d}Z", "latency_s": latency}

    lines = [mk("P1", 0, 1.0), mk("P2", 0, 2.0), mk("P3", 0, 3.0), mk("P4", 0, 10.0)]
    stats = bench_agents.run_latency_stats(lines)
    assert stats["n_papers_with_timing"] == 4
    assert stats["p50"] > 0
    assert stats["p95"] >= stats["p50"]


def test_run_latency_stats_unparsable_timestamp_is_skipped():
    lines = [{"run_id": "r1", "paper_id": "P1", "ts": "not-a-date", "latency_s": 1.0}]
    stats = bench_agents.run_latency_stats(lines)
    assert stats["n_papers_with_timing"] == 0
    assert stats["p50"] == 0.0


# ---------------------------------------------------------------------------
# cost_stats / model_ids
# ---------------------------------------------------------------------------
def test_cost_stats_per_paper_and_per_run():
    lines = [
        {"run_id": "r1", "paper_id": "P1", "cost_usd": 0.01},
        {"run_id": "r1", "paper_id": "P1", "cost_usd": 0.02},
        {"run_id": "r1", "paper_id": "P2", "cost_usd": 0.03},
        {"run_id": "r2", "paper_id": "P3", "cost_usd": 0.04},
    ]
    stats = bench_agents.cost_stats(lines)
    assert stats["total_usd"] == pytest.approx(0.10)
    assert stats["per_paper_usd"] == pytest.approx(0.10 / 3)
    assert stats["per_run_usd"] == pytest.approx(0.10 / 2)


def test_cost_stats_empty_input():
    stats = bench_agents.cost_stats([])
    assert stats == {"total_usd": 0.0, "per_paper_usd": 0.0, "per_run_usd": 0.0}


def test_model_ids_deduplicates_and_sorts():
    lines = [{"model": "claude-sonnet-5"}, {"model": "claude-haiku-4-5"},
              {"model": "claude-sonnet-5"}]
    assert bench_agents.model_ids(lines) == ["claude-haiku-4-5", "claude-sonnet-5"]


# ---------------------------------------------------------------------------
# key-alias tolerance (the schema-gap mitigation)
# ---------------------------------------------------------------------------
def test_get_tolerates_alternate_key_spellings():
    line = {"input_tokens": 100, "cost": 0.5, "timestamp": "2026-01-01T00:00:00Z"}
    assert bench_agents._get(line, "tokens_in") == 100
    assert bench_agents._get(line, "cost_usd") == 0.5
    assert bench_agents._get(line, "ts") == "2026-01-01T00:00:00Z"


def test_get_returns_default_when_absent():
    assert bench_agents._get({}, "model", default="unknown") == "unknown"


# ---------------------------------------------------------------------------
# build_report end-to-end on a small fixture
# ---------------------------------------------------------------------------
def test_build_report_end_to_end():
    lines = [
        {"run_id": "r1", "paper_id": "P1", "agent": "extraction", "model": "claude-haiku-4-5",
         "attempt": 1, "verdict": "accepted", "cost_usd": 0.01,
         "ts": "2026-01-01T00:00:00Z", "latency_s": 1.0},
        {"run_id": "r1", "paper_id": "P1", "agent": "synthesis", "model": "claude-sonnet-5",
         "attempt": 1, "verdict": "accepted", "cost_usd": 0.05,
         "ts": "2026-01-01T00:00:02Z", "latency_s": 3.0},
    ]
    report = bench_agents.build_report(lines)
    assert report["n_papers"] == 1
    assert report["n_runs"] == 1
    assert report["n_trace_lines"] == 2
    assert report["model_ids"] == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert report["papers_included"] == ["P1"]
    assert report["cost_usd"]["total_usd"] == pytest.approx(0.06)
    assert "generated_at" in report


# ---------------------------------------------------------------------------
# main() writes eval/agent_bench.json-shaped output
# ---------------------------------------------------------------------------
def test_main_writes_report_file_respecting_n_cap(trace_dir, tmp_path, monkeypatch, capsys):
    rows = []
    for i in range(5):
        rows.append({"run_id": "r1", "paper_id": f"P{i}", "agent": "extraction",
                     "model": "claude-haiku-4-5", "attempt": 1, "verdict": "accepted",
                     "cost_usd": 0.01, "ts": "2026-01-01T00:00:00Z", "latency_s": 1.0})
    _write_jsonl(trace_dir / "r1.jsonl", rows)

    out_path = tmp_path / "agent_bench.json"
    monkeypatch.setattr(
        "sys.argv",
        ["bench_agents.py", "--n", "3", "--trace-dir", str(trace_dir), "--out", str(out_path)],
    )
    bench_agents.main()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["n_papers"] == 3   # capped by --n, not the 5 on disk
    capsys.readouterr()


def test_main_run_id_filters_to_one_run(trace_dir, tmp_path, monkeypatch, capsys):
    rows_r1 = [{"run_id": "r1", "paper_id": "P1", "agent": "extraction",
                "model": "claude-haiku-4-5", "attempt": 1, "verdict": "accepted",
                "cost_usd": 0.01, "ts": "2026-01-01T00:00:00Z", "latency_s": 1.0}]
    rows_r2 = [{"run_id": "r2", "paper_id": "P2", "agent": "extraction",
                "model": "claude-haiku-4-5", "attempt": 1, "verdict": "accepted",
                "cost_usd": 0.02, "ts": "2026-01-01T00:00:00Z", "latency_s": 1.0}]
    _write_jsonl(trace_dir / "r1.jsonl", rows_r1)
    _write_jsonl(trace_dir / "r2.jsonl", rows_r2)

    out_path = tmp_path / "agent_bench.json"
    monkeypatch.setattr(
        "sys.argv",
        ["bench_agents.py", "--run-id", "r1", "--trace-dir", str(trace_dir), "--out", str(out_path)],
    )
    bench_agents.main()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["n_runs"] == 1
    assert data["papers_included"] == ["P1"]
    capsys.readouterr()


def test_main_no_traces_raises_systemexit(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty_runs"
    empty_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["bench_agents.py", "--trace-dir", str(empty_dir), "--out", str(tmp_path / "out.json")],
    )
    with pytest.raises(SystemExit):
        bench_agents.main()
