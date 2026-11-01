"""Offline tests for scripts/ai_second_grader_sample.py's pure helpers.

The actual second-grading run makes real LLM calls against the eval DB and
is not exercised here -- only sampling and cost estimation.
"""
from __future__ import annotations

from scripts import ai_second_grader_sample as second_grader


def test_sample_query_indices_is_sorted_unique_and_in_range():
    idx = second_grader.sample_query_indices(51, 10)
    assert len(idx) == 10
    assert idx == sorted(set(idx))
    assert all(0 <= i < 51 for i in idx)


def test_sample_query_indices_deterministic():
    a = second_grader.sample_query_indices(51, 10)
    b = second_grader.sample_query_indices(51, 10)
    assert a == b


def test_sample_query_indices_caps_at_total():
    idx = second_grader.sample_query_indices(5, 10)
    assert idx == [0, 1, 2, 3, 4]


def test_estimate_cost_scales_with_candidate_count():
    data = {"queries": [
        {"query": "q0", "judgments": {f"P{i}": 1 for i in range(5)}},
        {"query": "q1", "judgments": {f"P{i}": 1 for i in range(10)}},
    ]}
    small = second_grader.estimate_cost(data, [0])
    big = second_grader.estimate_cost(data, [1])
    assert small > 0
    assert big > small
