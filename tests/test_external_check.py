from __future__ import annotations

from axiom import external_check


def test_build_query_combines_both_labels():
    assert external_check.build_query("GPT-3", "retrieve-and-read") == "GPT-3 retrieve-and-read"


def test_is_artifact_below_threshold_false():
    assert external_check.is_artifact(external_check.ARTIFACT_WORKS_THRESHOLD - 1) is False


def test_is_artifact_at_threshold_true():
    assert external_check.is_artifact(external_check.ARTIFACT_WORKS_THRESHOLD) is True


class _FakeClient:
    def __init__(self, count: int):
        self.count = count

    def count_works(self, query: str) -> int:
        return self.count


def test_check_gap_builds_result_from_client():
    result = external_check.check_gap(_FakeClient(9), "dense passage retrieval", "retrieve-and-read")
    assert result.query == "dense passage retrieval retrieve-and-read"
    assert result.works_count == 9
    assert result.is_likely_artifact is False


def test_check_gap_flags_high_count():
    result = external_check.check_gap(_FakeClient(5000), "hallucination", "retrieval-augmented generation")
    assert result.is_likely_artifact is True
