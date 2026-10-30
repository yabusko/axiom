"""Tests for the router-worker agent pipeline -- fully offline.

`axiom.llm_providers.complete_json` is monkeypatched with a small fake queue
(`FakeLLM`) so coordinator/extraction exercise their real retry/verifier
logic without any network or model call. Enrichment gets a `FakeEncoder`
with an exact-text->vector lookup so cosine similarity is fully
deterministic. Linking/pipeline use a real (tmp-file) SQLite db via
`axiom.db`, per the existing test suite's convention (see conftest.py).
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from pydantic import ValidationError

from axiom import config, db
from axiom.agents import coordinator, enrichment, extraction, linking, pipeline, verifier
from axiom.agents.contracts import (
    EnrichedEntity,
    Entity,
    EnrichmentResult,
    ExtractionResult,
    RouteDecision,
    VerifierError,
    VerifierResult,
)
from axiom import llm_providers as lp


# --- Shared fakes ----------------------------------------------------------------

class FakeLLM:
    """Programmable stand-in for `llm_providers.complete_json`, keyed by agent."""

    def __init__(self):
        self.queues: dict[str, list] = {}
        self.calls: list[dict] = []

    def push(self, agent: str, item) -> None:
        self.queues.setdefault(agent, []).append(item)

    async def __call__(self, tier, system, prompt, model_cls, *, run_id, agent,
                        paper_id=None, attempt=1, temperature=None):
        self.calls.append({"agent": agent, "prompt": prompt, "attempt": attempt,
                            "tier": tier, "paper_id": paper_id})
        queue = self.queues.get(agent, [])
        if not queue:
            raise AssertionError(f"no fake response queued for agent={agent!r}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(lp, "complete_json", fake)
    return fake


class FakeEncoder:
    """Exact-text -> vector lookup; unknown text falls back to `default`."""

    def __init__(self, vectors: dict[str, list[float]], default: list[float] | None = None):
        self.vectors = vectors
        self.default = default or [0.0, 0.0]

    def encode(self, texts, abstracts=None, batch_size=16, max_length=512):
        return [self.vectors.get(t, self.default) for t in texts]


@pytest.fixture(autouse=True)
def _small_retry_budget(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_RETRIES", 2)
    yield


@pytest.fixture(autouse=True)
def _isolated_trace_dir(tmp_path, monkeypatch):
    """Every agent call logs via axiom.tracing regardless of whether the LLM
    provider itself is faked -- point TRACE_DIR at a tmp dir so tests never
    write into the repo's real `eval/runs/`."""
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path / "trace")
    yield


def _validation_error(model_cls, **bad_kwargs) -> ValidationError:
    try:
        model_cls(**bad_kwargs)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


# --- verifier.py -------------------------------------------------------------------

def test_verify_extraction_passes_when_span_is_substring():
    result = ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="low-rank adaptation",
               confidence=0.9),
    ])
    verdict = verifier.verify_extraction(result, abstract="We propose low-rank adaptation of LLMs.", attempt=1)
    assert verdict.ok
    assert verdict.errors == []


def test_verify_extraction_fails_when_span_missing():
    result = ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="not in the abstract", confidence=0.9),
    ])
    verdict = verifier.verify_extraction(result, abstract="Something else entirely.", attempt=1)
    assert not verdict.ok
    assert verdict.errors[0].code == "evidence_span"
    assert verdict.errors[0].field == "entities[0].evidence_span"


def test_verify_enrichment_fails_on_unmatched():
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="x", kind="method", evidence_span="x", confidence=0.5,
                       canonical="", cosine=-1.0, ontology_source="unmatched"),
    ])
    verdict = verifier.verify_enrichment(result, attempt=1)
    assert not verdict.ok
    assert verdict.errors[0].code == "ontology_membership"


def test_verify_enrichment_fails_below_cosine_threshold(monkeypatch):
    monkeypatch.setattr(config, "VERIFIER_COSINE_MIN", 0.9)
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="x", kind="method", evidence_span="x", confidence=0.5,
                       canonical="Y", cosine=0.5, ontology_source="concept_canonical"),
    ])
    verdict = verifier.verify_enrichment(result, attempt=1)
    assert not verdict.ok
    assert verdict.errors[0].code == "cosine_threshold"


def test_verify_enrichment_passes_above_threshold(monkeypatch):
    monkeypatch.setattr(config, "VERIFIER_COSINE_MIN", 0.7)
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="x", kind="method", evidence_span="x", confidence=0.5,
                       canonical="Y", cosine=0.95, ontology_source="concept_canonical"),
    ])
    verdict = verifier.verify_enrichment(result, attempt=1)
    assert verdict.ok


def test_errors_from_validation_maps_to_schema_code():
    exc = _validation_error(RouteDecision, paper_id="W1", action="bogus", reason="x")
    errors = verifier.errors_from_validation(exc)
    assert errors
    assert all(e.code == "schema" for e in errors)


def test_format_errors_for_retry_renders_each_error():
    errors = [VerifierError(field="entities[0].evidence_span", code="evidence_span", message="nope")]
    text = verifier.format_errors_for_retry(errors)
    assert "evidence_span" in text
    assert "nope" in text


def test_verification_failed_carries_attempts_and_errors():
    errors = [VerifierError(field="f", code="schema", message="m")]
    exc = verifier.VerificationFailed("extraction", 3, errors)
    assert exc.attempts == 3
    assert exc.errors == errors
    assert "extraction" in str(exc)


def test_log_attempt_writes_accepted_and_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    from axiom import tracing
    verifier.log_attempt("run1", "extraction", VerifierResult(ok=True, attempt=1, errors=[]))
    verifier.log_attempt("run1", "extraction", VerifierResult(
        ok=False, attempt=2, errors=[VerifierError(field="f", code="schema", message="m")]))
    traces = tracing.read_trace("run1")
    assert traces[0]["verdict"] == "accepted"
    assert traces[1]["verdict"] == "rejected"
    assert traces[1]["reasons"]


# --- coordinator.py ----------------------------------------------------------------

def test_route_skips_without_llm_call_when_no_text(fake_llm):
    decision = asyncio.run(coordinator.route("W1", None, None, run_id="r1"))
    assert decision.action == "skip"
    assert fake_llm.calls == []


def test_route_calls_fast_tier_and_returns_decision(fake_llm):
    fake_llm.push("coordinator", RouteDecision(paper_id="W1", action="full", reason="ok"))
    decision = asyncio.run(coordinator.route("W1", "Title", "Abstract text.", run_id="r1"))
    assert decision.action == "full"
    assert fake_llm.calls[0]["tier"] == "fast"
    assert fake_llm.calls[0]["agent"] == "coordinator"


# --- extraction.py -------------------------------------------------------------------

def test_extract_succeeds_first_attempt(fake_llm):
    abstract = "We propose low-rank adaptation for efficient tuning."
    fake_llm.push("extraction", ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="low-rank adaptation", confidence=0.9),
    ]))
    result = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1"))
    assert result.entities[0].name == "LoRA"
    assert len(fake_llm.calls) == 1


def test_extract_retries_after_bad_evidence_span_then_succeeds(fake_llm):
    abstract = "We propose low-rank adaptation for efficient tuning."
    fake_llm.push("extraction", ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="not present at all", confidence=0.9),
    ]))
    fake_llm.push("extraction", ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="low-rank adaptation", confidence=0.9),
    ]))
    result = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    assert result.entities[0].evidence_span == "low-rank adaptation"
    assert len(fake_llm.calls) == 2
    assert "failed verification" in fake_llm.calls[1]["prompt"]
    assert "evidence_span" in fake_llm.calls[1]["prompt"]


def test_extract_exhausts_retries_raises_verification_failed(fake_llm):
    abstract = "Something unrelated."
    bad = ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="never in abstract", confidence=0.9),
    ])
    fake_llm.push("extraction", bad)
    fake_llm.push("extraction", bad)
    with pytest.raises(verifier.VerificationFailed) as excinfo:
        asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    assert excinfo.value.attempts == 2
    assert len(fake_llm.calls) == 2


def test_extract_treats_validation_error_as_schema_retry(fake_llm):
    abstract = "We propose low-rank adaptation."
    bad_exc = _validation_error(ExtractionResult, paper_id="W1", entities="not-a-list")
    fake_llm.push("extraction", bad_exc)
    fake_llm.push("extraction", ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="low-rank adaptation", confidence=0.9),
    ]))
    result = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    assert result.entities[0].name == "LoRA"
    assert len(fake_llm.calls) == 2


def test_extract_reraises_llm_error_on_last_attempt(fake_llm):
    fake_llm.push("extraction", lp.LLMTransportError("down"))
    fake_llm.push("extraction", lp.LLMTransportError("down"))
    with pytest.raises(lp.LLMTransportError):
        asyncio.run(extraction.extract("W1", "Title", "abstract", run_id="r1", max_retries=2))


# --- enrichment.py -------------------------------------------------------------------

def test_enrich_matches_best_candidate_first_attempt():
    extraction_result = ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ])
    ontology = enrichment.Ontology(
        labels=["Low-Rank Adaptation"], sources=["concept_canonical"],
        vectors=np.array([[1.0, 0.0]]),
    )
    encoder = FakeEncoder({"LoRA": [1.0, 0.0]})
    result = asyncio.run(enrichment.enrich(
        extraction_result, encoder=encoder, ontology=ontology, run_id="r1", max_retries=2,
    ))
    assert result.enriched[0].canonical == "Low-Rank Adaptation"
    assert result.enriched[0].cosine == pytest.approx(1.0)
    assert result.enriched[0].ontology_source == "concept_canonical"


def test_enrich_retries_with_richer_context_to_cross_threshold(monkeypatch):
    monkeypatch.setattr(config, "VERIFIER_COSINE_MIN", 0.9)
    extraction_result = ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ])
    ontology = enrichment.Ontology(
        labels=["Low-Rank Adaptation"], sources=["concept_canonical"],
        vectors=np.array([[1.0, 0.0]]),
    )
    # attempt 1 embeds "LoRA" -> weak match (cosine ~0.707, below the 0.9 floor);
    # attempt 2 embeds "LoRA (method)" -> exact match.
    encoder = FakeEncoder({"LoRA": [0.5, 0.5], "LoRA (method)": [1.0, 0.0]})
    result = asyncio.run(enrichment.enrich(
        extraction_result, encoder=encoder, ontology=ontology, run_id="r1", max_retries=2,
    ))
    assert result.enriched[0].cosine == pytest.approx(1.0)


def test_enrich_empty_ontology_exhausts_retries():
    extraction_result = ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ])
    ontology = enrichment.Ontology(labels=[], sources=[], vectors=np.zeros((0, 1)))
    encoder = FakeEncoder({})
    with pytest.raises(verifier.VerificationFailed):
        asyncio.run(enrichment.enrich(
            extraction_result, encoder=encoder, ontology=ontology, run_id="r1", max_retries=2,
        ))


def test_enrich_no_entities_returns_empty_result():
    extraction_result = ExtractionResult(paper_id="W1", entities=[])
    ontology = enrichment.Ontology(labels=["X"], sources=["concept_canonical"], vectors=np.array([[1.0, 0.0]]))
    result = asyncio.run(enrichment.enrich(
        extraction_result, encoder=FakeEncoder({}), ontology=ontology, run_id="r1",
    ))
    assert result.enriched == []


def test_build_ontology_unions_canonical_and_openalex_concepts(empty_conn):
    db.upsert_canonical(empty_conn, "lora", "Low-Rank Adaptation", source="manual")
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                     publication_year=2024, venue_id=None, venue="ACL")
    db.insert_concepts(empty_conn, "W1", [("Low-Rank Adaptation", 2), ("Retrieval", 1)])
    empty_conn.commit()

    encoder = FakeEncoder({"Low-Rank Adaptation": [1.0, 0.0], "Retrieval": [0.0, 1.0]})
    ontology = enrichment.build_ontology(empty_conn, encoder)

    assert "Low-Rank Adaptation" in ontology.labels
    assert "Retrieval" in ontology.labels
    # "Low-Rank Adaptation" comes from concept_canonical, not duplicated as openalex.
    assert ontology.labels.count("Low-Rank Adaptation") == 1
    idx = ontology.labels.index("Low-Rank Adaptation")
    assert ontology.sources[idx] == "concept_canonical"
    idx2 = ontology.labels.index("Retrieval")
    assert ontology.sources[idx2] == "openalex_concept"


# --- linking.py ----------------------------------------------------------------------

def test_link_writes_concepts_and_cooccurrence_skips_unmatched(empty_conn):
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                     publication_year=2024, venue_id=None, venue="ACL")
    empty_conn.commit()

    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
                       canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="concept_canonical"),
        EnrichedEntity(name="Retrieval", kind="task", evidence_span="Retrieval", confidence=0.8,
                       canonical="Retrieval", cosine=0.9, ontology_source="openalex_concept"),
        EnrichedEntity(name="junk", kind="metric", evidence_span="junk", confidence=0.1,
                       canonical="", cosine=-1.0, ontology_source="unmatched"),
    ])
    written = linking.link(empty_conn, result)
    assert written == 2

    concept_rows = empty_conn.execute("SELECT * FROM paper_concepts WHERE paper_id = 'W1'").fetchall()
    assert len(concept_rows) == 2

    cooc_rows = empty_conn.execute("SELECT * FROM concept_cooccurrence WHERE paper_id = 'W1'").fetchall()
    assert len(cooc_rows) == 1
    row = cooc_rows[0]
    assert row["concept_a"] < row["concept_b"]
    assert {row["concept_a"], row["concept_b"]} == {"Low-Rank Adaptation", "Retrieval"}


def test_link_no_cooccurrence_for_single_concept(empty_conn):
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                     publication_year=2024, venue_id=None, venue="ACL")
    empty_conn.commit()
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
                       canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="concept_canonical"),
    ])
    written = linking.link(empty_conn, result)
    assert written == 1
    assert empty_conn.execute("SELECT COUNT(*) AS c FROM concept_cooccurrence").fetchone()["c"] == 0


# --- pipeline.py (end to end, offline) ------------------------------------------------

def _seed_ontology_source(conn):
    db.upsert_canonical(conn, "lora", "Low-Rank Adaptation", source="manual")


def test_pipeline_full_happy_path(empty_conn, tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "AGENT_CONCURRENCY", 2)
    db_path = tmp_path / "pipeline.db"
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="LoRA paper",
                         abstract="We propose low-rank adaptation for tuning.",
                         publication_year=2024, venue_id=None, venue="ACL")
        _seed_ontology_source(conn)
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("coordinator", RouteDecision(paper_id="W1", action="full", reason="ok"))
    fake_llm.push("extraction", ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="low-rank adaptation", confidence=0.9),
    ]))

    encoder = FakeEncoder({"Low-Rank Adaptation": [1.0, 0.0], "LoRA": [1.0, 0.0]})
    paper = pipeline.PaperInput(paper_id="W1", title="LoRA paper",
                                 abstract="We propose low-rank adaptation for tuning.")

    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=encoder, conn_factory=lambda: db.connect(db_path), run_id="runX",
    ))

    assert summary["run_id"] == "runX"
    assert summary["items"][0]["verdict"] == "accepted"

    conn = db.connect(db_path)
    try:
        status = pipeline.get_run_status(conn, "runX")
        assert status.status == "done"
        assert status.papers_total == 1
        assert status.papers_done == 1

        item = conn.execute(
            "SELECT * FROM agent_run_items WHERE run_id = 'runX' AND paper_id = 'W1'"
        ).fetchone()
        assert item["verdict"] == "accepted"
        assert item["route"] == "full"

        concepts = conn.execute("SELECT * FROM paper_concepts WHERE paper_id = 'W1'").fetchall()
        assert len(concepts) == 1
        assert concepts[0]["canonical"] == "Low-Rank Adaptation"
    finally:
        conn.close()


def test_pipeline_skips_paper_with_no_text(tmp_path, fake_llm):
    db_path = tmp_path / "pipeline.db"
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
    finally:
        conn.close()

    paper = pipeline.PaperInput(paper_id="W1", title=None, abstract=None)
    encoder = FakeEncoder({})
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=encoder, conn_factory=lambda: db.connect(db_path), run_id="runY",
    ))
    assert summary["items"][0]["verdict"] == "skipped"
    assert fake_llm.calls == []  # coordinator short-circuits; extraction never runs


def test_pipeline_records_rejected_item_on_verification_failure(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_RETRIES", 1)
    db_path = tmp_path / "pipeline.db"
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="t", abstract="unrelated text",
                         publication_year=2024, venue_id=None, venue="ACL")
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("coordinator", RouteDecision(paper_id="W1", action="full", reason="ok"))
    fake_llm.push("extraction", ExtractionResult(paper_id="W1", entities=[
        Entity(name="LoRA", kind="method", evidence_span="never present", confidence=0.9),
    ]))

    paper = pipeline.PaperInput(paper_id="W1", title="t", abstract="unrelated text")
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=FakeEncoder({}), conn_factory=lambda: db.connect(db_path), run_id="runZ",
    ))
    assert summary["items"][0]["verdict"] == "rejected"

    conn = db.connect(db_path)
    try:
        item = conn.execute(
            "SELECT * FROM agent_run_items WHERE run_id = 'runZ' AND paper_id = 'W1'"
        ).fetchone()
        assert item["verdict"] == "rejected"
        assert item["verifier_attempts"] == 1
    finally:
        conn.close()
