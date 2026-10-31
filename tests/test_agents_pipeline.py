"""Tests for the router-worker agent pipeline -- fully offline.

`axiom.llm_providers.complete_json` is monkeypatched with a small fake queue
(`FakeLLM`) so extraction exercises its real retry/verifier logic without
any network or model call (A5 made the coordinator's routing fully
rule-based with no LLM call at all in the common path -- see
test_coordinator_* below). Enrichment gets a `FakeEncoder` with an
exact-text->vector lookup so cosine similarity is fully deterministic.
Linking/pipeline use a real (tmp-file) SQLite db via `axiom.db`, per the
existing test suite's convention (see conftest.py).
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
from axiom.agents.extraction import ExtractionOutcome
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
                        agent_type=None, paper_id=None, attempt=1, temperature=None,
                        client=None, conn=None):
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


LOW_WORDS_ABSTRACT = "Short abstract."  # well under ROUTE_TITLE_ONLY_MIN_WORDS
LONG_ABSTRACT = " ".join(["word"] * 50)  # clears the title_only floor


# --- verifier.py -------------------------------------------------------------------

def test_normalize_text_unifies_dashes_quotes_and_case():
    span_curly = "Low‐Rank “Adaptation”"
    span_ascii = "low-rank \"adaptation\""
    assert verifier.normalize_text(span_curly) == verifier.normalize_text(span_ascii)


def test_verify_extraction_passes_when_span_is_substring_of_abstract():
    result = ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA",
               confidence=0.9),
    ])
    verdict = verifier.verify_extraction(
        result, abstract="We propose LoRA, a low-rank adaptation method.", attempt=1,
    )
    assert verdict.ok
    assert verdict.errors == []


def test_verify_extraction_passes_when_span_is_substring_of_title_only():
    """A1: title is checked too, not just the abstract."""
    result = ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA: Low-Rank Adaptation",
               confidence=0.9),
    ])
    verdict = verifier.verify_extraction(
        result, title="LoRA: Low-Rank Adaptation of Large Language Models",
        abstract="Something unrelated.", attempt=1,
    )
    assert verdict.ok


def test_verify_extraction_fails_when_span_missing_from_both():
    result = ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="not in either", confidence=0.9),
    ])
    verdict = verifier.verify_extraction(
        result, title="Title", abstract="Something else entirely.", attempt=1,
    )
    assert not verdict.ok
    assert verdict.errors[0].code == "evidence_span"
    assert verdict.errors[0].field == "entities[0].evidence_span"


def test_verify_extraction_name_span_mismatch():
    """A3: the span is real text, but the claimed name has nothing to do with it."""
    abstract = "We benchmark on the SQuAD dataset for question answering."
    result = ExtractionResult(entities=[
        Entity(name="Gradient Descent", kind="method",
               evidence_span="SQuAD dataset for question answering", confidence=0.9),
    ])
    verdict = verifier.verify_extraction(result, abstract=abstract, attempt=1)
    assert not verdict.ok
    assert verdict.errors[0].code == "name_span_mismatch"


def test_verify_extraction_name_span_acronym_match():
    abstract = "We propose low-rank adaptation (LoRA) for efficient tuning."
    result = ExtractionResult(entities=[
        Entity(name="LoRA", kind="method",
               evidence_span="low-rank adaptation (LoRA)", confidence=0.9),
    ])
    verdict = verifier.verify_extraction(result, abstract=abstract, attempt=1)
    assert verdict.ok


def test_verify_enrichment_passes_new_concept_unconditionally():
    """B2: new_concept has no real ontology match to fail on."""
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="x", kind="method", evidence_span="x", confidence=0.5,
                       canonical="x", cosine=-1.0, ontology_source="new_concept"),
    ])
    verdict = verifier.verify_enrichment(result, attempt=1)
    assert verdict.ok


def test_verify_enrichment_fails_below_cosine_threshold_for_embedding_source(monkeypatch):
    monkeypatch.setattr(config, "VERIFIER_COSINE_MIN", 0.9)
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="x", kind="method", evidence_span="x", confidence=0.5,
                       canonical="Y", cosine=0.5, ontology_source="embedding"),
    ])
    verdict = verifier.verify_enrichment(result, attempt=1)
    assert not verdict.ok
    assert verdict.errors[0].code == "cosine_threshold"


def test_verify_enrichment_passes_above_threshold(monkeypatch):
    monkeypatch.setattr(config, "VERIFIER_COSINE_MIN", 0.7)
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="x", kind="method", evidence_span="x", confidence=0.5,
                       canonical="Y", cosine=0.95, ontology_source="embedding"),
    ])
    verdict = verifier.verify_enrichment(result, attempt=1)
    assert verdict.ok


def test_errors_from_validation_maps_to_schema_code():
    exc = _validation_error(RouteDecision, action="bogus", reason="x")
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


# --- coordinator.py (A5: rule-based, no LLM call in the common path) --------------

def test_route_skips_without_llm_call_when_no_text(fake_llm):
    decision = asyncio.run(coordinator.route("W1", None, None, run_id="r1"))
    assert decision.action == "skip"
    assert decision.route_source == "rule"
    assert fake_llm.calls == []


def test_route_title_only_when_abstract_missing(fake_llm):
    decision = asyncio.run(coordinator.route("W1", "A title", None, run_id="r1"))
    assert decision.action == "title_only"
    assert decision.route_source == "rule"
    assert fake_llm.calls == []


def test_route_title_only_when_abstract_under_word_floor(fake_llm):
    decision = asyncio.run(coordinator.route("W1", "A title", LOW_WORDS_ABSTRACT, run_id="r1"))
    assert decision.action == "title_only"
    assert fake_llm.calls == []


def test_route_full_when_title_and_long_abstract(fake_llm):
    decision = asyncio.run(coordinator.route("W1", "A title", LONG_ABSTRACT, run_id="r1"))
    assert decision.action == "full"
    assert fake_llm.calls == []


def test_route_via_llm_stamps_llm_route_source(fake_llm):
    fake_llm.push("coordinator", coordinator._LLMRouteJSON(action="full", reason="ok"))
    decision = asyncio.run(coordinator.route_via_llm("W1", "Title", "Abstract text.", run_id="r1"))
    assert decision.action == "full"
    assert decision.route_source == "llm"
    assert fake_llm.calls[0]["tier"] == "fast"
    assert fake_llm.calls[0]["agent"] == "coordinator"


# --- extraction.py -----------------------------------------------------------------

def test_extract_succeeds_first_attempt(fake_llm):
    abstract = "We propose LoRA, a low-rank adaptation method for efficient tuning."
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ]))
    outcome = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1"))
    assert outcome.paper_id == "W1"
    assert outcome.attempts == 1
    assert outcome.entities[0].name == "LoRA"
    assert len(fake_llm.calls) == 1


def test_extract_retries_after_bad_evidence_span_then_succeeds(fake_llm):
    abstract = "We propose LoRA, a low-rank adaptation method for efficient tuning."
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="not present at all", confidence=0.9),
    ]))
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ]))
    outcome = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    assert outcome.entities[0].evidence_span == "LoRA"
    assert outcome.attempts == 2
    assert len(fake_llm.calls) == 2
    assert "failed verification" in fake_llm.calls[1]["prompt"]
    assert "evidence_span" in fake_llm.calls[1]["prompt"]


def test_extract_accumulates_passing_entities_across_attempts(fake_llm):
    """A2: a mixed response (one good entity, one bad) no longer drags the
    good one down -- the good entity from attempt 1 survives into the final
    result even though attempt 2 only fixes the bad one."""
    abstract = "We propose LoRA and evaluate on GLUE."
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
        Entity(name="GLUE", kind="dataset", evidence_span="not in abstract", confidence=0.9),
    ]))
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="GLUE", kind="dataset", evidence_span="GLUE", confidence=0.9),
    ]))
    outcome = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    names = {e.name for e in outcome.entities}
    assert names == {"LoRA", "GLUE"}
    assert outcome.attempts == 2


def test_extract_exhausts_retries_raises_verification_failed_with_empty_after_retry_marker(fake_llm):
    abstract = "Something unrelated."
    bad = ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="never in abstract", confidence=0.9),
    ])
    fake_llm.push("extraction", bad)
    fake_llm.push("extraction", bad)
    with pytest.raises(verifier.VerificationFailed) as excinfo:
        asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    assert excinfo.value.attempts == 2
    assert len(fake_llm.calls) == 2
    assert any(e.code == "empty_after_retry" for e in excinfo.value.errors)


def test_extract_returns_empty_without_error_when_nothing_was_ever_proposed(fake_llm):
    """A2: a paper that genuinely has nothing to extract is accepted with
    zero entities -- distinct from the empty_after_retry rejection above."""
    fake_llm.push("extraction", ExtractionResult(entities=[]))
    outcome = asyncio.run(extraction.extract("W1", "Title", "abstract", run_id="r1", max_retries=2))
    assert outcome.entities == []
    assert outcome.attempts == 1


def test_extract_treats_validation_error_as_schema_retry(fake_llm):
    abstract = "We propose LoRA."
    bad_exc = _validation_error(ExtractionResult, entities="not-a-list")
    fake_llm.push("extraction", bad_exc)
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ]))
    outcome = asyncio.run(extraction.extract("W1", "Title", abstract, run_id="r1", max_retries=2))
    assert outcome.entities[0].name == "LoRA"
    assert len(fake_llm.calls) == 2


def test_extract_reraises_llm_error_on_last_attempt(fake_llm):
    fake_llm.push("extraction", lp.LLMTransportError("down"))
    fake_llm.push("extraction", lp.LLMTransportError("down"))
    with pytest.raises(lp.LLMTransportError):
        asyncio.run(extraction.extract("W1", "Title", "abstract", run_id="r1", max_retries=2))


def test_extract_title_only_mode_ignores_abstract_and_checks_title(fake_llm):
    """A5: title_only mode verifies against the title, never the abstract --
    even if a (too-thin, per the route decision) abstract string is still
    technically present on the input."""
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA paper", confidence=0.9),
    ]))
    outcome = asyncio.run(extraction.extract(
        "W1", "LoRA paper", "Unrelated abstract text that LoRA paper never mentions.",
        run_id="r1", mode="title_only",
    ))
    assert outcome.entities[0].name == "LoRA"
    # the prompt sent to the model must not even mention "abstract"
    assert "abstract" not in fake_llm.calls[0]["prompt"].lower()


def test_extract_title_only_mode_rejects_span_only_found_in_abstract(fake_llm):
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="GLUE", kind="dataset", evidence_span="evaluate on GLUE", confidence=0.9),
    ]))
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="GLUE", kind="dataset", evidence_span="evaluate on GLUE", confidence=0.9),
    ]))
    # "evaluate on GLUE" only appears in the abstract, not the title -- in
    # title_only mode that's not valid evidence, so the entity never
    # survives any attempt and the whole call is rejected (A2: the one
    # entity ever proposed failed every attempt).
    with pytest.raises(verifier.VerificationFailed) as excinfo:
        asyncio.run(extraction.extract(
            "W1", "A short title", "We evaluate on GLUE extensively.",
            run_id="r1", mode="title_only", max_retries=2,
        ))
    assert any(e.code == "evidence_span" for e in excinfo.value.errors)


# --- enrichment.py (B1 cascade) ----------------------------------------------------

def _outcome(*entities: Entity, paper_id: str = "W1") -> ExtractionOutcome:
    return ExtractionOutcome(paper_id=paper_id, entities=list(entities), attempts=1)


def test_enrich_exact_lexical_match_needs_no_embedding_call():
    entity = Entity(name="low-rank adaptation", kind="method", evidence_span="x", confidence=0.9)
    ontology = enrichment.Ontology(
        labels=["Low-Rank Adaptation"], sources=["concept_canonical"],
        vectors=np.array([[1.0, 0.0]]),
    )

    def _boom(texts, **kw):
        raise AssertionError("lexical match should not call the embedding encoder")

    encoder = FakeEncoder({})
    encoder.encode = _boom
    outcome = asyncio.run(enrichment.enrich(_outcome(entity), encoder=encoder, ontology=ontology, run_id="r1"))
    enriched = outcome.result.enriched[0]
    assert enriched.ontology_source == "lexical"
    assert enriched.canonical == "Low-Rank Adaptation"
    assert enriched.cosine == pytest.approx(1.0)
    assert outcome.attempts == 1


def test_enrich_embedding_match_above_cosine_and_margin():
    entity = Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9)
    ontology = enrichment.Ontology(
        labels=["Low-Rank Adaptation", "Translation"], sources=["concept_canonical"] * 2,
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    encoder = FakeEncoder({"LoRA": [1.0, 0.0]})
    outcome = asyncio.run(enrichment.enrich(_outcome(entity), encoder=encoder, ontology=ontology, run_id="r1"))
    enriched = outcome.result.enriched[0]
    assert enriched.ontology_source == "embedding"
    assert enriched.canonical == "Low-Rank Adaptation"
    assert enriched.cosine == pytest.approx(1.0)


def test_enrich_low_margin_falls_through_to_new_concept(monkeypatch):
    """Two near-identical candidates (small top1-top2 margin) must NOT be
    accepted even if both clear the cosine floor -- B1's confidence margin."""
    monkeypatch.setattr(config, "VERIFIER_COSINE_MIN", 0.5)
    monkeypatch.setattr(config, "ENRICHMENT_MARGIN_DELTA", 0.2)
    entity = Entity(name="Widget", kind="method", evidence_span="Widget", confidence=0.9)
    ontology = enrichment.Ontology(
        labels=["Candidate A", "Candidate B"], sources=["concept_canonical"] * 2,
        vectors=np.array([[1.0, 0.0], [0.99, 0.01411]]),  # nearly identical cosine to the query
    )
    encoder = FakeEncoder({
        "Widget": [1.0, 0.0], "Widget (method)": [1.0, 0.0], "Widget: Widget": [1.0, 0.0],
    })
    outcome = asyncio.run(enrichment.enrich(_outcome(entity), encoder=encoder, ontology=ontology, run_id="r1"))
    enriched = outcome.result.enriched[0]
    assert enriched.ontology_source == "new_concept"


def test_enrich_no_match_becomes_new_concept_and_is_kept():
    """B1's behavior change: an unmatched entity is KEPT, not dropped."""
    entity = Entity(name="Brand New Method", kind="method", evidence_span="x", confidence=0.9)
    ontology = enrichment.Ontology(
        labels=["Translation"], sources=["concept_canonical"], vectors=np.array([[0.0, 1.0]]),
    )
    encoder = FakeEncoder({}, default=[1.0, 0.0])  # orthogonal to the only candidate -> cosine 0
    outcome = asyncio.run(enrichment.enrich(_outcome(entity), encoder=encoder, ontology=ontology, run_id="r1"))
    enriched = outcome.result.enriched[0]
    assert enriched.ontology_source == "new_concept"
    assert enriched.canonical == verifier.normalize_text("Brand New Method")


def test_enrich_embedding_tries_richer_context_before_falling_back():
    entity = Entity(name="LoRA", kind="method", evidence_span="low-rank adaptation", confidence=0.9)
    ontology = enrichment.Ontology(
        labels=["Low-Rank Adaptation"], sources=["concept_canonical"], vectors=np.array([[1.0, 0.0]]),
    )
    # bare name embeds orthogonally (no match); name+evidence_span embeds exactly.
    encoder = FakeEncoder({
        "LoRA": [0.0, 1.0],
        "LoRA (method)": [0.0, 1.0],
        "LoRA: low-rank adaptation": [1.0, 0.0],
    })
    outcome = asyncio.run(enrichment.enrich(_outcome(entity), encoder=encoder, ontology=ontology, run_id="r1"))
    enriched = outcome.result.enriched[0]
    assert enriched.ontology_source == "embedding"
    assert enriched.cosine == pytest.approx(1.0)


def test_enrich_no_entities_returns_empty_result():
    outcome = asyncio.run(enrichment.enrich(
        _outcome(), encoder=FakeEncoder({}),
        ontology=enrichment.Ontology(labels=["X"], sources=["concept_canonical"], vectors=np.array([[1.0, 0.0]])),
        run_id="r1",
    ))
    assert outcome.result.enriched == []
    assert outcome.attempts == 1


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


# --- linking.py (B2: nothing is filtered anymore) ---------------------------------

def test_link_writes_every_entity_including_new_concept(empty_conn):
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                     publication_year=2024, venue_id=None, venue="ACL")
    empty_conn.commit()

    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
                       canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="embedding"),
        EnrichedEntity(name="Retrieval", kind="task", evidence_span="Retrieval", confidence=0.8,
                       canonical="Retrieval", cosine=1.0, ontology_source="lexical"),
        EnrichedEntity(name="junk method", kind="metric", evidence_span="junk method", confidence=0.1,
                       canonical="junk method", cosine=-1.0, ontology_source="new_concept"),
    ])
    written = linking.link(empty_conn, result)
    assert written == 3

    concept_rows = empty_conn.execute("SELECT * FROM paper_concepts WHERE paper_id = 'W1'").fetchall()
    assert len(concept_rows) == 3

    cooc_rows = empty_conn.execute("SELECT * FROM concept_cooccurrence WHERE paper_id = 'W1'").fetchall()
    assert len(cooc_rows) == 3  # 3 canonicals -> C(3,2) pairs


def test_link_no_cooccurrence_for_single_concept(empty_conn):
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                     publication_year=2024, venue_id=None, venue="ACL")
    empty_conn.commit()
    result = EnrichmentResult(paper_id="W1", enriched=[
        EnrichedEntity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
                       canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="embedding"),
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
    # A5: "full" route needs the abstract to clear ROUTE_TITLE_ONLY_MIN_WORDS
    # (40) words.
    abstract = "LoRA " + "word " * 44
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="LoRA paper",
                         abstract=abstract,
                         publication_year=2024, venue_id=None, venue="ACL")
        _seed_ontology_source(conn)
        conn.commit()
    finally:
        conn.close()

    # route is now rule-based (A5) -- the abstract clears the word floor, so
    # no "coordinator" fake response is needed at all.
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ]))

    phrase_encoder = FakeEncoder({"Low-Rank Adaptation": [1.0, 0.0], "LoRA": [1.0, 0.0]})
    paper = pipeline.PaperInput(paper_id="W1", title="LoRA paper", abstract=abstract)

    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=object(), phrase_encoder=phrase_encoder,
        conn_factory=lambda: db.connect(db_path), run_id="runX",
    ))

    assert summary["run_id"] == "runX"
    assert summary["items"][0]["verdict"] == "accepted"
    assert "coordinator" not in fake_llm.queues or not fake_llm.queues.get("coordinator")

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
        assert item["verifier_attempts"] == 1

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
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=object(), phrase_encoder=FakeEncoder({}),
        conn_factory=lambda: db.connect(db_path), run_id="runY",
    ))
    assert summary["items"][0]["verdict"] == "skipped"
    assert fake_llm.calls == []  # no LLM call at all: rule-based route, extraction never runs


def test_pipeline_routes_title_only_and_extracts_from_title(tmp_path, fake_llm):
    db_path = tmp_path / "pipeline.db"
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="LoRA Paper", abstract=LOW_WORDS_ABSTRACT,
                         publication_year=2024, venue_id=None, venue="ACL")
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA Paper", confidence=0.9),
    ]))

    paper = pipeline.PaperInput(paper_id="W1", title="LoRA Paper", abstract=LOW_WORDS_ABSTRACT)
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=object(), phrase_encoder=FakeEncoder({"LoRA": [1.0, 0.0]}),
        conn_factory=lambda: db.connect(db_path), run_id="runTO",
    ))
    assert summary["items"][0]["route"] == "title_only"
    assert summary["items"][0]["verdict"] == "accepted"
    assert "abstract" not in fake_llm.calls[0]["prompt"].lower()


def test_pipeline_records_rejected_item_on_verification_failure(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_RETRIES", 1)
    db_path = tmp_path / "pipeline.db"
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="t", abstract=LONG_ABSTRACT,
                         publication_year=2024, venue_id=None, venue="ACL")
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="never present", confidence=0.9),
    ]))

    paper = pipeline.PaperInput(paper_id="W1", title="t", abstract=LONG_ABSTRACT)
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=object(), phrase_encoder=FakeEncoder({}),
        conn_factory=lambda: db.connect(db_path), run_id="runZ",
    ))
    assert summary["items"][0]["verdict"] == "rejected"
    assert "empty_after_retry" in summary["items"][0]["error"]

    conn = db.connect(db_path)
    try:
        item = conn.execute(
            "SELECT * FROM agent_run_items WHERE run_id = 'runZ' AND paper_id = 'W1'"
        ).fetchone()
        assert item["verdict"] == "rejected"
        assert item["verifier_attempts"] == 1
    finally:
        conn.close()


def test_pipeline_accepts_paper_with_zero_genuine_entities(tmp_path, fake_llm):
    """A2: a paper that legitimately has nothing to extract is accepted
    (not rejected) with zero concepts written."""
    db_path = tmp_path / "pipeline.db"
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="t", abstract=LONG_ABSTRACT,
                         publication_year=2024, venue_id=None, venue="ACL")
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("extraction", ExtractionResult(entities=[]))

    paper = pipeline.PaperInput(paper_id="W1", title="t", abstract=LONG_ABSTRACT)
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=object(), phrase_encoder=FakeEncoder({}),
        conn_factory=lambda: db.connect(db_path), run_id="runEmpty",
    ))
    assert summary["items"][0]["verdict"] == "accepted"
    assert summary["items"][0]["concepts_written"] == 0


def test_pipeline_records_real_attempt_count_on_accepted_path(tmp_path, fake_llm, monkeypatch):
    """D3: verifier_attempts on the accepted row reflects real retries, not
    a hardcoded 1."""
    monkeypatch.setattr(config, "AGENT_MAX_RETRIES", 2)
    db_path = tmp_path / "pipeline.db"
    abstract = "LoRA " + "word " * 44
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="t", abstract=abstract,
                         publication_year=2024, venue_id=None, venue="ACL")
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="not present yet", confidence=0.9),
    ]))
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9),
    ]))

    paper = pipeline.PaperInput(paper_id="W1", title="t", abstract=abstract)
    summary = asyncio.run(pipeline.run_pipeline(
        [paper], encoder=object(), phrase_encoder=FakeEncoder({}),
        conn_factory=lambda: db.connect(db_path), run_id="runAttempts",
    ))
    assert summary["items"][0]["verdict"] == "accepted"

    conn = db.connect(db_path)
    try:
        item = conn.execute(
            "SELECT * FROM agent_run_items WHERE run_id = 'runAttempts' AND paper_id = 'W1'"
        ).fetchone()
        assert item["verifier_attempts"] == 2
    finally:
        conn.close()


def test_pipeline_paper_id_always_comes_from_input_not_model(tmp_path, fake_llm):
    """A4: even though ExtractionResult has no paper_id field for a model to
    misreport, confirm end-to-end that every write lands under the caller's
    own paper_id for a multi-paper run (nothing could plausibly cross wires)."""
    db_path = tmp_path / "pipeline.db"
    abstract_w1 = "MethodOne " + "word " * 44
    abstract_w2 = "MethodTwo " + "word " * 44
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.insert_paper(conn, openalex_id="W1", title="t", abstract=abstract_w1,
                         publication_year=2024, venue_id=None, venue="ACL")
        db.insert_paper(conn, openalex_id="W2", title="t", abstract=abstract_w2,
                         publication_year=2024, venue_id=None, venue="ACL")
        conn.commit()
    finally:
        conn.close()

    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="MethodOne", kind="method", evidence_span="MethodOne", confidence=0.9),
    ]))
    fake_llm.push("extraction", ExtractionResult(entities=[
        Entity(name="MethodTwo", kind="method", evidence_span="MethodTwo", confidence=0.9),
    ]))

    papers = [
        pipeline.PaperInput(paper_id="W1", title="t", abstract=abstract_w1),
        pipeline.PaperInput(paper_id="W2", title="t", abstract=abstract_w2),
    ]
    asyncio.run(pipeline.run_pipeline(
        papers, encoder=object(), phrase_encoder=FakeEncoder({}),
        conn_factory=lambda: db.connect(db_path), run_id="runMulti",
    ))

    conn = db.connect(db_path)
    try:
        rows = {r["paper_id"]: r["name"] for r in conn.execute(
            "SELECT paper_id, name FROM paper_concepts WHERE run_id = 'runMulti'"
        ).fetchall()}
        assert rows == {"W1": "MethodOne", "W2": "MethodTwo"}
    finally:
        conn.close()
