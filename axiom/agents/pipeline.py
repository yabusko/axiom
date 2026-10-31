"""Per-paper state machine + run-level fan-out (R1/R5).

`run_pipeline()` is the whole thing: coordinator -> extraction -> enrichment
-> linking per paper, concurrently via `asyncio.gather` under
`asyncio.Semaphore(config.AGENT_CONCURRENCY)`; `agent_runs`/`agent_run_items`
rows track status transitions, verifier attempts, and verdicts. `synthesis`
(cluster labels + hypothesis pitches) runs once per run, after every paper in
it has finished linking -- only when the caller supplies cluster/gap input,
since that data comes from the corpus-level concept/citation graphs (a
sibling workstream's `axiom/concept_graph.py` for clusters, the existing
`axiom/gaps.py` for gap candidates), not from this run's papers alone. See
the agent-implementer patch note for how `api/main.py`'s `POST /agents/run`
should supply them.
"""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import networkx as nx

from axiom import config, db
from axiom.agents import coordinator, enrichment, extraction, linking, synthesis, verifier
from axiom.agents.contracts import RunStatusOut
from axiom.agents.enrichment import Ontology
from axiom.gaps import GapCandidate

ConnFactory = Callable[[], sqlite3.Connection]


@dataclass
class PaperInput:
    paper_id: str
    title: str | None
    abstract: str | None


def _default_conn_factory(db_path) -> ConnFactory:
    return lambda: db.connect(db_path)




async def _process_paper(
    paper: PaperInput,
    *,
    run_id: str,
    conn_factory: ConnFactory,
    encoder: Any,
    ontology: Ontology,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        route_action: str | None = None
        try:
            decision = await coordinator.route(
                paper.paper_id, paper.title, paper.abstract, run_id=run_id,
            )
            route_action = decision.action
        except Exception as exc:  # coordinator failure: nothing else can run for this paper
            conn = conn_factory()
            try:
                db.upsert_agent_run_item(conn, run_id, paper.paper_id, route=None,
                                  verifier_attempts=0, verdict="error", error=str(exc))
            finally:
                conn.close()
            return {"paper_id": paper.paper_id, "route": None, "verdict": "error", "error": str(exc)}

        if route_action == "skip":
            conn = conn_factory()
            try:
                db.upsert_agent_run_item(conn, run_id, paper.paper_id, route="skip",
                                  verifier_attempts=0, verdict="skipped", error=None)
            finally:
                conn.close()
            return {"paper_id": paper.paper_id, "route": "skip", "verdict": "skipped", "error": None}

        try:
            extraction_result = await extraction.extract(
                paper.paper_id, paper.title, paper.abstract, run_id=run_id,
            )
        except verifier.VerificationFailed as exc:
            conn = conn_factory()
            try:
                db.upsert_agent_run_item(conn, run_id, paper.paper_id, route=route_action,
                                  verifier_attempts=exc.attempts, verdict="rejected", error=str(exc))
            finally:
                conn.close()
            return {"paper_id": paper.paper_id, "route": route_action, "verdict": "rejected", "error": str(exc)}
        except Exception as exc:
            conn = conn_factory()
            try:
                db.upsert_agent_run_item(conn, run_id, paper.paper_id, route=route_action,
                                  verifier_attempts=0, verdict="error", error=str(exc))
            finally:
                conn.close()
            return {"paper_id": paper.paper_id, "route": route_action, "verdict": "error", "error": str(exc)}

        try:
            enrichment_result = await enrichment.enrich(
                extraction_result, encoder=encoder, ontology=ontology, run_id=run_id,
            )
        except verifier.VerificationFailed as exc:
            conn = conn_factory()
            try:
                db.upsert_agent_run_item(conn, run_id, paper.paper_id, route=route_action,
                                  verifier_attempts=exc.attempts, verdict="rejected", error=str(exc))
            finally:
                conn.close()
            return {"paper_id": paper.paper_id, "route": route_action, "verdict": "rejected", "error": str(exc)}
        except Exception as exc:
            conn = conn_factory()
            try:
                db.upsert_agent_run_item(conn, run_id, paper.paper_id, route=route_action,
                                  verifier_attempts=0, verdict="error", error=str(exc))
            finally:
                conn.close()
            return {"paper_id": paper.paper_id, "route": route_action, "verdict": "error", "error": str(exc)}

        conn = conn_factory()
        try:
            written = linking.link(conn, enrichment_result, run_id=run_id)
            db.upsert_agent_run_item(conn, run_id, paper.paper_id, route=route_action,
                              verifier_attempts=1, verdict="accepted", error=None)
        finally:
            conn.close()
        return {
            "paper_id": paper.paper_id, "route": route_action, "verdict": "accepted",
            "error": None, "concepts_written": written,
        }


async def run_pipeline(
    papers: list[PaperInput],
    *,
    encoder: Any,
    db_path=None,
    conn_factory: ConnFactory | None = None,
    run_id: str | None = None,
    model_fast: str | None = None,
    model_strong: str | None = None,
    clusters: list[dict] | None = None,
    gaps: list[tuple[GapCandidate, nx.DiGraph]] | None = None,
    trend_context: list[str] | None = None,
) -> dict:
    """Run the full agentic pipeline over `papers`. `encoder` is a
    `Specter2Encoder`-shaped object (or a test double) -- constructed once by
    the caller, never by this function, since loading real SPECTER2 weights
    per run would be far too slow for a per-paper loop.

    `model_fast`/`model_strong` default to `config.MODEL_FAST`/`MODEL_STRONG`,
    read at call time -- they're only recorded on the `agent_runs` row as a
    label, but kept consistent with the same pattern used elsewhere in this
    module (see extraction.py's `extract()`).

    `clusters`/`gaps` are optional synthesis inputs (see module docstring);
    when omitted, synthesis is skipped and the run still completes normally.
    """
    run_id = run_id or uuid.uuid4().hex
    conn_factory = conn_factory or _default_conn_factory(db_path)
    model_fast = model_fast if model_fast is not None else config.MODEL_FAST
    model_strong = model_strong if model_strong is not None else config.MODEL_STRONG

    started_at = datetime.now(timezone.utc).isoformat()
    conn = conn_factory()
    try:
        db.create_agent_run(
            conn, run_id, papers_total=len(papers),
            model_fast=model_fast, model_strong=model_strong, started_at=started_at,
        )
    finally:
        conn.close()

    def _build_ontology() -> Ontology:
        # sqlite3 connections are thread-affine (`check_same_thread=True` by
        # default in `axiom.db.connect`) -- open it INSIDE the worker thread
        # asyncio.to_thread runs this in, not before, or every query raises
        # "SQLite objects created in a thread can only be used in that same thread".
        conn = conn_factory()
        try:
            return enrichment.build_ontology(conn, encoder)
        finally:
            conn.close()

    ontology = await asyncio.to_thread(_build_ontology)

    semaphore = asyncio.Semaphore(config.AGENT_CONCURRENCY)
    items = await asyncio.gather(*[
        _process_paper(
            paper, run_id=run_id, conn_factory=conn_factory,
            encoder=encoder, ontology=ontology, semaphore=semaphore,
        )
        for paper in papers
    ])

    cluster_labels: list[dict] = []
    if clusters:
        cluster_labels = await synthesis.label_clusters(clusters, run_id=run_id)

    hypotheses: list[dict] = []
    if gaps:
        conn = conn_factory()
        try:
            for gap, g in gaps:
                pitch = await synthesis.generate_hypothesis(
                    gap, g, run_id=run_id, trend_context=trend_context,
                )
                review_id = synthesis.queue_hypothesis(conn, gap, pitch)
                hypotheses.append({"review_id": review_id, "title": pitch.title, "claim": pitch.claim})
        finally:
            conn.close()

    finished_at = datetime.now(timezone.utc).isoformat()
    conn = conn_factory()
    try:
        db.finish_agent_run(conn, run_id, papers_done=len(items), finished_at=finished_at)
    finally:
        conn.close()

    return {
        "run_id": run_id,
        "items": items,
        "cluster_labels": cluster_labels,
        "hypotheses": hypotheses,
    }


def get_run_status(conn: sqlite3.Connection, run_id: str) -> RunStatusOut | None:
    """For `GET /agents/runs/{id}` -- see the patch note."""
    row = db.get_agent_run(conn, run_id)
    if row is None:
        return None
    return RunStatusOut(**dict(row))
