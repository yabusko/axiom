"""Real-Redis integration test for axiom/workers.py (AXIOM_DISTRIBUTED_TASK.md):
10 papers end-to-end through all 4 roles (coordinator -> extraction ->
enrichment -> linking), with the one LLM call (extraction) stubbed via the
same `FakeLLM`/monkeypatch pattern tests/test_agents_pipeline.py already
uses for the in-process pipeline -- so this proves the Redis plumbing
(serialization, routing, ack, run-status bookkeeping) for real, without any
network/LLM spend.

Skipped unless REDIS_URL is set AND a real `redis` package is importable --
this is intentionally NOT part of the default `pytest tests/` run (CI has no
Redis service); run it locally with Redis up:
    docker compose up -d redis
    REDIS_URL=redis://localhost:6379/0 pytest tests/test_workers_integration.py -v -m redis
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

redis_url = os.environ.get("REDIS_URL")
redis_asyncio = pytest.importorskip("redis.asyncio") if redis_url else None

pytestmark = pytest.mark.redis
if not redis_url:
    pytestmark = pytest.mark.skip(reason="REDIS_URL not set")

if redis_url:
    from axiom import config, db, workers
    from axiom.agents import enrichment as enrichment_mod
    from axiom.agents.contracts import Entity, ExtractionResult
    from axiom import llm_providers as lp

    class FakeLLM:
        async def __call__(self, tier, system, prompt, model_cls, *, run_id, agent,
                            agent_type=None, paper_id=None, attempt=1, temperature=None,
                            client=None, conn=None):
            return ExtractionResult(entities=[
                Entity(name="retrieval", kind="method", evidence_span="retrieval",
                       confidence=0.9),
            ])

    class FakePhraseEncoder:
        def encode(self, texts, abstracts=None, batch_size=16, max_length=512):
            return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "axiom.db"
    conn = db.connect(db_path)
    db.init_db(conn)
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return db_path


async def _flush_streams(client):
    for stream in (workers.STREAM_ROUTE, workers.STREAM_EXTRACT, workers.STREAM_ENRICH,
                   workers.STREAM_LINK, workers.STREAM_DEAD):
        await client.delete(stream)


def test_ten_papers_end_to_end_through_all_four_roles(isolated_db, monkeypatch):
    monkeypatch.setattr(lp, "complete_json", FakeLLM())
    # This test drives run_consumer_loop() directly (not the CLI's
    # _main_async, which builds these at process startup) -- set the same
    # per-worker-process globals handle_enrichment()/handle_linking() read.
    workers._worker_phrase_encoder = FakePhraseEncoder()
    workers._worker_ontology = enrichment_mod.Ontology([], [], np.zeros((0, 2)))

    run_id = "integration-test-run"

    async def run():
        client = redis_asyncio.from_url(redis_url, decode_responses=True)
        try:
            await _flush_streams(client)

            conn = db.connect(isolated_db)
            try:
                db.create_agent_run(conn, run_id, papers_total=10, model_fast="fast",
                                     model_strong="strong", started_at="2026-11-01T00:00:00Z")
            finally:
                conn.close()

            from axiom.agents.pipeline import PaperInput
            # 40+ words so coordinator.route() picks "full" (not "title_only",
            # config.ROUTE_TITLE_ONLY_MIN_WORDS), and it literally contains
            # "retrieval" so the stubbed entity's evidence_span verifies.
            abstract = (
                "This paper studies retrieval methods for open domain question "
                "answering over a large corpus of documents using dense and "
                "sparse retrieval combined with a reader model trained end to "
                "end on real data with careful evaluation across several "
                "benchmarks and a thorough ablation of every design choice made."
            )
            assert len(abstract.split()) >= 40  # coordinator.route() needs this to pick "full"
            inputs = [
                PaperInput(paper_id=f"P{i}", title=f"Paper {i}", abstract=abstract)
                for i in range(10)
            ]
            # paper_concepts.paper_id has a real FK -> papers.openalex_id
            # (linking writes there) -- a synthetic paper_id with no row in
            # `papers` fails that constraint, same as it would for the
            # in-process pipeline against a paper never actually ingested.
            conn = db.connect(isolated_db)
            try:
                for p in inputs:
                    db.insert_paper(conn, openalex_id=p.paper_id, title=p.title, abstract=p.abstract,
                                     publication_year=2026, venue_id=None, venue=None)
                conn.commit()
            finally:
                conn.close()
            await workers.enqueue_run(client, run_id, inputs)

            workers._worker_link_conn = db.connect(isolated_db)
            try:
                tasks = [
                    asyncio.create_task(workers.run_consumer_loop(
                        client, role=role, consumer_name=f"{role}-test", max_iterations=20,
                    ))
                    for role in workers.ROLES
                ]
                await asyncio.gather(*tasks)
                # a couple more short passes so messages that chained through
                # 2+ streams within one iteration window all settle
                for _ in range(3):
                    tasks = [
                        asyncio.create_task(workers.run_consumer_loop(
                            client, role=role, consumer_name=f"{role}-test", max_iterations=5,
                        ))
                        for role in workers.ROLES
                    ]
                    await asyncio.gather(*tasks)
            finally:
                workers._worker_link_conn.close()
                workers._worker_link_conn = None
        finally:
            await client.aclose()

    asyncio.run(run())

    conn = db.connect(isolated_db)
    try:
        run_row = db.get_agent_run(conn, run_id)
        items = conn.execute(
            "SELECT paper_id, verdict FROM agent_run_items WHERE run_id = ?", (run_id,)
        ).fetchall()
    finally:
        conn.close()

    assert run_row["papers_done"] == 10
    assert len(items) == 10
    assert all(r["verdict"] == "accepted" for r in items)
