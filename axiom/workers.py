"""Distributed agent workers (AXIOM_DISTRIBUTED_TASK.md): each agent role as
an independent process, talking only through Redis Streams -- so the
pipeline scales horizontally (`docker compose up --scale worker-extraction=3`)
instead of being one process's `asyncio.gather` fan-out
(`axiom/agents/pipeline.py`, unchanged, still the default).

Four streams, one per stage: axiom:route -> axiom:extract -> axiom:enrich ->
axiom:link, plus axiom:dead for messages that exhausted `STREAM_MAX_DELIVERIES`.
Message body = the existing Pydantic contracts (axiom/agents/contracts.py)
serialized to JSON, wrapped in a small envelope carrying `run_id`, `paper_id`,
and `attempt` (the retry counter -- see `run_consumer_loop`'s docstring for
why this lives in the message rather than relying on Redis's own PEL delivery
count).

**Single-writer SQLite (OD43 below).** Only the `linking` role opens a sqlite
connection for `agent_runs`/`agent_run_items`/`paper_concepts`/
`concept_cooccurrence` -- the real corpus-write tables. `coordinator` and
`enrichment` are pure and stateless (no DB at all -- neither ever calls an
LLM: `coordinator.route()` is rule-based, B1 moved entity matching off
embeddings-via-LLM onto a local phrase-model cascade). `extraction` is the
one role that calls an LLM (`axiom/agents/extraction.py`'s `extract()`), so
it opens its own connection ONLY for `llm_providers.complete_json`'s existing
durable budget ledger (`llm_spend`) -- OD43 explains why, deliberately
diverging from "linking is the one durable writer" for that one table.
Every terminal outcome (skip / rejected / error / accepted) is published
straight to `axiom:link` -- the one place that ever calls
`db.upsert_agent_run_item`/`db.increment_agent_run_papers_done` -- instead of
each stage writing its own failures directly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom.agents import coordinator, enrichment, extraction, linking, verifier
from axiom.agents.contracts import EnrichedEntity, EnrichmentResult, Entity

log = logging.getLogger("axiom.workers")

STREAM_ROUTE = "axiom:route"
STREAM_EXTRACT = "axiom:extract"
STREAM_ENRICH = "axiom:enrich"
STREAM_LINK = "axiom:link"
STREAM_DEAD = "axiom:dead"
FIELD = "data"          # the one Redis Streams field each message body lives in
GROUP = "workers"        # one consumer group per stream, shared name is fine (different streams)

ROLES = ("coordinator", "extraction", "enrichment", "linking")


# ---------------------------------------------------------------------------
# Envelope: {run_id, paper_id, attempt, payload} -- payload is stage-specific,
# built from/into the existing Pydantic contracts via .model_dump()/(**kwargs).
# ---------------------------------------------------------------------------
def make_envelope(run_id: str, paper_id: str, attempt: int, payload: dict) -> str:
    return json.dumps({"run_id": run_id, "paper_id": paper_id, "attempt": attempt, "payload": payload})


def parse_envelope(raw: str) -> dict:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# The narrow Redis Streams surface every role uses -- 5 calls, named to match
# redis.asyncio.Redis exactly, so a real client or a tiny in-memory fake
# (tests/test_workers.py) both satisfy this without any adapter layer.
# ---------------------------------------------------------------------------
class StreamClient(Protocol):
    async def xgroup_create(self, name: str, groupname: str, id: str = "0", mkstream: bool = False) -> Any: ...
    async def xadd(self, name: str, fields: dict) -> Any: ...
    async def xreadgroup(self, groupname: str, consumername: str, streams: dict,
                          count: int | None = None, block: int | None = None) -> Any: ...
    async def xack(self, name: str, groupname: str, *ids: str) -> Any: ...
    async def xautoclaim(self, name: str, groupname: str, consumername: str,
                          min_idle_time: int, start_id: str, count: int | None = None) -> Any: ...


async def ensure_group(client: StreamClient, stream: str, group: str = GROUP) -> None:
    try:
        await client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


# ---------------------------------------------------------------------------
# Per-role handlers. Each takes the parsed envelope's `payload` dict plus
# `run_id`/`paper_id`, returns `(next_stream, next_payload)` to publish, or
# `(None, None)` for a terminal outcome already recorded by this handler's
# side effect (linking only -- see module docstring).
# ---------------------------------------------------------------------------
async def handle_coordinator(run_id: str, paper_id: str, payload: dict) -> tuple[str | None, dict | None]:
    decision = await coordinator.route(paper_id, payload.get("title"), payload.get("abstract"), run_id=run_id)
    if decision.action == "skip":
        return STREAM_LINK, {"terminal": {"verdict": "skipped", "route": "skip", "error": None}}
    return STREAM_EXTRACT, {
        "title": payload.get("title"), "abstract": payload.get("abstract"), "route": decision.action,
    }


def _make_budget_conn():
    """extraction/enrichment's own connection, scoped ONLY to the llm_spend
    ledger via llm_providers.complete_json's `conn` param -- OD43."""
    return db.connect(config.DB_PATH)


async def handle_extraction(run_id: str, paper_id: str, payload: dict) -> tuple[str | None, dict | None]:
    route = payload.get("route", "full")
    budget_conn = _make_budget_conn()
    try:
        outcome = await extraction.extract(
            paper_id, payload.get("title"), payload.get("abstract"), run_id=run_id,
            mode=route, conn=budget_conn,
        )
    except verifier.VerificationFailed as exc:
        return STREAM_LINK, {"terminal": {"verdict": "rejected", "route": route, "error": str(exc)}}
    except Exception as exc:
        return STREAM_LINK, {"terminal": {"verdict": "error", "route": route, "error": str(exc)}}
    finally:
        budget_conn.close()
    return STREAM_ENRICH, {
        "route": route,
        "entities": [e.model_dump() for e in outcome.entities],
        "attempts": outcome.attempts,
    }


_worker_ontology = None   # built once per enrichment worker process, see main()
_worker_phrase_encoder = None


async def handle_enrichment(run_id: str, paper_id: str, payload: dict) -> tuple[str | None, dict | None]:
    route = payload.get("route", "full")
    from axiom.agents.extraction import ExtractionOutcome
    extraction_outcome = ExtractionOutcome(
        paper_id=paper_id,
        entities=[Entity(**e) for e in payload["entities"]],
        attempts=payload["attempts"],
    )
    try:
        outcome = await enrichment.enrich(
            extraction_outcome, encoder=_worker_phrase_encoder, ontology=_worker_ontology, run_id=run_id,
        )
    except verifier.VerificationFailed as exc:
        return STREAM_LINK, {"terminal": {"verdict": "rejected", "route": route, "error": str(exc)}}
    except Exception as exc:
        return STREAM_LINK, {"terminal": {"verdict": "error", "route": route, "error": str(exc)}}
    verifier_attempts = max(payload["attempts"], outcome.attempts)
    return STREAM_LINK, {
        "route": route,
        "enriched": [e.model_dump() for e in outcome.result.enriched],
        "verifier_attempts": verifier_attempts,
    }


def handle_linking_sync(conn, run_id: str, paper_id: str, payload: dict) -> None:
    """Synchronous -- linking.link() and the db.* calls are all plain sqlite,
    no await needed; called from the async handler via asyncio.to_thread
    only if the caller wants it off the event loop (main() doesn't bother --
    this worker's event loop has nothing else to do while it writes)."""
    terminal = payload.get("terminal")
    if terminal is not None:
        db.upsert_agent_run_item(
            conn, run_id, paper_id, route=terminal.get("route"),
            verifier_attempts=0, verdict=terminal["verdict"], error=terminal.get("error"),
        )
        db.increment_agent_run_papers_done(conn, run_id)
        return
    enrichment_result = EnrichmentResult(
        paper_id=paper_id, enriched=[EnrichedEntity(**e) for e in payload["enriched"]],
    )
    linking.link(conn, enrichment_result, run_id=run_id)
    db.upsert_agent_run_item(
        conn, run_id, paper_id, route=payload.get("route"),
        verifier_attempts=payload.get("verifier_attempts", 0), verdict="accepted", error=None,
    )
    db.increment_agent_run_papers_done(conn, run_id)
    _maybe_finish_run(conn, run_id)


def _maybe_finish_run(conn, run_id: str) -> None:
    """The in-process pipeline calls db.finish_agent_run() once, after
    asyncio.gather returns -- there's no single "the run is over" moment in
    distributed mode, so each linking write checks for it instead. A real
    race (two papers finishing their last write at the same moment, both
    seeing papers_done==papers_total) just calls finish_agent_run() twice;
    it's an idempotent UPDATE, not a bug."""
    row = db.get_agent_run(conn, run_id)
    if row is not None and row["papers_done"] >= row["papers_total"]:
        db.finish_agent_run(conn, run_id, papers_done=row["papers_done"],
                             finished_at=datetime.now(timezone.utc).isoformat())


_worker_link_conn = None  # the one sqlite connection this process ever opens, see main()


async def handle_linking(run_id: str, paper_id: str, payload: dict) -> tuple[str | None, dict | None]:
    handle_linking_sync(_worker_link_conn, run_id, paper_id, payload)
    return None, None


HANDLERS: dict[str, Callable[[str, str, dict], Awaitable[tuple[str | None, dict | None]]]] = {
    "coordinator": handle_coordinator,
    "extraction": handle_extraction,
    "enrichment": handle_enrichment,
    "linking": handle_linking,
}
ROLE_STREAM_IN = {
    "coordinator": STREAM_ROUTE, "extraction": STREAM_EXTRACT,
    "enrichment": STREAM_ENRICH, "linking": STREAM_LINK,
}


# ---------------------------------------------------------------------------
# Generic consumer loop: read -> handle -> publish+ack, or requeue-with-
# incremented-attempt+ack on failure, or dead-letter+ack once
# `max_deliveries` is exhausted. A separate periodic XAUTOCLAIM pass recovers
# messages stuck in another (dead) consumer's PEL after `claim_idle_ms`
# (e.g. a worker that was SIGKILLed mid-message, not a handled exception --
# the normal retry path above already covers handled failures and never
# leaves anything pending).
# ---------------------------------------------------------------------------
async def _process_one(
    client: StreamClient, *, role: str, stream_in: str, group: str,
    msg_id: str, raw: str, handler, max_deliveries: int,
) -> str:
    """Returns 'ok' | 'retried' | 'dead' for the caller's bookkeeping/tests."""
    envelope = parse_envelope(raw)
    run_id, paper_id, attempt = envelope["run_id"], envelope["paper_id"], envelope["attempt"]
    try:
        next_stream, next_payload = await handler(run_id, paper_id, envelope["payload"])
        if next_stream is not None:
            await client.xadd(next_stream, {FIELD: make_envelope(run_id, paper_id, 1, next_payload)})
        await client.xack(stream_in, group, msg_id)
        return "ok"
    except Exception as exc:
        log.warning("worker=%s run=%s paper=%s attempt=%d failed: %s", role, run_id, paper_id, attempt, exc)
        if attempt >= max_deliveries:
            await client.xadd(STREAM_DEAD, {FIELD: make_envelope(
                run_id, paper_id, attempt, {"stage": role, "error": str(exc), "original_payload": envelope["payload"]},
            )})
            await client.xack(stream_in, group, msg_id)
            return "dead"
        await client.xadd(stream_in, {FIELD: make_envelope(run_id, paper_id, attempt + 1, envelope["payload"])})
        await client.xack(stream_in, group, msg_id)
        return "retried"


async def reclaim_stuck(client: StreamClient, stream: str, group: str, consumer: str,
                         min_idle_ms: int) -> list[tuple[str, dict]]:
    """XAUTOCLAIM every pending message idle longer than `min_idle_ms`
    (a worker that died mid-message, never acked) onto `consumer`."""
    claimed: list[tuple[str, dict]] = []
    cursor = "0-0"
    while True:
        cursor, messages, *_ = await client.xautoclaim(stream, group, consumer, min_idle_ms, cursor, count=50)
        claimed.extend(messages)
        if not messages or cursor in ("0-0", 0):
            break
    if claimed:
        # eval/distributed_bench.json's "redelivery count" greps this --
        # the only source of truth for "a message got reclaimed" in this
        # design (retries from a *handled* failure re-XADD explicitly and
        # are counted by the caller separately; this is specifically the
        # "a consumer went dark mid-message" case).
        log.warning("reclaimed %d stuck message(s) from %s/%s onto %s", len(claimed), stream, group, consumer)
    return claimed


async def run_consumer_loop(
    client: StreamClient, *, role: str, consumer_name: str,
    max_deliveries: int = config.STREAM_MAX_DELIVERIES,
    claim_idle_ms: int = config.STREAM_CLAIM_IDLE_MS,
    stop_event: asyncio.Event | None = None,
    max_iterations: int | None = None,
) -> None:
    """The role's whole lifecycle. `stop_event` (set by a SIGTERM handler in
    main(), or directly by a test) ends the loop after the in-flight batch
    finishes -- never mid-message. `max_iterations` is test-only (run N
    read/reclaim cycles then return, instead of forever)."""
    stream_in = ROLE_STREAM_IN[role]
    handler = HANDLERS[role]
    await ensure_group(client, stream_in)
    iterations = 0
    stop_event = stop_event or asyncio.Event()
    while not stop_event.is_set():
        reclaimed = await reclaim_stuck(client, stream_in, GROUP, consumer_name, claim_idle_ms)
        for msg_id, fields in reclaimed:
            await _process_one(client, role=role, stream_in=stream_in, group=GROUP,
                                msg_id=msg_id, raw=fields[FIELD], handler=handler, max_deliveries=max_deliveries)

        resp = await client.xreadgroup(GROUP, consumer_name, {stream_in: ">"}, count=10, block=1000)
        for _stream_name, messages in (resp or []):
            for msg_id, fields in messages:
                await _process_one(client, role=role, stream_in=stream_in, group=GROUP,
                                    msg_id=msg_id, raw=fields[FIELD], handler=handler, max_deliveries=max_deliveries)

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            return


# ---------------------------------------------------------------------------
# Enqueue-only entry point for distributed mode (axiom/agents/pipeline.py
# calls this when config.AXIOM_EXECUTION == "distributed").
# ---------------------------------------------------------------------------
async def enqueue_run(client: StreamClient, run_id: str, papers: list) -> None:
    for paper in papers:
        await client.xadd(STREAM_ROUTE, {FIELD: make_envelope(
            run_id, paper.paper_id, 1, {"title": paper.title, "abstract": paper.abstract},
        )})


def get_redis_client():
    import redis.asyncio as aioredis
    return aioredis.from_url(config.REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# CLI entry point: python -m axiom.workers <role>
# ---------------------------------------------------------------------------
async def _main_async(role: str) -> None:
    global _worker_ontology, _worker_phrase_encoder, _worker_link_conn

    client = get_redis_client()
    consumer_name = f"{role}-{uuid.uuid4().hex[:8]}"

    if role == "enrichment":
        conn = db.connect(config.DB_PATH)
        try:
            _worker_phrase_encoder = enrichment.get_default_phrase_encoder()
            _worker_ontology = enrichment.build_ontology(conn, _worker_phrase_encoder)
        finally:
            conn.close()
    elif role == "linking":
        _worker_link_conn = db.connect(config.DB_PATH)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        log.info("worker=%s consumer=%s received shutdown signal", role, consumer_name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass  # Windows: no add_signal_handler for SIGTERM; Ctrl-C still raises KeyboardInterrupt

    log.info("worker=%s consumer=%s starting", role, consumer_name)
    try:
        await run_consumer_loop(client, role=role, consumer_name=consumer_name, stop_event=stop_event)
    finally:
        if _worker_link_conn is not None:
            _worker_link_conn.close()
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("role", choices=ROLES)
    args = ap.parse_args(argv)
    try:
        asyncio.run(_main_async(args.role))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
