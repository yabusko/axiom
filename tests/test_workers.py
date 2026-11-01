"""Offline tests for axiom/workers.py: envelope/contract round-trips and the
ack/retry/dead-letter/reclaim logic against a tiny in-memory fake of the 5
Redis Streams calls workers.py actually uses (xgroup_create, xadd,
xreadgroup, xack, xautoclaim) -- no real Redis involved. The real-Redis
integration test lives in tests/test_workers_integration.py, gated behind
@pytest.mark.redis.
"""
from __future__ import annotations

import asyncio

from axiom import workers
from axiom.agents.contracts import EnrichedEntity, Entity


# ---------------------------------------------------------------------------
# Envelope / contract round-trips
# ---------------------------------------------------------------------------
def test_envelope_round_trip():
    raw = workers.make_envelope("run1", "P1", 2, {"a": 1, "b": [1, 2, 3]})
    parsed = workers.parse_envelope(raw)
    assert parsed == {"run_id": "run1", "paper_id": "P1", "attempt": 2, "payload": {"a": 1, "b": [1, 2, 3]}}


def test_entity_round_trips_through_envelope_payload():
    entity = Entity(name="RAG", kind="method", evidence_span="we use RAG", confidence=0.9)
    raw = workers.make_envelope("r", "P1", 1, {"entities": [entity.model_dump()]})
    payload = workers.parse_envelope(raw)["payload"]
    restored = Entity(**payload["entities"][0])
    assert restored == entity


def test_enriched_entity_round_trips_through_envelope_payload():
    e = EnrichedEntity(
        name="RAG", kind="method", evidence_span="we use RAG", confidence=0.9,
        canonical="retrieval-augmented generation", cosine=0.87, ontology_source="embedding",
    )
    raw = workers.make_envelope("r", "P1", 1, {"enriched": [e.model_dump()]})
    payload = workers.parse_envelope(raw)["payload"]
    restored = EnrichedEntity(**payload["enriched"][0])
    assert restored == e


# ---------------------------------------------------------------------------
# A tiny in-memory fake of exactly the 5 Redis Streams calls workers.py uses.
# ---------------------------------------------------------------------------
class FakeStreamClient:
    def __init__(self):
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self.groups: dict[tuple[str, str], dict] = {}
        self._next = 0

    def _new_id(self) -> str:
        self._next += 1
        return f"{self._next}-0"

    async def xgroup_create(self, name, groupname, id="0", mkstream=False):
        self.streams.setdefault(name, [])
        key = (name, groupname)
        if key in self.groups:
            raise Exception("BUSYGROUP Consumer Group name already exists")
        self.groups[key] = {"pending": {}, "delivered": set(), "force_idle": []}

    async def xadd(self, name, fields):
        msg_id = self._new_id()
        self.streams.setdefault(name, []).append((msg_id, dict(fields)))
        return msg_id

    async def xreadgroup(self, groupname, consumername, streams, count=None, block=None):
        out = []
        for stream_name in streams:
            key = (stream_name, groupname)
            group = self.groups[key]
            msgs = []
            for msg_id, fields in self.streams.get(stream_name, []):
                if msg_id in group["delivered"]:
                    continue
                group["delivered"].add(msg_id)
                group["pending"][msg_id] = (consumername, fields)
                msgs.append((msg_id, fields))
                if count and len(msgs) >= count:
                    break
            if msgs:
                out.append((stream_name, msgs))
        return out

    async def xack(self, name, groupname, *ids):
        group = self.groups.get((name, groupname))
        if group:
            for i in ids:
                group["pending"].pop(i, None)

    async def xautoclaim(self, name, groupname, consumername, min_idle_time, start_id, count=None):
        group = self.groups.setdefault((name, groupname), {"pending": {}, "delivered": set(), "force_idle": []})
        claimed = []
        for msg_id in list(group["force_idle"]):
            if msg_id in group["pending"]:
                _, fields = group["pending"][msg_id]
                group["pending"][msg_id] = (consumername, fields)
                claimed.append((msg_id, fields))
        group["force_idle"] = []
        return "0-0", claimed, []

    def mark_idle(self, stream: str, group: str, msg_id: str) -> None:
        """Test hook: pretend `msg_id` has been pending long enough to reclaim."""
        self.groups[(stream, group)]["force_idle"].append(msg_id)


# ---------------------------------------------------------------------------
# _process_one: ok / retry / dead-letter
# ---------------------------------------------------------------------------
def test_process_one_ok_publishes_to_next_stream_and_acks():
    client = FakeStreamClient()

    async def handler(run_id, paper_id, payload):
        return workers.STREAM_ENRICH, {"ok": True}

    async def run():
        await workers.ensure_group(client, workers.STREAM_EXTRACT)
        raw = workers.make_envelope("r1", "P1", 1, {"x": 1})
        return await workers._process_one(
            client, role="extraction", stream_in=workers.STREAM_EXTRACT, group=workers.GROUP,
            msg_id="5-0", raw=raw, handler=handler, max_deliveries=3,
        )

    result = asyncio.run(run())
    assert result == "ok"
    assert len(client.streams[workers.STREAM_ENRICH]) == 1
    forwarded = workers.parse_envelope(client.streams[workers.STREAM_ENRICH][0][1][workers.FIELD])
    assert forwarded["payload"] == {"ok": True}
    assert forwarded["run_id"] == "r1" and forwarded["paper_id"] == "P1"
    # acked -- not left pending
    assert "5-0" not in client.groups[(workers.STREAM_EXTRACT, workers.GROUP)]["pending"]


def test_process_one_failure_requeues_with_incremented_attempt():
    client = FakeStreamClient()

    async def failing_handler(run_id, paper_id, payload):
        raise RuntimeError("boom")

    async def run():
        await workers.ensure_group(client, workers.STREAM_EXTRACT)
        raw = workers.make_envelope("r1", "P1", 1, {"x": 1})
        return await workers._process_one(
            client, role="extraction", stream_in=workers.STREAM_EXTRACT, group=workers.GROUP,
            msg_id="5-0", raw=raw, handler=failing_handler, max_deliveries=3,
        )

    result = asyncio.run(run())
    assert result == "retried"
    requeued = [m for m in client.streams[workers.STREAM_EXTRACT] if m[0] != "5-0"]
    assert len(requeued) == 1
    envelope = workers.parse_envelope(requeued[0][1][workers.FIELD])
    assert envelope["attempt"] == 2
    assert workers.STREAM_DEAD not in client.streams


def test_process_one_dead_letters_after_max_deliveries():
    client = FakeStreamClient()

    async def failing_handler(run_id, paper_id, payload):
        raise RuntimeError("boom")

    async def run():
        await workers.ensure_group(client, workers.STREAM_EXTRACT)
        raw = workers.make_envelope("r1", "P1", 3, {"x": 1})  # already at the limit
        return await workers._process_one(
            client, role="extraction", stream_in=workers.STREAM_EXTRACT, group=workers.GROUP,
            msg_id="5-0", raw=raw, handler=failing_handler, max_deliveries=3,
        )

    result = asyncio.run(run())
    assert result == "dead"
    assert len(client.streams[workers.STREAM_DEAD]) == 1
    dead_envelope = workers.parse_envelope(client.streams[workers.STREAM_DEAD][0][1][workers.FIELD])
    assert dead_envelope["payload"]["stage"] == "extraction"
    assert "boom" in dead_envelope["payload"]["error"]
    # not left on the original stream for redelivery
    assert all(m[0] == "5-0" for m in client.streams[workers.STREAM_EXTRACT])


# ---------------------------------------------------------------------------
# reclaim_stuck: a message pending from a dead consumer gets claimed
# ---------------------------------------------------------------------------
def test_reclaim_stuck_picks_up_idle_pending_message():
    client = FakeStreamClient()

    async def run():
        await workers.ensure_group(client, workers.STREAM_LINK)
        msg_id = await client.xadd(workers.STREAM_LINK, {workers.FIELD: workers.make_envelope("r", "P1", 1, {})})
        # consumer "dead-worker" reads it, then never acks (simulating a crash)
        await client.xreadgroup(workers.GROUP, "dead-worker", {workers.STREAM_LINK: ">"})
        assert msg_id in client.groups[(workers.STREAM_LINK, workers.GROUP)]["pending"]

        client.mark_idle(workers.STREAM_LINK, workers.GROUP, msg_id)
        claimed = await workers.reclaim_stuck(client, workers.STREAM_LINK, workers.GROUP, "live-worker", 30_000)
        return msg_id, claimed

    msg_id, claimed = asyncio.run(run())
    assert [m[0] for m in claimed] == [msg_id]
    consumer, _ = client.groups[(workers.STREAM_LINK, workers.GROUP)]["pending"][msg_id]
    assert consumer == "live-worker"
