"""Tests for axiom.fields (the registry + build job) and the /fields/* API
routes (stage-3 product redesign, docs/product/ARCHITECTURE_NOTE.md)."""
from __future__ import annotations

import pytest

from axiom import config, db, fields as fieldsmod
from tests.conftest import SEED_VECTORS


# --- axiom.fields: registry -----------------------------------------------------

@pytest.fixture
def fields_conn(tmp_path, monkeypatch):
    path = tmp_path / "fields.db"
    monkeypatch.setattr(config, "FIELDS_DB_PATH", path)
    monkeypatch.setattr(config, "FIELDS_DIR", tmp_path / "fields")
    conn = fieldsmod.connect_fields()
    fieldsmod.init_fields_db(conn)
    yield conn
    conn.close()


def test_create_field_generates_id_and_paths(fields_conn):
    row = fieldsmod.create_field(fields_conn, topic="graph neural networks", target=100)
    assert row["status"] == "queued"
    assert row["step"] is None
    assert row["topic"] == "graph neural networks"
    assert row["papers_total"] == 100
    assert row["papers_done"] == 0
    assert row["id"] in row["db_path"]          # data/fields/<id>/axiom.db
    assert row["id"] in row["collection"]        # axiom_field_<id>


def test_get_field_missing_returns_none(fields_conn):
    assert fieldsmod.get_field(fields_conn, "nope") is None


def test_list_fields_orders_by_last_opened_then_created(fields_conn):
    a = fieldsmod.create_field(fields_conn, topic="a", target=10)
    b = fieldsmod.create_field(fields_conn, topic="b", target=10)
    fieldsmod.touch_last_opened(fields_conn, a["id"])
    ordered = fieldsmod.list_fields(fields_conn)
    assert ordered[0]["id"] == a["id"]           # most recently opened first
    assert ordered[1]["id"] == b["id"]           # never opened -> falls back to created_at


def test_set_status_step_papers_cost(fields_conn):
    row = fieldsmod.create_field(fields_conn, topic="x", target=10)
    fieldsmod.set_status(fields_conn, row["id"], "building")
    fieldsmod.set_step(fields_conn, row["id"], "fetching")
    fieldsmod.set_papers(fields_conn, row["id"], done=5, total=10)
    fieldsmod.set_cost(fields_conn, row["id"], 0.04)
    updated = fieldsmod.get_field(fields_conn, row["id"])
    assert updated["status"] == "building"
    assert updated["step"] == "fetching"
    assert updated["papers_done"] == 5
    assert updated["cost_so_far"] == 0.04


def test_set_status_failed_records_error(fields_conn):
    row = fieldsmod.create_field(fields_conn, topic="x", target=10)
    fieldsmod.set_status(fields_conn, row["id"], "failed", error="OpenAlex timed out")
    updated = fieldsmod.get_field(fields_conn, row["id"])
    assert updated["status"] == "failed"
    assert updated["error"] == "OpenAlex timed out"


def test_ensure_default_field_idempotent_and_counts_real_papers(fields_conn, seeded_db, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", seeded_db)
    fieldsmod.ensure_default_field(fields_conn)
    row = fieldsmod.get_field(fields_conn, config.FIELD_DEFAULT_ID)
    assert row["status"] == "ready"
    assert row["papers_total"] == 8              # the seeded corpus's real paper count
    assert row["db_path"] == str(seeded_db)
    # calling again must not raise or duplicate the row (migration is a no-op after first run)
    fieldsmod.ensure_default_field(fields_conn)
    assert len(fieldsmod.list_fields(fields_conn)) == 1


def test_live_status_reads_agent_run_during_extraction(fields_conn, seeded_db):
    row = fieldsmod.create_field(fields_conn, topic="x", target=8)
    fields_conn.execute("UPDATE fields SET db_path = ? WHERE id = ?", (str(seeded_db), row["id"]))
    fields_conn.commit()

    conn = db.connect(seeded_db)
    try:
        db.create_agent_run(conn, "run-1", papers_total=8, model_fast="m", model_strong="m", started_at="t")
        db.increment_agent_run_papers_done(conn, "run-1")
        db.increment_agent_run_papers_done(conn, "run-1")
        db.record_llm_spend(conn, "run-1", ts="t", model="m", tokens_in=10, tokens_out=10, cost=0.01)
    finally:
        conn.close()

    fieldsmod.set_step(fields_conn, row["id"], "extracting")
    fieldsmod.set_extraction_run(fields_conn, row["id"], "run-1")
    out = fieldsmod.live_status(fields_conn, row["id"])
    assert out["papers_done"] == 2                # live from agent_runs, not the stale fields row
    assert out["cost_so_far"] == 0.01


def test_live_status_unknown_field_returns_none(fields_conn):
    assert fieldsmod.live_status(fields_conn, "nope") is None


class _FakeOpenAlexClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_build_field_reindex_step_does_not_cross_threads(fields_conn, monkeypatch):
    """Regression test for a real bug hit live (2026-11-01): reindex_qdrant
    was called via asyncio.to_thread with a sqlite3.Connection created on the
    event-loop thread, which crashes with "SQLite objects created in a
    thread can only be used in that same thread." Also covers the papers
    count: ingest.snowball's 2-hop expansion can overshoot the requested
    target, and papers_total must end up reflecting the real count, not the
    original request, once it's known."""
    import asyncio

    field = fieldsmod.create_field(fields_conn, topic="contrastive learning", target=5)
    field_id = field["id"]

    def fake_snowball(client, *, seed_query, target, keep=None, **kwargs):
        # Overshoots the requested target, like the real OpenAlex snowball does.
        return {
            f"W{i}": {"id": f"W{i}", "title": f"paper {i}", "abstract": "abstract text"}
            for i in range(target + 3)
        }

    def fake_ingest_to_sqlite(conn, works, **kwargs):
        for i, w in enumerate(works):
            db.insert_paper(
                conn, openalex_id=w["id"], title=w["title"], abstract=w["abstract"],
                publication_year=2024, venue_id=None, venue="ACL", cited_by_count=0,
            )
        conn.commit()
        return len(works)

    def fake_reindex_qdrant(conn, *, collection=None, log=print):
        # The actual regression check: this runs inside asyncio.to_thread
        # (see axiom/fields.py's `_reindex` closure) -- using `conn` here
        # must not raise the cross-thread sqlite3 error.
        conn.execute("SELECT COUNT(*) FROM papers").fetchone()
        return 0

    async def fake_run_pipeline(papers, *, encoder, db_path=None, conn_factory=None, run_id=None, **kwargs):
        return {}

    monkeypatch.setattr(fieldsmod.ingest, "OpenAlexClient", lambda *a, **k: _FakeOpenAlexClient())
    monkeypatch.setattr(fieldsmod.ingest, "snowball", fake_snowball)
    monkeypatch.setattr(fieldsmod.ingest, "ingest_to_sqlite", fake_ingest_to_sqlite)
    monkeypatch.setattr(fieldsmod.ingest, "make_venue_filter", lambda *a, **k: None)
    monkeypatch.setattr("axiom.indexer.reindex_qdrant", fake_reindex_qdrant)
    monkeypatch.setattr(fieldsmod.agents_pipeline, "run_pipeline", fake_run_pipeline)

    asyncio.run(fieldsmod.build_field(field_id, "contrastive learning", 5))

    updated = fieldsmod.get_field(fields_conn, field_id)
    assert updated["status"] == "ready", updated["error"]
    assert updated["papers_done"] == 8          # 5 requested + 3 overshoot, the real count
    assert updated["papers_total"] == 8         # corrected to the real count, not the original 5


def test_estimate_extraction_scales_with_papers_and_concurrency(monkeypatch):
    monkeypatch.setattr(config, "AGENT_CONCURRENCY", 4)
    small = fieldsmod.estimate_extraction(100)
    large = fieldsmod.estimate_extraction(300)
    assert large["est_cost_usd"] > small["est_cost_usd"]
    assert large["est_seconds"] > small["est_seconds"]
    assert small["est_cost_usd"] > 0


# --- /fields/* API routes --------------------------------------------------------

@pytest.fixture
def empty_fields_db(monkeypatch, tmp_path):
    """FastAPI's TestClient (used bare, not as a context manager, by api_client)
    never runs the app's lifespan -- so fields.db's schema must be created
    explicitly here, same as the real startup hook would."""
    monkeypatch.setattr(config, "FIELDS_DB_PATH", tmp_path / "fields.db")
    conn = fieldsmod.connect_fields()
    try:
        fieldsmod.init_fields_db(conn)
    finally:
        conn.close()


def _patch_corpus(main, monkeypatch, patched_db_path, *, query_vector, min_papers=None):
    """A4: POST /fields slices the cached base-corpus graph/vectors, not a
    live download -- fake those two process-wide singletons (same seeded db
    the rest of this fixture module already uses as a stand-in corpus) plus
    the encoder's query embedding."""
    from axiom import graph as graphmod

    for fn in (main.get_corpus_graph, main.get_corpus_vectors):
        fn.cache_clear()
    g = graphmod.load_graph(db.connect(patched_db_path))
    monkeypatch.setattr(main, "get_corpus_graph", lambda: g)
    monkeypatch.setattr(main, "get_corpus_vectors", lambda: dict(SEED_VECTORS))

    class _QueryEncoder:
        def encode_query(self, query: str) -> list[float]:
            return query_vector

    monkeypatch.setattr(main, "get_encoder", lambda: _QueryEncoder())
    if min_papers is not None:
        monkeypatch.setattr(config, "FIELD_MIN_PAPERS", min_papers)


def test_create_field_is_instant_and_ready(api_client, empty_fields_db, patched_db_path, monkeypatch):
    import api.main as main

    # Query aligned with cluster A ([1,0,0]) -> A1-A4 clear the 0.80 floor;
    # 4 papers clears a lowered FIELD_MIN_PAPERS so this field lands 'ready'.
    _patch_corpus(main, monkeypatch, patched_db_path, query_vector=[1.0, 0.0, 0.0], min_papers=2)

    r = api_client.post("/fields", json={"topic": "retrieval", "target": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["papers_total"] == 4            # A1-A4, see tests/conftest.py's seeded corpus
    assert body["total_relevant"] == 4
    assert body["extend_command"] is None

    # And it's immediately usable -- no polling a background build.
    status = api_client.get(f"/fields/{body['field_id']}").json()
    assert status["status"] == "ready"


def test_create_field_thin_coverage_includes_extend_command(api_client, empty_fields_db, patched_db_path, monkeypatch):
    import api.main as main

    # Same aligned query, but default FIELD_MIN_PAPERS (20) -- the seeded
    # corpus only has 8 papers total, so this always lands 'thin'.
    _patch_corpus(main, monkeypatch, patched_db_path, query_vector=[1.0, 0.0, 0.0])

    r = api_client.post("/fields", json={"topic": "a very niche topic", "target": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "thin"
    assert body["extend_command"] == 'python scripts/build_corpus.py --extend "a very niche topic"'


def test_create_field_validates_target_range(api_client, empty_fields_db):
    assert api_client.post("/fields", json={"topic": "x", "target": 1}).status_code == 422
    assert api_client.post("/fields", json={"topic": "x", "target": 5000}).status_code == 422
    assert api_client.post("/fields", json={"topic": ""}).status_code == 422


def test_field_status_404_when_unknown(api_client, empty_fields_db):
    assert api_client.get("/fields/nope").status_code == 404


def test_field_status_get_does_not_touch_last_opened(api_client, empty_fields_db, tmp_path, monkeypatch):
    """B4: GET /fields/{id} must stay read-only; POST /fields/{id}/open is the
    one explicit 'opened' signal."""
    monkeypatch.setattr(config, "FIELDS_DB_PATH", tmp_path / "fields.db")
    conn = fieldsmod.connect_fields()
    fieldsmod.init_fields_db(conn)
    row = fieldsmod.create_field(conn, topic="x", target=10)
    conn.execute("UPDATE fields SET status = 'ready' WHERE id = ?", (row["id"],))
    conn.commit()
    before = fieldsmod.get_field(conn, row["id"])["last_opened_at"]
    conn.close()

    assert api_client.get(f"/fields/{row['id']}").status_code == 200

    conn = fieldsmod.connect_fields()
    assert fieldsmod.get_field(conn, row["id"])["last_opened_at"] == before

    assert api_client.post(f"/fields/{row['id']}/open").status_code == 204
    assert fieldsmod.get_field(conn, row["id"])["last_opened_at"] != before
    conn.close()


def test_field_open_404_when_unknown(api_client, empty_fields_db):
    assert api_client.post("/fields/nope/open").status_code == 404


def test_field_overview(field_client):
    client, field_id = field_client
    r = client.get(f"/fields/{field_id}/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["papers"] == 8
    assert body["topic_communities"] == 2
    assert body["candidate_gaps"] == 1
    assert len(body["topics"]) == 2


def test_field_gaps_list(field_client):
    client, field_id = field_client
    r = client.get(f"/fields/{field_id}/gaps")
    assert r.status_code == 200
    assert len(r.json()["gaps"]) == 1


def test_field_gap_detail_has_per_side_methods_and_datasets(field_client):
    client, field_id = field_client
    r = client.get(f"/fields/{field_id}/gaps/0")
    assert r.status_code == 200
    body = r.json()
    assert body["a"]["size"] == 4 and body["b"]["size"] == 4
    assert "methods" in body["a"] and "datasets" in body["a"]
    assert body["inter_citations"] == 1


def test_field_gap_detail_404_out_of_range(field_client):
    client, field_id = field_client
    assert client.get(f"/fields/{field_id}/gaps/99").status_code == 404


def test_field_paper_detail_with_entities(field_client, patched_db_path):
    client, field_id = field_client
    conn = db.connect(patched_db_path)
    try:
        from datetime import datetime, timezone
        conn.execute(
            "INSERT INTO paper_concepts (run_id, paper_id, name, kind, canonical, cosine, "
            "evidence_span, confidence, source, created_at) VALUES "
            "('r1', 'A1', 'RAG', 'method', 'retrieval-augmented generation', 0.9, 'RAG', 0.9, "
            "'agent', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.get(f"/fields/{field_id}/papers/A1")
    assert r.status_code == 200
    body = r.json()
    assert body["paper_id"] == "A1"
    assert len(body["entities"]) == 1
    assert body["entities"][0]["kind"] == "method"


def test_field_paper_detail_404(field_client):
    client, field_id = field_client
    assert client.get(f"/fields/{field_id}/papers/NOPE").status_code == 404


def test_shortlist_create_and_list(field_client, monkeypatch):
    client, field_id = field_client
    from axiom.hypothesis import HypothesisPitch
    import api.main as main

    async def fake_generate(gap, g, *, run_id, trend_context=None, conn=None, **_):
        return HypothesisPitch(
            title="Bridge retrieval and translation", claim="c", method_sketch="m",
            datasets=["D1"], supporting_paper_ids=["A1", "B1"],
        )

    monkeypatch.setattr(main.synthesis_agent, "generate_hypothesis", fake_generate)

    r = client.post(f"/fields/{field_id}/shortlist", json={"gap_index": 0})
    assert r.status_code == 200
    item = r.json()
    assert item["status"] == "draft"             # not "pending" -- product-facing label
    assert item["title"] == "Bridge retrieval and translation"

    listed = client.get(f"/fields/{field_id}/shortlist", params={"status": "draft"}).json()
    assert any(i["id"] == item["id"] for i in listed)
    assert client.get(f"/fields/{field_id}/shortlist", params={"status": "kept"}).json() == []


def test_shortlist_create_404_bad_gap_index(field_client):
    client, field_id = field_client
    assert client.post(f"/fields/{field_id}/shortlist", json={"gap_index": 99}).status_code == 404


def test_shortlist_patch_status_notes_papers(field_client, monkeypatch):
    client, field_id = field_client
    from axiom.hypothesis import HypothesisPitch
    import api.main as main

    async def fake_generate(gap, g, *, run_id, trend_context=None, conn=None, **_):
        return HypothesisPitch(title="T", claim="c", method_sketch="m",
                                datasets=[], supporting_paper_ids=["A1"])
    monkeypatch.setattr(main.synthesis_agent, "generate_hypothesis", fake_generate)

    item = client.post(f"/fields/{field_id}/shortlist", json={"gap_index": 0}).json()

    r = client.patch(
        f"/fields/{field_id}/shortlist/{item['id']}",
        json={"status": "kept", "notes": "worth pursuing", "papers_to_read": ["A1", "B1"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "kept"
    assert body["notes"] == "worth pursuing"
    assert body["papers_to_read"] == ["A1", "B1"]


def test_shortlist_patch_404_unknown_item(field_client):
    client, field_id = field_client
    r = client.patch(f"/fields/{field_id}/shortlist/999", json={"status": "kept"})
    assert r.status_code == 404


def test_shortlist_export_markdown(field_client, monkeypatch):
    client, field_id = field_client
    from axiom.hypothesis import HypothesisPitch
    import api.main as main

    async def fake_generate(gap, g, *, run_id, trend_context=None, conn=None, **_):
        return HypothesisPitch(title="Bridge the gap", claim="A real claim.", method_sketch="m",
                                datasets=["D1"], supporting_paper_ids=["A1"])
    monkeypatch.setattr(main.synthesis_agent, "generate_hypothesis", fake_generate)

    item = client.post(f"/fields/{field_id}/shortlist", json={"gap_index": 0}).json()
    client.patch(f"/fields/{field_id}/shortlist/{item['id']}", json={"status": "kept"})

    r = client.get(f"/fields/{field_id}/shortlist/export.md")
    assert r.status_code == 200
    assert "Bridge the gap" in r.text
    assert "A real claim." in r.text
    assert "unverified" in r.text.lower()


def test_shortlist_export_empty(field_client):
    client, field_id = field_client
    r = client.get(f"/fields/{field_id}/shortlist/export.md")
    assert r.status_code == 200
    assert "No kept directions yet" in r.text


# --- B4: /map/similarity on a bad topic id -------------------------------------
def test_field_map_similarity_bad_topic_id_is_422_not_500(field_client):
    """Stage 4 B4: a non-numeric topic id used to crash `int(ident)` uncaught,
    surfacing as an unhandled 500 instead of a client error."""
    client, field_id = field_client
    r = client.get(f"/fields/{field_id}/map/similarity?a_kind=topic&a_id=not-a-number&b_kind=topic&b_id=0")
    assert r.status_code == 422


def test_field_map_similarity_unknown_topic_cid_is_404(field_client):
    client, field_id = field_client
    r = client.get(f"/fields/{field_id}/map/similarity?a_kind=topic&a_id=99999&b_kind=topic&b_id=0")
    assert r.status_code == 404
