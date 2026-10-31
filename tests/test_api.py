"""Tests for api.main routes via TestClient, with encoder/store/graph/gap faked."""
from __future__ import annotations

from axiom import db


def test_health_ok(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_503_when_store_down(api_client, fake_store):
    fake_store.raise_on_count = True
    assert api_client.get("/health").status_code == 503


def test_search_hybrid_and_dense(api_client):
    assert api_client.get("/search", params={"q": "retrieval"}).status_code == 200
    assert api_client.get("/search", params={"q": "retrieval", "mode": "dense"}).status_code == 200
    hits = api_client.get("/search", params={"q": "retrieval", "top_k": 3}).json()
    assert len(hits) <= 3


def test_search_falls_back_when_no_hybrid(api_client, fake_store):
    fake_store._hybrid = False
    r = api_client.get("/search", params={"q": "retrieval", "mode": "hybrid"})
    assert r.status_code == 200                # falls back to dense, no error


def test_search_validates_top_k(api_client):
    assert api_client.get("/search", params={"q": "x", "top_k": 0}).status_code == 422
    assert api_client.get("/search", params={"q": "x", "mode": "bogus"}).status_code == 422


def test_get_paper_found_and_missing(api_client):
    assert api_client.get("/papers/A1").json()["paper_id"] == "A1"
    assert api_client.get("/papers/NOPE").status_code == 404


def test_trends_top(api_client):
    data = api_client.get("/trends/top", params={"n": 5}).json()
    assert "keywords" in data and "prior_window" in data


def test_graph_stats_and_gaps(api_client):
    stats = api_client.get("/graph/stats").json()
    assert stats["papers"] == 8
    gaps = api_client.get("/graph/gaps").json()
    assert len(gaps["gaps"]) == 1
    assert gaps["gaps"][0]["inter_citations"] == 1


def test_reading_list_crud(api_client):
    assert api_client.post("/reading-list/A1").status_code == 204
    assert api_client.post("/reading-list/NOPE").status_code == 404   # unknown paper
    ids = [b["paper_id"] for b in api_client.get("/reading-list").json()]
    assert "A1" in ids
    assert api_client.delete("/reading-list/A1").status_code == 204
    assert "A1" not in [b["paper_id"] for b in api_client.get("/reading-list").json()]


def test_summarize_cached_path(api_client, patched_db_path):
    conn = db.connect(patched_db_path)
    try:
        db.save_summary(conn, "A1", ["b1", "b2", "b3"], model="qwen")
    finally:
        conn.close()
    body = api_client.post("/reading-list/A1/summarize").json()
    assert body["cached"] is True and body["bullets"] == ["b1", "b2", "b3"]


def test_summarize_fresh_path_writes_db(api_client, patched_db_path, fake_llm):
    fake_llm.responses = [{"bullets": ["one", "two", "three"]}]
    body = api_client.post("/reading-list/A1/summarize").json()
    assert body["cached"] is False and body["bullets"] == ["one", "two", "three"]
    conn = db.connect(patched_db_path)
    try:
        assert db.get_summary(conn, "A1") == ["one", "two", "three"]
    finally:
        conn.close()


def test_hypothesize_creates_review_row(api_client, patched_db_path, monkeypatch):
    import api.main as main
    from axiom.hypothesis import HypothesisPitch

    async def fake_generate_hypothesis(gap, g, *, run_id, trend_context=None, **_):
        return HypothesisPitch(
            title="Bridge", claim="c", method_sketch="m",
            datasets=[], supporting_paper_ids=["A1", "B1"],
        )

    monkeypatch.setattr(main.synthesis_agent, "generate_hypothesis", fake_generate_hypothesis)

    r = api_client.post("/graph/gaps/0/hypothesize")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Bridge" and isinstance(body["review_id"], int)

    pending = api_client.get("/review-queue", params={"status": "pending"}).json()
    assert any(item["id"] == body["review_id"] for item in pending)


def test_hypothesize_out_of_range(api_client):
    assert api_client.post("/graph/gaps/99/hypothesize").status_code == 404


def test_review_queue_approve_reject(api_client, patched_db_path):
    conn = db.connect(patched_db_path)
    try:
        rid = db.add_to_review_queue(
            conn, gap_a_label="A", gap_b_label="B", title="t", claim="c",
            method_sketch="m", datasets=[], supporting_paper_ids=["A1", "B1"])
    finally:
        conn.close()
    assert api_client.post(f"/review-queue/{rid}/approve").status_code == 204
    approved = api_client.get("/review-queue", params={"status": "approved"}).json()
    assert any(i["id"] == rid for i in approved)

    assert api_client.post(f"/review-queue/{rid}/reject").status_code == 204
    rejected = api_client.get("/review-queue", params={"status": "rejected"}).json()
    assert any(i["id"] == rid for i in rejected)


def test_review_queue_status_validation(api_client):
    assert api_client.get("/review-queue", params={"status": "bogus"}).status_code == 422


def test_agents_run_is_backgrounded_and_status_reports_progress(api_client, patched_db_path, monkeypatch):
    import api.main as main

    async def fake_run_pipeline(papers, *, encoder, run_id, clusters=None, gaps=None, trend_context=None, **_):
        conn = db.connect(patched_db_path)
        try:
            db.create_agent_run(conn, run_id, papers_total=len(papers),
                                 model_fast="fast", model_strong="strong", started_at="t0")
            db.upsert_agent_run_item(conn, run_id, papers[0].paper_id, route="full",
                                      verifier_attempts=1, verdict="accepted", error=None)
            db.finish_agent_run(conn, run_id, papers_done=len(papers), finished_at="t1")
        finally:
            conn.close()
        return {"run_id": run_id, "items": [], "cluster_labels": [], "hypotheses": []}

    monkeypatch.setattr(main.agents_pipeline, "run_pipeline", fake_run_pipeline)

    r = api_client.post("/agents/run", json={"limit": 1})
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    status = api_client.get(f"/agents/runs/{run_id}").json()
    assert status["status"] == "done"
    assert status["verdict_counts"] == {"accepted": 1}
    assert status["cost_so_far"] == 0.0


def test_agents_run_marks_failed_on_pipeline_error(api_client, patched_db_path, monkeypatch):
    import api.main as main

    async def failing_run_pipeline(papers, *, encoder, run_id, **_):
        # real run_pipeline creates its agent_runs row before doing anything else that
        # could fail -- mirror that here so the failure path has a row to mark 'failed'.
        conn = db.connect(patched_db_path)
        try:
            db.create_agent_run(conn, run_id, papers_total=len(papers),
                                 model_fast="fast", model_strong="strong", started_at="t0")
        finally:
            conn.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(main.agents_pipeline, "run_pipeline", failing_run_pipeline)

    run_id = api_client.post("/agents/run", json={"limit": 1}).json()["run_id"]
    status = api_client.get(f"/agents/runs/{run_id}").json()
    assert status["status"] == "failed"


def test_agents_run_validates_limit(api_client):
    assert api_client.post("/agents/run", json={"limit": 0}).status_code == 422
    assert api_client.post("/agents/run", json={"limit": 51}).status_code == 422


def test_agents_run_404_when_no_matching_papers(api_client):
    assert api_client.post("/agents/run", json={"paper_ids": ["nope"]}).status_code == 404


def test_agents_run_status_404_when_unknown(api_client):
    assert api_client.get("/agents/runs/does-not-exist").status_code == 404
