"""FastAPI service layer (Backlog PBI 6, Task 6.1) — the stable REST seam a
future React frontend will use, so it stops importing `axiom/*` directly the
way `app/streamlit_app.py` does.

Scope (OD12): wraps what actually exists — search (dense/hybrid), paper
lookup, the OD10 velocity/trends engine, the OD9 citation graph + gap
detector, (OD13) reading-list bookmarks, (OD14) paper summaries, and (OD16)
hypothesis pitches + a HITL review queue. `/gap/evaluate`'s original 3-step
per-hypothesis design is still not built — see docs/DECISIONS.md OD9/OD16 for
why Step 3 is scoped to narrating an existing OD9 gap candidate instead.

Pydantic response models mirror the `axiom/*` dataclasses (SearchHit,
PaperRank, KeywordVelocity/VelocityAnalysis, Community/GapCandidate/
GapAnalysis, Neighbors) via `from_attributes=True` — the dataclasses stay the
single source of truth; this module only adds a JSON-serializable boundary.

Run from the repo root:
    uvicorn api.main:app --reload
Docs at http://localhost:8000/docs
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from axiom import concept_graph, config, db, llm
from axiom import gaps as gapsmod
from axiom import graph as graphmod
from axiom import hypothesis as hypothesismod
from axiom import summarize as summarizemod
from axiom import velocity as velocitymod
from axiom.agents import pipeline as agents_pipeline
from axiom.agents.contracts import RunStatusOut
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant


# --- Response models (mirror axiom/* dataclasses) -----------------------------
class SearchHitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    paper_id: str
    title: str
    year: int
    venue: str
    cited_by_count: int
    score: float
    concepts: list[str]


class PaperOut(BaseModel):
    paper_id: str
    title: str | None
    abstract: str | None
    year: int | None
    venue: str | None
    cited_by_count: int
    doi: str | None


class BookmarkOut(BaseModel):
    paper_id: str
    added_at: str
    title: str | None
    abstract: str | None
    year: int | None
    venue: str | None
    cited_by_count: int
    doi: str | None


class SummaryOut(BaseModel):
    paper_id: str
    bullets: list[str]
    cached: bool


class NodeBriefOut(BaseModel):
    paper_id: str
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    cited_by_count: int = 0
    in_corpus: bool = False


class NeighborsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    references: list[NodeBriefOut]
    cited_by: list[NodeBriefOut]


class PaperRankOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    paper_id: str
    title: str | None
    year: int | None
    venue: str | None
    cited_by_count: int
    local_in_degree: int
    pagerank: float


class KeywordVelocityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    concept: str
    recent_count: int
    prior_count: int
    recent_share: float
    prior_share: float
    velocity: float
    low_confidence: bool
    year_counts: dict[int, int]


class VelocityAnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    prior_window: tuple[int, int]
    recent_window: tuple[int, int]
    total_prior: int
    total_recent: int
    insufficient_year_spread: bool
    keywords: list[KeywordVelocityOut]


class CommunityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    cid: int
    members: list[str]
    labels: list[str]
    size: int


class GapCandidateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    a: CommunityOut
    b: CommunityOut
    semantic_similarity: float
    inter_citations: int
    gap_score: float
    g_score: float                       # OD17 composite score
    components: dict[str, float]         # {similarity, disconnection, velocity, authority}
    meets_threshold: bool               # g_score >= calibrated τ


class HypothesisPitchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    review_id: int
    title: str
    claim: str
    method_sketch: str
    datasets: list[str]
    supporting_paper_ids: list[str]
    disclaimer: str


class ReviewItemOut(BaseModel):
    id: int
    gap_a_label: str
    gap_b_label: str
    title: str
    claim: str
    method_sketch: str
    datasets: list[str]
    supporting_paper_ids: list[str]
    status: str
    created_at: str


class GapAnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    node2c: dict[str, int]
    communities: dict[int, CommunityOut]
    gaps: list[GapCandidateOut]


class ConceptCommunityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    cid: int
    members: list[str]
    size: int


class ConceptCentralityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    concept: str
    pagerank: float
    degree: int
    weighted_degree: int


class AgentRunIn(BaseModel):
    paper_ids: list[str] | None = None
    limit: int | None = None


class AgentRunOut(BaseModel):
    run_id: str


# --- App -----------------------------------------------------------------------
app = FastAPI(
    title="Axiom API",
    version="0.1.0",
    description="REST seam over axiom/*: search, trends (OD10), citation graph + gaps (OD9).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lazy singletons (mirrors app/streamlit_app.py's cached-singleton pattern,
# using lru_cache in place of st.cache_resource/st.cache_data) -----------------
@lru_cache(maxsize=1)
def get_encoder() -> Specter2Encoder:
    return Specter2Encoder()


@lru_cache(maxsize=1)
def get_store() -> AxiomQdrant:
    return AxiomQdrant()


@lru_cache(maxsize=1)
def get_graph():
    return graphmod.load_graph()


@lru_cache(maxsize=1)
def get_gap_analysis():
    conn = db.connect()
    try:
        return gapsmod.analyze(get_graph(), conn, get_store().fetch_dense_vectors())
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_concept_analysis() -> concept_graph.ConceptGraphAnalysis:
    """R7: Louvain + PageRank over the agent-extracted concept co-occurrence
    graph (OD19/OD20) -- separate from get_graph()/get_gap_analysis() above,
    which stay on the citation graph. Cached like the others; a fresh run's
    new concept_cooccurrence rows need a process restart (or a future
    explicit invalidation hook) to show up here, same tradeoff OD12 already
    accepted for get_graph()/get_gap_analysis().
    """
    conn = db.connect()
    try:
        return concept_graph.analyze(conn)
    finally:
        conn.close()


def _year_range(year_from: int | None, year_to: int | None) -> tuple[int, int] | None:
    if year_from is None or year_to is None:
        return None
    return (min(year_from, year_to), max(year_from, year_to))


# --- Routes --------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        count = get_store().count()
    except Exception:
        raise HTTPException(status_code=503, detail="Qdrant unreachable")
    return {"status": "ok", "collection": config.COLLECTION_NAME, "points": count}


@app.get("/search", response_model=list[SearchHitOut])
def search(
    q: str,
    top_k: int = Query(config.DEFAULT_TOP_K, ge=1, le=200),
    venues: list[str] | None = Query(None),
    year_from: int | None = None,
    year_to: int | None = None,
    mode: str = Query("hybrid", pattern="^(hybrid|dense)$"),
):
    store = get_store()
    year_range = _year_range(year_from, year_to)
    vec = get_encoder().encode_query(q)
    if mode == "hybrid" and store.supports_hybrid():
        hits = store.search_hybrid(query_vector=vec, query_text=q, top_k=top_k,
                                    venues=venues, year_range=year_range)
    else:
        hits = store.search(query_vector=vec, top_k=top_k,
                             venues=venues, year_range=year_range)
    return [SearchHitOut.model_validate(h) for h in hits]


@app.get("/papers/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str):
    conn = db.connect()
    try:
        rows = db.papers_by_ids(conn, [paper_id])
    finally:
        conn.close()
    row = rows.get(paper_id)
    if row is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return PaperOut(
        paper_id=row["openalex_id"], title=row["title"], abstract=row["abstract"],
        year=row["publication_year"], venue=row["venue"],
        cited_by_count=row["cited_by_count"], doi=row["doi"],
    )


@app.get("/papers/{paper_id}/similar", response_model=list[SearchHitOut])
def similar_papers(paper_id: str, top_k: int = Query(config.DEFAULT_TOP_K, ge=1, le=200)):
    hits = get_store().similar_papers(paper_id, top_k=top_k)
    return [SearchHitOut.model_validate(h) for h in hits]


@app.get("/trends/top", response_model=VelocityAnalysisOut)
def trends_top(
    n: int = Query(config.VELOCITY_TOP_K, ge=1, le=500),
    venue: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
):
    conn = db.connect()
    try:
        analysis = velocitymod.get_top_velocity_keywords(
            conn, n=n, venue=venue, year_range=_year_range(year_from, year_to)
        )
    finally:
        conn.close()
    return VelocityAnalysisOut.model_validate(analysis)


@app.get("/graph/stats")
def graph_stats():
    return graphmod.stats(get_graph())


@app.get("/graph/influence", response_model=list[PaperRankOut])
def graph_influence(top_k: int = Query(50, ge=1, le=500)):
    ranking = graphmod.influence(get_graph(), top_k=top_k)
    return [PaperRankOut.model_validate(r) for r in ranking]


@app.get("/graph/papers/{paper_id}/neighbors", response_model=NeighborsOut)
def graph_neighbors(paper_id: str):
    return NeighborsOut.model_validate(graphmod.neighbors(get_graph(), paper_id))


@app.get("/graph/gaps", response_model=GapAnalysisOut)
def graph_gaps():
    """Candidate research gaps (OD9): semantically-close, weakly-citing community pairs."""
    return GapAnalysisOut.model_validate(get_gap_analysis())


# --- Reading list (OD13) + cached LLM summaries (OD14) -----------------------
@app.get("/reading-list", response_model=list[BookmarkOut])
def reading_list():
    conn = db.connect()
    try:
        rows = db.list_bookmarks(conn)
    finally:
        conn.close()
    return [
        BookmarkOut(
            paper_id=r["paper_id"], added_at=r["added_at"], title=r["title"],
            abstract=r["abstract"], year=r["publication_year"], venue=r["venue"],
            cited_by_count=r["cited_by_count"], doi=r["doi"],
        )
        for r in rows
    ]


@app.post("/reading-list/{paper_id}", status_code=204)
def add_bookmark(paper_id: str):
    conn = db.connect()
    try:
        if not db.papers_by_ids(conn, [paper_id]):
            raise HTTPException(status_code=404, detail="paper not found")
        db.add_bookmark(conn, paper_id)
    finally:
        conn.close()


@app.delete("/reading-list/{paper_id}", status_code=204)
def remove_bookmark(paper_id: str):
    conn = db.connect()
    try:
        db.remove_bookmark(conn, paper_id)
    finally:
        conn.close()


@app.post("/reading-list/{paper_id}/summarize", response_model=SummaryOut)
def summarize(paper_id: str, force: bool = False):
    """3-bullet summary (OD14, local Ollama). Cached in SQLite; force=true to refresh."""
    conn = db.connect()
    try:
        if not force:
            cached = db.get_summary(conn, paper_id)
            if cached:
                return SummaryOut(paper_id=paper_id, bullets=cached, cached=True)

        rows = db.papers_by_ids(conn, [paper_id])
        row = rows.get(paper_id)
        if row is None:
            raise HTTPException(status_code=404, detail="paper not found")
        if not row["abstract"]:
            raise HTTPException(status_code=422, detail="paper has no abstract to summarize")

        try:
            result = summarizemod.summarize_paper(paper_id, row["title"], row["abstract"])
        except llm.OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        db.save_summary(conn, paper_id, result.bullets, config.OLLAMA_MODEL)
        return SummaryOut(paper_id=paper_id, bullets=result.bullets, cached=False)
    finally:
        conn.close()


# --- Hypothesis pitch + HITL review queue (OD16) ------------------------------
@app.post("/graph/gaps/{gap_index}/hypothesize", response_model=HypothesisPitchOut)
def hypothesize(gap_index: int):
    """Generate a grounded hypothesis pitch for one OD9 gap candidate.

    Stores the pitch in review_queue as `pending` -- never auto-promoted.
    `gap_index` is the position in `GET /graph/gaps`'s `gaps` list.
    """
    analysis = get_gap_analysis()
    if gap_index < 0 or gap_index >= len(analysis.gaps):
        raise HTTPException(status_code=404, detail="no gap candidate at that index")
    gap = analysis.gaps[gap_index]

    conn = db.connect()
    try:
        v = velocitymod.get_top_velocity_keywords(conn, n=10)
        trend_context = [k.concept for k in v.keywords if k.velocity > 0][:5]
    finally:
        conn.close()

    try:
        pitch = hypothesismod.generate_hypothesis(gap, get_graph(), trend_context=trend_context)
    except llm.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except hypothesismod.VerificationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    conn = db.connect()
    try:
        review_id = db.add_to_review_queue(
            conn, gap_a_label=gap.a.label, gap_b_label=gap.b.label,
            title=pitch.title, claim=pitch.claim, method_sketch=pitch.method_sketch,
            datasets=pitch.datasets, supporting_paper_ids=pitch.supporting_paper_ids,
        )
    finally:
        conn.close()

    return HypothesisPitchOut(
        review_id=review_id, title=pitch.title, claim=pitch.claim,
        method_sketch=pitch.method_sketch, datasets=pitch.datasets,
        supporting_paper_ids=pitch.supporting_paper_ids, disclaimer=pitch.disclaimer,
    )


@app.get("/review-queue", response_model=list[ReviewItemOut])
def review_queue(status: str | None = Query(None, pattern="^(pending|approved|rejected)$")):
    conn = db.connect()
    try:
        rows = db.list_review_queue(conn, status=status)
    finally:
        conn.close()
    return [
        ReviewItemOut(
            id=r["id"], gap_a_label=r["gap_a_label"], gap_b_label=r["gap_b_label"],
            title=r["title"], claim=r["claim"], method_sketch=r["method_sketch"],
            datasets=json.loads(r["datasets_json"]),
            supporting_paper_ids=json.loads(r["supporting_ids_json"]),
            status=r["status"], created_at=r["created_at"],
        )
        for r in rows
    ]


@app.post("/review-queue/{item_id}/approve", status_code=204)
def approve_review_item(item_id: int):
    conn = db.connect()
    try:
        db.set_review_status(conn, item_id, "approved")
    finally:
        conn.close()


@app.post("/review-queue/{item_id}/reject", status_code=204)
def reject_review_item(item_id: int):
    conn = db.connect()
    try:
        db.set_review_status(conn, item_id, "rejected")
    finally:
        conn.close()


# --- Agentic pipeline (R1-R6, docs/agentic/ARCHITECTURE.md) -------------------
# Async, unlike every route above (R5) -- the pipeline awaits LLM calls;
# everything else in this file stays CPU-bound sync `def` on purpose (OD12).
@app.post("/agents/run", response_model=AgentRunOut)
async def agents_run(body: AgentRunIn | None = None):
    body = body or AgentRunIn()
    conn = db.connect()
    try:
        if body.paper_ids:
            rows = db.papers_by_ids(conn, body.paper_ids)
        else:
            limit = body.limit or 50
            rows = {
                r["openalex_id"]: r for r in conn.execute(
                    "SELECT openalex_id, title, abstract FROM papers LIMIT ?", (limit,)
                ).fetchall()
            }
    finally:
        conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="no matching papers")

    papers = [
        agents_pipeline.PaperInput(paper_id=pid, title=r["title"], abstract=r["abstract"])
        for pid, r in rows.items()
    ]
    summary = await agents_pipeline.run_pipeline(papers, encoder=get_encoder())
    return AgentRunOut(run_id=summary["run_id"])


@app.get("/agents/runs/{run_id}", response_model=RunStatusOut)
def agents_run_status(run_id: str):
    conn = db.connect()
    try:
        status = agents_pipeline.get_run_status(conn, run_id)
    finally:
        conn.close()
    if status is None:
        raise HTTPException(status_code=404, detail="no such run")
    return status


@app.get("/concepts/communities", response_model=list[ConceptCommunityOut])
def concepts_communities():
    return get_concept_analysis().communities


@app.get("/concepts/central", response_model=list[ConceptCentralityOut])
def concepts_central():
    return get_concept_analysis().central
