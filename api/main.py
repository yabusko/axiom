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
import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from axiom import concept_graph, config, db, embedding_map as mapmod, fields as fieldsmod, llm_providers
from axiom import gaps as gapsmod
from axiom import graph as graphmod
from axiom import hypothesis as hypothesismod
from axiom import summarize as summarizemod
from axiom import velocity as velocitymod
from axiom.agents import pipeline as agents_pipeline
from axiom.agents import synthesis as synthesis_agent
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
    gap_id: str                          # B4: stable id, replaces list-position gap_index
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
    paper_ids: list[str] | None = Field(default=None, max_length=50)
    limit: int | None = Field(default=None, ge=1, le=50)
    synthesize: bool = False


class AgentRunOut(BaseModel):
    run_id: str


class AgentRunStatusOut(BaseModel):
    run_id: str
    status: str
    papers_total: int
    papers_done: int
    model_fast: str | None
    model_strong: str | None
    started_at: str | None
    finished_at: str | None
    verdict_counts: dict[str, int] = {}
    cost_so_far: float = 0.0


# --- Fields (stage-3 product redesign, docs/product/ARCHITECTURE_NOTE.md) -----
class FieldCreateIn(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    # Stage 4 (A4): a field is now a relevance slice of the shared base
    # corpus, computed synchronously -- `target` overrides the default slice
    # size (config.FIELD_SLICE_TARGET), it no longer sizes a download.
    target: int = Field(default=config.FIELD_SLICE_TARGET, ge=10, le=1000)


class FieldCreateOut(BaseModel):
    field_id: str
    status: str                  # ready | thin (A5) -- instant, no more queued/building here
    papers_total: int
    total_relevant: int          # how many papers in the WHOLE corpus cleared the floor
    extend_command: str | None   # A5: the exact CLI command to grow coverage, only when thin


class FieldOut(BaseModel):
    id: str
    topic: str
    status: str
    papers_total: int | None
    last_opened_at: str | None


class CorpusManifestOut(BaseModel):
    papers_count: int
    query: str
    from_year: int
    to_year: int
    updated_at: str
    spend_total_usd: float
    budget_usd: float             # B4: the cap scripts/build_corpus.py ran under
    extraction_complete: bool     # False if the last run stopped on 'partial' (budget reached)


class FieldStatusOut(BaseModel):
    id: str
    topic: str
    status: str                 # queued | building | ready | thin | failed
    step: str | None            # fetching | embedding | extracting | None
    papers_total: int | None
    papers_done: int
    cost_so_far: float
    error: str | None
    created_at: str


class TopicOut(BaseModel):
    cid: int
    labels: list[str]
    size: int


class OverviewOut(BaseModel):
    field_id: str
    topic: str
    papers: int
    in_corpus_citations: int
    topic_communities: int
    candidate_gaps: int
    topics: list[TopicOut]
    rising: list[KeywordVelocityOut]
    key_papers: list[PaperRankOut]


class SideDetailOut(BaseModel):
    cid: int                     # community id -- lets the frontend link to Map's ?a=/?b=
    labels: list[str]
    size: int
    top_papers: list[PaperRankOut]
    methods: list[str]
    datasets: list[str]


class GapDetailOut(BaseModel):
    gap_id: str                          # B4: stable id, replaces list-position gap_index
    a: SideDetailOut
    b: SideDetailOut
    semantic_similarity: float
    inter_citations: int
    total_in_corpus_citations: int
    gap_score: float
    g_score: float
    components: dict[str, float]
    meets_threshold: bool


class EntityOut(BaseModel):
    name: str
    kind: str
    canonical: str
    evidence_span: str
    confidence: float


class FieldPaperOut(BaseModel):
    paper_id: str
    title: str | None
    abstract: str | None
    year: int | None
    venue: str | None
    cited_by_count: int
    doi: str | None
    entities: list[EntityOut]


# --- Map screen (stage-3 product redesign) ------------------------------------
class MapNodeOut(BaseModel):
    paper_id: str
    title: str | None
    cid: int                     # community id, for topic coloring
    x: float
    y: float
    z: float


class MapTopicOut(BaseModel):
    cid: int
    label: str
    size: int                    # papers from this topic present in the map (post-cap)


class MapOut(BaseModel):
    nodes: list[MapNodeOut]
    edges: list[tuple[str, str]]
    topics: list[MapTopicOut]
    total_papers: int            # papers in the field, before the node cap
    capped: bool                 # True if total_papers > len(nodes)


class MapSimilarityOut(BaseModel):
    a_label: str
    b_label: str
    cosine: float
    explanation: str


# draft/kept/discarded is the product-facing vocabulary (PRODUCT.md); the
# underlying review_queue.status column keeps its OD16 HITL values
# (pending/approved/rejected) -- translated at this boundary only, so the
# approve/reject HITL code paths above stay untouched.
_SHORTLIST_TO_DB = {"draft": "pending", "kept": "approved", "discarded": "rejected"}
_DB_TO_SHORTLIST = {v: k for k, v in _SHORTLIST_TO_DB.items()}


class ShortlistItemOut(BaseModel):
    id: int
    gap_a_label: str
    gap_b_label: str
    title: str
    claim: str
    method_sketch: str
    datasets: list[str]
    supporting_paper_ids: list[str]
    status: str
    notes: str | None
    papers_to_read: list[str]
    created_at: str


class ShortlistCreateIn(BaseModel):
    gap_id: str = Field(min_length=1)   # B4: stable id, replaces list-position gap_index


class ShortlistPatchIn(BaseModel):
    status: str | None = Field(default=None, pattern="^(draft|kept|discarded)$")
    notes: str | None = None
    papers_to_read: list[str] | None = None


# --- App -----------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    # D2: load the sentence encoder once at process startup instead of inside the
    # first request that happens to need it -- that first caller shouldn't pay a
    # multi-second model-load latency hit just for being first.
    get_encoder()
    # Stage-3 fields registry: ensure its db exists and the pre-existing corpus
    # is registered as field "default" (ARCHITECTURE_NOTE.md §1) -- idempotent.
    fconn = fieldsmod.connect_fields()
    try:
        fieldsmod.init_fields_db(fconn)
        fieldsmod.ensure_default_field(fconn)
        # B4: a field left 'queued'/'building' belonged to a background task
        # that died with whatever process crashed/restarted last -- it will
        # never finish on its own; say so instead of leaving it stuck.
        fieldsmod.mark_interrupted_jobs_failed(fconn)
    finally:
        fconn.close()
    yield


app = FastAPI(
    title="Axiom API",
    version="0.1.0",
    description="REST seam over axiom/*: search, trends (OD10), citation graph + gaps (OD9).",
    lifespan=_lifespan,
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


# --- Stage 4: the shared base corpus (A4 instant fields) -----------------------
# Separate seam from get_graph()/get_store() above (those stay pointed at the
# legacy "default" field's DB_PATH/COLLECTION_NAME) -- process-wide singletons
# cached for the process lifetime, same rationale as the rest of this block:
# a field slice needs to be fast (p50 < 5s, A7), and the base corpus only
# changes via an explicit scripts/build_corpus.py run (which requires a
# process restart to pick up, same limitation get_graph()/get_store() already
# have for the "default" field).
@lru_cache(maxsize=1)
def get_corpus_graph():
    return graphmod.load_graph(db.connect(config.CORPUS_DB_PATH))


@lru_cache(maxsize=1)
def get_corpus_store() -> AxiomQdrant:
    return AxiomQdrant(collection=config.CORPUS_COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_corpus_vectors() -> dict[str, list[float]]:
    return get_corpus_store().fetch_dense_vectors()


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
    """3-bullet summary (OD14). Cached in SQLite; force=true to refresh."""
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
        except llm_providers.LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        db.save_summary(conn, paper_id, result.bullets, llm_providers.model_for_tier("fast"))
        return SummaryOut(paper_id=paper_id, bullets=result.bullets, cached=False)
    finally:
        conn.close()


# --- Hypothesis pitch + HITL review queue (OD16) ------------------------------
# D4: this calls axiom.agents.synthesis.generate_hypothesis -- the one real
# implementation (strong tier, via llm_providers, real spend tracking) -- not the
# older axiom.hypothesis.generate_hypothesis (Ollama-only) this replaced. Async,
# like the other LLM-calling routes in this file (OD12), unlike the rest.
@app.post("/graph/gaps/{gap_index}/hypothesize", response_model=HypothesisPitchOut)
async def hypothesize(gap_index: int):
    """Generate a grounded hypothesis pitch for one OD9 gap candidate.

    Stores the pitch in review_queue as `pending` -- never auto-promoted.
    `gap_index` is the position in `GET /graph/gaps`'s `gaps` list.
    """
    analysis = await run_in_threadpool(get_gap_analysis)  # B4: cheap on cache hit, real work on miss
    if gap_index < 0 or gap_index >= len(analysis.gaps):
        raise HTTPException(status_code=404, detail="no gap candidate at that index")
    gap = analysis.gaps[gap_index]

    def _trend_context() -> list[str]:
        conn = db.connect()
        try:
            v = velocitymod.get_top_velocity_keywords(conn, n=10)
            return [k.concept for k in v.keywords if k.velocity > 0][:5]
        finally:
            conn.close()

    trend_context = await run_in_threadpool(_trend_context)

    conn = db.connect()
    try:
        try:
            pitch = await synthesis_agent.generate_hypothesis(
                gap, get_graph(), run_id=uuid.uuid4().hex, trend_context=trend_context, conn=conn,
            )
        except llm_providers.LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except hypothesismod.VerificationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

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
async def _run_agents_background(
    papers: list[agents_pipeline.PaperInput], run_id: str, *,
    clusters: list[dict] | None, gaps: list, trend_context: list[str] | None,
) -> None:
    """D1: the actual pipeline run, scheduled as a background task so the request
    handler can return 202 immediately. Mirrors run_pipeline's own try/finally shape
    for the one failure mode it can't cover itself -- an exception raised before (or
    after) its own agent_runs bookkeeping, which would otherwise leave the row stuck
    at 'running' forever with no record of what happened."""
    try:
        await agents_pipeline.run_pipeline(
            papers, encoder=get_encoder(), run_id=run_id,
            clusters=clusters, gaps=gaps, trend_context=trend_context,
        )
        get_concept_analysis.cache_clear()
    except Exception as exc:
        conn = db.connect()
        try:
            db.fail_agent_run(conn, run_id, error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        finally:
            conn.close()


@app.post("/agents/run", response_model=AgentRunOut, status_code=202)
def agents_run(body: AgentRunIn | None, background_tasks: BackgroundTasks):
    body = body or AgentRunIn()
    conn = db.connect()
    try:
        if body.paper_ids:
            rows = db.papers_by_ids(conn, body.paper_ids)
        else:
            rows = db.recent_papers(conn, body.limit or 50)
    finally:
        conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="no matching papers")

    papers = [
        agents_pipeline.PaperInput(paper_id=pid, title=r["title"], abstract=r["abstract"])
        for pid, r in rows.items()
    ]

    clusters: list[dict] | None = None
    gaps: list | None = None
    trend_context: list[str] | None = None
    if body.synthesize:
        concept_analysis = get_concept_analysis()
        clusters = [
            {"cluster_id": c.cid, "concepts": c.members} for c in concept_analysis.communities
        ]
        citation_graph = get_graph()
        gaps = [(g, citation_graph) for g in get_gap_analysis().gaps[:3]]
        conn = db.connect()
        try:
            v = velocitymod.get_top_velocity_keywords(conn, n=10)
            trend_context = [k.concept for k in v.keywords if k.velocity > 0][:5]
        finally:
            conn.close()

    run_id = uuid.uuid4().hex
    background_tasks.add_task(
        _run_agents_background, papers, run_id,
        clusters=clusters, gaps=gaps, trend_context=trend_context,
    )
    return AgentRunOut(run_id=run_id)


@app.get("/agents/runs/{run_id}", response_model=AgentRunStatusOut)
def agents_run_status(run_id: str):
    conn = db.connect()
    try:
        row = db.get_agent_run(conn, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no such run")
        verdict_counts = db.agent_run_verdict_counts(conn, run_id)
        cost_so_far = db.run_llm_spend(conn, run_id)
    finally:
        conn.close()
    return AgentRunStatusOut(**dict(row), verdict_counts=verdict_counts, cost_so_far=cost_so_far)


@app.get("/concepts/communities", response_model=list[ConceptCommunityOut])
def concepts_communities():
    return get_concept_analysis().communities


@app.get("/concepts/central", response_model=list[ConceptCentralityOut])
def concepts_central():
    return get_concept_analysis().central


# --- Fields (stage-3 product redesign, docs/product/ARCHITECTURE_NOTE.md) -----
# No field_id-keyed caching yet (deliberate scope cut for this pass, logged in
# .claude/product/PROGRESS.md): every field-scoped read below recomputes from
# that field's own db/collection per request. ARCHITECTURE_NOTE.md's
# alternative -- keying the existing lru_cache singletons above by field_id --
# is a reasonable follow-up once there's more than one field to actually
# notice the recompute cost on; a single researcher's few-hundred-paper
# corpus analyzes in low single-digit seconds, not a latency problem yet.

def _resolve_field(field_id: str) -> sqlite3.Row:
    fconn = fieldsmod.connect_fields()
    try:
        row = fieldsmod.get_field(fconn, field_id)
    finally:
        fconn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="no such field")
    return row


def _find_gap(analysis, gap_id: str):
    """B4: look up a gap candidate by its stable gap_id (a hash of both sides'
    member sets -- see GapCandidate.gap_id), replacing the old list-position
    gap_index lookup everywhere a gap is addressed from outside analyze()'s
    own ranked list (a URL, a shortlist request body)."""
    for candidate in analysis.gaps:
        if candidate.gap_id == gap_id:
            return candidate
    raise HTTPException(status_code=404, detail=f"no gap candidate with id {gap_id!r}")


def _slice_paper_ids(row: sqlite3.Row) -> set[str] | None:
    """A4: a 'slice' field is a (topic, paper_ids) pair over the SHARED base
    corpus, not its own download -- None for a legacy/'download' field (no
    paper_ids_json), which reads its own private db_path/collection whole."""
    raw = row["paper_ids_json"] if "paper_ids_json" in row.keys() else None
    return set(json.loads(raw)) if raw else None


def _field_graph(row: sqlite3.Row):
    ids = _slice_paper_ids(row)
    if ids is not None:
        # Subgraph induced by the slice's own members only -- a citation to a
        # paper outside the slice drops out entirely (not even a dangling
        # node), same effect as "outside the corpus" has elsewhere: structure
        # and communities are scoped to what this field actually contains.
        return get_corpus_graph().subgraph(ids).copy()
    conn = db.connect(row["db_path"])
    try:
        return graphmod.load_graph(conn)
    finally:
        conn.close()


def _field_store(row: sqlite3.Row) -> AxiomQdrant:
    """Separate seam from get_store() (the "default"/legacy global singleton)
    so tests can monkeypatch a fake store per field without a real Qdrant
    connection, same pattern as get_store() itself."""
    return AxiomQdrant(collection=row["collection"])


def _field_vectors(row: sqlite3.Row) -> dict[str, list[float]]:
    """A slice field reuses the cached corpus-wide vector set (no live Qdrant
    scroll per field -- the point of A4's "instant") rather than calling
    `_field_store(row).fetch_dense_vectors()`, which a legacy/download field
    (its own small collection) still does live, same as before stage 4."""
    if _slice_paper_ids(row) is not None:
        return get_corpus_vectors()
    return _field_store(row).fetch_dense_vectors()


def _field_gap_analysis(row: sqlite3.Row, g=None):
    g = g or _field_graph(row)
    conn = db.connect(row["db_path"])
    try:
        return gapsmod.analyze(g, conn, _field_vectors(row))
    finally:
        conn.close()


# B4: "results cached per field/slice" -- a field's graph + gap analysis are
# recomputed from scratch on nearly every field endpoint (overview, gaps,
# gap detail, map, map/similarity, shortlist create); a slice field's
# paper_ids never change after creation (A4) and a legacy field is only ever
# queried by these endpoints once its build reaches 'ready', so caching by
# field_id alone is safe -- same precedent `_field_map_data` below already
# set. `maxsize=32` keyed by field_id, not unbounded, so a long-running
# server building many fields over time doesn't leak memory per field forever.
@lru_cache(maxsize=32)
def _cached_field_graph(field_id: str):
    return _field_graph(_resolve_field(field_id))


@lru_cache(maxsize=32)
def _cached_field_gap_analysis(field_id: str):
    row = _resolve_field(field_id)
    return _field_gap_analysis(row, _cached_field_graph(field_id))


@lru_cache(maxsize=8)
def _field_map_data(field_id: str) -> MapOut:
    """Nodes/edges/topics for the Map screen, capped at `config.MAP_NODE_CAP`
    by PageRank (same ranking `graphmod.influence`/`top_influential_subgraph`
    already use elsewhere) so a laptop GPU can render the 3D view. PCA is
    computed once over just the capped set and cached here per field_id --
    deterministic given the same vectors (`embedding_map.project`), so this is
    safe to cache indefinitely for a field that isn't being rebuilt.
    """
    row = _resolve_field(field_id)
    g = _cached_field_graph(field_id)
    analysis = _cached_field_gap_analysis(field_id)
    total_papers = graphmod.stats(g)["papers"]

    capped = graphmod.top_influential_subgraph(g, n=config.MAP_NODE_CAP)
    node_ids = list(capped.nodes())

    vectors = _field_vectors(row)
    coords = mapmod.project({pid: vectors[pid] for pid in node_ids if pid in vectors})

    nodes = [
        MapNodeOut(
            paper_id=pid, title=g.nodes[pid].get("title"),
            cid=analysis.node2c.get(pid, -1),
            x=coords.get(pid, [0.0, 0.0, 0.0])[0],
            y=coords.get(pid, [0.0, 0.0, 0.0])[1],
            z=coords.get(pid, [0.0, 0.0, 0.0])[2],
        )
        for pid in node_ids
    ]
    edges = [(u, v) for u, v in capped.edges()]

    sizes: dict[int, int] = {}
    for n in nodes:
        sizes[n.cid] = sizes.get(n.cid, 0) + 1
    topics = [
        MapTopicOut(cid=cid, label=analysis.communities[cid].label if cid in analysis.communities else f"cluster {cid}", size=size)
        for cid, size in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return MapOut(nodes=nodes, edges=edges, topics=topics, total_papers=total_papers, capped=total_papers > len(nodes))


def _entities_for_papers(conn, paper_ids: list[str], kinds: tuple[str, ...]) -> list[str]:
    return db.concepts_for_papers(conn, paper_ids, kinds)


def _top_papers(g, member_ids: list[str], top_k: int = 5) -> list:
    ranked = sorted(
        member_ids, key=lambda pid: g.nodes[pid].get("cited_by_count", 0) if pid in g.nodes else 0,
        reverse=True,
    )[:top_k]
    out = []
    for pid in ranked:
        node = g.nodes.get(pid, {}) if pid in g.nodes else {}
        out.append(graphmod.PaperRank(
            paper_id=pid, title=node.get("title"), year=node.get("year"),
            venue=node.get("venue"), cited_by_count=node.get("cited_by_count", 0),
            local_in_degree=g.in_degree(pid) if pid in g.nodes else 0, pagerank=0.0,
        ))
    return out


@app.post("/fields", response_model=FieldCreateOut)
def create_field(body: FieldCreateIn):
    """A4: instant -- a relevance-ranked slice of the shared base corpus,
    computed here synchronously (no download, no background task). A5: a
    thin-coverage topic still gets a field row (status='thin'), with the
    exact `--extend` command to grow the base corpus around it."""
    query_vector = get_encoder().encode_query(body.topic)
    fconn = fieldsmod.connect_fields()
    try:
        row = fieldsmod.create_instant_field(
            fconn, topic=body.topic, corpus_graph=get_corpus_graph(),
            corpus_vectors=get_corpus_vectors(), query_vector=query_vector,
            target_n=body.target,
        )
    finally:
        fconn.close()
    extend_command = (
        f'python scripts/build_corpus.py --extend "{body.topic}"' if row["status"] == "thin" else None
    )
    return FieldCreateOut(
        field_id=row["id"], status=row["status"], papers_total=row["papers_total"] or 0,
        total_relevant=row["total_relevant"] or 0, extend_command=extend_command,
    )


@app.get("/corpus", response_model=CorpusManifestOut)
def corpus_manifest():
    """A6: the Start screen headline ("Instant search over N papers...") and
    the app footer both read this -- the one place the shared base corpus's
    own build manifest (scripts/build_corpus.py, corpus_meta table) surfaces
    in the product."""
    conn = db.connect(config.CORPUS_DB_PATH)
    try:
        meta = db.get_corpus_meta(conn)
    finally:
        conn.close()
    if meta is None:
        raise HTTPException(status_code=404, detail="base corpus not built yet")
    filters = json.loads(meta["filters_json"])
    steps = json.loads(meta["steps_json"])
    return CorpusManifestOut(
        papers_count=meta["papers_count"], query=meta["query"],
        from_year=int(filters["from_publication_date"][:4]),
        to_year=int(filters["to_publication_date"][:4]),
        updated_at=meta["updated_at"], spend_total_usd=meta["spend_total"],
        budget_usd=meta["budget_usd"],
        extraction_complete=steps.get("extract") == "done",
    )


@app.get("/fields", response_model=list[FieldOut])
def list_fields():
    fconn = fieldsmod.connect_fields()
    try:
        rows = fieldsmod.list_fields(fconn)
    finally:
        fconn.close()
    return [FieldOut(**dict(r)) for r in rows]


@app.get("/fields/{field_id}", response_model=FieldStatusOut)
def field_status(field_id: str):
    """B4: read-only -- this is polled (FieldNav, build-progress) and a GET
    must stay safe/idempotent. `last_opened_at` updates only via the explicit
    POST /fields/{id}/open below, not as a side effect of every poll."""
    fconn = fieldsmod.connect_fields()
    try:
        out = fieldsmod.live_status(fconn, field_id)
        if out is None:
            raise HTTPException(status_code=404, detail="no such field")
    finally:
        fconn.close()
    return FieldStatusOut(**out)


@app.post("/fields/{field_id}/open", status_code=204)
def field_open(field_id: str):
    """The one explicit 'this field was opened' signal -- call once when a
    user actually navigates into a field, not on every status poll. Drives
    `list_fields`'s "Recent fields" ordering (`COALESCE(last_opened_at,
    created_at) DESC`)."""
    fconn = fieldsmod.connect_fields()
    try:
        if fieldsmod.get_field(fconn, field_id) is None:
            raise HTTPException(status_code=404, detail="no such field")
        fieldsmod.touch_last_opened(fconn, field_id)
    finally:
        fconn.close()


@app.delete("/fields/{field_id}", status_code=204)
def field_delete(field_id: str):
    """B4: delete a field (a saved query). The default field can't be
    deleted -- see `fieldsmod.delete_field`'s docstring."""
    fconn = fieldsmod.connect_fields()
    try:
        try:
            found = fieldsmod.delete_field(fconn, field_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not found:
            raise HTTPException(status_code=404, detail="no such field")
    finally:
        fconn.close()
    # No per-field cache eviction needed: field_ids are fresh UUIDs, never
    # reused, so a deleted field's cached graph/analysis entry (if any) just
    # becomes harmless dead weight until LRU eviction -- no endpoint can
    # reach it again (_resolve_field 404s first). Clearing the whole cache
    # would only punish every OTHER field's hit rate for no benefit.


@app.get("/fields/{field_id}/overview", response_model=OverviewOut)
def field_overview(field_id: str):
    row = _resolve_field(field_id)
    g = _cached_field_graph(field_id)
    gap_analysis = _cached_field_gap_analysis(field_id)
    stats = graphmod.stats(g)
    conn = db.connect(row["db_path"])
    try:
        velocity_analysis = velocitymod.get_top_velocity_keywords(conn, n=8, paper_ids=_slice_paper_ids(row))
    finally:
        conn.close()
    rising = [k for k in velocity_analysis.keywords if k.velocity > 0][:6]
    key_papers = graphmod.influence(g, top_k=6)
    topics = sorted(gap_analysis.communities.values(), key=lambda c: c.size, reverse=True)
    return OverviewOut(
        field_id=field_id, topic=row["topic"], papers=stats["papers"],
        in_corpus_citations=stats["edges_in_corpus"],
        topic_communities=len(gap_analysis.communities), candidate_gaps=len(gap_analysis.gaps),
        topics=[TopicOut(cid=c.cid, labels=c.labels, size=c.size) for c in topics],
        rising=[KeywordVelocityOut.model_validate(k) for k in rising],
        key_papers=[PaperRankOut.model_validate(p) for p in key_papers],
    )


@app.get("/fields/{field_id}/gaps", response_model=GapAnalysisOut)
def field_gaps(field_id: str):
    _resolve_field(field_id)  # 404s before touching the cache
    return GapAnalysisOut.model_validate(_cached_field_gap_analysis(field_id))


@app.get("/fields/{field_id}/gaps/{gap_id}", response_model=GapDetailOut)
def field_gap_detail(field_id: str, gap_id: str):
    row = _resolve_field(field_id)
    g = _cached_field_graph(field_id)
    analysis = _cached_field_gap_analysis(field_id)
    gap = _find_gap(analysis, gap_id)
    stats = graphmod.stats(g)

    conn = db.connect(row["db_path"])
    try:
        sides = []
        for community in (gap.a, gap.b):
            top_papers = _top_papers(g, community.members)
            sides.append(SideDetailOut(
                cid=community.cid, labels=community.labels, size=community.size, top_papers=[
                    PaperRankOut.model_validate(p) for p in top_papers
                ],
                methods=_entities_for_papers(conn, community.members, ("method",)),
                datasets=_entities_for_papers(conn, community.members, ("dataset",)),
            ))
    finally:
        conn.close()

    return GapDetailOut(
        gap_id=gap.gap_id, a=sides[0], b=sides[1],
        semantic_similarity=gap.semantic_similarity, inter_citations=gap.inter_citations,
        total_in_corpus_citations=stats["edges_in_corpus"],
        gap_score=gap.gap_score, g_score=gap.g_score, components=gap.components,
        meets_threshold=gap.meets_threshold,
    )


@app.get("/fields/{field_id}/map", response_model=MapOut)
def field_map(field_id: str):
    _resolve_field(field_id)  # 404s before touching the cache
    return _field_map_data(field_id)


@app.get("/fields/{field_id}/map/similarity", response_model=MapSimilarityOut)
def field_map_similarity(
    field_id: str,
    a_kind: str = Query(pattern="^(paper|topic)$"),
    a_id: str = Query(min_length=1),
    b_kind: str = Query(pattern="^(paper|topic)$"),
    b_id: str = Query(min_length=1),
):
    row = _resolve_field(field_id)
    g = _cached_field_graph(field_id)
    analysis = _cached_field_gap_analysis(field_id)
    vectors = _field_vectors(row)

    def resolve(kind: str, ident: str) -> tuple[list[float], str]:
        if kind == "paper":
            if ident not in vectors:
                raise HTTPException(status_code=404, detail=f"no vector for paper {ident!r}")
            return vectors[ident], g.nodes.get(ident, {}).get("title") or ident
        try:
            cid = int(ident)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"topic id must be an integer, got {ident!r}")
        community = analysis.communities.get(cid)
        if community is None:
            raise HTTPException(status_code=404, detail=f"no topic community {cid}")
        member_vectors = [vectors[pid] for pid in community.members if pid in vectors]
        if not member_vectors:
            raise HTTPException(status_code=422, detail=f"topic {cid} has no embedded papers")
        centroid = np.asarray(member_vectors, dtype=float).mean(axis=0).tolist()
        return centroid, community.label

    vec_a, label_a = resolve(a_kind, a_id)
    vec_b, label_b = resolve(b_kind, b_id)
    cos = mapmod.cosine(vec_a, vec_b)
    return MapSimilarityOut(a_label=label_a, b_label=label_b, cosine=cos, explanation=mapmod.explain(cos))


@app.get("/fields/{field_id}/papers/{paper_id}", response_model=FieldPaperOut)
def field_paper_detail(field_id: str, paper_id: str):
    row = _resolve_field(field_id)
    conn = db.connect(row["db_path"])
    try:
        rows = db.papers_by_ids(conn, [paper_id])
        paper = rows.get(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found")
        entity_rows = db.paper_entities(conn, paper_id)
    finally:
        conn.close()
    return FieldPaperOut(
        paper_id=paper["openalex_id"], title=paper["title"], abstract=paper["abstract"],
        year=paper["publication_year"], venue=paper["venue"],
        cited_by_count=paper["cited_by_count"], doi=paper["doi"],
        entities=[EntityOut(**dict(r)) for r in entity_rows],
    )


@app.get("/fields/{field_id}/shortlist", response_model=list[ShortlistItemOut])
def field_shortlist(field_id: str, status: str | None = Query(None, pattern="^(draft|kept|discarded)$")):
    row = _resolve_field(field_id)
    db_status = _SHORTLIST_TO_DB.get(status) if status else None
    conn = db.connect(row["db_path"])
    try:
        rows = db.list_review_queue(conn, status=db_status)
    finally:
        conn.close()
    return [_shortlist_item_out(r) for r in rows]


def _shortlist_item_out(r: sqlite3.Row) -> ShortlistItemOut:
    return ShortlistItemOut(
        id=r["id"], gap_a_label=r["gap_a_label"], gap_b_label=r["gap_b_label"],
        title=r["title"], claim=r["claim"], method_sketch=r["method_sketch"],
        datasets=json.loads(r["datasets_json"]),
        supporting_paper_ids=json.loads(r["supporting_ids_json"]),
        status=_DB_TO_SHORTLIST.get(r["status"], r["status"]),
        notes=r["notes"],
        papers_to_read=json.loads(r["papers_to_read_json"]) if r["papers_to_read_json"] else [],
        created_at=r["created_at"],
    )


@app.post("/fields/{field_id}/shortlist", response_model=ShortlistItemOut)
async def field_shortlist_create(field_id: str, body: ShortlistCreateIn):
    """Combines what the current app does as two independent call sites
    (AUDIT.md: Streamlit's inline button and POST /graph/gaps/{i}/hypothesize
    both call generate_hypothesis + add_to_review_queue separately) into one."""
    row = _resolve_field(field_id)
    # B4: no heavy sync work inside this async def -- on a cache miss, a full
    # graph load + gap analysis (community detection, concept-label queries)
    # over a 1000-paper corpus is real CPU/IO time that would otherwise block
    # every other request this process is serving. Each threadpooled callable
    # is fully self-contained (opens and closes its own sqlite connection) --
    # never hands a connection across the thread boundary, since sqlite3
    # connections are thread-affine (same caveat axiom/fields.py's build job
    # already documents).
    g, analysis = await run_in_threadpool(
        lambda: (_cached_field_graph(field_id), _cached_field_gap_analysis(field_id))
    )
    gap = _find_gap(analysis, body.gap_id)

    def _trend_context() -> list[str]:
        conn = db.connect(row["db_path"])
        try:
            v = velocitymod.get_top_velocity_keywords(conn, n=10, paper_ids=_slice_paper_ids(row))
            return [k.concept for k in v.keywords if k.velocity > 0][:5]
        finally:
            conn.close()

    trend_context = await run_in_threadpool(_trend_context)

    conn = db.connect(row["db_path"])
    try:
        try:
            pitch = await synthesis_agent.generate_hypothesis(
                gap, g, run_id=uuid.uuid4().hex, trend_context=trend_context, conn=conn,
            )
        except llm_providers.LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except hypothesismod.VerificationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        review_id = db.add_to_review_queue(
            conn, gap_a_label=gap.a.label, gap_b_label=gap.b.label,
            title=pitch.title, claim=pitch.claim, method_sketch=pitch.method_sketch,
            datasets=pitch.datasets, supporting_paper_ids=pitch.supporting_paper_ids,
        )
        item = db.get_review_item(conn, review_id)
    finally:
        conn.close()
    return _shortlist_item_out(item)


@app.patch("/fields/{field_id}/shortlist/{item_id}", response_model=ShortlistItemOut)
def field_shortlist_patch(field_id: str, item_id: int, body: ShortlistPatchIn):
    row = _resolve_field(field_id)
    conn = db.connect(row["db_path"])
    try:
        if db.get_review_item(conn, item_id) is None:
            raise HTTPException(status_code=404, detail="no such shortlist item")
        db.update_review_item(
            conn, item_id,
            status=_SHORTLIST_TO_DB.get(body.status) if body.status else None,
            notes=body.notes, papers_to_read=body.papers_to_read,
        )
        item = db.get_review_item(conn, item_id)
    finally:
        conn.close()
    return _shortlist_item_out(item)


@app.get("/fields/{field_id}/shortlist/export.md")
def field_shortlist_export(field_id: str):
    row = _resolve_field(field_id)
    conn = db.connect(row["db_path"])
    try:
        kept = db.list_review_queue(conn, status="approved")
    finally:
        conn.close()
    lines = [f"# {row['topic']} — shortlist", ""]
    if not kept:
        lines.append("_No kept directions yet._")
    for r in kept:
        lines.append(f"## {r['title']}")
        lines.append("")
        lines.append(r["claim"])
        lines.append("")
        lines.append(f"**Method sketch:** {r['method_sketch']}")
        datasets = json.loads(r["datasets_json"])
        if datasets:
            lines.append(f"**Datasets:** {', '.join(datasets)}")
        supporting = json.loads(r["supporting_ids_json"])
        if supporting:
            lines.append(f"**Supporting papers:** {', '.join(supporting)}")
        if r["notes"]:
            lines.append(f"**Notes:** {r['notes']}")
        lines.append("")
        lines.append("_AI-generated, unverified — review before relying on it._")
        lines.append("")
    return PlainTextResponse("\n".join(lines), media_type="text/markdown")
