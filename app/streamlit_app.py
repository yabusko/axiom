"""Axiom Streamlit UI.

Demo tab order: Trending → Research gaps → Search → Reading list →
Review queue → Vector geometry. Visual polish (navy theme, hero, chips,
empty states) is from the ui-ux-polish work.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import html
import json
import re
import sys
import uuid
from pathlib import Path

# Make the repo root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Make the app/ directory importable so sibling modules (vector_tools) resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from axiom import config, db, gaps, graph, hypothesis, llm_providers, summarize, velocity
from axiom.agents import synthesis as synthesis_agent
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant, SearchHit
from vector_tools import render_proximity_analyzer, render_vector_gap_discovery

st.set_page_config(
    page_title="Axiom · Thesis Discovery",
    page_icon="🔭",
    layout="wide",
)


def _resolve_theme() -> str:
    """App-wide light/dark. Query param wins so a refresh keeps the choice."""
    qp = st.query_params.get("theme")
    if isinstance(qp, list):
        qp = qp[0] if qp else None
    if qp in ("light", "dark"):
        st.session_state["ax_theme"] = qp
    st.session_state.setdefault("ax_theme", "dark")
    return st.session_state["ax_theme"]


def _set_theme(next_theme: str) -> None:
    st.session_state["ax_theme"] = next_theme
    st.query_params["theme"] = next_theme


def _chart_theme() -> str | None:
    """Streamlit chart theming: follow the in-app toggle, not config.toml."""
    return "streamlit" if st.session_state.get("ax_theme") == "dark" else None


# --- Visual polish -----------------------------------------------------------
# Palettes swap via one button. Custom chrome used to be hardcoded navy, which
# stayed dark even when Streamlit itself was Light.
_CSS_DARK = """
:root {
  --ax-accent: #5b8cff;
  --ax-accent-soft: rgba(91,140,255,0.14);
  --ax-bg: #0b0f19;
  --ax-hero-bg: #131a29;
  --ax-hero-tint: rgba(91,140,255,0.16);
  --ax-panel: #151b2b;
  --ax-panel-2: #1b2334;
  --ax-border: rgba(146,164,205,0.18);
  --ax-border-strong: rgba(146,164,205,0.32);
  --ax-muted: #c5d0e0;
  --ax-text: #eef2f8;
  --ax-heading: #f4f7fb;
  --background-color: #0b0f19;
  --secondary-background-color: #151b2b;
  --text-color: #eef2f8;
  --ax-tab-active-fg: #0b0f19;
  --ax-tab-hover: #d6deec;
  --ax-code: #cdd6e4;
  --ax-scroll: #2a3550;
  --ax-scroll-hover: #364365;
  --ax-chip-score: #bcd0ff;
}
"""

_CSS_LIGHT = """
:root {
  --ax-accent: #3b6fd8;
  --ax-accent-soft: rgba(59,111,216,0.12);
  --ax-bg: #f4f6fb;
  --ax-hero-bg: #ffffff;
  --ax-hero-tint: rgba(59,111,216,0.08);
  --ax-panel: #ffffff;
  --ax-panel-2: #eef1f7;
  --ax-border: rgba(40,55,85,0.14);
  --ax-border-strong: rgba(40,55,85,0.28);
  --ax-muted: #3a4556;
  --ax-text: #121821;
  --ax-heading: #121821;
  --ax-tab-active-fg: #ffffff;
  --ax-tab-hover: #121821;
  --ax-code: #1e293b;
  --ax-scroll: #c5cddb;
  --ax-scroll-hover: #a8b2c4;
  --ax-chip-score: #2f5eb8;
  --background-color: #f4f6fb;
  --secondary-background-color: #ffffff;
  --text-color: #121821;
}
"""

_CSS = """
<style>
__PALETTE__

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], .stApp header {
  background-color: var(--ax-bg) !important;
}
.stApp { color: var(--ax-text); }

/* Captions/labels otherwise keep Streamlit Light greys on our dark page. */
[data-testid="stCaption"],
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
label, small {
  color: var(--ax-muted) !important;
  opacity: 1 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p, h1, h2, h3, h4, h5, h6 {
  color: var(--ax-heading) !important;
}
[data-testid="stAlert"] {
  background-color: var(--ax-panel-2) !important;
  color: var(--ax-text) !important;
  border: 1px solid var(--ax-border-strong) !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
  color: var(--ax-text) !important;
}

.block-container {
  position: relative;
  padding: 1.6rem 2rem 3rem 2rem !important;
  max-width: 1180px;
}

/* Marker + the following st.button (theme toggle). Streamlit 1.37 has no st-key-* class. */
.element-container:has(#ax-theme-marker) {
  display: none;
}
.element-container:has(#ax-theme-marker) + .element-container {
  position: absolute;
  top: 1.7rem;
  right: 2rem;
  z-index: 20;
  width: auto !important;
  background: transparent !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.element-container:has(#ax-theme-marker) + .element-container .stButton {
  width: auto !important;
  background: transparent !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.element-container:has(#ax-theme-marker) + .element-container [data-testid="stTooltipHoverTarget"],
.element-container:has(#ax-theme-marker) + .element-container [data-testid="stTooltipHoverTarget"] > div {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}
.element-container:has(#ax-theme-marker) + .element-container button {
  width: auto !important;
  min-width: 6.6rem;
  background: var(--ax-panel-2) !important;
  color: var(--ax-text) !important;
  border: 1px solid var(--ax-border-strong) !important;
  border-radius: 9px !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Tab labels sit inside a padded bar; panel content was flush left. Match the hero. */
.stTabs [data-baseweb="tab-panel"],
div[data-testid="stTab"] {
  padding: 1.15rem 1.25rem 1.25rem 1.25rem !important;
  box-sizing: border-box;
}

.ax-hero {
  display: flex; align-items: center; gap: 16px;
  padding: 16px 8.5rem 16px 20px;
  margin: 0 0 4px 0;
  border: 1px solid var(--ax-border);
  border-radius: 16px;
  background:
    radial-gradient(120% 140% at 0% 0%, var(--ax-hero-tint) 0%, transparent 55%),
    var(--ax-hero-bg);
}
.ax-hero-copy { flex: 1; min-width: 0; }
.ax-hero .ax-mark {
  font-size: 34px; line-height: 1;
  width: 58px; height: 58px; flex: none;
  display: grid; place-items: center;
  border-radius: 14px;
  background: var(--ax-accent-soft);
  border: 1px solid var(--ax-border-strong);
}
.ax-hero h1 {
  font-size: 1.7rem; font-weight: 750; margin: 0;
  letter-spacing: -0.02em; color: var(--ax-text);
}
.ax-hero p { margin: 3px 0 0; color: var(--ax-muted) !important; font-size: 0.92rem; }

.ax-meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 10px; }
.ax-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 500;
  color: var(--ax-muted);
  background: var(--ax-panel); border: 1px solid var(--ax-border);
}
.ax-pill code { background: none; color: var(--ax-code); padding: 0; font-size: 0.78rem; }
.ax-pill.ok  { color: #2f9e62; border-color: rgba(81,207,102,0.45); background: rgba(81,207,102,0.12); }
.ax-pill.off { color: #c24a3a; border-color: rgba(255,135,135,0.45); background: rgba(255,135,135,0.12); }
.ax-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

.stTabs [data-baseweb="tab-list"] {
  gap: 6px; border-bottom: none; margin: 8px 0 4px;
  background: var(--ax-panel); padding: 6px; border-radius: 12px;
  border: 1px solid var(--ax-border);
}
.stTabs [data-baseweb="tab"] {
  height: auto; padding: 8px 16px; border-radius: 8px;
  color: var(--ax-muted); font-weight: 550; font-size: 0.9rem;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--ax-accent-soft); color: var(--ax-tab-hover); }
.stTabs [aria-selected="true"] {
  background: var(--ax-accent) !important; color: var(--ax-tab-active-fg) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

[data-testid="stExpander"] {
  border: 1px solid var(--ax-border) !important;
  border-radius: 12px !important;
  background: var(--ax-panel);
  margin-bottom: 10px; overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 12px 16px; font-weight: 550; color: var(--ax-text); }
[data-testid="stExpander"] summary:hover { color: var(--ax-accent); }
[data-testid="stExpander"]:hover { border-color: var(--ax-border-strong) !important; }

[data-testid="stMetric"] {
  background: var(--ax-panel); border: 1px solid var(--ax-border);
  border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: var(--ax-muted); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.01em; color: var(--ax-text); }

[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 14px; border-color: var(--ax-border) !important;
  background: var(--ax-panel);
}

.stButton {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
.stButton > button {
  border-radius: 9px !important;
  border: 1px solid var(--ax-border-strong) !important;
  font-weight: 550;
  background: var(--ax-panel) !important;
  color: var(--ax-text) !important;
  outline: none !important;
  box-shadow: none !important;
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button:hover { border-color: var(--ax-accent); color: var(--ax-text); }

[data-testid="stTextInput"] input, [data-baseweb="select"] > div {
  border-radius: 9px;
  background-color: var(--ax-panel) !important;
  color: var(--ax-text) !important;
}
.stTextInput input:focus { border-color: var(--ax-accent) !important; }

h5 { margin-top: 0.4rem; color: var(--ax-heading); }

.ax-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px; }
.ax-chip {
  font-size: 0.76rem; color: var(--ax-muted);
  padding: 2px 9px; border-radius: 6px;
  background: var(--ax-panel-2); border: 1px solid var(--ax-border);
}
.ax-chip.score { color: var(--ax-chip-score); border-color: rgba(91,140,255,0.4); background: var(--ax-accent-soft); }
.ax-chip.grow { color: #2f9e62; border-color: rgba(81,207,102,0.45); background: rgba(81,207,102,0.12); }
.ax-chip.fade { color: #c24a3a; border-color: rgba(255,135,135,0.45); background: rgba(255,135,135,0.12); }
.ax-chip.warn { color: #b8860b; border-color: rgba(255,212,59,0.45); background: rgba(255,212,59,0.14); }

.ax-empty { text-align: center; padding: 20px 12px 8px; }
.ax-empty-icon {
  font-size: 40px; line-height: 1; opacity: 0.9;
  display: inline-grid; place-items: center;
  width: 72px; height: 72px; border-radius: 18px;
  background: var(--ax-accent-soft); border: 1px solid var(--ax-border);
}
.ax-empty-title { font-weight: 650; font-size: 1.08rem; color: var(--ax-text); margin-top: 12px; }
.ax-empty-sub {
  color: var(--ax-muted); font-size: 0.9rem; line-height: 1.55;
  margin: 8px auto 4px; max-width: 540px;
}
.ax-empty-sub code {
  background: var(--ax-panel-2); color: var(--ax-code);
  padding: 1px 6px; border-radius: 5px; font-size: 0.85em;
}

[data-testid="stDecoration"] {
  background: var(--ax-accent);
}

.ax-draft-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
  letter-spacing: 0.02em; text-transform: uppercase;
  color: #b8860b; background: rgba(255,212,59,0.14); border: 1px solid rgba(255,212,59,0.45);
}
.ax-draft-title { font-size: 1.15rem; font-weight: 700; color: var(--ax-text); margin: 8px 0 2px; letter-spacing: -0.01em; }

.ax-list { display: flex; flex-direction: column; gap: 6px; }
.ax-listrow {
  display: flex; align-items: center; gap: 10px; padding: 8px 14px;
  border: 1px solid var(--ax-border-strong); border-radius: 10px;
  background: var(--ax-panel-2);
}
.ax-listrow:hover { border-color: var(--ax-border-strong); background: var(--ax-panel-2); }
.ax-listrow .ax-rank {
  color: var(--ax-muted); font-size: 0.82rem; font-variant-numeric: tabular-nums;
  width: 2em; text-align: right; flex: none;
}
.ax-listrow .ax-name {
  font-weight: 550; color: var(--ax-text); flex: 1;
  min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ax-listrow .ax-chip { flex: none; margin: 0; }

@media (max-width: 640px) {
  .ax-listrow { flex-wrap: wrap; }
  .ax-listrow .ax-name { flex: 1 1 100%; white-space: normal; }
  .ax-listrow .ax-rank { width: auto; }
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--ax-scroll); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--ax-scroll-hover); }
</style>
"""
_THEME = _resolve_theme()
st.markdown(
    _CSS.replace("__PALETTE__", _CSS_DARK if _THEME == "dark" else _CSS_LIGHT),
    unsafe_allow_html=True,
)

_next = "light" if _THEME == "dark" else "dark"
_label = "Light" if _THEME == "dark" else "Dark"
_icon = "☀" if _THEME == "dark" else "🌙"
st.markdown(
    '<div class="ax-hero">'
    '<div class="ax-mark">🔭</div>'
    '<div class="ax-hero-copy">'
    "<h1>Axiom</h1>"
    "<p>Not a paper search engine. Two signals: which concepts are accelerating, "
    "and which related literatures barely cite each other.</p>"
    "</div></div>",
    unsafe_allow_html=True,
)
st.markdown('<div id="ax-theme-marker"></div>', unsafe_allow_html=True)
if st.button(f"{_icon} {_label}", key="ax_theme_toggle"):
    _set_theme(_next)
    st.rerun()


# --- Cached singletons -------------------------------------------------------
@st.cache_resource(show_spinner="Loading SPECTER2 encoder…")
def get_encoder() -> Specter2Encoder:
    return Specter2Encoder()


@st.cache_resource(show_spinner=False)
def get_store() -> AxiomQdrant:
    return AxiomQdrant()


@st.cache_resource(show_spinner="Loading citation graph…")
def get_graph() -> nx.DiGraph:
    """In-memory citation graph from SQLite (decision OD6, NetworkX-first)."""
    return graph.load_graph()


@st.cache_data(show_spinner=False)
def get_influence_ranking(top_k: int = 50) -> list:
    """PageRank-ranked influential papers (cached; slice for the UI)."""
    return graph.influence(get_graph(), top_k=top_k)


@st.cache_data(show_spinner="Detecting communities & research gaps…")
def get_gap_analysis() -> gaps.GapAnalysis:
    """Communities + candidate research gaps (cached). Needs Qdrant vectors."""
    conn = db.connect()
    try:
        return gaps.analyze(get_graph(), conn, get_store().fetch_dense_vectors())
    finally:
        conn.close()


@st.cache_data(show_spinner="Computing keyword velocity…")
def get_velocity_analysis(venue: str | None, year_range: tuple[int, int] | None) -> velocity.VelocityAnalysis:
    """Concept velocity: normalized-frequency log2-ratio, recent vs prior window (OD10)."""
    conn = db.connect()
    try:
        return velocity.get_top_velocity_keywords(conn, n=50, venue=venue, year_range=year_range)
    finally:
        conn.close()


# Distinct, dark-background-friendly palette for community coloring.
_COMMUNITY_PALETTE = [
    "#4dabf7", "#ff922b", "#51cf66", "#cc5de8", "#ffd43b", "#ff6b6b",
    "#22b8cf", "#a9e34b", "#f783ac", "#748ffc", "#ffa94d", "#63e6be",
    "#e599f7", "#94d82d", "#ff8787", "#3bc9db",
]


def _community_color(cid: int) -> str:
    return _COMMUNITY_PALETTE[cid % len(_COMMUNITY_PALETTE)]


@st.cache_data(show_spinner=False)
def get_filter_options() -> tuple[list[str], tuple[int, int] | None]:
    """Venue list + year bounds from SQLite (drives the filter bar)."""
    conn = db.connect()
    try:
        venues = db.distinct_venues(conn)
        bounds = db.year_bounds(conn)
        return venues, bounds
    except Exception:
        return [], None
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def get_meta() -> dict[str, dict]:
    """Per-paper abstract/DOI/title from SQLite, keyed by paper_id.

    Qdrant payloads are intentionally lean; the cards enrich hits from SQLite,
    the durable source of truth.
    """
    conn = db.connect()
    try:
        rows = conn.execute("SELECT openalex_id, title, abstract, doi FROM papers").fetchall()
        return {
            r["openalex_id"]: {"title": r["title"], "abstract": r["abstract"], "doi": r["doi"]}
            for r in rows
        }
    except Exception:
        return {}
    finally:
        conn.close()

def render_year_filter(
    bounds: tuple[int, int] | None,
    key_prefix: str,
    min_year: int | None = None,
) -> tuple[int, int] | None:
    """Year range selector. `min_year` clamps the earliest selectable year —
    e.g. the Trending tab pins the floor to config.VELOCITY_MIN_YEAR so velocity
    never spans the noisy pre-2020 references pulled in by snowball ingestion."""
    if not bounds:
        return None
    lo, hi = bounds
    floor = lo if min_year is None else min(max(min_year, lo), hi)
    years = list(range(floor, hi + 1))
    ycol1, ycol2 = st.columns(2)
    with ycol1:
        y_from = st.selectbox("From year", years, index=0, key=f"{key_prefix}_from")
    with ycol2:
        y_to = st.selectbox("To year", years, index=len(years) - 1, key=f"{key_prefix}_to")
    return (min(y_from, y_to), max(y_from, y_to))


# --- Guard rails: Qdrant only gates the vector-backed tabs -------------------
# Search + Citation graph need Qdrant vectors; Trending / Reading list / Review
# queue are SQLite-only. A missing or empty index must NOT stop the whole app
# (a top-level st.stop() here previously blacked out every tab, including
# Trending). Record the status and let the two vector tabs surface it themselves.
store = get_store()
try:
    point_count = store.count()
    qdrant_ok = point_count > 0
    qdrant_msg = "" if qdrant_ok else (
        "The vector index is empty. Run the embed/index step "
        "(`python scripts/ingest_openalex.py` without `--no-index`, or "
        "`python scripts/bootstrap_synthetic.py`) to enable Search and the "
        "citation graph. Trending, Reading list and Review queue work without it."
    )
except Exception:
    point_count = 0
    qdrant_ok = False
    qdrant_msg = (
        "⚠️ Cannot reach Qdrant. Start it with `docker compose up -d`, then run "
        "the embed/index step. Search and the citation graph stay disabled until "
        "then — Trending, Reading list and Review queue work without it."
    )

meta = get_meta()
st.session_state.setdefault("similar_to", None)

# Corpus / model / index status as a compact pill row under the hero.
if qdrant_ok:
    _index_pill = (
        f'<span class="ax-pill ok"><span class="dot"></span>'
        f"index · {point_count:,} vectors</span>"
    )
else:
    _index_pill = (
        '<span class="ax-pill off"><span class="dot"></span>index offline</span>'
    )
st.markdown(
    '<div class="ax-meta">'
    '<span class="ax-pill">ACL · EMNLP · COLING · NAACL · 2020–2025</span>'
    f'<span class="ax-pill">collection <code>{html.escape(config.COLLECTION_NAME)}</code></span>'
    f'<span class="ax-pill">model <code>{html.escape(config.MODEL_ID)}</code></span>'
    f"{_index_pill}"
    "</div>",
    unsafe_allow_html=True,
)


# --- Reading list (OD13): bookmarks only, no LLM summaries (PBI 5 not built) --
def get_bookmarked_ids() -> set[str]:
    """Not cached — must reflect adds/removes immediately on the next rerun."""
    conn = db.connect()
    try:
        return {r["paper_id"] for r in db.list_bookmarks(conn)}
    finally:
        conn.close()


def toggle_bookmark(paper_id: str, *, add: bool) -> None:
    conn = db.connect()
    try:
        if add:
            db.add_bookmark(conn, paper_id)
        else:
            db.remove_bookmark(conn, paper_id)
    finally:
        conn.close()


# --- Result rendering --------------------------------------------------------
def _chips(*chips: tuple[str, str]) -> None:
    """Horizontal chip row. Each chip is (text, css_suffix) — suffix is "" or
    a modifier like score/grow/fade/warn."""
    inner = "".join(
        f'<span class="ax-chip {html.escape(cls)}">{html.escape(txt)}</span>'
        for txt, cls in chips
    )
    st.markdown(f'<div class="ax-chips">{inner}</div>', unsafe_allow_html=True)


def _empty_state(icon: str, title: str, body_md: str | None = None) -> None:
    """Card-styled empty/offline placeholder with optional markdown-ish body."""
    body_html = ""
    if body_md:
        safe = html.escape(body_md.lstrip().removeprefix("⚠️").lstrip())
        safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
        safe = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", safe)
        safe = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", safe)
        body_html = f'<div class="ax-empty-sub">{safe}</div>'
    with st.container(border=True):
        st.markdown(
            f'<div class="ax-empty"><div class="ax-empty-icon">{icon}</div>'
            f'<div class="ax-empty-title">{html.escape(title)}</div>'
            f"{body_html}</div>",
            unsafe_allow_html=True,
        )


def render_hits(hits: list[SearchHit], *, score_label: str = "score") -> None:
    """Render hits as expandable cards with concepts, abstract, and a pivot."""
    if not hits:
        _empty_state(
            "🔎",
            "No papers match these filters",
            "Try loosening the venue or year filters, or broaden the query.",
        )
        return
    meta = get_meta()
    bookmarked = get_bookmarked_ids()
    st.caption(f"{len(hits)} result{'s' if len(hits) != 1 else ''}")
    for h in hits:
        m = meta.get(h.paper_id, {})
        with st.expander(h.title):
            _chips(
                (f"{h.score:.3f} {score_label}", "score"),
                (str(h.year), ""),
                (str(h.venue), ""),
                (f"{h.cited_by_count:,} cites", ""),
            )
            if m.get("doi"):
                st.markdown(f"[Open paper (DOI)](https://doi.org/{m['doi']})")
            if h.concepts:
                st.markdown(" ".join(f"`{c}`" for c in h.concepts))
            st.write(m.get("abstract") or "_No abstract available._")
            bcol, scol = st.columns([1, 1])
            with bcol:
                if h.paper_id in bookmarked:
                    if st.button("📚 Remove bookmark", key=f"unbm_{h.paper_id}"):
                        toggle_bookmark(h.paper_id, add=False)
                        st.rerun()
                else:
                    if st.button("📚 Add to reading list", key=f"bm_{h.paper_id}"):
                        toggle_bookmark(h.paper_id, add=True)
                        st.rerun()
            with scol:
                # Pivot: explore semantic neighbors of this paper.
                if st.button("🔎 Similar papers", key=f"sim_{h.paper_id}"):
                    st.session_state["similar_to"] = h.paper_id
                    st.rerun()


# --- Search tab --------------------------------------------------------------
def render_search() -> None:
    """Semantic/hybrid search with the venue/year filter bar."""
    if not qdrant_ok:
        _empty_state("🔌", "Search needs the vector index", qdrant_msg)
        return
    venues, bounds = get_filter_options()
    with st.container():
        fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
        with fcol1:
            # Empty selection = all venues (no silent filtering).
            selected_venues = st.multiselect(
                "Venue", options=venues, default=[],
                help="Leave empty to search all venues.",
            )
        with fcol2:
            year_range = render_year_filter(bounds, "search")
        with fcol3:
            # Default to DEFAULT_TOP_K; cap at corpus size (no magic number).
            top_k = st.number_input(
                "Top-K", min_value=1, max_value=point_count,
                value=min(config.DEFAULT_TOP_K, point_count), step=1,
            )

    # Hybrid is only offered if the index has sparse vectors.
    if store.supports_hybrid():
        mode = st.radio(
            "Retrieval", ["Hybrid (dense + sparse)", "Dense only"],
            index=0, horizontal=True,
            help="Hybrid adds keyword/acronym matching (e.g. 'LoRA') on top of dense.",
        )
        use_hybrid = mode.startswith("Hybrid")
    else:
        use_hybrid = False
        st.caption("Retrieval: dense only (index has no sparse vectors — re-run the bootstrap for hybrid).")

    common = dict(
        top_k=int(top_k),
        venues=selected_venues or None,
        year_range=tuple(year_range) if year_range else None,
    )

    # Similar-papers pivot vs normal query view.
    similar_to = st.session_state.get("similar_to")
    if similar_to:
        title = meta.get(similar_to, {}).get("title", similar_to)
        st.subheader(f"Papers similar to: {title}")
        if st.button("← Back to search"):
            st.session_state["similar_to"] = None
            st.rerun()
        render_hits(store.similar_papers(similar_to, **common), score_label="cosine")
    else:
        query = st.text_input(
            "Search query",
            placeholder="e.g. reducing hallucination in retrieval-augmented generation",
        )
        if query:
            qvec = get_encoder().encode_query(query)
            if use_hybrid:
                hits = store.search_hybrid(query_vector=qvec, query_text=query, **common)
            else:
                hits = store.search(query_vector=qvec, **common)
            render_hits(hits, score_label="RRF rank" if use_hybrid else "cosine")
        else:
            _empty_state(
                "🔎",
                "Search the corpus",
                "Type a research question or an acronym (`LoRA`, `RAG`). "
                "Hybrid retrieval matches both meaning and exact terms.",
            )


# --- Citation-graph tab ------------------------------------------------------
_GRAPH3D_TEMPLATE = """
<div id="graph3d" style="width:100%;height:HEIGHTpx;background:#0b0f19;border-radius:10px;"></div>
<script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js" integrity="sha384-GNPicn8pBA2/PGSyPTpxIlPurgLUYcNYJ2zskIq782dE9+gp5E32WSyuxZqA7J+u" crossorigin="anonymous"></script>
<script>
(function () {
  var el = document.getElementById('graph3d');
  function fail(msg) {
    el.innerHTML = '<p style="color:#ff8787;font-family:sans-serif;padding:1rem">' + msg + '</p>';
  }
  if (typeof ForceGraph3D === 'undefined') {
    fail('3D graph library could not load — check the browser has internet access.');
    return;
  }
  try {
    var data = __DATA__;
    el.style.position = 'relative';
    // Two background themes for the scene; toggled via the on-screen button.
    var THEMES = {
      dark:  { bg: '#0b0f19', link: 'rgba(205,222,248,0.85)', arrow: 'rgba(220,230,250,0.8)', particle: '#ffd43b' },
      light: { bg: '#f4f6fb', link: 'rgba(60,72,95,0.7)',     arrow: 'rgba(45,55,75,0.75)',   particle: '#e8590c' }
    };
    var theme = THEMES['__APP_THEME__'] || THEMES.dark;
    el.style.background = theme.bg;
    var Graph = ForceGraph3D()(el)
      .width(el.clientWidth || 800)
      .height(HEIGHT)
      .showNavInfo(false)
      .backgroundColor(theme.bg)
      .graphData(data)
      .nodeLabel('name')
      .nodeVal('val')
      .nodeColor('color')
      .nodeOpacity(0.95)
      .nodeResolution(16)
      .linkColor(function () { return theme.link; })
      .linkWidth(1.2)
      .linkDirectionalArrowLength(3.6)
      .linkDirectionalArrowRelPos(1)
      .linkDirectionalArrowColor(function () { return theme.arrow; })
      .linkDirectionalParticles(3)
      .linkDirectionalParticleWidth(2)
      .linkDirectionalParticleColor(function () { return theme.particle; })
      .linkDirectionalParticleSpeed(0.006)
      .onNodeClick(function (node) {
        var dist = 70;
        var ratio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
        Graph.cameraPosition(
          {x: (node.x || 0) * ratio, y: (node.y || 0) * ratio, z: (node.z || 0) * ratio},
          node, 800);
      });

    // Explicit starting camera so the scene is never a blank black frame.
    Graph.cameraPosition({ z: 320 });

    // Disable the (TrackballControls) wheel-zoom so the wheel scrolls the PAGE
    // instead of the canvas; zoom is via the on-screen buttons. (Guarded.)
    try {
      var ctrls = Graph.controls();
      if (ctrls) { ctrls.noZoom = true; ctrls.enableZoom = false; }
    } catch (e) {}
    // Cap long-range repulsion so disconnected clusters don't drift far apart.
    try { if (Graph.d3Force('charge')) Graph.d3Force('charge').strength(-45).distanceMax(220); } catch (e) {}
    // Auto-fit once the layout settles (guarded; runs after physics).
    setTimeout(function () { try { Graph.zoomToFit(600, 30); } catch (e) {} }, 1800);

    function dolly(f) {
      var c = Graph.cameraPosition();
      Graph.cameraPosition({x: c.x * f, y: c.y * f, z: c.z * f}, undefined, 250);
    }
    function mkBtn(txt, fn) {
      var b = document.createElement('button');
      b.textContent = txt;
      b.style.cssText = 'width:32px;height:32px;border:1px solid #93a4bd;border-radius:6px;' +
        'background:#5b6b85;color:#ffffff;font-size:17px;font-weight:600;line-height:1;' +
        'cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.4);';
      b.onmouseenter = function () { b.style.background = '#74859f'; };
      b.onmouseleave = function () { b.style.background = '#5b6b85'; };
      b.onclick = fn;
      return b;
    }
    // Zoom is via the on-screen buttons. (Guarded.)
    var bar = document.createElement('div');
    bar.style.cssText = 'position:absolute;top:10px;right:12px;display:flex;gap:6px;z-index:5;';
    bar.appendChild(mkBtn('+', function () { dolly(0.8); }));
    bar.appendChild(mkBtn('–', function () { dolly(1.25); }));
    bar.appendChild(mkBtn('▣', function () { try { Graph.zoomToFit(400, 30); } catch (e) {} }));
    el.appendChild(bar);

    window.addEventListener('resize', function () { Graph.width(el.clientWidth || 800); });
  } catch (err) {
    fail('3D graph error: ' + (err && err.message ? err.message : err));
  }
})();
</script>
"""


def _graph_html(sub: nx.DiGraph, node_colors: dict[str, str] | None = None,
                node_cids: dict[str, int] | None = None, height: int = 500) -> str:
    """
    Renders sub to an interactive 3d-force-graph HTML snippet.
    node_colors dict maps openalex_id -> hex color.
    """
    indeg = dict(sub.in_degree())
    max_in = max(indeg.values(), default=0) or 1
    nodes = []
    for nid, d in sub.nodes(data=True):
        title = d.get("title") or nid
        if node_colors and nid in node_colors:
            color = node_colors[nid]
        else:
            color = "#4dabf7"
            
        cluster_str = f"[Cluster: {node_cids[nid]}] " if node_cids and nid in node_cids else ""
        
        nodes.append({
            "id": nid,
            "name": (f"{cluster_str}{title} ({d.get('year', '?')}) — "
                     f"{d.get('cited_by_count', 0)} cites · "
                     f"{indeg.get(nid, 0)} citing within view"),
            "val": 2 + 9 * (indeg.get(nid, 0) / max_in),
            "color": color,
        })
    links = [{"source": u, "target": v} for u, v in sub.edges()]
    payload = json.dumps({"nodes": nodes, "links": links})
    payload = payload.replace("</", "<\\/")
    app_theme = st.session_state.get("ax_theme", "dark")
    return (
        _GRAPH3D_TEMPLATE
        .replace("HEIGHT", str(height))
        .replace("__DATA__", payload)
        .replace("__APP_THEME__", "light" if app_theme == "light" else "dark")
    )


def _short(comm, k: int = 2) -> str:
    """First k distinctive concepts of a community, for compact labels."""
    return " · ".join(comm.labels[:k]) if comm.labels else f"cluster {comm.cid}"


def _render_gap_detail(g, gap) -> None:
    """Explain a candidate gap and list the top papers on each side."""
    st.markdown(
        f"**Why this is a candidate gap:** these two sub-topics are close in "
        f"meaning (centroid cosine **{gap.semantic_similarity:.2f}**) but only "
        f"**{gap.inter_citations}** citation(s) connect **{gap.a.size}** and "
        f"**{gap.b.size}** papers — a bridge largely unbuilt."
    )
    if getattr(gap, "components", None):
        c = gap.components
        flag = " ✅ meets threshold" if gap.meets_threshold else ""
        st.caption(
            f"**G-score {gap.g_score:.2f}**{flag} — similarity {c['similarity']:.2f} · "
            f"disconnection {c['disconnection']:.2f} · velocity {c['velocity']:.2f} · "
            f"authority {c['authority']:.2f}. ⚠️ Uncalibrated defaults (OD17); weights "
            f"and threshold need human labels — see scripts/calibrate_gap_thresholds.py."
        )
    cola, colb = st.columns(2)
    for col, comm in ((cola, gap.a), (colb, gap.b)):
        with col:
            st.markdown(f"**{comm.label}** · {comm.size} papers")
            tops = sorted(comm.members,
                          key=lambda p: g.nodes[p].get("cited_by_count", 0),
                          reverse=True)[:5]
            for p in tops:
                d = g.nodes[p]
                st.markdown(f"- {d.get('title')}  _({d.get('year')})_")


def render_graph_view() -> None:
    """Research gaps + citation structure (communities, gaps, influence)."""
    if not qdrant_ok:
        st.subheader("Research gaps")
        _empty_state("🔌", "Research gaps need the vector index", qdrant_msg)
        return
    st.subheader("Research gaps")
    st.caption(
        "The product's citation-backed gap view. Sub-topics (communities) come "
        "from the citation graph; a candidate **research gap** is a pair of "
        "communities close in meaning yet barely citing each other. The 3D "
        "graph below is the map — not the product."
    )

    g = get_graph()
    gs = graph.stats(g)
    if gs["edges_in_corpus"] == 0:
        _empty_state(
            "🕸️",
            "No in-corpus citations yet",
            "Ingest a connected corpus first so the citation graph has edges to cluster.",
        )
        return

    analysis = get_gap_analysis()
    node_colors = {pid: _community_color(cid) for pid, cid in analysis.node2c.items()}

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Papers", gs["papers"])
    mcol2.metric("In-corpus citations", gs["edges_in_corpus"])
    mcol3.metric("Sub-topics", len(analysis.communities))
    mcol4.metric("Candidate gaps", len(analysis.gaps))

    st.markdown("##### Interactive 3D graph")

    with st.container(border=True):
        ccol1, ccol2, ccol3 = st.columns([1.4, 2, 1])
        with ccol1:
            view = st.radio(
                "View", ["Research gaps", "Top influential"],
                help="Research gaps (default) highlights related-but-disconnected "
                     "clusters — the openings. 'Top influential' is for orientation: "
                     "the field's central papers. Click any node to inspect it.",
            )
        sel_gap = None
        sel_gap_idx = None
        with ccol2:
            if view == "Research gaps":
                if analysis.gaps:
                    opts = [
                        f"#{i} {_short(gp.a)} ⟷ {_short(gp.b)}  "
                        f"(sim {gp.semantic_similarity:.2f} · {gp.inter_citations} cites)"
                        for i, gp in enumerate(analysis.gaps)
                    ]
                    sel = st.selectbox("Candidate gap", range(len(opts)),
                                       format_func=lambda i: opts[i])
                    sel_gap_idx = sel
                    sel_gap = analysis.gaps[sel_gap_idx]
                    members = set(sel_gap.a.members) | set(sel_gap.b.members)
                    sub = g.subgraph(members).copy()
                else:
                    sub = g.subgraph([]).copy()
            else:  # Top influential (orientation)
                v_n = st.slider("Papers to draw", 10, 50, 25, step=5)
                sub = graph.top_influential_subgraph(g, n=v_n)
        with ccol3:
            # Escape hatch: the WebGL canvas captures the mouse wheel (to zoom),
            # which blocks page scrolling while hovering it. Toggle off to scroll.
            show_3d = st.toggle("Show 3D graph", value=True,
                                help="Turn off to scroll the page freely past this section.")

    # Legend split:
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;justify-content:space-between;'
        'align-items:center;gap:16px;font-size:0.86rem;margin:2px 0 8px;">'
        # left group — what you're looking at
        '<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;">'
        f'<span><b>{sub.number_of_nodes()}</b> papers · '
        f'<b>{sub.number_of_edges()}</b> citations in view</span>'
        '<span>🎨 colour = sub-topic cluster</span>'
        '<span style="display:inline-flex;align-items:center;gap:8px;">'
        '<span style="width:7px;height:7px;background:#868e96;border-radius:50%;display:inline-block;"></span>fewer'
        '<span style="width:17px;height:17px;background:#868e96;border-radius:50%;display:inline-block;"></span>'
        'more citations</span>'
        '</div>'
        # right group
        '<div style="color:#9ca3af;white-space:nowrap;">'
        'drag rotate · +/–/▣ zoom · click node to fly · wheel scrolls page'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if show_3d:
        components.html(
            _graph_html(sub, node_colors=node_colors, node_cids=analysis.node2c, height=520),
            height=540, scrolling=False)

    # Gap detail + ranked list (gaps view only).
    if view == "Research gaps" and sel_gap is not None:
        _render_gap_detail(g, sel_gap)

        st.info(
            "⚠️ **Unverified Candidate** — a hypothesis pitch below is an "
            "LLM-generated narrative over this gap candidate, not a validated "
            "research direction. Nothing is promoted without an explicit "
            "approve in the Review queue tab."
        )
        if st.button("💡 Generate hypothesis pitch", key=f"hyp_{sel_gap_idx}"):
            v = get_velocity_analysis(None, None)
            trend_context = [k.concept for k in v.keywords if k.velocity > 0][:5]
            with st.spinner("Generating + verifying pitch…"):
                try:
                    # D4: the one real hypothesis implementation (strong tier, via
                    # llm_providers) -- not the older Ollama-only axiom.hypothesis
                    # path this replaced. generate_hypothesis is async; Streamlit's
                    # script execution isn't, so run it to completion here.
                    pitch = asyncio.run(synthesis_agent.generate_hypothesis(
                        sel_gap, g, run_id=uuid.uuid4().hex, trend_context=trend_context,
                    ))
                    conn = db.connect()
                    try:
                        db.add_to_review_queue(
                            conn, gap_a_label=sel_gap.a.label, gap_b_label=sel_gap.b.label,
                            title=pitch.title, claim=pitch.claim,
                            method_sketch=pitch.method_sketch, datasets=pitch.datasets,
                            supporting_paper_ids=pitch.supporting_paper_ids,
                        )
                    finally:
                        conn.close()
                    st.session_state["last_pitch"] = (sel_gap_idx, pitch)
                except llm_providers.LLMError as exc:
                    st.error(f"LLM provider error: {exc}")
                except hypothesis.VerificationError as exc:
                    st.error(f"Verifier rejected every attempt: {exc}")

        _stored = st.session_state.get("last_pitch")
        if _stored is not None and _stored[0] == sel_gap_idx:
            last_pitch = _stored[1]
            with st.container(border=True):
                st.markdown(
                    '<span class="ax-draft-badge">Draft hypothesis</span>'
                    f'<div class="ax-draft-title">{html.escape(last_pitch.title)}</div>',
                    unsafe_allow_html=True,
                )
                st.write(last_pitch.claim)
                st.markdown(f"**Method sketch:** {last_pitch.method_sketch}")
                if last_pitch.datasets:
                    st.markdown("**Datasets:** " + ", ".join(f"`{d}`" for d in last_pitch.datasets))
                st.markdown(
                    "**Supporting papers:** " +
                    ", ".join(f"`{pid}`" for pid in last_pitch.supporting_paper_ids)
                )
                st.caption(f"⚠️ {last_pitch.disclaimer} Sent to the Review queue as pending.")

        with st.expander(f"All {len(analysis.gaps)} candidate gaps (ranked)"):
            for i, gp in enumerate(analysis.gaps, 1):
                st.markdown(
                    f"{i}. **{_short(gp.a)}** ⟷ **{_short(gp.b)}** — "
                    f"semantic sim `{gp.semantic_similarity:.2f}`, "
                    f"`{gp.inter_citations}` citations between them"
                )

    st.divider()
    st.markdown("##### Influence ranking")
    n = st.slider("How many to rank", min_value=5, max_value=50, value=20, step=5)
    ranking = get_influence_ranking(50)[:n]

    for i, r in enumerate(ranking, 1):
        with st.expander(f"{i}. {r.title}"):
            _chips(
                (f"PR {r.pagerank:.4f}", "score"),
                (str(r.year), ""),
                (f"{r.local_in_degree} local cites", ""),
                (f"{r.cited_by_count} global cites", ""),
            )
            m = meta.get(r.paper_id, {})
            if m.get("doi"):
                st.markdown(f"[Open paper (DOI)](https://doi.org/{m['doi']})")
            nb = graph.neighbors(g, r.paper_id)
            ext = sum(1 for x in nb.references if not x["in_corpus"])
            st.markdown(
                f"**Cited by {len(nb.cited_by)} papers in the corpus** · "
                f"cites {len(nb.references)} works ({ext} outside the corpus)"
            )
            if nb.cited_by:
                st.markdown("**Top corpus papers citing this:**")
                for c in nb.cited_by[:8]:
                    st.markdown(f"- {c['title']}  _({c['year']} · {c['cited_by_count']} cites)_")


# --- Trending tab -------------------------------------------------------------
def _velocity_bar_chart(items: list, color: str) -> None:
    """One horizontal bar per concept, input order preserved (items arrive sorted).

    Bars diverge from a shared centered zero baseline. Each bar is labelled at its
    tip with its share change as a percentage (2**velocity - 1), so risers read as
    +N% and faders as -N%. The raw velocity stays in the tooltip only.
    """
    # log2 share-ratio -> percent change in corpus share.
    pct = [(2 ** k.velocity - 1) * 100.0 for k in items]
    df = pd.DataFrame(
        {"concept": [k.concept for k in items],
         "velocity": [k.velocity for k in items],
         "pct": pct,
         "pct_label": [f"{p:+.0f}%" for p in pct]}
    )
    # Symmetric domain so the zero baseline sits in the visual center and the
    # magnitude of a rise reads the same as an equal fade.
    bound = max((abs(v) for v in df["velocity"]), default=1.0) * 1.15 or 1.0
    xscale = alt.Scale(domain=[-bound, bound])
    # All items in one chart share a sign (risers positive, faders negative), so
    # the labels sit just beyond every bar tip: past the right end for risers,
    # past the left end for faders.
    rising = (df["velocity"] >= 0).all()
    align, dx = ("left", 12) if rising else ("right", -12)

    base = alt.Chart(df).encode(
        y=alt.Y("concept:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=220)),  # sort=None => keep input order
    )
    bars = base.mark_bar(color=color).encode(
        x=alt.X("velocity:Q",
                title="share change (velocity = log2 share ratio)",
                scale=xscale),
        tooltip=["concept",
                 alt.Tooltip("pct:Q", title="share change", format="+.1f"),
                 alt.Tooltip("velocity:Q", title="velocity", format="+.2f")],
    )
    pct_labels = base.mark_text(
        align=align, baseline="middle", dx=dx, fontWeight="bold",
    ).encode(x=alt.X("velocity:Q", scale=xscale), text="pct_label:N")

    chart = (bars + pct_labels).properties(height=alt.Step(30))
    st.altair_chart(chart, use_container_width=True, theme=_chart_theme())


def render_trending() -> None:
    """Concepts ranked by velocity: normalized-frequency log2-ratio, recent vs prior window."""
    st.subheader("Trending concepts")
    st.caption(
        "Concepts ranked by velocity: how much their share of the corpus has "
        "changed between the older and newer half of the available years. "
        "Positive = rising, negative = fading. Counts below "
        f"{config.VELOCITY_MIN_FREQ} recent papers are flagged low-confidence."
    )
    st.info(
        "ℹ️ **Heuristic signal** — velocity ranks concepts by corpus frequency "
        "change, not by verified research impact. Treat as directional, not definitive."
    )

    venues, bounds = get_filter_options()
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        venue_choice = st.selectbox("Venue", ["All venues"] + venues, index=0,
                                    key="trend_venue")
        venue = None if venue_choice == "All venues" else venue_choice
    with fcol2:
        # Pin the earliest selectable year to the modern floor so velocity
        # windows never span the noisy pre-2020 references (config note).
        year_range = render_year_filter(bounds, "trend", min_year=config.VELOCITY_MIN_YEAR)

    analysis = get_velocity_analysis(venue, year_range)
    if not analysis.keywords:
        _empty_state(
            "📈",
            "Not enough dated papers",
            "This venue/year filter doesn't have enough dated papers to "
            "compute velocity. Widen the filter to see trends.",
        )
        return
    if analysis.insufficient_year_spread:
        st.warning("Selected range spans a single year — velocity needs at least two years to compare.")

    st.caption(
        f"Prior window {analysis.prior_window[0]}–{analysis.prior_window[1]} "
        f"({analysis.total_prior} papers) → Recent window "
        f"{analysis.recent_window[0]}–{analysis.recent_window[1]} "
        f"({analysis.total_recent} papers)"
    )

    # Charts show only meaningful movers. A concept must have >= MIN papers in
    # BOTH windows: without a real base in the prior window, a 0->N concept's
    # share-change % is dominated by the epsilon smoothing constant (e.g. +13000%)
    # and those degenerate values crowd out genuine risers. Requiring presence in
    # both windows yields real growth/decline ratios. The full ranked list below
    # still shows everything, including brand-new/vanished concepts.
    min_ct = config.VELOCITY_MIN_CHART_COUNT
    rising = [k for k in analysis.keywords
              if k.velocity > 0 and k.prior_count >= min_ct and k.recent_count >= min_ct][:15]
    fading = [k for k in reversed(analysis.keywords)
              if k.velocity < 0 and k.prior_count >= min_ct and k.recent_count >= min_ct][:10]

    if rising or fading:
        st.caption(
            f"Charts show concepts with ≥{min_ct} papers in *both* windows "
            "(real base → meaningful % change); brand-new or vanished concepts "
            "are excluded here but still listed below."
        )
    else:
        st.info(
            f"No concept reaches ≥{min_ct} papers in a window for this filter — "
            "the corpus is too sparse to chart a trend. See the ranked list below."
        )
    if rising:
        st.markdown("##### Top risers")
        _velocity_bar_chart(rising, color="#4c78a8")
    if fading:
        st.markdown("##### Top faders")
        _velocity_bar_chart(fading, color="#d62728")

    st.markdown("##### Ranked keywords")
    rows_html = []
    for i, k in enumerate(analysis.keywords, 1):
        vel_cls = "grow" if k.velocity >= 0 else "fade"
        warn = '<span class="ax-chip warn">low volume</span>' if k.low_confidence else ""
        rows_html.append(
            '<div class="ax-listrow">'
            f'<span class="ax-rank">{i}</span>'
            f'<span class="ax-name">{html.escape(k.concept)}</span>'
            f"{warn}"
            f'<span class="ax-chip">{k.prior_count} → {k.recent_count} papers</span>'
            f'<span class="ax-chip {vel_cls}">{k.velocity:+.2f}</span>'
            "</div>"
        )
    st.markdown(f'<div class="ax-list">{"".join(rows_html)}</div>', unsafe_allow_html=True)


# --- Reading list tab ---------------------------------------------------------
def render_reading_list() -> None:
    """Bookmarked papers, with 3-bullet local-LLM summaries (OD14)."""
    st.subheader("Reading list")
    st.caption(
        f"Papers bookmarked from Search. Summaries are generated locally via "
        f"Ollama (`{config.OLLAMA_MODEL}`, OD14) — grounded only in the "
        f"paper's own abstract, cached after the first request."
    )
    conn = db.connect()
    try:
        rows = db.list_bookmarks(conn)
    finally:
        conn.close()

    if not rows:
        _empty_state(
            "📚",
            "Your reading list is empty",
            "Bookmark papers from a **Search** result card and they will "
            "collect here for local summarization.",
        )
        return

    st.caption(f"{len(rows)} bookmarked paper{'s' if len(rows) != 1 else ''}")
    for r in rows:
        with st.expander(r["title"]):
            _chips(
                (str(r["publication_year"]), ""),
                (str(r["venue"]), ""),
                (f"{r['cited_by_count']:,} cites", ""),
            )
            if r["doi"]:
                st.markdown(f"[Open paper (DOI)](https://doi.org/{r['doi']})")
            st.write(r["abstract"] or "_No abstract available._")

            conn3 = db.connect()
            try:
                bullets = db.get_summary(conn3, r["paper_id"])
            finally:
                conn3.close()

            bcol, scol = st.columns([1, 1])
            with bcol:
                if st.button("📚 Remove bookmark", key=f"rl_unbm_{r['paper_id']}"):
                    toggle_bookmark(r["paper_id"], add=False)
                    st.rerun()

            if bullets:
                st.markdown("**LLM-generated Summary:**")
                for b in bullets:
                    st.markdown(f"- {b}  _(cites `{r['paper_id']}`)_")
            else:
                with scol:
                    if st.button("🧠 Summarize", key=f"rl_sum_{r['paper_id']}"):
                        if not r["abstract"]:
                            st.error("No abstract available to summarize.")
                        else:
                            with st.spinner("Summarizing locally…"):
                                try:
                                    result = summarize.summarize_paper(
                                        r["paper_id"], r["title"], r["abstract"]
                                    )
                                    conn2 = db.connect()
                                    try:
                                        db.save_summary(conn2, r["paper_id"], result.bullets,
                                                         llm_providers.model_for_tier("fast"))
                                    finally:
                                        conn2.close()
                                    st.rerun()
                                except llm_providers.LLMError as exc:
                                    st.error(f"Summarization failed: {exc}")


# --- Review queue tab (OD16) ---------------------------------------------------
def render_review_queue() -> None:
    """HITL queue for hypothesis pitches (Task 5.2, OD16). Nothing auto-promotes."""
    st.subheader("Review queue")
    st.caption(
        "HITL backoffice — approve or reject pitches. Items are generated from "
        "the Research gaps tab. Every item starts **pending**; nothing is "
        "promoted automatically."
    )
    status_filter = st.radio("Filter", ["pending", "approved", "rejected", "all"],
                             horizontal=True)
    conn = db.connect()
    try:
        rows = db.list_review_queue(conn, status=None if status_filter == "all" else status_filter)
    finally:
        conn.close()

    if not rows:
        _empty_state(
            "🗂️",
            f"No {status_filter} pitches",
            "Generate a hypothesis pitch from **Research gaps** and it will "
            "queue here for review.",
        )
        return

    for r in rows:
        try:
            datasets = json.loads(r["datasets_json"] or "[]")
            supporting = json.loads(r["supporting_ids_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            datasets, supporting = [], []

        badge = {"pending": "🟡", "approved": "🟢", "rejected": "🔴"}[r["status"]]
        with st.expander(f"{badge} {r['title']}  ({r['status']})"):
            st.caption(f"{r['gap_a_label']} ⟷ {r['gap_b_label']}")
            st.write(r["claim"])
            st.markdown(f"**Method sketch:** {r['method_sketch']}")
            if datasets:
                st.markdown("**Datasets:** " + ", ".join(f"`{d}`" for d in datasets))
            st.markdown("**Supporting papers:** " + ", ".join(f"`{p}`" for p in supporting))
            st.warning("⚠️ Unverified Candidate — not a validated research gap.")
            if r["status"] == "pending":
                acol, rcol = st.columns(2)
                with acol:
                    if st.button("✅ Approve", key=f"approve_{r['id']}"):
                        conn2 = db.connect()
                        try:
                            db.set_review_status(conn2, r["id"], "approved")
                        finally:
                            conn2.close()
                        st.rerun()
                with rcol:
                    if st.button("❌ Reject", key=f"reject_{r['id']}"):
                        conn2 = db.connect()
                        try:
                            db.set_review_status(conn2, r["id"], "rejected")
                        finally:
                            conn2.close()
                        st.rerun()


# --- Vector tools tab --------------------------------------------------------
def render_vector_tools() -> None:
    """Embedding-geometry analysis: similarity + KMeans gap discovery (PR #3)."""
    if not qdrant_ok:
        _empty_state("🔌", "Vector geometry needs the vector index", qdrant_msg)
        return

    st.subheader("Advanced: vector geometry")
    st.caption(
        "Second opinion from embedding space only — no citation evidence. "
        "The product gap view is **Research gaps** (citation communities). "
        "This tab is a complementary geometry check, not a replacement."
    )

    st.subheader("🔗 Paper Similarity Analyzer")
    st.caption(
        "Pick any two papers to measure how close they are in meaning, visualise "
        "each paper's research neighbourhood, and see what (if anything) currently "
        "bridges the space between them."
    )
    render_proximity_analyzer(store)

    st.divider()

    st.subheader("🕳️ Vector Gap Discovery")
    st.caption(
        "Groups every paper into topics (KMeans) and finds pairs of related topics "
        "with no paper bridging them — the places new research could go."
    )
    render_vector_gap_discovery(store)


# --- Tab dispatch ------------------------------------------------------------
tab_trending, tab_graph, tab_search, tab_reading, tab_review, tab_vector = st.tabs(
    ["📈 Trending", "🕸️ Research gaps", "🔍 Search",
     "📚 Reading list", "🗂️ Review queue", "🔬 Advanced: vector geometry"]
)
with tab_trending:
    render_trending()
with tab_graph:
    render_graph_view()
with tab_search:
    render_search()
with tab_reading:
    render_reading_list()
with tab_review:
    render_review_queue()
with tab_vector:
    render_vector_tools()
