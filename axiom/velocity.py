"""Research-trends velocity: which concepts are accelerating.

Trends complement research gaps (axiom/gaps.py): gaps show WHERE the literature
hasn't connected; velocity shows WHAT is heating up. There is no PROJECT_PLAN.md
in this repo and no persisted KEYWORD table (decision OD10) — velocity is
computed on demand from `concepts` + `papers`, the same on-demand pattern OD9
already established for community detection.

Pipeline:
  1. split the (optionally venue/year-filtered) corpus's available publication
     years into two contiguous windows at the midpoint: PRIOR (older half) and
     RECENT (newer half). This generalizes the backlog's "ACL 2024 vs 2025"
     example to a corpus that isn't venue-year-cadenced (OD7 topic-snowball).
  2. per OpenAlex concept (level >= 1 — level-0 concepts like "Computer
     science" are too broad to trend, same rationale OD9 used to drop generic
     community labels), compute its normalized frequency in each window: share
     of that window's papers carrying the concept, not a raw count, so windows
     of different sizes stay comparable.
  3. velocity = log2((recent_share + eps) / (prior_share + eps)), epsilon
     smoothed so a brand-new or vanished concept doesn't blow up to +-inf.
     Positive = rising, negative = fading.
  4. `recent_count < min_freq` flags a row low_confidence (included, not
     dropped — matches the backlog's "confidence warning when f_k < 5").

Pure-Python: stdlib only (no pandas/numpy needed for this computation).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import Collection

from axiom import config, db


@dataclass
class KeywordVelocity:
    concept: str
    recent_count: int
    prior_count: int
    recent_share: float
    prior_share: float
    velocity: float              # log2(recent_share/prior_share), smoothed
    low_confidence: bool         # recent_count < min_freq
    year_counts: dict[int, int]  # raw per-year counts, full range (sparkline data)


@dataclass
class VelocityAnalysis:
    prior_window: tuple[int, int]
    recent_window: tuple[int, int]
    total_prior: int
    total_recent: int
    insufficient_year_spread: bool   # True when the filtered corpus spans <2 years
    keywords: list[KeywordVelocity]  # sorted by velocity, descending


_EMPTY_ANALYSIS = VelocityAnalysis(
    prior_window=(0, 0), recent_window=(0, 0),
    total_prior=0, total_recent=0,
    insufficient_year_spread=True, keywords=[],
)


def _windows(years: list[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Split sorted distinct years into two contiguous halves at the midpoint."""
    lo, hi = years[0], years[-1]
    mid = lo + (hi - lo) // 2
    prior = (lo, mid)
    recent = (mid + 1, hi) if mid < hi else (mid, hi)
    return prior, recent


def compute_velocity(
    conn,
    *,
    venue: str | None = None,
    year_range: tuple[int, int] | None = None,
    paper_ids: Collection[str] | None = None,
    min_concept_level: int = config.VELOCITY_MIN_CONCEPT_LEVEL,
    min_freq: int = config.VELOCITY_MIN_FREQ,
    top_k: int = config.VELOCITY_TOP_K,
) -> VelocityAnalysis:
    """Rank concepts by velocity: normalized-frequency log2-ratio, recent vs prior window.

    `paper_ids`, when given, restricts the computation to that set (stage 4:
    a field 'slice' field shares the base corpus's db_path, so without this
    every field would otherwise see the WHOLE corpus's trends regardless of
    its own topic -- found running the real corpus for real, see OD39).
    """
    paper_year: dict[str, int] = {}
    for p in db.iter_papers(conn):
        if paper_ids is not None and p["openalex_id"] not in paper_ids:
            continue
        year = p["publication_year"]
        if year is None:
            continue
        if venue and p["venue"] != venue:
            continue
        if year_range and not (year_range[0] <= year <= year_range[1]):
            continue
        paper_year[p["openalex_id"]] = year

    if not paper_year:
        return _EMPTY_ANALYSIS

    years = sorted(set(paper_year.values()))
    prior_window, recent_window = _windows(years)
    insufficient = prior_window == recent_window

    total_prior = sum(1 for y in paper_year.values()
                       if prior_window[0] <= y <= prior_window[1])
    total_recent = sum(1 for y in paper_year.values()
                        if recent_window[0] <= y <= recent_window[1])

    concept_rows = conn.execute(
        "SELECT paper_id, concept FROM concepts WHERE level >= ?",
        (min_concept_level,),
    ).fetchall()

    # Canonicalize labels (T3.2/OD14) so synonyms (e.g. "LoRA" / "Low-Rank
    # Adaptation") accumulate into one trend line instead of splitting it.
    # Concepts with no row in concept_canonical map to themselves (identity) --
    # canonicalization is optional; velocity works fine before it's ever run.
    canon = db.canonical_map(conn)

    prior_counts: Counter = Counter()
    recent_counts: Counter = Counter()
    year_series: dict[str, Counter] = defaultdict(Counter)
    for r in concept_rows:
        year = paper_year.get(r["paper_id"])
        if year is None:
            continue
        concept = canon.get(r["concept"], r["concept"])
        year_series[concept][year] += 1
        if prior_window[0] <= year <= prior_window[1]:
            prior_counts[concept] += 1
        if recent_window[0] <= year <= recent_window[1]:
            recent_counts[concept] += 1

    eps = config.VELOCITY_EPSILON
    scored: list[KeywordVelocity] = []
    for concept in set(prior_counts) | set(recent_counts):
        rc, pc = recent_counts.get(concept, 0), prior_counts.get(concept, 0)
        r_share = rc / total_recent if total_recent else 0.0
        p_share = pc / total_prior if total_prior else 0.0
        scored.append(KeywordVelocity(
            concept=concept,
            recent_count=rc, prior_count=pc,
            recent_share=r_share, prior_share=p_share,
            velocity=log2((r_share + eps) / (p_share + eps)),
            low_confidence=rc < min_freq,
            year_counts=dict(year_series[concept]),
        ))
    scored.sort(key=lambda k: k.velocity, reverse=True)

    return VelocityAnalysis(
        prior_window=prior_window, recent_window=recent_window,
        total_prior=total_prior, total_recent=total_recent,
        insufficient_year_spread=insufficient,
        keywords=scored[:top_k],
    )


def get_top_velocity_keywords(
    conn, n: int = 50, venue: str | None = None,
    year_range: tuple[int, int] | None = None,
    paper_ids: Collection[str] | None = None,
) -> VelocityAnalysis:
    """Backlog-contract wrapper: top-n keywords by velocity for a venue/year filter."""
    return compute_velocity(conn, venue=venue, year_range=year_range, paper_ids=paper_ids, top_k=n)
