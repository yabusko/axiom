"""OpenAlex ingestion — the data-layer seam (decision: OpenAlex -> SQLite).

Builds the corpus by SNOWBALL / 2-hop expansion (decision OD7): seed on a topic,
then pull in the papers the seeds cite *and* the papers that cite the seeds, so
the citation graph is connected rather than a star of dangling external edges.

Writes through the frozen db.insert_* helpers, so the shared schema/contract is
untouched. Citation edges keep dangling externals (dst outside the corpus) by
design — the schema has no FK on citation_edges.dst_id for exactly this reason.

This module is PURE-PYTHON (httpx only): it does NOT import torch/transformers,
so the fetch path runs on any Python. Embedding/upsert lives in axiom.indexer.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import httpx

from axiom import config, db


def make_venue_filter(patterns: list[str] | None) -> Callable[[dict], bool] | None:
    """Build a predicate: keep a raw OpenAlex work iff its venue display_name
    contains one of `patterns` (case-insensitive substring). None => keep all."""
    if not patterns:
        return None
    pats = [p.lower() for p in patterns]

    def keep(work: dict) -> bool:
        source = ((work.get("primary_location") or {}).get("source") or {})
        name = (source.get("display_name") or "").lower()
        return any(p in name for p in pats)

    return keep


# ---------------------------------------------------------------------------
# Parsed shapes
# ---------------------------------------------------------------------------
@dataclass
class ParsedWork:
    """One OpenAlex work, normalized to our schema's columns."""
    openalex_id: str
    title: str | None
    abstract: str | None
    publication_year: int | None
    venue_id: str | None
    venue: str | None
    cited_by_count: int
    doi: str | None
    concepts: list[tuple[str, int]] = field(default_factory=list)   # (name, level)
    references: list[str] = field(default_factory=list)             # short dst ids


# ---------------------------------------------------------------------------
# Pure helpers (no network) — easy to unit-test
# ---------------------------------------------------------------------------
def short_id(openalex_id: str | None) -> str | None:
    """'https://openalex.org/W2741809807' -> 'W2741809807' (idempotent).

    Accepts any OpenAlex entity id (W works, S sources/venues, A authors, ...):
    parse_work() reuses this to shorten both work ids and the venue's source id.
    """
    if not openalex_id:
        return None
    val = openalex_id.rsplit("/", 1)[-1]
    if not re.match(r"^[A-Z]\d+$", val):
        raise ValueError(f"Invalid OpenAlex ID format: {val}")
    return val


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain text from OpenAlex's abstract_inverted_index.

    OpenAlex stores abstracts as {word: [positions]}. We invert it back to a
    position-ordered string (decision: abstracts reconstructed before insert).
    """
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


def parse_work(work: dict) -> ParsedWork:
    """Map a raw OpenAlex work JSON onto our schema columns."""
    venue_id = None
    venue = None
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    if source:
        venue_id = short_id(source.get("id"))
        venue = source.get("display_name")

    concepts = [
        (c["display_name"], c.get("level"))
        for c in (work.get("concepts") or [])
        if c.get("display_name")
    ]
    references = [short_id(r) for r in (work.get("referenced_works") or [])]
    references = [r for r in references if r]

    return ParsedWork(
        openalex_id=short_id(work.get("id")),
        title=work.get("title"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        publication_year=work.get("publication_year"),
        venue_id=venue_id,
        venue=venue,
        cited_by_count=work.get("cited_by_count") or 0,
        doi=work.get("doi"),
        concepts=concepts,
        references=references,
    )


def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# OpenAlex HTTP client (polite pool, paging, retries)
# ---------------------------------------------------------------------------
# Only the fields we actually use, to keep responses small/fast.
_WORK_FIELDS = ",".join([
    "id", "title", "abstract_inverted_index", "publication_year",
    "primary_location", "cited_by_count", "doi", "concepts", "referenced_works",
])


class OpenAlexClient:
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None):
        self.base = config.OPENALEX_BASE
        self.api_key = api_key or config.OPENALEX_API_KEY
        self._client = client or httpx.Client(
            base_url=self.base, timeout=config.OPENALEX_TIMEOUT
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAlexClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get(self, path: str, params: dict) -> dict:
        params = {**params, **({"api_key": self.api_key} if self.api_key else {})}
        last_exc: Exception | None = None
        for attempt in range(config.OPENALEX_MAX_RETRIES):
            try:
                resp = self._client.get(path, params=params)
                if resp.status_code == 429:               # rate limited — back off
                    # Honour Retry-After when present; else exponential backoff
                    # capped so a sustained throttle waits patiently, not forever.
                    retry_after = resp.headers.get("retry-after")
                    delay = (float(retry_after) if retry_after
                             else min(2 ** attempt, config.OPENALEX_BACKOFF_CAP))
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError,) as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, config.OPENALEX_BACKOFF_CAP))
        raise RuntimeError(f"OpenAlex GET {path} failed after retries") from last_exc

    def search_seeds(self, query: str, count: int) -> list[dict]:
        """Most *relevant* works matching `query` — the snowball seed set.

        Uses the `title_and_abstract.search` filter with OpenAlex's default
        relevance ranking. A plain `cited_by_count:desc` sort is wrong here: it
        surfaces the most-cited papers in the whole index that merely contain
        the tokens (SciPy, QUANTUM ESPRESSO, ...), not on-topic work.
        """
        data = self._get("/works", {
            "filter": f"title_and_abstract.search:{query}",
            "per-page": min(count, 200),
            "select": _WORK_FIELDS,
        })
        return data.get("results", [])[:count]

    def count_works(self, query: str) -> int:
        """Total OpenAlex works matching `query` (stage 4 B1's external
        corpus-artifact check) -- `meta.count`, not the page of results
        itself, so `per-page=1` is enough and this costs the same one
        request as any other `/works` call regardless of how big the real
        total is. `title_and_abstract.search`'s multi-word filter value
        requires every word present (not an OR relevance match like the
        top-level `search=` param), which is exactly "works that combine
        both sides' key concepts" -- see axiom/external_check.py for how
        `query` gets built from a gap's two labels."""
        data = self._get("/works", {
            "filter": f"title_and_abstract.search:{query}",
            "per-page": 1,
            "select": "id",
        })
        return int(data.get("meta", {}).get("count", 0))

    def works_by_ids(self, ids: list[str], log=print) -> list[dict]:
        """Fetch metadata for specific works via an OR filter on openalex_id.

        A single batch failing (transient OpenAlex 5xx / rate-limit) skips that
        batch rather than aborting a long snowball — partial metadata is fine.
        """
        out: list[dict] = []
        for batch in _chunked(ids, config.OPENALEX_BATCH_SIZE):
            try:
                data = self._get("/works", {
                    "filter": "openalex_id:" + "|".join(batch),
                    "per-page": len(batch),
                    "select": _WORK_FIELDS,
                })
            except RuntimeError as exc:
                log(f"[ingest] works_by_ids batch skipped: {exc}")
                continue
            out.extend(data.get("results", []))
        return out

    def citers(self, paper_id: str, limit: int) -> list[dict]:
        """Works that cite `paper_id`, most-cited first (capped at `limit`)."""
        data = self._get("/works", {
            "filter": f"cites:{paper_id}",
            "sort": "cited_by_count:desc",
            "per-page": min(limit, 200),
            "select": _WORK_FIELDS,
        })
        return data.get("results", [])[:limit]

    def resolve_source_ids(self, patterns: list[str], log=print) -> list[str]:
        """Resolve `config.RELEVANT_VENUE_PATTERNS`-style name substrings to
        concrete OpenAlex source ids, ONE small /sources lookup per pattern --
        so the venue allow-list can be applied server-side
        (`primary_location.source.id:S1|S2|...`) instead of over-fetching
        /works pages and discarding most of them client-side (OD38: the old
        `make_venue_filter`/`keep()` post-filter burned through most of the
        keyless daily budget on admitted-but-discarded pages; this makes each
        /works request count fully)."""
        ids: list[str] = []
        for pattern in patterns:
            data = self._get("/sources", {
                "search": pattern, "select": "id,display_name", "per-page": 25,
            })
            matched = [short_id(s["id"]) for s in data.get("results", [])
                       if pattern.lower() in (s.get("display_name") or "").lower()]
            ids.extend(matched)
            log(f"[ingest] venue pattern {pattern!r} -> {len(matched)} source ids")
        return sorted(set(i for i in ids if i))

    def search_relevant(
        self, query: str, filter_str: str, size: int, *,
        skip_ids: set[str] | None = None, keep: Callable[[dict], bool] | None = None, log=print,
    ) -> Iterator[dict]:
        """Relevance-ranked works matching `query` within `filter_str`, cursor-paginated
        past OpenAlex's ~10k offset-pagination ceiling so `size` can grow to 20k+ without
        a different code path (stage 4 A2's requirement). Only yields NEW works (not
        already in `skip_ids`, which this mutates as it goes) that pass `keep` (e.g. a
        venue allow-list) -- `size` counts exactly what gets yielded, same contract as
        `snowball()`'s `keep`-filtered counting loop, so a caller doesn't have to
        over-fetch to compensate for a filter dropping candidates after the fact.

        This also gives scripts/build_corpus.py its resumability (OD37): a paper already
        in the corpus is cheap to skip, so re-running a fetch that was interrupted
        mid-page just re-walks pages faster rather than restarting real work.
        """
        skip_ids = skip_ids if skip_ids is not None else set()
        keep = keep or (lambda _w: True)
        cursor = "*"
        seen_new = 0
        while cursor and seen_new < size:
            data = self._get("/works", {
                "search": query,
                "filter": filter_str,
                "sort": "relevance_score:desc",
                "per-page": 200,
                "cursor": cursor,
                "select": _WORK_FIELDS,
            })
            results = data.get("results", [])
            if not results:
                break
            for w in results:
                sid = short_id(w.get("id"))
                if not sid or sid in skip_ids or not keep(w):
                    continue
                skip_ids.add(sid)
                seen_new += 1
                yield w
                if seen_new >= size:
                    break
            cursor = (data.get("meta") or {}).get("next_cursor")
            log(f"[ingest] search_relevant: {seen_new}/{size} new works so far")


# ---------------------------------------------------------------------------
# Snowball: seed -> expand cited + citers -> ~target papers
# ---------------------------------------------------------------------------
def snowball(
    client: OpenAlexClient,
    *,
    seed_query: str | None = None,
    target: int | None = None,
    seed_count: int | None = None,
    citers_per_seed: int | None = None,
    keep: Callable[[dict], bool] | None = None,
    log=print,
) -> dict[str, dict]:
    """Return {short_id: raw_work} for a connected ~`target`-paper neighborhood.

    Strategy per hop:
      1. From the current frontier, count how often each *cited* work is
         referenced (co-citation frequency) — central papers rank highest.
      2. Pull citers of frontier papers (already-fetched metadata, free edges).
      3. Fetch metadata for the top cited candidates until we reach `target`.

    `keep`: optional predicate on a raw work — only works it accepts are admitted
    to the corpus (e.g. a venue allow-list). Because the frontier only ever grows
    from admitted papers, a strict filter terminates early (under `target`) rather
    than looping forever; it never drifts off-topic.
    """
    seed_query = seed_query or config.SEED_QUERY
    target = target or config.CORPUS_TARGET
    seed_count = seed_count or config.SEED_COUNT
    citers_per_seed = citers_per_seed if citers_per_seed is not None else config.CITERS_PER_SEED
    filtering = keep is not None
    keep = keep or (lambda _w: True)

    works: dict[str, dict] = {}
    # When filtering, most search hits are off-venue, so cast a wider seed net
    # (OpenAlex caps per-page at 200) and keep only the matching ones.
    seed_fetch = max(seed_count * 4, 200) if filtering else seed_count
    for w in client.search_seeds(seed_query, seed_fetch):
        sid = short_id(w.get("id"))
        if sid and sid not in works and keep(w):
            works[sid] = w
            if len(works) >= seed_count:
                break
    log(f"[ingest] {len(works)} seeds for query={seed_query!r}")

    frontier = list(works.keys())
    hop = 0
    while frontier and len(works) < target:
        hop += 1
        next_frontier: list[str] = []

        # 1. citers of frontier papers — adds citer->frontier edges. One request
        # per frontier paper, so a growing frontier explodes request count; cap
        # how many we expand per hop to keep runtime bounded (the rest still feed
        # the co-citation step below).
        for sid in frontier[:config.SNOWBALL_MAX_CITERS_PER_HOP]:
            if len(works) >= target:
                break
            try:
                citers = client.citers(sid, citers_per_seed)
            except RuntimeError as exc:
                log(f"[ingest] citers({sid}) skipped: {exc}")
                continue
            for w in citers:
                cid = short_id(w.get("id"))
                if cid and cid not in works and keep(w):
                    works[cid] = w
                    next_frontier.append(cid)
                    if len(works) >= target:
                        break

        # 2. cited works ranked by co-citation frequency across the frontier.
        cited_freq: Counter[str] = Counter()
        for sid in frontier:
            for ref in (works[sid].get("referenced_works") or []):
                rid = short_id(ref)
                if rid and rid not in works:
                    cited_freq[rid] += 1

        need = target - len(works)
        if need > 0 and cited_freq:
            # Over-fetch candidates since many will be filtered out by venue.
            to_fetch = [rid for rid, _ in cited_freq.most_common(need * 4)]
            for w in client.works_by_ids(to_fetch, log=log):
                wid = short_id(w.get("id"))
                if wid and wid not in works and keep(w):
                    works[wid] = w
                    next_frontier.append(wid)
                    if len(works) >= target:
                        break

        log(f"[ingest] hop {hop}: corpus={len(works)} (+{len(next_frontier)})")
        frontier = next_frontier

    return works


# ---------------------------------------------------------------------------
# Write parsed works into SQLite (idempotent: wipes prior rows first)
# ---------------------------------------------------------------------------
def write_works(
    conn,
    works: Iterable[dict],
    *,
    source: str = "openalex",
    log=print,
) -> int:
    """Parse + UPSERT works into SQLite. Returns the paper count written.

    Non-destructive (no reset_corpus) -- `insert_paper` is already `INSERT OR
    REPLACE`, so calling this again with overlapping works is safe. Used by
    both the legacy per-field downloader (via `ingest_to_sqlite` below, which
    wraps a reset around this) and scripts/build_corpus.py's fetch/--extend
    steps, which need to ADD to an existing corpus rather than wipe it.

    Citation edges are written for EVERY referenced_work, including dangling
    ones whose dst is outside the corpus (kept by design — flagged downstream
    in the graph layer as in_corpus=False). Concepts/citation-edges have no
    uniqueness constraint, so re-running this for a paper already present
    duplicates its concept/edge rows -- callers that care about true
    idempotency (scripts/build_corpus.py) only pass NEW works in the first
    place (see OpenAlexClient.search_relevant's skip_ids).
    """
    parsed = [parse_work(w) for w in works]
    parsed = [p for p in parsed if p.openalex_id]

    for p in parsed:
        db.insert_paper(
            conn,
            openalex_id=p.openalex_id,
            title=p.title,
            abstract=p.abstract,
            publication_year=p.publication_year,
            venue_id=p.venue_id,
            venue=p.venue,
            cited_by_count=p.cited_by_count,
            doi=p.doi,
        )
        if p.concepts:
            db.insert_concepts(conn, p.openalex_id, p.concepts)
        db.insert_provenance(
            conn, paper_id=p.openalex_id, source=source,
            abstract=p.abstract, has_fulltext=False,
        )
        if p.references:
            db.insert_citation_edges(
                conn,
                [(p.openalex_id, dst, p.publication_year) for dst in p.references],
            )
    conn.commit()
    return len(parsed)


def ingest_to_sqlite(
    conn,
    works: Iterable[dict],
    *,
    source: str = "openalex",
    log=print,
) -> int:
    """Legacy per-field downloader: wipe the corpus, then write_works() the
    given set. Kept for the --extend path's internal snowball reuse and any
    remaining synthetic-bootstrap callers; scripts/build_corpus.py never calls
    this (it must not wipe the shared base corpus)."""
    db.init_db(conn)
    db.reset_corpus(conn)
    n = write_works(conn, works, source=source, log=log)
    edge_count = conn.execute("SELECT COUNT(*) AS n FROM citation_edges").fetchone()["n"]
    log(f"[ingest] wrote {n} papers, {edge_count} citation edges to {config.DB_PATH}")
    return n
