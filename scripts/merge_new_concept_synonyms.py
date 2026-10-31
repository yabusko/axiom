"""Merge synonymous `new_concept` labels in `paper_concepts` (B3 follow-up, OD28).

B3's real calibration run found that most of B1's `new_concept` fallback rate (96% of
entities in the reference run) is an ontology-gap problem, not a miscalibrated threshold:
loosening tau/delta to the calibrated point only flipped 6/208 `new_concept` entities to a
real ontology match. Those entities genuinely aren't in the ontology -- but many of them are
the SAME concept named differently across papers (e.g. "RAG" in one paper's new_concept
canonical vs "retrieval augmented generation" in another's), which fragments the concept
graph into near-duplicate nodes instead of one real node.

This script runs the same matching logic B1 uses against the ontology, except against the
`new_concept` canonicals themselves: exact/normalized match, then a short-phrase embedding
match gated on the same calibrated cosine threshold (no margin requirement here -- there's
no "runner-up" concept to compare against when clustering a set against itself, just pairwise
similarity). Connected components under that threshold get merged to one representative
canonical (the most-occurring raw name in the cluster, ties broken alphabetically).

Idempotent: safe to re-run after a pipeline run that adds more new_concept entities -- an
already-merged cluster has nothing left to cluster.

Usage (from repo root):
    python scripts/merge_new_concept_synonyms.py --tau 0.93 --dry-run
    python scripts/merge_new_concept_synonyms.py --tau 0.93
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom.agents import enrichment, verifier

DEFAULT_TAU = 0.93  # OD28's calibrated embedding-match threshold


def union_find_clusters(labels: list[str], adjacency: list[tuple[int, int]]) -> list[list[int]]:
    """Plain union-find over `labels`' indices, connected by `adjacency` edges."""
    parent = list(range(len(labels)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in adjacency:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(labels)):
        groups.setdefault(find(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]


def pick_representative(members: list[str], counts: dict[str, int]) -> str:
    return sorted(members, key=lambda m: (-counts.get(m, 0), m))[0]


async def find_clusters(conn, *, tau: float) -> tuple[list[list[str]], dict[str, int]]:
    rows = conn.execute(
        "SELECT canonical, COUNT(*) AS n FROM paper_concepts WHERE cosine = -1.0 "
        "GROUP BY canonical"
    ).fetchall()
    labels = [r["canonical"] for r in rows]
    counts = {r["canonical"]: r["n"] for r in rows}
    if len(labels) < 2:
        return [], counts

    encoder = enrichment.PhraseEncoder()
    import numpy as np
    raw = await asyncio.to_thread(encoder.encode, labels)
    vectors = enrichment._normalize_rows(np.array(raw, dtype=float))
    sims = vectors @ vectors.T

    # Exact-normalized match first (free, catches case/whitespace-only variants already
    # missed because `new_concept`'s own canonical is already normalize_text'd once --
    # still worth a second pass since two DIFFERENT raw names can normalize identically).
    normalized = [verifier.normalize_text(label) for label in labels]
    edges: list[tuple[int, int]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if normalized[i] == normalized[j] or sims[i, j] >= tau:
                edges.append((i, j))

    index_clusters = union_find_clusters(labels, edges)
    label_clusters = [[labels[i] for i in members] for members in index_clusters]
    return label_clusters, counts


def apply_merge(conn, clusters: list[list[str]], counts: dict[str, int]) -> dict:
    """Rewrites paper_concepts/concept_cooccurrence to collapse each cluster onto one
    representative canonical. Returns summary stats. Caller commits."""
    merges = {}  # old canonical -> representative, every non-representative member
    for members in clusters:
        rep = pick_representative(members, counts)
        for m in members:
            if m != rep:
                merges[m] = rep

    rows_updated = 0
    rows_deleted_as_duplicates = 0
    for old, new in merges.items():
        # A paper that already has BOTH `old` and `new` as separate new_concept rows (same
        # kind) would collide with the unique index once `old` is renamed to `new` -- drop
        # the `old` row for that (paper_id, kind) first, keep `new`'s.
        conflicts = conn.execute(
            "SELECT a.id FROM paper_concepts a JOIN paper_concepts b "
            "ON a.paper_id = b.paper_id AND a.kind = b.kind "
            "WHERE a.canonical = ? AND b.canonical = ? AND a.cosine = -1.0 AND b.cosine = -1.0",
            (old, new),
        ).fetchall()
        if conflicts:
            ids = [r["id"] for r in conflicts]
            conn.executemany("DELETE FROM paper_concepts WHERE id = ?", [(i,) for i in ids])
            rows_deleted_as_duplicates += len(ids)

        cur = conn.execute(
            "UPDATE paper_concepts SET canonical = ? WHERE canonical = ? AND cosine = -1.0",
            (new, old),
        )
        rows_updated += cur.rowcount

        conn.execute(
            "UPDATE concept_cooccurrence SET concept_a = ? WHERE concept_a = ?", (new, old)
        )
        conn.execute(
            "UPDATE concept_cooccurrence SET concept_b = ? WHERE concept_b = ?", (new, old)
        )

    # A merge can make concept_a == concept_b (the two sides of a co-occurrence pair were
    # merged into the same concept) or create an exact duplicate pair -- clean up both.
    # concept_cooccurrence has no declared id column; SQLite's implicit rowid stands in.
    conn.execute("DELETE FROM concept_cooccurrence WHERE concept_a = concept_b")
    conn.execute("""
        DELETE FROM concept_cooccurrence WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM concept_cooccurrence GROUP BY run_id, paper_id, concept_a, concept_b
        )
    """)

    return {
        "clusters_merged": len(merges) and len({v for v in merges.values()}),
        "canonicals_absorbed": len(merges),
        "paper_concepts_rows_updated": rows_updated,
        "paper_concepts_rows_deleted_as_duplicates": rows_deleted_as_duplicates,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    conn = db.connect(config.DB_PATH)
    try:
        clusters, counts = asyncio.run(find_clusters(conn, tau=args.tau))
        print(f"[merge] {len(clusters)} cluster(s) found at tau={args.tau}")
        for members in clusters:
            rep = pick_representative(members, counts)
            others = [m for m in members if m != rep]
            print(f"  -> {rep!r} absorbs {others!r}")

        if args.dry_run or not clusters:
            print("[merge] dry run -- no changes written" if args.dry_run else "[merge] nothing to merge")
            return 0

        stats = apply_merge(conn, clusters, counts)
        conn.commit()
        print(f"[merge] {stats}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
