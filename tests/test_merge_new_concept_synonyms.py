"""Offline tests for scripts/merge_new_concept_synonyms.py's pure helpers.

find_clusters/apply_merge touch a real database and real embeddings; only
the deterministic union-find and representative-picking logic are tested
here.
"""
from __future__ import annotations

from scripts import merge_new_concept_synonyms as merge_mod


def test_union_find_clusters_merges_transitively():
    labels = ["a", "b", "c", "d"]
    # a-b and b-c connected (transitively one cluster); d alone.
    clusters = merge_mod.union_find_clusters(labels, [(0, 1), (1, 2)])
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2]


def test_union_find_clusters_no_edges_means_no_clusters():
    labels = ["a", "b", "c"]
    assert merge_mod.union_find_clusters(labels, []) == []


def test_union_find_clusters_multiple_independent_clusters():
    labels = ["a", "b", "c", "d"]
    clusters = merge_mod.union_find_clusters(labels, [(0, 1), (2, 3)])
    assert len(clusters) == 2
    index_sets = {frozenset(c) for c in clusters}
    assert index_sets == {frozenset({0, 1}), frozenset({2, 3})}


def test_pick_representative_prefers_higher_count():
    members = ["rag", "retrieval augmented generation"]
    counts = {"rag": 2, "retrieval augmented generation": 8}
    assert merge_mod.pick_representative(members, counts) == "retrieval augmented generation"


def test_pick_representative_ties_broken_alphabetically():
    members = ["zed concept", "alpha concept"]
    counts = {"zed concept": 3, "alpha concept": 3}
    assert merge_mod.pick_representative(members, counts) == "alpha concept"
