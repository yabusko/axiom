"""Embedding map (stage-3 product redesign, Map screen): a 2D/3D PCA
projection of paper vectors for spatial browsing, plus cosine-similarity
explanations for the "select two papers/topics" comparison UI.

PCA, not UMAP: UMAP needs numba/pynndescent, which breaks this repo's
established "no new heavy deps" bar (`gaps.py`: "Pure-Python: networkx +
numpy only"). PCA itself is plain `numpy.linalg.svd` (stage 4 B5) -- the
original draft used `sklearn.decomposition.PCA` since scikit-learn was
already pinned, but that broke the "whole suite runs on bare Python" CI
requirement: sklearn has no wheel for the 3.14 dev environment this repo's
test stubs already accommodate for torch/transformers/qdrant_client
(tests/conftest.py), and pytest collecting this module would crash on
import before a single test ran. A plain SVD is numerically identical to
what PCA does internally for a matrix this shape (sklearn's own 'full'
solver IS `numpy.linalg.svd`) and is deterministic with no randomness to
seed, so dropping it costs nothing.
"""
from __future__ import annotations

import numpy as np


def project(vectors: dict[str, list[float]], n_components: int = 3) -> dict[str, list[float]]:
    """PCA-project each paper's dense vector to `n_components` axes, rescaled
    to roughly [-50, 50] per axis so a 3D viewer gets a sane default camera
    framing regardless of the embedding model's native scale.

    Deterministic given the same `vectors` (mean-center + SVD, no random
    source anywhere) -- the API layer caches this per field_id on that
    assumption.
    """
    ids = list(vectors)
    if len(ids) < 2:
        return {pid: [0.0] * n_components for pid in ids}
    X = np.asarray([vectors[pid] for pid in ids], dtype=float)
    k = min(n_components, X.shape[0], X.shape[1])
    centered = X - X.mean(axis=0)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :k] * s[:k]
    if coords.shape[1] < n_components:
        coords = np.hstack([coords, np.zeros((coords.shape[0], n_components - coords.shape[1]))])
    span = float(np.abs(coords).max()) or 1.0
    coords = coords / span * 50.0
    return {pid: coords[i].tolist() for i, pid in enumerate(ids)}


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# (floor, plain-language band) -- checked high to low.
_BANDS = (
    (0.85, "near-identical in topic — expect heavy vocabulary and method overlap"),
    (0.65, "closely related — similar problem space, likely shared methods"),
    (0.40, "loosely related — some shared vocabulary, different core focus"),
    (0.00, "distant — little topical overlap in this embedding space"),
)


def explain(cos: float) -> str:
    """One plain-language sentence for a cosine similarity value, for readers
    with no stats background."""
    for floor, phrase in _BANDS:
        if cos >= floor:
            return f"{cos * 100:.0f}% cosine similarity — {phrase}."
    return f"{cos * 100:.0f}% cosine similarity."
