"""Embedding map (stage-3 product redesign, Map screen): a 2D/3D PCA
projection of paper vectors for spatial browsing, plus cosine-similarity
explanations for the "select two papers/topics" comparison UI.

PCA, not UMAP: UMAP needs numba/pynndescent, which breaks this repo's
established "no new heavy deps" bar (`gaps.py`: "Pure-Python: networkx +
numpy only"). `scikit-learn` is already pinned (requirements.txt) and unused
elsewhere, so `sklearn.decomposition.PCA` costs nothing new. Deferred, not
decided against -- revisit if a real need for UMAP's better global structure
shows up.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def project(vectors: dict[str, list[float]], n_components: int = 3) -> dict[str, list[float]]:
    """PCA-project each paper's dense vector to `n_components` axes, rescaled
    to roughly [-50, 50] per axis so a 3D viewer gets a sane default camera
    framing regardless of the embedding model's native scale.

    Deterministic given the same `vectors` (fixed `random_state`, no other
    source of randomness) -- the API layer caches this per field_id on that
    assumption.
    """
    ids = list(vectors)
    if len(ids) < 2:
        return {pid: [0.0] * n_components for pid in ids}
    X = np.asarray([vectors[pid] for pid in ids], dtype=float)
    k = min(n_components, X.shape[0], X.shape[1])
    coords = PCA(n_components=k, random_state=0).fit_transform(X)
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
