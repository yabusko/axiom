"""Qdrant wrapper for Axiom — the vector-store seam.

Owns collection lifecycle (versioned name `axiom_v1`), upserts, and
search-with-payload-filter. The UI talks to this module, never to raw Qdrant
objects, so the frontend stays decoupled from the vector store.

NOTE: this file is intentionally named `qdrant_client.py` to mirror the spec.
Python 3 uses absolute imports, so `from qdrant_client import QdrantClient`
below still resolves to the installed package, not this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from axiom import config
from axiom import sparse


@dataclass
class SearchHit:
    """One result row, rendered directly by the Streamlit page."""
    paper_id: str
    title: str
    year: int
    venue: str
    cited_by_count: int
    score: float
    concepts: list[str] = field(default_factory=list)


class AxiomQdrant:
    def __init__(self, host: str | None = None, port: int | None = None,
                 collection: str | None = None):
        self.collection = collection or config.COLLECTION_NAME
        self.client = QdrantClient(
            host=host or config.QDRANT_HOST,
            port=port or config.QDRANT_PORT,
        )

    # --- lifecycle ----------------------------------------------------------
    def recreate_collection(self) -> None:
        """Drop and recreate the collection. Used by the idempotent bootstrap.

        Uses NAMED vectors: a dense SPECTER2 vector plus a sparse BM25-style
        vector, so the same collection serves both pure-dense `search` and
        `search_hybrid`.
        """
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                config.DENSE_VECTOR_NAME: VectorParams(
                    size=config.VECTOR_SIZE, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                config.SPARSE_VECTOR_NAME: SparseVectorParams()
            },
        )

    def delete_collection(self) -> None:
        """B4: drop this collection outright (no recreate) -- used when a
        legacy/download field that owns its own private collection is
        deleted. Never call this on a shared collection (e.g. corpus_v1)."""
        self.client.delete_collection(collection_name=self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def supports_hybrid(self) -> bool:
        """True if the collection is configured with a sparse vector."""
        try:
            info = self.client.get_collection(self.collection)
            sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
            return bool(sparse_cfg) and config.SPARSE_VECTOR_NAME in sparse_cfg
        except Exception:
            return False

    # --- writes -------------------------------------------------------------
    def upsert_papers(
        self,
        points: list[dict],
        vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
        batch_size: int = 256,
    ) -> None:
        """Upsert one point per paper.

        points: list of payload dicts with keys
            paper_id, title, year, venue, cited_by_count, concepts
        vectors: parallel list of 768-dim dense embeddings.
        sparse_vectors: optional parallel list of (indices, values) BM25 vectors.
            When provided, each point also gets the named sparse vector so
            `search_hybrid` works.

        Points are upserted in batches of `batch_size` — a single request for a
        few thousand dense+sparse vectors overruns the client write timeout
        (fine at the 30-paper synthetic scale, not at real-corpus scale).
        """
        structs = []
        for idx, (payload, vec) in enumerate(zip(points, vectors)):
            named: dict = {config.DENSE_VECTOR_NAME: vec}
            if sparse_vectors is not None:
                s_idx, s_val = sparse_vectors[idx]
                named[config.SPARSE_VECTOR_NAME] = SparseVector(
                    indices=s_idx, values=s_val
                )
            structs.append(PointStruct(id=idx, vector=named, payload=payload))
        for start in range(0, len(structs), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=structs[start:start + batch_size],
            )

    # --- reads --------------------------------------------------------------
    @staticmethod
    def _build_filter(
        venues: list[str] | None, year_range: tuple[int, int] | None
    ) -> Filter | None:
        """Payload filter shared by dense and hybrid search.

        Facets combine as AND; venues match as OR within the facet.
        """
        conditions = []
        if venues:
            conditions.append(
                FieldCondition(key="venue", match=MatchAny(any=list(venues)))
            )
        if year_range:
            lo, hi = year_range
            conditions.append(
                FieldCondition(key="year", range=Range(gte=lo, lte=hi))
            )
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _to_hit(payload: dict | None, score: float) -> SearchHit:
        p = payload or {}
        return SearchHit(
            paper_id=p.get("paper_id", ""),
            title=p.get("title", ""),
            year=p.get("year", 0),
            venue=p.get("venue", ""),
            cited_by_count=p.get("cited_by_count", 0),
            concepts=p.get("concepts", []) or [],
            score=score,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = config.DEFAULT_TOP_K,
        venues: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> list[SearchHit]:
        """Dense search + native payload filter.

        Filters are applied by Qdrant (not post-filtered) so top_k is honored
        after filtering. Signature is the frozen team contract — unchanged; only
        the internal query now targets the named dense vector.
        """
        query_filter = self._build_filter(venues, year_range)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=NamedVector(
                name=config.DENSE_VECTOR_NAME, vector=query_vector
            ),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._to_hit(r.payload, r.score) for r in results]

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = config.DEFAULT_TOP_K,
        venues: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> list[SearchHit]:
        """Dense + sparse (BM25-style) retrieval fused with Reciprocal Rank Fusion.

        Runs two native Qdrant searches (the pinned client predates server-side
        fusion) and fuses by RRF in Python. The dense arm supplies semantics; the
        sparse arm supplies exact terms/acronyms. Returns the top_k by fused rank.
        """
        query_filter = self._build_filter(venues, year_range)
        n = max(config.HYBRID_CANDIDATES, top_k)

        dense_hits = self.client.search(
            collection_name=self.collection,
            query_vector=NamedVector(
                name=config.DENSE_VECTOR_NAME, vector=query_vector
            ),
            query_filter=query_filter,
            limit=n,
            with_payload=True,
        )

        s_idx, s_val = sparse.encode_query_sparse(query_text)
        sparse_hits = []
        if s_idx:  # skip the sparse arm if the query has no usable terms
            sparse_hits = self.client.search(
                collection_name=self.collection,
                query_vector=NamedSparseVector(
                    name=config.SPARSE_VECTOR_NAME,
                    vector=SparseVector(indices=s_idx, values=s_val),
                ),
                query_filter=query_filter,
                limit=n,
                with_payload=True,
            )
            # With a payload filter, Qdrant returns ALL filtered points for a
            # sparse query — including zero-overlap ones scored 0.0. Those are
            # non-matches; keep only genuine keyword hits so they don't earn RRF
            # rank credit and flatten the fusion.
            sparse_hits = [r for r in sparse_hits if r.score > 0.0]

        # Reciprocal Rank Fusion: score = sum 1/(RRF_K + rank), rank 1-based.
        fused: dict[int, float] = {}
        payloads: dict[int, dict] = {}
        for arm in (dense_hits, sparse_hits):
            for rank, r in enumerate(arm, start=1):
                fused[r.id] = fused.get(r.id, 0.0) + 1.0 / (config.RRF_K + rank)
                payloads[r.id] = r.payload

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self._to_hit(payloads[pid], score) for pid, score in ranked]

    def similar_papers(
        self,
        paper_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        venues: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> list[SearchHit]:
        """Semantic 'more like this': nearest dense neighbors of a given paper.

        Fetches the paper's stored dense vector, reuses the dense search path,
        and drops the paper itself from the results.
        """
        found, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            with_vectors=True,
            limit=1,
        )
        if not found:
            return []
        vectors = found[0].vector or {}
        dense_vec = vectors.get(config.DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if dense_vec is None:
            return []
        # Pull one extra, then drop the self-hit.
        hits = self.search(dense_vec, top_k=top_k + 1, venues=venues, year_range=year_range)
        return [h for h in hits if h.paper_id != paper_id][:top_k]

    def fetch_dense_vectors(self, batch: int = 256) -> dict[str, list[float]]:
        """Return {paper_id: dense_vector} for every point in the collection.

        Used by the gap-detection track to build per-cluster embedding centroids.
        Scrolls the whole collection (one point per paper) in pages.
        """
        out: dict[str, list[float]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                with_vectors=True,
                with_payload=True,
                limit=batch,
                offset=offset,
            )
            for p in points:
                vectors = p.vector or {}
                dense = (vectors.get(config.DENSE_VECTOR_NAME)
                         if isinstance(vectors, dict) else None)
                pid = (p.payload or {}).get("paper_id")
                if pid and dense is not None:
                    out[pid] = dense
            if offset is None:
                break
        return out


# NOTE: the P2 tracks landed differently than originally sketched here — velocity
# (OD10) reads concepts/papers straight from SQLite rather than Qdrant payloads,
# and gap detection (OD9) consumes fetch_dense_vectors() above for community
# centroids. No shared payload-filter extension was needed.
