"""Shared test fixtures + heavy-dependency stubs.

The crux is the stubbing block below: it runs at import time (before any test
module imports `axiom.*`/`api.*`) and installs lightweight `sys.modules` entries
for torch/transformers/qdrant_client **only when those packages are absent**.

Why not `pytest.importorskip`? The primary dev machine runs Python 3.14, where
the pinned torch/transformers/qdrant-client have no wheels. Skipping there would
hide the entire API/search surface from the suite. Stubbing lets the *same* tests
run on 3.12 (real deps installed → stubs are never created) and on 3.14 (stubs
active). Tests never construct `Specter2Encoder`/`AxiomQdrant`; they only need the
modules importable so `axiom.qdrant_client` / `api.main` import cleanly, and use
fakes (below) in their place.
"""
from __future__ import annotations

import importlib.util
import sys
import types


def _missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Parent package missing → the submodule is missing too.
        return True


class _AnyCtor:
    """Stand-in class/type that swallows any constructor args."""

    def __init__(self, *args, **kwargs):
        pass


def _install_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


# torch: embed.py imports it and touches torch.cuda / torch.backends only inside
# Specter2Encoder (never instantiated in tests), BUT `@torch.no_grad()` decorates
# a method at class-definition time, so the stub must supply a no-op decorator.
if _missing("torch") and "torch" not in sys.modules:
    def _no_grad(*args, **kwargs):
        def _decorate(fn):
            return fn
        return _decorate
    _install_stub("torch", no_grad=_no_grad)

# transformers: embed.py imports AutoModel/AutoTokenizer at module level.
if _missing("transformers") and "transformers" not in sys.modules:
    _install_stub("transformers", AutoModel=_AnyCtor, AutoTokenizer=_AnyCtor)

# qdrant_client + qdrant_client.models: qdrant_client.py imports many model names
# at module level. They only need to exist as importable symbols.
if _missing("qdrant_client") and "qdrant_client" not in sys.modules:
    qc = _install_stub("qdrant_client", QdrantClient=_AnyCtor)
    model_names = (
        "Distance", "FieldCondition", "Filter", "MatchAny", "MatchValue",
        "NamedSparseVector", "NamedVector", "PointStruct", "Range",
        "SparseVector", "SparseVectorParams", "VectorParams",
    )
    models = _install_stub(
        "qdrant_client.models",
        **{n: type(n, (_AnyCtor,), {}) for n in model_names},
    )
    qc.models = models


# --- Fixtures ----------------------------------------------------------------
import pytest  # noqa: E402  (must follow the stub block)

from axiom import config, db  # noqa: E402


def _init_db(path) -> None:
    conn = db.connect(path)
    try:
        db.init_db(conn)
    finally:
        conn.close()


@pytest.fixture
def empty_conn(tmp_path):
    """A fresh SQLite connection over the real schema, empty. Test owns the corpus."""
    path = tmp_path / "empty.db"
    _init_db(path)
    conn = db.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _seed_corpus(conn) -> None:
    """A small deterministic corpus: two 4-paper clusters + one dangling external.

    Cluster A (retrieval / LoRA) papers A1..A4, cluster B (translation) B1..B4.
    Dense intra-cluster citations, a single A1->B1 bridge, and A1->WEXT1 pointing
    at a paper outside the corpus (exercises the dangling-node path).
    """
    papers = [
        # id, title, abstract, year, venue, cites
        ("A1", "Retrieval augmented generation", "dense retrieval for QA", 2020, "ACL", 100),
        ("A2", "Dense passage retrieval", "retrieval over passages", 2021, "ACL", 40),
        ("A3", "LoRA for retrieval", "low rank adaptation of retrievers", 2022, "EMNLP", 20),
        ("A4", "Retrieval reranking", "reranking retrieved passages", 2024, "EMNLP", 10),
        ("B1", "Neural machine translation", "translation with transformers", 2020, "ACL", 90),
        ("B2", "Low resource translation", "translation for low resource langs", 2021, "ACL", 35),
        ("B3", "Multilingual translation", "many to many translation", 2023, "EMNLP", 15),
        ("B4", "Translation evaluation", "metrics for translation quality", 2025, "EMNLP", 5),
    ]
    for pid, title, abstract, year, venue, cites in papers:
        db.insert_paper(conn, openalex_id=pid, title=title, abstract=abstract,
                        publication_year=year, venue_id=None, venue=venue,
                        cited_by_count=cites, doi=f"10.0/{pid}")

    concepts = {
        "A1": [("Computer science", 0), ("Information retrieval", 1), ("LoRA", 2)],
        "A2": [("Computer science", 0), ("Information retrieval", 1)],
        "A3": [("Computer science", 0), ("Information retrieval", 1), ("Low-Rank Adaptation", 2)],
        "A4": [("Computer science", 0), ("Information retrieval", 1)],
        "B1": [("Computer science", 0), ("Machine translation", 1)],
        "B2": [("Computer science", 0), ("Machine translation", 1)],
        "B3": [("Computer science", 0), ("Machine translation", 1)],
        "B4": [("Computer science", 0), ("Machine translation", 1)],
    }
    for pid, cs in concepts.items():
        db.insert_concepts(conn, pid, cs)

    # Dense intra-cluster edges + a single inter-cluster bridge + a dangling ext.
    edges = [
        ("A2", "A1", 2021), ("A3", "A1", 2022), ("A3", "A2", 2022), ("A4", "A1", 2024),
        ("A4", "A2", 2024), ("A4", "A3", 2024),
        ("B2", "B1", 2021), ("B3", "B1", 2023), ("B3", "B2", 2023), ("B4", "B1", 2025),
        ("B4", "B2", 2025), ("B4", "B3", 2025),
        ("A1", "B1", 2020),          # single inter-cluster bridge
        ("A1", "WEXT1", 2020),       # dangling external target (not in papers)
    ]
    db.insert_citation_edges(conn, edges)
    conn.commit()


# Vectors for the seeded corpus. Cluster A ~ [1,0,0], cluster B ~ [1,1,0]:
# centroid cosine = 1/sqrt(2) ≈ 0.7071, a stable value the gap/API tests assert on.
SEED_VECTORS = {
    "A1": [1.0, 0.0, 0.0], "A2": [1.0, 0.0, 0.0], "A3": [1.0, 0.0, 0.0], "A4": [1.0, 0.0, 0.0],
    "B1": [1.0, 1.0, 0.0], "B2": [1.0, 1.0, 0.0], "B3": [1.0, 1.0, 0.0], "B4": [1.0, 1.0, 0.0],
}


@pytest.fixture
def seeded_db(tmp_path):
    """Path to a tmp SQLite db populated with the standard two-cluster corpus."""
    path = tmp_path / "axiom.db"
    _init_db(path)
    conn = db.connect(path)
    try:
        _seed_corpus(conn)
    finally:
        conn.close()
    return path


@pytest.fixture
def seeded_conn(seeded_db):
    conn = db.connect(seeded_db)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def patched_db_path(seeded_db, monkeypatch):
    """Point every db.connect() (routes, UI helpers) at the seeded tmp db."""
    monkeypatch.setattr(config, "DB_PATH", seeded_db)
    return seeded_db


class FakeEncoder:
    def encode_query(self, query: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class FakeStore:
    """Minimal AxiomQdrant stand-in backed by the seeded vectors."""

    def __init__(self, vectors=None, hybrid=True):
        self._vectors = dict(vectors or SEED_VECTORS)
        self._hybrid = hybrid
        self.raise_on_count = False

    def count(self) -> int:
        if self.raise_on_count:
            raise RuntimeError("qdrant down")
        return len(self._vectors)

    def supports_hybrid(self) -> bool:
        return self._hybrid

    def fetch_dense_vectors(self, batch: int = 256):
        return dict(self._vectors)

    def _hits(self, top_k):
        from axiom.qdrant_client import SearchHit
        out = []
        for i, pid in enumerate(list(self._vectors)[:top_k]):
            out.append(SearchHit(paper_id=pid, title=f"title {pid}", year=2020,
                                 venue="ACL", cited_by_count=1, score=1.0 - i * 0.01,
                                 concepts=["information retrieval"]))
        return out

    def search(self, *, query_vector, top_k=10, venues=None, year_range=None):
        return self._hits(top_k)

    def search_hybrid(self, *, query_vector, query_text, top_k=10, venues=None, year_range=None):
        return self._hits(top_k)

    def similar_papers(self, paper_id, top_k=10, venues=None, year_range=None):
        return self._hits(top_k)


@pytest.fixture
def fake_store():
    return FakeStore()


@pytest.fixture
def fake_llm(monkeypatch):
    """Programmable axiom.llm.chat_json. Push dicts (or Exceptions) to `.responses`."""
    from axiom import llm

    class _Queue:
        def __init__(self):
            self.responses = []
            self.calls = []

        def __call__(self, prompt, *, system=None, model=None, temperature=0.2, timeout=None):
            self.calls.append(prompt)
            item = self.responses.pop(0) if self.responses else {}
            if isinstance(item, Exception):
                raise item
            return item

    q = _Queue()
    monkeypatch.setattr(llm, "chat_json", q)
    return q


@pytest.fixture
def api_client(patched_db_path, fake_store, monkeypatch):
    """FastAPI TestClient with the encoder/store/graph/gap singletons faked.

    Graph + gap analysis run for real over the seeded db (cheap, pure Python);
    only the torch encoder and Qdrant store are replaced.
    """
    from fastapi.testclient import TestClient

    import api.main as main
    from axiom import gaps as gapsmod
    from axiom import graph as graphmod

    for fn in (main.get_encoder, main.get_store, main.get_graph, main.get_gap_analysis):
        fn.cache_clear()

    monkeypatch.setattr(main, "get_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(main, "get_store", lambda: fake_store)

    def _graph():
        conn = db.connect(patched_db_path)
        try:
            return graphmod.load_graph(conn)
        finally:
            conn.close()

    def _gaps():
        conn = db.connect(patched_db_path)
        try:
            return gapsmod.analyze(_graph(), conn, fake_store.fetch_dense_vectors(),
                                   min_community_size=4)
        finally:
            conn.close()

    monkeypatch.setattr(main, "get_graph", _graph)
    monkeypatch.setattr(main, "get_gap_analysis", _gaps)

    return TestClient(main.app)


@pytest.fixture
def field_client(api_client, patched_db_path, fake_store, monkeypatch, tmp_path):
    """api_client, plus a registered field ("test-field") pointing at the same
    seeded db/fake store the global endpoints use -- for the /fields/* routes,
    which read a field row's own db_path/collection rather than config.DB_PATH/
    get_store(). Returns (client, field_id)."""
    import api.main as main
    from axiom import fields as fieldsmod

    monkeypatch.setattr(config, "FIELDS_DB_PATH", tmp_path / "fields.db")
    monkeypatch.setattr(main, "_field_store", lambda row: fake_store)

    fconn = fieldsmod.connect_fields()
    try:
        fieldsmod.init_fields_db(fconn)
        row = fieldsmod.create_field(fconn, topic="retrieval augmented generation", target=8)
        fconn.execute(
            "UPDATE fields SET db_path = ?, collection = ?, status = 'ready' WHERE id = ?",
            (str(patched_db_path), "test-collection", row["id"]),
        )
        fconn.commit()
        field_id = row["id"]
    finally:
        fconn.close()

    return api_client, field_id
