"""Stage 4 A7: measure time-to-field (POST /fields' real work) across 20
topics, p50/p95, against the real prebuilt base corpus. Writes
eval/field_latency.json.

Calls the same functions the API endpoint calls (axiom.corpus_slice.
compute_slice + axiom.fields.create_instant_field) directly rather than over
HTTP, so this measures the actual computation (encode query -> slice ->
write field row) without conflating it with a live server's request
overhead -- the number the API would add on top is a constant per-request
cost (FastAPI dispatch + JSON serialization), not what A7's p50 < 5s target
is about.

    python scripts/measure_field_latency.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db, fields as fieldsmod, graph as graphmod
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant

# 20 topics: a mix of squarely-in-corpus (retrieval/RAG/QA/LLM knowledge) and
# adjacent-but-different NLP/ML topics, so this also exercises some 'thin'
# slices, not just the easy case -- time-to-field should be fast either way,
# since both paths do the same fixed-cost work (encode + matrix-vector
# cosine + a sqlite insert), not an amount of work that scales with how many
# papers end up relevant.
TOPICS = [
    "retrieval augmented generation",
    "open domain question answering",
    "dense passage retrieval",
    "large language model hallucination",
    "in-context learning",
    "knowledge-intensive NLP tasks",
    "chain of thought prompting",
    "multi-hop question answering",
    "semantic parsing",
    "document reranking",
    "few-shot learning for language models",
    "instruction tuning",
    "vector database indexing",
    "hybrid sparse-dense retrieval",
    "long-context language models",
    "retrieval-augmented code generation",
    "fact verification with retrieval",
    "conversational question answering",
    "graph neural networks",             # adjacent, likely thinner
    "protein structure prediction",      # off-topic, should be thin
]


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    meta_conn = db.connect(config.CORPUS_DB_PATH)
    meta = db.get_corpus_meta(meta_conn)
    meta_conn.close()
    if meta is None or meta["papers_count"] == 0:
        print("Base corpus not built yet -- run scripts/build_corpus.py first.")
        return

    print(f"Loading base corpus (graph + vectors, {meta['papers_count']} papers)...")
    g = graphmod.load_graph(db.connect(config.CORPUS_DB_PATH))
    vectors = AxiomQdrant(collection=config.CORPUS_COLLECTION_NAME).fetch_dense_vectors()
    encoder = Specter2Encoder()
    encoder.encode_query("warmup")  # exclude one-time model-load cost from the measured latencies

    # A throwaway fields registry -- this script creates 20 real field rows
    # purely to measure the write path; they must never show up in the app's
    # actual "Recent fields" list.
    tmp_fields_db = Path(tempfile.mkstemp(suffix=".db")[1])
    config.FIELDS_DB_PATH = tmp_fields_db
    fconn = fieldsmod.connect_fields()
    fieldsmod.init_fields_db(fconn)

    results = []
    for topic in TOPICS:
        t0 = time.perf_counter()
        query_vector = encoder.encode_query(topic)
        row = fieldsmod.create_instant_field(
            fconn, topic=topic, corpus_graph=g, corpus_vectors=vectors, query_vector=query_vector,
        )
        elapsed = time.perf_counter() - t0
        results.append({
            "topic": topic, "seconds": round(elapsed, 4), "status": row["status"],
            "papers_total": row["papers_total"], "total_relevant": row["total_relevant"],
        })
        print(f"  {elapsed:6.3f}s  {row['status']:5s}  {row['papers_total']:4d} papers  {topic}")
    fconn.close()
    try:
        tmp_fields_db.unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows sometimes holds the sqlite file handle a moment longer; harmless, it's in temp/

    seconds = [r["seconds"] for r in results]
    report = {
        "n_topics": len(results), "corpus_papers": meta["papers_count"],
        "p50_seconds": round(percentile(seconds, 0.50), 4),
        "p95_seconds": round(percentile(seconds, 0.95), 4),
        "max_seconds": round(max(seconds), 4),
        "target_p50_seconds": 5.0,
        "meets_target": percentile(seconds, 0.50) < 5.0,
        "results": results,
    }
    out_path = config.PROJECT_ROOT / "eval" / "field_latency.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\np50={report['p50_seconds']}s  p95={report['p95_seconds']}s  "
          f"(target p50 < 5s: {'PASS' if report['meets_target'] else 'FAIL'})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
