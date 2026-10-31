"""Central configuration for Axiom P1.

Single source of truth for paths, Qdrant connection, collection name, and the
embedding model id. Every other module imports from here so there are no magic
strings scattered across the codebase.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import truststore

# OD24/OD26: make every httpx/requests call in the app trust the OS certificate
# store instead of certifi's bundled list. On a network with a TLS-inspecting
# proxy (the dev laptop this was found on), certifi lacks that proxy's root
# CA while Windows already trusts it -- without this, every outbound HTTPS
# call (Anthropic API, HuggingFace Hub) fails with CERTIFICATE_VERIFY_FAILED
# even though a plain OS-level TLS handshake succeeds. Must run before any
# module constructs an SSLContext, so it lives at the top of config.py, which
# everything else imports first. Opt-in only (OD26): this rewires trust for
# the whole process, which is the right call on this inspected-TLS laptop but
# would be a silent, surprising security-posture change on a network that
# doesn't need it -- so it's gated behind an explicit env var rather than
# unconditional, and documented in the README troubleshooting section.
if os.environ.get("AXIOM_OS_TRUSTSTORE") == "1":
    truststore.inject_into_ssl()

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "axiom.db"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def _load_dotenv(path: Path) -> None:
    """Minimal `.env` loader -- no new dependency (repo keeps the stack pinned
    and minimal, see requirements.txt). Only fills in vars not already set in
    the real environment, so an exported ANTHROPIC_API_KEY always wins.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")

# --- Qdrant ------------------------------------------------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# Versioned collection name: bump the suffix when the embedding model or vector
# dimensionality changes so old/new points never mix.
COLLECTION_NAME = "axiom_v1"

# --- OpenAlex ingestion ------------------------------------------------------
# Source of truth (decision: Data layer = OpenAlex -> SQLite). No API key needed;
# OpenAlex asks only for a `mailto` to join the faster "polite pool".
#
# Corpus is built by SNOWBALL / 2-hop expansion (decision OD7): seed on a topic,
# then pull in the papers the seeds cite (referenced_works[]) AND papers that
# cite the seeds. This densifies *internal* citation edges so the graph is
# actually connected, instead of a star of dangling external edges.
OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_MAILTO = "yakub.yakubov.business@gmail.com"   # polite-pool contact
OPENALEX_TIMEOUT = 30.0                                 # per-request seconds
OPENALEX_MAX_RETRIES = 6                                # transient-error retries (rides out 429 throttles)
OPENALEX_BACKOFF_CAP = 30.0                             # max seconds between retries

SEED_QUERY = "retrieval-augmented generation"           # default seed topic
SEED_COUNT = 50                                         # most-cited seeds to start
CORPUS_TARGET = 500                                     # snowball stops at ~this many papers
CITERS_PER_SEED = 25                                    # capped citers pulled per seed (per hop)
OPENALEX_BATCH_SIZE = 50                                # ids per `openalex_id:a|b|...` fetch
SNOWBALL_MAX_CITERS_PER_HOP = 300                       # cap citer-expansion calls/hop (bounds runtime)

# Venue allow-list for the snowball (case-insensitive substring match on the
# OpenAlex source display_name). A topic snowball drifts far off-seed via citers
# and cited works (a RAG seed pulls in clinical-trial and manufacturing papers),
# which floods gap/velocity analysis with irrelevant industries. When active,
# only papers whose venue matches one of these patterns are admitted, keeping the
# corpus in the NLP/ML world the project targets (ACL Anthology + top ML venues).
RELEVANT_VENUE_PATTERNS = [
    "computational linguistics",                    # ACL, NAACL, EACL, COLING, TACL, Findings, CL journal
    "empirical methods in natural language",        # EMNLP
    "neural information processing systems",        # NeurIPS
    "learning representations",                     # ICLR
    "international conference on machine learning",  # ICML
    "aaai",                                          # AAAI
]

# --- Embeddings --------------------------------------------------------------
# Committed team decision: SPECTER2, 768-dim. Do NOT silently swap models.
MODEL_ID = "allenai/specter2_base"
VECTOR_SIZE = 768

# Default search fan-out.
DEFAULT_TOP_K = 10

# --- Query-side embedding ----------------------------------------------------
# Terse queries ("LoRA", "TOOLS for AI") embed poorly when treated as documents
# and land in a mushy region of the space (flat ~0.82-0.88 scores). Two paths
# nudge the query vector toward the document region:
#   - "adapter":   SPECTER2 ad-hoc query adapter. Best, BUT the `adapters` lib
#                  pins transformers~=4.57, conflicting with our pinned 4.40.2,
#                  so it is unavailable without a committed-decision pin bump.
#   - "expansion": wrap the terse query in an abstract-shaped sentence. No new
#                  deps; works under the current pins. This is the active path.
QUERY_EXPANSION_TEMPLATE = "This paper presents {query} for natural language processing."

# --- Hybrid retrieval (dense + sparse) ---------------------------------------
# Dense SPECTER2 captures meaning; a sparse BM25-style vector captures exact
# terms/acronyms (LoRA, RAG, dataset names) that dense misses. Each point stores
# two named vectors; results are fused with Reciprocal Rank Fusion.
#
# NOTE: qdrant-client==1.9.0 has sparse vectors but NOT the Query-API fusion
# primitives (Prefetch/FusionQuery, client >= 1.10). We therefore run two native
# searches and fuse with RRF in Python — no pin bump required.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_VOCAB_SIZE = 2 ** 20      # hashing-trick index space (fits uint32)
BM25_K1 = 1.5                    # BM25 term-frequency saturation
BM25_B = 0.75                    # BM25 length normalization
RRF_K = 60                       # reciprocal-rank-fusion constant
HYBRID_CANDIDATES = 20           # candidates pulled from each arm before fusion
# Tiny stopword set so terse keyword queries aren't dominated by glue words.
SPARSE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "by", "as", "at", "that", "this", "from", "how",
})

# --- Trends / velocity --------------------------------------------------------
# No PROJECT_PLAN.md exists in this repo (decision OD10) so there is no formula
# to inherit; velocity is computed on-demand (no persisted KEYWORD table) as a
# two-window normalized-frequency log2-ratio. See axiom/velocity.py.
VELOCITY_MIN_CONCEPT_LEVEL = 1   # drop level-0 concepts (e.g. "Computer science") — too broad to trend
VELOCITY_MIN_FREQ = 5            # recent_count below this => low_confidence flag
VELOCITY_TOP_K = 50              # default keywords returned by get_top_velocity_keywords
VELOCITY_EPSILON = 1e-4          # smoothing so the log2 ratio never blows up on a 0 count
# A concept seen in only ONE paper in a window isn't a trend — its velocity pins
# to the epsilon ceiling (every 0->1 concept ties at the same value), so the
# Trending charts would show a flat wall of identical bars. Require at least this
# many papers in the window a concept is moving from/to before charting it; the
# full ranked list below the charts still shows everything, low-volume flagged.
VELOCITY_MIN_CHART_COUNT = 2

# Floor for the Trending tab's year filter. Snowball ingestion drags in cited
# references back decades, so the full span (e.g. 1975-2026) splits into a
# near-empty pre-2000 "prior" window and a huge "recent" one, making almost every
# concept a fake 0->N riser. We clamp the selectable "From year" to this floor so
# velocity is only ever computed over the modern, densely-populated years.
VELOCITY_MIN_YEAR = 2020

# --- Eval corpus (OD11, PBI 8 / Task 8.2) -------------------------------------
# Real ACL/EMNLP/COLING/NAACL corpus (2020-2025, ~23k papers), pulled from the
# team's `main` branch for retrieval eval only. Kept in a SEPARATE SQLite db +
# Qdrant collection from the gaps/velocity corpus (data/axiom.db, axiom_v1):
# this dataset has no citation edges and no concept/keyword tags, so it cannot
# feed OD9 gap detection or the OD10 velocity engine — only search/nDCG@10 eval.
EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "FULL_DATASET.jsonl"
EVAL_DB_PATH = PROJECT_ROOT / "data" / "eval.db"
EVAL_COLLECTION_NAME = "axiom_eval_v1"
EVAL_SAMPLE_SIZE = 1500          # stratified across (venue, year) buckets

# --- Composite gap G-score + threshold calibration (OD17, PBI 8 / Task 8.1) ---
# OD16 deferred a composite G-score for lack of a formula to calibrate. OD17
# reverses that now that a calibration scaffold exists (scripts/export_gap_labels
# + scripts/calibrate_gap_thresholds): G blends four normalized signals already
# on the OD9 gap candidates. These weights and τ are UNCALIBRATED DEFAULTS — the
# calibration script overwrites eval/calibration.json from human labels, and
# load_calibration() below prefers that file when present.
G_SCORE_WEIGHTS = {
    "similarity":    0.4,   # centroid cosine — how topically related the two clusters are
    "disconnection": 0.3,   # 1/(1+inter_citations) — how weakly they cite each other
    "velocity":      0.2,   # logistic of mean concept velocity — how "hot" the region is
    "authority":     0.1,   # normalized mean log(cited_by) — how substantial the clusters are
}
GAP_TAU_DEFAULT = 0.65      # G >= τ flags a candidate "meets threshold" (dev default, uncalibrated)
CALIBRATION_PATH = PROJECT_ROOT / "eval" / "calibration.json"


def load_calibration() -> dict:
    """Return {'tau': float, 'weights': {...}}, calibrated file merged over defaults.

    Reads eval/calibration.json (written by scripts/calibrate_gap_thresholds.py)
    when it exists and parses; otherwise — or on any malformed file — falls back
    to the uncalibrated defaults so gap ranking always works pre-calibration.
    """
    tau = GAP_TAU_DEFAULT
    weights = dict(G_SCORE_WEIGHTS)
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {"tau": tau, "weights": weights}
    if isinstance(data.get("tau"), (int, float)):
        tau = float(data["tau"])
    if isinstance(data.get("weights"), dict):
        weights.update({k: float(v) for k, v in data["weights"].items()
                        if k in weights and isinstance(v, (int, float))})
    return {"tau": tau, "weights": weights}


# --- API (PBI 6, OD12) --------------------------------------------------------
# CORS allowlist for local React dev servers (Vite default 5173, CRA default 3000).
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]

# --- Local LLM via Ollama (OD14) ----------------------------------------------
# No API key, no cost -- runs on this machine. Unblocks T3.2 (keyword
# canonicalization) and reading-list summaries (T7.4); PBI 5's hypothesis
# pipeline will use the same client later.
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"        # good instruction-following/JSON at 7B, fast locally
OLLAMA_TIMEOUT = 120.0

# --- Agentic pipeline (OD19-OD23, docs/agentic/ARCHITECTURE.md) --------------
# Provider priority: anthropic iff ANTHROPIC_API_KEY is set, else ollama (mirrors OD14's
# no-key philosophy -- the pipeline always runs, quality/latency differ by provider).
LLM_PROVIDER_PRIORITY = ("anthropic", "ollama")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_TIMEOUT = 60.0
ANTHROPIC_MAX_RETRIES = 5
ANTHROPIC_BACKOFF_CAP = 30.0

# Model tiers (R6). IDs and prices re-verified live against platform.claude.com/docs
# (models/overview + pricing pages) on 2026-10-31 (OD26) -- re-verify before relying on
# these if this drifts far from that date. claude-haiku-5-5 is ~10x cheaper than the
# claude-haiku-4-5 this superseded for prompts <=100k tokens (the only case this app hits);
# claude-sonnet-5-5 is the same price as the claude-sonnet-5 it superseded ($2/$10), so this
# is a quality upgrade at equal cost, not a cost tradeoff.
MODEL_PRICE_CHECKED_DATE = "2026-10-31"
MODEL_FAST = "claude-haiku-5-5"
MODEL_FAST_PRICE_IN = 0.10          # $ / 1M input tokens (prompts <=100k tokens tier)
MODEL_FAST_PRICE_OUT = 0.50         # $ / 1M output tokens (prompts <=100k tokens tier)
MODEL_STRONG = "claude-sonnet-5-5"
MODEL_STRONG_PRICE_IN = 2.00        # $ / 1M input tokens
MODEL_STRONG_PRICE_OUT = 10.00      # $ / 1M output tokens
# Third tier, not part of the pipeline's normal fast/strong split -- a second, stronger
# independent labeler for AI-vs-AI calibration review (B3/E3, stage-2 fixes), same
# verification date as the two above.
MODEL_OPUS = "claude-opus-5-5"
MODEL_OPUS_PRICE_IN = 4.00           # $ / 1M input tokens
MODEL_OPUS_PRICE_OUT = 20.00         # $ / 1M output tokens

LLM_BUDGET_USD = 3.00               # hard cap on total agentic-pipeline spend (laptop rule)

# C4: per-agent-type max_tokens instead of one constant for every call. Routing only ever
# returns a one-field RouteDecision; extraction returns a handful of entities. Synthesis
# (cluster labels + hypothesis pitch, strong tier) is prose, so it keeps the old default.
AGENT_MAX_TOKENS = {
    "route": 200,
    "extraction": 1500,
    "synthesis": 4096,
    "pair_judgment": 150,            # B3/E3: a short structured {label, reason} judgment
}
_DEFAULT_MAX_TOKENS = 4096          # fallback for any agent type not listed above

AGENT_MAX_RETRIES = 3               # R4: self-correction attempts per agent call
AGENT_CONCURRENCY = 4               # R5: asyncio.Semaphore limit for worker fan-out

ENTITY_KINDS = ("method", "task", "dataset", "metric")
ROUTE_ACTIONS = ("full", "title_only", "skip")
ROUTE_TITLE_ONLY_MIN_WORDS = 40     # A5: abstract shorter than this (or missing) -> title_only

# Calibrated 2026-10-31 (OD23) by inspecting all 109 real (entity, ontology-match) pairs
# produced by a live 30-paper pipeline run, sorted by cosine. Matches are clean and correct
# down to ~0.95 (e.g. 0.9748 'translation'->'Machine translation', 0.9608 'Chain-of-Thought
# Prompting'->'Prompting'); below that, clear false positives start appearing (0.9477
# 'Classification'->'Quantization', 0.9447 'summarization'->'Text segmentation'). Not a
# perfect cutoff on this small (~40-label) ontology -- one false positive still slips through
# just above the line (0.9531 'Dual Encoder'->'Decoding') and some correct matches near the
# boundary may get rejected and retried -- but it's where true/false visibly separate overall.
VERIFIER_COSINE_MIN = 0.95

# B1: short-phrase embedding model for the entity-name matching cascade (lexical match first,
# then this, else new_concept) -- SPECTER2 above is a document-level encoder (title+abstract
# concat) and is the wrong tool for embedding a bare entity name like "Chain-of-Thought
# Prompting". ~130MB, CPU-fine. Loaded via the same plain transformers AutoModel/AutoTokenizer
# pattern as axiom/embed.py (no new heavy dependency).
ENRICHMENT_PHRASE_MODEL = "BAAI/bge-small-en-v1.5"
# Placeholder pending the B3 gate: ask the user to label eval/enrichment_pairs.csv (>=60 real
# (entity, candidate, cos, margin) rows), then set by "highest recall at precision >= 0.9" and
# log the replacement decision (supersedes this placeholder, same pattern as OD23 superseding
# VERIFIER_COSINE_MIN's old 0.70 placeholder). top1-top2 margin, not a cosine floor by itself.
ENRICHMENT_MARGIN_DELTA = 0.05

TRACE_DIR = PROJECT_ROOT / "eval" / "runs"
EVAL_V2_QUERIES_PATH = PROJECT_ROOT / "eval" / "ndcg_queries_v2.json"
EVAL_V2_SCALE_MAX = 2
AGENT_BENCH_PATH = PROJECT_ROOT / "eval" / "agent_bench.json"
STRUCT_RETRIEVAL_CANDIDATES = 200   # R8: candidate pool size for the eval-only concept-overlap arm
