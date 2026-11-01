"""Run LLM keyword canonicalization (Task 3.2, OD14) over the corpus in
data/axiom.db, persisting concept -> canonical mappings so the OD10 velocity
engine merges synonyms (e.g. "LoRA" / "Low-Rank Adaptation") into one trend.

Routed through axiom/llm_providers.py (Anthropic if ANTHROPIC_API_KEY is set,
else local Ollama -- no separate --model flag any more; see axiom/config.py's
MODEL_FAST / OLLAMA_MODEL to change which model that resolves to).

Usage (from repo root):
    python scripts/canonicalize_concepts.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import canonicalize, db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    conn = db.connect()
    try:
        canonicalize.run(conn, batch_size=args.batch_size)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
