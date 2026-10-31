"""LLM keyword canonicalization (Task 3.2, OD14): merge concept-label synonyms
(e.g. "LoRA" / "Low-Rank Adaptation") into one canonical form.

Batches distinct concept labels (default 50/batch, per the backlog spec),
asks the model to group true synonyms and name each group's canonical label,
and persists the mapping to `concept_canonical` (db/schema.sql) so it's
computed once per corpus, not on every velocity query. Idempotent: re-running
overwrites prior `source='auto'` rows but never touches `source='manual'`
rows (the manual-override path the backlog's acceptance criteria call for).

Routed through `axiom/llm_providers.py` (the one pooled LLM client, picking
Anthropic if `ANTHROPIC_API_KEY` is set else local Ollama -- same
provider-selection and spend/trace ledger as the rest of the agentic
pipeline) rather than calling `axiom/llm.py` directly, which this module did
until stage-3 cleanup item A.1.6. Ollama is still the default with no key
set, so a dev box keeps the original "no API key, no cost" behavior.
"""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from axiom import db, llm_providers

_SYSTEM = (
    "You are grouping near-duplicate machine learning research concept labels "
    "into canonical forms. Group ONLY true synonyms or abbreviation/full-name "
    "pairs (e.g. 'LoRA' and 'Low-Rank Adaptation'). Never merge merely-related "
    "but distinct concepts (e.g. 'Machine translation' and 'Low-resource NLP' "
    "must stay separate groups). Every input label must appear in exactly one "
    "group, even if that group has only one member."
)


def _prompt(labels: list[str]) -> str:
    joined = "\n".join(f"- {label}" for label in labels)
    return (
        f"Concept labels:\n{joined}\n\n"
        'Return JSON: {"groups": [{"canonical": "<label>", "members": '
        '["<label>", ...]}, ...]}. `canonical` must be one of `members` '
        "(pick the clearer/more complete form)."
    )


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str
    members: list[str]


class _CanonicalGroups(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[_Group]


@dataclass
class CanonicalizationResult:
    mapping: dict[str, str]      # concept -> canonical, one entry per input label
    groups: list[list[str]]      # non-trivial groups only (size > 1), for review


async def canonicalize_batch_async(labels: list[str]) -> CanonicalizationResult:
    result = await llm_providers.complete_json(
        "fast", _SYSTEM, _prompt(labels), _CanonicalGroups,
        agent="canonicalize", agent_type="canonicalize",
    )
    mapping: dict[str, str] = {}
    groups: list[list[str]] = []
    label_set = set(labels)
    for g in result.groups:
        members = [m for m in g.members if m in label_set]
        if not g.canonical or not members:
            continue
        for m in members:
            mapping.setdefault(m, g.canonical)
        if len(members) > 1:
            groups.append(members)
    # Any label the model dropped or malformed maps to itself (safe fallback).
    for label in labels:
        mapping.setdefault(label, label)
    return CanonicalizationResult(mapping=mapping, groups=groups)


def canonicalize_batch(labels: list[str]) -> CanonicalizationResult:
    """Sync wrapper -- every caller today (the batch script, tests) is sync."""
    return asyncio.run(canonicalize_batch_async(labels))


def run(conn: sqlite3.Connection, *, batch_size: int = 50, log=print) -> int:
    """Canonicalize every distinct concept in the corpus; persist to SQLite.

    Skips concepts already manually overridden. Returns the number of
    (concept -> canonical) rows written.
    """
    labels = [r["concept"] for r in conn.execute(
        "SELECT DISTINCT concept FROM concepts ORDER BY concept"
    ).fetchall()]
    manual = db.manual_canonical_concepts(conn)
    todo = [label for label in labels if label not in manual]

    written = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        result = canonicalize_batch(batch)
        for concept, canonical in result.mapping.items():
            db.upsert_canonical(conn, concept, canonical, source="auto")
            written += 1
        if result.groups:
            merged = sum(len(g) for g in result.groups)
            log(f"[canonicalize] batch {start // batch_size + 1}: "
                f"merged {merged} labels into {len(result.groups)} groups")
    conn.commit()
    log(f"[canonicalize] {written} concepts mapped "
        f"({len(manual)} manual overrides preserved)")
    return written
