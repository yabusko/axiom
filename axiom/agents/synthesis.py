"""Synthesis agent (R1): strong tier, runs once per run after every paper in
it has finished linking. Two independent jobs:

1. `label_clusters` -- short human-readable labels for concept co-occurrence
   communities. Takes already-computed cluster membership as input rather
   than importing `axiom/concept_graph.py` directly: that module (Louvain +
   PageRank over `concept_cooccurrence`, R7) is owned by a sibling
   workstream in a different worktree and may not exist yet here. Whoever
   wires `axiom/agents/pipeline.py` into the real corpus should call
   `concept_graph.get_communities(conn)` (once it exists) and pass the
   result straight in -- see the agent-implementer patch note.
2. `generate_hypothesis` + `queue_hypothesis` -- OD16's grounded hypothesis
   pitch, relocated from `axiom/hypothesis.py`. Same verifier rule (rule-based,
   not LLM self-report: >=2 supporting ids, both sides of the gap
   represented) and the same HITL `review_queue` write path
   (`axiom/db.py.add_to_review_queue`) -- only the transport changed, from a
   direct `axiom.llm.chat_json` call to the strong tier via
   `axiom/llm_providers.py` (so it actually uses `config.MODEL_STRONG` when
   an Anthropic key is set, instead of always falling back to local Ollama).
   `axiom/hypothesis.py` itself is untouched (not in this agent's owned
   files) -- its prompt builder, side-paper selection, and rule-based
   `_verify` are imported and reused as-is, not copy-pasted, so "same
   verifier rules" is literally true, not just similar.
"""
from __future__ import annotations

import sqlite3

import networkx as nx
from pydantic import BaseModel, ConfigDict, ValidationError

from axiom import config, db, llm_providers
from axiom.gaps import GapCandidate
from axiom.hypothesis import (
    MAX_PAPERS_PER_SIDE,  # noqa: F401  (re-exported for callers that want it)
    MAX_RETRIES as _HYPOTHESIS_MAX_RETRIES,
    TEMPERATURE as _HYPOTHESIS_TEMPERATURE,
    HypothesisPitch,
    VerificationError,
    _SYSTEM as _HYPOTHESIS_SYSTEM,
    _prompt as _hypothesis_prompt,
    _side_papers,
    _verify as _verify_hypothesis,
)

# --- Cluster labeling -----------------------------------------------------------

_CLUSTER_SYSTEM = (
    "You label clusters of research concepts. Each cluster is a set of concept "
    "names that co-occur across papers. Give each cluster a short (<=6 word) "
    "human-readable topic label grounded ONLY in the concept names given -- "
    "never invent a concept or domain not present in the list."
)


class _ClusterLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_id: int
    label: str


class _ClusterLabelsJSON(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[_ClusterLabel]


def _cluster_prompt(clusters: list[dict]) -> str:
    lines = []
    for c in clusters:
        concepts = ", ".join(c["concepts"])
        lines.append(f"  - cluster_id {c['cluster_id']}: [{concepts}]")
    return (
        "Clusters:\n" + "\n".join(lines) + "\n\n"
        'Return JSON: {"labels": [{"cluster_id": 0, "label": "..."}, ...]} -- '
        "exactly one entry per cluster_id listed above."
    )


async def label_clusters(
    clusters: list[dict],
    *,
    run_id: str,
    max_retries: int | None = None,
) -> list[dict]:
    """`clusters`: `[{"cluster_id": int, "concepts": [canonical_label, ...]}, ...]`
    (e.g. from a future `concept_graph.get_communities(conn)`). Returns
    `[{"cluster_id": int, "label": str}, ...]`, one per input cluster.

    No deterministic verifier here (ARCHITECTURE.md scopes the verifier to
    extraction/enrichment only) -- just a provider-level retry on transport
    or schema failure. `max_retries` defaults to `config.AGENT_MAX_RETRIES`,
    read at call time (see extraction.py's `extract()`).
    """
    if not clusters:
        return []
    max_retries = max_retries if max_retries is not None else config.AGENT_MAX_RETRIES
    prompt = _cluster_prompt(clusters)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = await llm_providers.complete_json(
                "strong", _CLUSTER_SYSTEM, prompt, _ClusterLabelsJSON,
                run_id=run_id, agent="synthesis.clusters", attempt=attempt,
            )
        except (ValidationError, llm_providers.LLMError) as exc:
            last_exc = exc
            continue
        return [item.model_dump() for item in result.labels]

    raise last_exc  # type: ignore[misc]  # loop always either returns or sets last_exc


# --- Hypothesis pitch (relocated from axiom/hypothesis.py, OD16) ---------------

class _HypothesisJSON(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    claim: str
    method_sketch: str
    datasets: list[str] = []
    supporting_paper_ids: list[str] = []


async def generate_hypothesis(
    gap: GapCandidate,
    g: nx.DiGraph,
    *,
    run_id: str,
    trend_context: list[str] | None = None,
    max_retries: int = _HYPOTHESIS_MAX_RETRIES,
) -> HypothesisPitch:
    """Same contract as `axiom.hypothesis.generate_hypothesis`: raises
    `VerificationError` if no attempt produces a pitch with >=2
    `supporting_paper_ids` drawn from both sides of the gap. Runs on the
    strong tier via `axiom/llm_providers.py` instead of calling
    `axiom.llm.chat_json` directly."""
    side_a = _side_papers(g, gap.a.members)
    side_b = _side_papers(g, gap.b.members)
    side_a_ids = {p["paper_id"] for p in side_a}
    side_b_ids = {p["paper_id"] for p in side_b}
    prompt = _hypothesis_prompt(gap, side_a, side_b, trend_context)

    last_data: dict | None = None
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        nudge = "" if attempt == 1 else (
            "\n\nYour previous attempt did not include >=2 valid paper ids "
            "with at least one from EACH side. Fix `supporting_paper_ids` -- "
            "use ONLY the [bracketed] ids shown above."
        )
        try:
            result = await llm_providers.complete_json(
                "strong", _HYPOTHESIS_SYSTEM, prompt + nudge, _HypothesisJSON,
                run_id=run_id, agent="synthesis.hypothesis", attempt=attempt,
                temperature=_HYPOTHESIS_TEMPERATURE,
            )
        except (ValidationError, llm_providers.LLMError) as exc:
            last_exc = exc
            continue

        data = result.model_dump()
        last_data = data
        verified = _verify_hypothesis(data, side_a_ids, side_b_ids)
        if verified is not None:
            return verified

    if last_data is None and last_exc is not None:
        raise last_exc

    raise VerificationError(
        f"Could not produce a verifiable pitch after {max_retries} attempts "
        f"(last supporting_paper_ids="
        f"{last_data.get('supporting_paper_ids') if last_data else None})"
    )


def queue_hypothesis(
    conn: sqlite3.Connection,
    gap: GapCandidate,
    pitch: HypothesisPitch,
) -> int:
    """Same HITL write path as the original `/graph/gaps/{i}/hypothesize`
    endpoint (api/main.py) -- `review_queue`, status always starts
    `'pending'`, never auto-promoted. Returns the new row id."""
    return db.add_to_review_queue(
        conn,
        gap_a_label=gap.a.label,
        gap_b_label=gap.b.label,
        title=pitch.title,
        claim=pitch.claim,
        method_sketch=pitch.method_sketch,
        datasets=pitch.datasets,
        supporting_paper_ids=pitch.supporting_paper_ids,
    )
