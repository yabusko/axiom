"""Stage 4 B1: external OpenAlex corpus-artifact check for a gap candidate.

A gap can be numerically well-formed (two sizable, semantically close,
weakly-cited communities within THIS corpus slice) and still not be a real
blind spot in the literature -- it can look that way purely because this
corpus is a few hundred papers, not because the two areas are actually
disconnected in the field at large. This module answers a narrower question
than `scripts/judge_gaps.py`'s subjective rubric: concretely, how many
OpenAlex works (the whole index, not just this corpus) already combine both
sides' key concepts? Many such works is evidence the "gap" is a corpus-size
artifact, not a real one -- flag it so a user can discount it, rather than
silently trusting the in-corpus citation count.
"""
from __future__ import annotations

from dataclasses import dataclass

# OD42 (docs/DECISIONS.md): calibrated against 5 real label pairs from the
# live corpus's actual gap candidates. Narrow/plausible pairings (genuinely
# few works combine them): "GPT-3"+"retrieve-and-read" -> 26,
# "dense passage retrieval"+"retrieve-and-read" -> 9,
# "2WikiMultihopQA"+"ScienceQA" -> 0. Pairings of two broad, independently
# well-established areas (not disconnected in the wider literature at all,
# just in this corpus's small citation subgraph): "hallucination"+
# "retrieval-augmented generation" -> 4655, "chain of thought prompting"+
# "reasoning" -> 3882. 50 sits cleanly in the gap between those two real
# clusters; revisit with more real gaps once more fields exist to check it
# against, same spirit as OD37's relevance floor.
ARTIFACT_WORKS_THRESHOLD = 50


@dataclass
class ExternalCheckResult:
    query: str
    works_count: int
    is_likely_artifact: bool


def build_query(label_a: str, label_b: str) -> str:
    """OpenAlex's `title_and_abstract.search` filter requires every word in
    its value to be present (not an OR relevance match) -- concatenating
    both labels as one filter value is literally "works that mention both
    concepts", not an approximation of it."""
    return f"{label_a} {label_b}"


def is_artifact(works_count: int, threshold: int = ARTIFACT_WORKS_THRESHOLD) -> bool:
    return works_count >= threshold


def check_gap(client, label_a: str, label_b: str) -> ExternalCheckResult:
    """`client` is an `axiom.ingest.OpenAlexClient` -- not type-hinted
    directly to avoid a hard import cycle (ingest.py doesn't need to know
    about gaps/external_check)."""
    query = build_query(label_a, label_b)
    count = client.count_works(query)
    return ExternalCheckResult(query=query, works_count=count, is_likely_artifact=is_artifact(count))
