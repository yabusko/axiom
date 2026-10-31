"""Enrichment agent (R1/R4): deterministic matching cascade, NO LLM call.

B1 (OD26): a 3-step cascade maps each extracted `Entity.name` to the
ontology -- `concept_canonical` rows union OpenAlex-sourced `concepts` rows:

  1. Exact/normalized match: normalize the entity name the same way
     `axiom.agents.verifier.normalize_text` normalizes everything else
     (casefold, NFKC, collapsed whitespace) and compare directly against
     normalized ontology labels. No embedding call for a hit here.
  2. Short-phrase embedding match: embed the name with `PhraseEncoder`
     (`config.ENRICHMENT_PHRASE_MODEL`, mean-pooled BAAI/bge-small-en-v1.5 --
     a document-level SPECTER2 concat-and-CLS-token encoding is the wrong
     tool for a 2-4 word entity name) against the SAME pool, also embedded
     with this model. Accepted only if the top candidate clears BOTH
     `cosine >= config.VERIFIER_COSINE_MIN` and a confidence margin over the
     runner-up (`top1 - top2 >= config.ENRICHMENT_MARGIN_DELTA` -- a close
     second place means the match is arbitrary, not confident). Tried with
     progressively richer context (bare name -> name+kind -> name+evidence_
     span) before giving up, the same escalation the old single-step design
     used on a verifier-driven retry; here it happens inside one cascade
     pass since there's no longer an LLM response to retry.
  3. Otherwise: `new_concept` -- KEPT (not dropped), `canonical` = the
     entity's own normalized name. A paper naming a method/dataset not yet
     in the ontology now still contributes that node to the concept graph,
     instead of being silently filtered out of linking.

`ontology_source` (B2) records which step produced the match --
`"lexical" | "embedding" | "new_concept"` -- not which table the label came
from (that's `Ontology.sources`, an internal pool-provenance detail the
cascade no longer surfaces on the result).

Still sits inside the attempt<=3 verifier retry box (ARCHITECTURE.md's
data-flow diagram) for defense in depth, but by construction every result
the cascade produces already passes `verifier.verify_enrichment` (a
`new_concept` has no ontology match to fail on; an `embedding` match only
exists once it already cleared the cosine+margin gate) -- the outer retry
loop is a backstop, not the normal path, now that step 2's own internal
escalation does the "try harder" work step 1-of-the-old-design's re-embed
used to do via the verifier.
"""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from axiom import config
from axiom.agents import verifier
from axiom.agents.contracts import EnrichedEntity, Entity, EnrichmentResult, VerifierError
from axiom.agents.extraction import ExtractionOutcome


@dataclass
class Ontology:
    """Flat candidate pool: `concept_canonical.canonical` union `concepts.concept`,
    each row's phrase-model embedding pre-normalized (unit L2 norm) so cosine
    similarity is a plain dot product. `sources` ("concept_canonical" |
    "openalex_concept") is which TABLE a label came from -- pool provenance
    only, independent of `EnrichedEntity.ontology_source` (B2), which
    records which CASCADE STEP produced a given entity's match."""

    labels: list[str]
    sources: list[str]
    vectors: np.ndarray           # (N, dim), L2-normalized rows; N may be 0
    normalized_labels: dict[str, int] = field(default_factory=dict)  # norm(label) -> index

    def __post_init__(self) -> None:
        if not self.normalized_labels:
            self.normalized_labels = {
                verifier.normalize_text(label): i for i, label in enumerate(self.labels)
            }


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_ontology(conn: sqlite3.Connection, encoder) -> Ontology:
    """Build the candidate pool once per pipeline run (not per paper/entity).

    `encoder` is a short-phrase encoder (`PhraseEncoder`-shaped, or a test
    double) -- B1 moved the ontology's own embedding off SPECTER2 onto the
    same phrase model step 2 of the cascade uses, so a name and its
    candidates are compared in the same embedding space.

    Raw SQL against the existing tables -- `axiom/db.py` has no helper for
    this read, and this module doesn't own `db.py` to add one.
    """
    canonical_rows = conn.execute(
        "SELECT DISTINCT canonical FROM concept_canonical WHERE canonical IS NOT NULL"
    ).fetchall()
    concept_rows = conn.execute(
        "SELECT DISTINCT concept FROM concepts WHERE concept IS NOT NULL"
    ).fetchall()

    canonical_labels = sorted({r["canonical"] for r in canonical_rows if r["canonical"]})
    seen = set(canonical_labels)
    concept_labels = sorted({r["concept"] for r in concept_rows if r["concept"]} - seen)

    labels = canonical_labels + concept_labels
    sources = (["concept_canonical"] * len(canonical_labels)
               + ["openalex_concept"] * len(concept_labels))

    if not labels:
        return Ontology(labels=[], sources=[], vectors=np.zeros((0, 1)))

    vectors = _normalize_rows(np.array(encoder.encode(labels), dtype=float))
    return Ontology(labels=labels, sources=sources, vectors=vectors)


_CONTEXT_LEVELS = 3  # bare name -> name+kind -> name+evidence_span


def _entity_text(entity: Entity, level: int) -> str:
    """Progressively richer context for step 2's embedding try (0-indexed)."""
    if level <= 0:
        return entity.name
    if level == 1:
        return f"{entity.name} ({entity.kind})"
    return f"{entity.name}: {entity.evidence_span}"


def _new_concept(entity: Entity) -> EnrichedEntity:
    """Step 3 of the cascade: kept, never dropped (B1's behavior change)."""
    return EnrichedEntity(
        **entity.model_dump(),
        canonical=verifier.normalize_text(entity.name) or entity.name,
        cosine=-1.0,
        ontology_source="new_concept",
    )


async def _embedding_match(
    entity: Entity, *, encoder, ontology: Ontology,
) -> EnrichedEntity | None:
    """Step 2: try progressively richer context; return the first candidate
    that clears both the cosine floor and the top1-top2 margin, or None if
    every context level fails (caller falls through to step 3)."""
    if ontology.vectors.shape[0] == 0:
        return None
    for level in range(_CONTEXT_LEVELS):
        text = _entity_text(entity, level)
        raw_vector = await asyncio.to_thread(encoder.encode, [text])
        vector = _normalize_rows(np.array(raw_vector, dtype=float))
        sims = (vector @ ontology.vectors.T)[0]
        order = np.argsort(sims)[::-1]
        top1_idx = int(order[0])
        top1 = max(-1.0, min(1.0, float(sims[top1_idx])))
        top2 = max(-1.0, min(1.0, float(sims[int(order[1])]))) if len(order) > 1 else -1.0
        if top1 >= config.VERIFIER_COSINE_MIN and (top1 - top2) >= config.ENRICHMENT_MARGIN_DELTA:
            return EnrichedEntity(
                **entity.model_dump(),
                canonical=ontology.labels[top1_idx],
                cosine=top1,
                ontology_source="embedding",
            )
    return None


async def _match_one(entity: Entity, *, encoder, ontology: Ontology) -> EnrichedEntity:
    """The 3-step cascade for one entity -- see module docstring."""
    name_norm = verifier.normalize_text(entity.name)
    idx = ontology.normalized_labels.get(name_norm)
    if idx is not None:
        return EnrichedEntity(
            **entity.model_dump(), canonical=ontology.labels[idx], cosine=1.0,
            ontology_source="lexical",
        )
    embedded = await _embedding_match(entity, encoder=encoder, ontology=ontology)
    if embedded is not None:
        return embedded
    return _new_concept(entity)


@dataclass
class EnrichmentOutcome:
    """`enrich()`'s real return value -- `EnrichmentResult` plus the real
    attempt count (D3: surfaced on the success path too, not just on
    failure via `VerificationFailed.attempts`)."""

    result: EnrichmentResult
    attempts: int


async def enrich(
    extraction: ExtractionOutcome,
    *,
    encoder,
    ontology: Ontology,
    run_id: str,
    max_retries: int | None = None,
) -> EnrichmentOutcome:
    """Enrich every entity in `extraction` via the cascade, verify, and (in
    the backstop case the cascade shouldn't normally hit) retry up to
    `max_retries` (defaults to `config.AGENT_MAX_RETRIES`, read at call
    time -- see extraction.py's `extract()` for why this isn't a bound
    default). Raises `verifier.VerificationFailed` if no attempt passes.
    """
    max_retries = max_retries if max_retries is not None else config.AGENT_MAX_RETRIES
    entities = extraction.entities
    last_errors: list[VerifierError] = []

    for attempt in range(1, max_retries + 1):
        enriched = await asyncio.gather(*[
            _match_one(e, encoder=encoder, ontology=ontology) for e in entities
        ])
        result = EnrichmentResult(paper_id=extraction.paper_id, enriched=list(enriched))
        verdict = verifier.verify_enrichment(result, attempt=attempt)
        verifier.log_attempt(run_id, "enrichment", verdict, paper_id=extraction.paper_id)
        if verdict.ok:
            return EnrichmentOutcome(result=result, attempts=attempt)
        last_errors = verdict.errors

    raise verifier.VerificationFailed("enrichment", max_retries, last_errors)


# --- B1: short-phrase encoder ---------------------------------------------------
#
# Same plain `transformers` AutoModel/AutoTokenizer pattern as
# `axiom/embed.py`'s `Specter2Encoder` -- no `sentence-transformers` dependency
# added. The recipe differs on purpose: SPECTER2 takes the [CLS] token of a
# title+abstract concatenation (a document-level embedding); the bge recipe
# mean-pools every token's last-hidden-state vector (masking out padding),
# which is what a short, CLS-token-poor phrase like an entity name needs.

def _mean_pool(last_hidden_state, attention_mask):
    """Attention-masked mean over the token dimension -- the bge recipe.
    Padding tokens (attention_mask == 0) are excluded from both the sum and
    the per-row token count, so batch padding never dilutes a short phrase's
    vector."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class PhraseEncoder:
    """Lazy-loaded short-phrase encoder. Construct once, reuse for all encode
    calls -- same "injected by the caller" convention `axiom.embed.
    Specter2Encoder` already uses, so loading model weights never happens
    per-paper or per-entity."""

    def __init__(self, model_id: str = config.ENRICHMENT_PHRASE_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32, max_length: int = 64) -> list[list[float]]:
        import torch

        vectors: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model(**inputs)
                pooled = _mean_pool(output.last_hidden_state, inputs["attention_mask"])
                vectors.extend(pooled.cpu().tolist())
        return vectors


_default_phrase_encoder: PhraseEncoder | None = None


def get_default_phrase_encoder() -> PhraseEncoder:
    """Process-wide singleton, built on first use. `axiom/agents/pipeline.py`
    calls this only when its caller doesn't inject a `phrase_encoder`
    explicitly (production default; tests always inject a fake one and never
    reach this, so no test ever downloads/loads the real model)."""
    global _default_phrase_encoder
    if _default_phrase_encoder is None:
        _default_phrase_encoder = PhraseEncoder()
    return _default_phrase_encoder
