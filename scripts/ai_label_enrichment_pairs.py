"""AI dual-labeling for `eval/enrichment_pairs.csv` (B3, stage-2 fixes).

Two independent blind judges, both stronger than the pipeline's own fast tier,
grade every (entity, candidate) pair: labeler A = the `strong` tier
(config.MODEL_STRONG), labeler B = the `opus` tier (config.MODEL_OPUS). Same
rubric, temperature 0, structured `{label, reason}` output. Each judge sees
only `entity`, `kind`, the entity's `evidence_span`, and `candidate` -- never
`cos`/`margin`, so the judgment can't anchor on the model's own confidence
score that produced the pair. Rows are judged in a fixed shuffled order (not
the file's cosine-sorted order), independently for every row (no shared
context between rows or between the two judges).

A deterministic, non-LLM helper separately scans each pair's paper's abstract
for an "Expansion Words (ACRONYM)" definition and flags whether the entity or
candidate matches either side of a detected pair. This is diagnostic only --
reported alongside A/B disagreements, never used to override a label (per
the task's instruction: real-world acronym definitions are suggestive, not
authoritative, since an abstract can define an acronym for something that
still isn't the same concept as the ontology's candidate label).

`label` = A and B agree on "1" or "0"; disagreements and any "unsure" are
excluded from calibration (both counted and reported). `label_a`/`label_b`/
`reason_a`/`reason_b` are kept so the raw per-judge output stays inspectable.
`provenance` is explicit: "AI-labeled (sonnet-5-5 + opus-5-5, agreement)" --
never call this human-labeled or hand-labeled, anywhere.

Usage (from repo root):
    python scripts/ai_label_enrichment_pairs.py --estimate-only   # cost estimate, no calls
    python scripts/ai_label_enrichment_pairs.py                   # run for real
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
import unicodedata
import uuid
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from axiom import config, db, llm_providers

PAIRS_PATH = Path(__file__).resolve().parent.parent / "eval" / "enrichment_pairs.csv"
PROVENANCE = "AI-labeled (sonnet-5-5 + opus-5-5, agreement)"
FIELDNAMES = [
    "paper_id", "entity", "kind", "evidence_span", "candidate", "cos", "margin",
    "label", "annotator", "label_a", "reason_a", "label_b", "reason_b",
    "acronym_evidence", "provenance",
]
SHUFFLE_SEED = 5678


class PairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["1", "0", "unsure"]
    reason: str


_SYSTEM = (
    "You are judging whether two concept labels refer to the SAME underlying concept. "
    "You are given an entity name, its kind, the sentence it was extracted from, and a "
    "candidate concept label from a fixed ontology. "
    "Return label=\"1\" if the entity and the candidate are the same concept -- including "
    "a synonym, an abbreviation/acronym vs. its full form, a spelling variant, or a plural/"
    "hyphenation variant. "
    "Return label=\"0\" if the candidate is broader or narrower than the entity, merely "
    "related but a genuinely different concept, or a different specific version/variant "
    "(e.g. GPT-3 vs GPT-4 are different concepts, not the same one). "
    "Return label=\"unsure\" only if there truly is not enough information in the entity "
    "name and sentence to decide. `reason` is one short sentence."
)


def _prompt(entity: str, kind: str, evidence_span: str, candidate: str) -> str:
    return (
        f"entity: {entity}\n"
        f"kind: {kind}\n"
        f"evidence (sentence it was extracted from): {evidence_span}\n"
        f"candidate: {candidate}\n\n"
        'Return JSON: {"label": "1"|"0"|"unsure", "reason": "..."}.'
    )


async def _judge(
    tier: str, entity: str, kind: str, evidence_span: str, candidate: str, *,
    run_id: str, client: httpx.AsyncClient, conn, semaphore: asyncio.Semaphore,
) -> PairJudgment:
    prompt = _prompt(entity, kind, evidence_span, candidate)
    async with semaphore:
        try:
            return await llm_providers.complete_json(
                tier, _SYSTEM, prompt, PairJudgment,
                run_id=run_id, agent=f"pair_judgment.{tier}", agent_type="pair_judgment",
                temperature=0, client=client, conn=conn,
            )
        except (ValidationError, llm_providers.LLMError) as exc:
            return PairJudgment(label="unsure", reason=f"provider failure: {exc}")


# ---------------------------------------------------------------------------
# Deterministic acronym helper (diagnostic only -- never overrides a label)
# ---------------------------------------------------------------------------
_PAREN_ACRONYM_RE = re.compile(r"\(([A-Z][A-Za-z0-9]{1,9})\)")
_WORD_RE = re.compile(r"[A-Za-z]+")
_LOOKBACK_CHARS = 120  # window searched before "(ACRONYM)" for its expansion


def _normalize(text: str) -> str:
    """Casefold + NFKC + fold hyphens to spaces, so "retrieval-augmented generation" and
    "retrieval augmented generation" compare equal -- find_acronym_pairs below reconstructs
    an expansion by joining tokenized words with spaces, losing any original hyphenation."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def find_acronym_pairs(abstract: str) -> list[tuple[str, str]]:
    """Every "Expansion Words (ACRONYM)" pattern in `abstract` -- (expansion, acronym).

    Heuristic: for each "(ACRONYM)", take the same number of words immediately preceding
    it as the acronym has letters, and accept it only if each word's first letter matches
    the acronym's letters in order (the standard initials-matching definition). Deliberately
    simple -- doesn't handle a skipped minor word (e.g. BERT dropping "from") -- acceptable
    since this is diagnostic only, never used to override a label.
    """
    if not abstract:
        return []
    pairs = []
    for m in _PAREN_ACRONYM_RE.finditer(abstract):
        acronym = m.group(1)
        letters = [c.upper() for c in acronym if c.isalpha()]
        if not letters:
            continue
        window = abstract[max(0, m.start() - _LOOKBACK_CHARS):m.start()]
        words = _WORD_RE.findall(window)
        if len(words) < len(letters):
            continue
        candidate_words = words[-len(letters):]
        if [w[0].upper() for w in candidate_words] == letters:
            pairs.append((" ".join(candidate_words), acronym))
    return pairs


def acronym_evidence_for(entity: str, candidate: str, acronym_pairs: list[tuple[str, str]]) -> str:
    """Non-authoritative diagnostic: does an acronym definition in this paper's abstract
    connect `entity` and `candidate` (one side matches the acronym, the other the
    expansion, in either direction)? Empty string if nothing found."""
    e, c = _normalize(entity), _normalize(candidate)
    for expansion, acronym in acronym_pairs:
        exp_n, acr_n = _normalize(expansion), _normalize(acronym)
        pair = {e, c}
        if (acr_n in pair and exp_n in pair) or (
            (e == acr_n and exp_n in c) or (c == acr_n and exp_n in e)
            or (e == exp_n and acr_n in c) or (c == exp_n and acr_n in e)
        ):
            return f"{expansion} ({acronym})"
    return ""


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in FIELDNAMES:
            row.setdefault(key, "")
    return rows


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def shuffled_order(n: int, seed: int = SHUFFLE_SEED) -> list[int]:
    import random
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def estimate_cost(rows: list[dict]) -> float:
    """Worst-case pre-run estimate: chars/4 input proxy + the pair_judgment max_tokens cap
    for output, for both tiers, over every row -- mirrors llm_providers' own pre-call
    reservation math (C3) so this number is the same kind of conservative upper bound."""
    max_tokens = config.AGENT_MAX_TOKENS["pair_judgment"]
    total = 0.0
    for row in rows:
        prompt = _prompt(row["entity"], row["kind"], row["evidence_span"], row["candidate"])
        chars = len(_SYSTEM) + len(prompt)
        tokens_in = chars / 4
        for model, price_in, price_out in (
            (config.MODEL_STRONG, config.MODEL_STRONG_PRICE_IN, config.MODEL_STRONG_PRICE_OUT),
            (config.MODEL_OPUS, config.MODEL_OPUS_PRICE_IN, config.MODEL_OPUS_PRICE_OUT),
        ):
            total += (tokens_in / 1_000_000) * price_in + (max_tokens / 1_000_000) * price_out
    return total


def compute_final_label(label_a: str, label_b: str) -> tuple[str, str]:
    """Returns (final_label, category). final_label is "1"/"0" only when A and B agree on
    a real label; otherwise "" (excluded from calibration). category is one of
    "agreed_1"/"agreed_0"/"unsure"/"disagreement", for the summary report."""
    if label_a == "unsure" or label_b == "unsure":
        return "", "unsure"
    if label_a == label_b:
        return label_a, f"agreed_{label_a}"
    return "", "disagreement"


async def run(rows: list[dict], *, estimate_only: bool) -> None:
    estimate = estimate_cost(rows)
    print(f"[ai_label] {len(rows)} pairs, 2 judges each ({len(rows) * 2} calls). "
          f"Worst-case cost estimate: ${estimate:.4f}")
    if estimate_only:
        return

    conn = db.connect(config.DB_PATH)
    try:
        paper_ids = sorted({r["paper_id"] for r in rows if r["paper_id"]})
        placeholders = ",".join("?" for _ in paper_ids)
        abstract_rows = conn.execute(
            f"SELECT openalex_id, abstract FROM papers WHERE openalex_id IN ({placeholders})",
            paper_ids,
        ).fetchall() if paper_ids else []
        abstracts = {r["openalex_id"]: r["abstract"] for r in abstract_rows}
        acronyms_by_paper = {pid: find_acronym_pairs(abstracts.get(pid) or "") for pid in paper_ids}

        run_id = uuid.uuid4().hex
        before = db.total_llm_spend(conn)
        semaphore = asyncio.Semaphore(config.AGENT_CONCURRENCY)

        async with httpx.AsyncClient(
            base_url=config.ANTHROPIC_BASE_URL, timeout=config.ANTHROPIC_TIMEOUT,
        ) as client:
            order = shuffled_order(len(rows))
            tasks_a = [
                _judge("strong", rows[i]["entity"], rows[i]["kind"], rows[i]["evidence_span"],
                       rows[i]["candidate"], run_id=run_id, client=client, conn=conn,
                       semaphore=semaphore)
                for i in order
            ]
            tasks_b = [
                _judge("opus", rows[i]["entity"], rows[i]["kind"], rows[i]["evidence_span"],
                       rows[i]["candidate"], run_id=run_id, client=client, conn=conn,
                       semaphore=semaphore)
                for i in order
            ]
            results_a = await asyncio.gather(*tasks_a)
            results_b = await asyncio.gather(*tasks_b)

        categories_by_row: dict[int, str] = {}
        for pos, i in enumerate(order):
            row = rows[i]
            ja, jb = results_a[pos], results_b[pos]
            final, category = compute_final_label(ja.label, jb.label)
            row["label_a"] = ja.label
            row["reason_a"] = ja.reason
            row["label_b"] = jb.label
            row["reason_b"] = jb.reason
            row["label"] = final
            row["provenance"] = PROVENANCE
            row["acronym_evidence"] = acronym_evidence_for(
                row["entity"], row["candidate"], acronyms_by_paper.get(row["paper_id"], [])
            )
            categories_by_row[i] = category

        after = db.total_llm_spend(conn)
    finally:
        conn.close()

    save_rows(PAIRS_PATH, rows)

    categories = {"agreed_1": 0, "agreed_0": 0, "unsure": 0, "disagreement": 0}
    for category in categories_by_row.values():
        categories[category] += 1
    disagreement_with_acronym = sum(
        1 for i, category in categories_by_row.items()
        if category == "disagreement" and rows[i]["acronym_evidence"]
    )

    from sklearn.metrics import cohen_kappa_score
    code = {"1": 1, "0": 0, "unsure": 2}
    a_codes = [code[r["label_a"]] for r in rows]
    b_codes = [code[r["label_b"]] for r in rows]
    kappa = cohen_kappa_score(a_codes, b_codes)

    print(f"\n[ai_label] wrote {len(rows)} judged pairs -> {PAIRS_PATH}")
    print(f"[ai_label] actual spend this run: ${after - before:.4f} "
          f"(cumulative total: ${after:.4f} / ${config.LLM_BUDGET_USD:.2f} cap)")
    print(f"[ai_label] Cohen's kappa (A vs B, all 3 categories): {kappa:.4f}")
    print(f"[ai_label] agreed_1={categories['agreed_1']} agreed_0={categories['agreed_0']} "
          f"unsure={categories['unsure']} disagreement={categories['disagreement']}")
    print(f"[ai_label] usable for calibration (agreed_1 + agreed_0): "
          f"{categories['agreed_1'] + categories['agreed_0']}/{len(rows)}")
    print(f"[ai_label] disagreements with acronym evidence available: "
          f"{disagreement_with_acronym}/{categories['disagreement']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=str, default=str(PAIRS_PATH))
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args(argv)

    rows = load_rows(Path(args.file))
    asyncio.run(run(rows, estimate_only=args.estimate_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
