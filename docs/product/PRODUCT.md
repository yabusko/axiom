# Axiom: product definition (stage 3, Phase B)

Validates and sharpens the stage-3 hypothesis against the real app (`docs/product/AUDIT.md`)
and four direct decisions from the user (recorded inline below, dated 2026-10-31). Supersedes
the hypothesis section of the stage-3 task brief where the two differ.

## Personas

**Primary: a master's or early PhD student** choosing a thesis direction in a field they
don't know well. Confirmed as primary user (not the supervisor or a general researcher) —
2026-10-31 decision. They know how to read a paper abstract and a citation count; they do
not know this sub-field's landscape, its open problems, or which venues/authors matter yet.

**Secondary: their supervisor**, who reviews the shortlist the student produces. Doesn't use
the exploration screens (Landscape, Research gaps) directly in the MVP — they see the
Markdown export in a meeting, not the live app.

## The job

"Find promising, under-explored directions in field X and give me the evidence to argue for
one in a supervisor meeting."

Two things make this different from a literature-search tool: (1) the output is a *direction*,
not a paper, and (2) every claim has to be defensible — a score a student can't explain to
their supervisor is worse than no score.

## Primary flow

1. **Start a field.** Type a topic. Axiom builds the corpus (fetch → embed → map topics →
   extract methods/datasets → find gaps), showing step-by-step progress and a time/cost
   estimate before the paid extraction step. One active field at a time; a "recent fields"
   list on Start reopens a previous one — **not** a multi-field comparison view (2026-10-31
   decision: the brief's "one active field or several" question, resolved toward simplicity;
   no validated use case yet for holding several fields open at once).
2. **See the landscape.** Main topics, what's rising and fading, key papers.
3. **Explore research gaps.** Pairs of related topics that rarely cite each other, explained
   in plain words, with evidence papers and the methods/datasets each side uses (from the
   agent pipeline) — e.g. "side A uses method M, side B works on dataset D." This evidence is
   what makes a gap actionable rather than just a statistic.
4. **Draft a direction.** A grounded AI pitch citing both sides, marked unverified, editable.
5. **Build a shortlist.** Keep or discard directions, attach papers to read, add notes.
   Export to Markdown (BibTeX is deferred — see MVP scope below).

**One mental model.** The topics on the Landscape are exactly the communities whose pairs
form the gaps. There is one clustering everywhere; concept clusters from the agent pipeline
label and describe those topics, they don't form a second map.

## Demo scenario (2026-10-31 decision)

The one 3-minute scenario this product is built to nail: **start from an already-built
field, go straight to a gap, draft a direction, shortlist it — no live build in the
critical path.** Not the live end-to-end build (that has unavoidable wait time in front of
an interviewer). The Start screen with real build progress still ships as part of the MVP
(students will actually use it to build their own field), but `docs/product/DEMO.md`
(Phase F) will script the fast path, picking up from a pre-built field.

## Information architecture (MVP scope)

Per the task's own fast-mode scope cut, the MVP is five screens. Everything else is deferred
(listed below), not built, not stubbed.

| Screen | What's on it | Replaces (from the current 6-tab Streamlit app) |
|---|---|---|
| Start | topic input, recent fields list, build progress + time/cost estimate | CLI ingest flag (`scripts/ingest_openalex.py --topic`, `bootstrap_synthetic.py`) — confirmed in `AUDIT.md` as the only way to seed a corpus today |
| Landscape | field summary, topics (ranked list or treemap — **not** a force graph), rising/fading concepts with sparklines, key papers | Trending tab, the influence/PageRank ranking |
| Research gaps (list + detail) | ranked gaps → gap "bridge" detail, plain-language reason, evidence papers, methods/datasets per side, "Draft direction" | Research gaps tab's citation-community gaps (kept as the primary gap signal) **and** the Advanced tab's Vector Gap Discovery, folded in as a secondary "embedding-only cross-check" panel inside gap detail rather than a separate top-level feature — per `AUDIT.md`'s explicit recommendation: both signals are genuinely complementary (citation-sparse corpora need the geometry-only fallback) but today's equal-tab presentation overstates Vector Gap Discovery's standing, since it has no path to a hypothesis pitch at all |
| Shortlist | directions by status (draft / kept / discarded), notes, papers to read, Markdown export | Reading list + Review queue (approve/reject becomes kept/discarded on a *direction*, not a bare pitch) |
| Paper panel (global side panel) | abstract with extracted entities highlighted at evidence spans, cites/cited by, similar papers, save | Search tab's result expanders |

**Deferred, not built this stage** (tracked in `.claude/product/PROGRESS.md`, not forgotten):
⌘K global search, "Under the hood" (pipeline runs, retrieval eval, vector geometry fine
detail, the standalone Paper Similarity Analyzer, dense/hybrid toggle), BibTeX export, dark
theme visual polish beyond carrying the existing token system over, the 5-task usability
script with classmates, and any git-history rewrite.

## Frontend (2026-10-31 decision)

**React + Vite + TypeScript**, in `web/`, on top of the existing FastAPI — confirmed by the
user. `AUDIT.md` found `api/main.py` currently has zero callers from the Streamlit app (every
widget talks straight to `axiom/*`/SQLite/Qdrant); building `web/` against it for real is what
finally uses that seam. Streamlit stays as an internal lab console, not deleted.

## Honesty by design

- One consistent badge system ("AI-generated", "unverified", "uncalibrated score"), each with
  a one-line explanation on hover/focus, replacing the repeated warning-banner pattern
  `AUDIT.md` found in at least 3 near-identical wordings across the current app (the Research
  gaps tab, the Review queue, and `axiom/hypothesis.py`'s own disclaimer string).
- Every score on screen has a plain-language explanation and its n — the G-score's
  "Uncalibrated defaults (OD17)" caveat is real and already present in the current app; it
  needs to survive the redesign, not get polished away.
- The corpus-identity pill must be computed from the real active field (`AUDIT.md` found the
  current pill is a hardcoded literal — "ACL · EMNLP · COLING · NAACL · 2020–2025" — that's
  wrong for the real 300-paper RAG corpus already built).

## Success metrics

- A new user reaches a kept direction in under 10 minutes, starting from an already-built
  field (matches the demo scenario above).
- Every number on screen has a one-line plain-language explanation reachable by hover/focus —
  zero unexplained scores.
- Zero of the "what happened, confirmed" findings in `AUDIT.md` regress: the agent pipeline's
  extracted entities/methods/datasets become visible somewhere in the product (Paper panel,
  gap detail) rather than staying API-only; the FastAPI seam gets at least one real caller
  (`web/`) instead of zero.
- Mobile: no dead first viewport — `AUDIT.md` found the current hero/metadata chrome alone
  fills the entire 844px mobile viewport before any tab content appears; the redesign must not
  repeat this.

## What this stage is not doing (explicitly out of scope, not an oversight)

- A G-score/gap-threshold calibration UI (the calibration *scaffold* stays as-is; no new
  screen for it).
- Multi-user accounts/auth.
- Re-ingesting or re-architecting the existing 300-paper real corpus or its agentic-pipeline
  output — Phase E's backend additions (fields, landscape, gap detail, paper detail,
  shortlist endpoints) are new read/write surface over data that already exists.

## Related decisions

- `docs/DECISIONS.md` OD31 — eval corpus source file choice (repo-cleanup, unrelated to
  product scope but logged the same stage).
- `docs/DECISIONS.md` OD32 — Start always builds the full ~300-paper corpus, no small-first
  tier.
