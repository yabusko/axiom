# Axiom: 3-minute demo script

For interviews. Matches the confirmed demo scenario in `docs/product/PRODUCT.md`:
start from an already-built field, go straight to a gap, draft a direction, shortlist
it — no live build in the critical path. Uses the real "retrieval-augmented
generation" field (an instant 220-paper slice of the prebuilt 1000-paper base
corpus) that's already in the running app; every number below is real, pulled
from that corpus on 2026-11-01, not staged.

## Setup (before the room, not during it)

1. Backend running: `uvicorn api.main:app --reload` (repo root, venv active) —
   against the real prebuilt corpus (`python scripts/build_corpus.py` already run;
   see the README for the one-time cost/time warning, not something to do live).
2. Frontend running: `npm run dev` in `web/` → `http://localhost:5173`.
3. Open the app to `/` (Start) so the first thing on screen is the product's name
   and the real field in "Recent fields."

## The script (~3 minutes)

**0:00 — Start.** "Axiom isn't a paper search engine — it surfaces two things:
which concepts are accelerating in a field, and which related areas of work barely
cite each other despite being closely related. That second one is the interesting
part — those are candidate thesis directions." Type "retrieval-augmented generation"
(or click the field already in Recent fields) — this is a *slice* of a prebuilt
1000-paper corpus, so it resolves instantly, no download or per-field LLM spend.

**0:20 — Landscape.** Point at the stat row: 220 papers, 79 in-corpus citations,
188 topic communities, 6 candidate gaps. "These aren't a second clustering —
they're the same communities the gaps are built from." Key papers: Dense Passage
Retrieval, Leveraging Passage Retrieval with Generative Models — ranked by
PageRank within this corpus's own citation graph, not raw citation count.

**0:50 — Research gaps.** Click through to the ranked list. Pick the top one:
"an 11-paper cluster on retrieve-and-read methods is closer than 83% of this
field's topic pairs to a 10-paper cluster on re-ranking-based open-domain QA —
yet only 7 of this field's 79 total in-corpus citations cross between them."
Point at the "Uncalibrated score" badge: "the G-score is a ranking signal, not a
verified measurement — we say that out loud instead of hiding it in a tooltip."

**1:20 — The bridge view.** Open the gap detail. Walk the two-cluster diagram:
box area is paper count, line thickness is cross-citations. Scroll to the
evidence: real papers on each side (*Leveraging Passage Retrieval with
Generative Models*, *Active Retrieval Augmented Generation* on one side;
*In-Context Retrieval-Augmented Language Models*, *REPLUG* on the other), and —
the part that makes this actionable — real methods and datasets the agent
pipeline actually extracted from those papers' abstracts. Click a paper to open
the panel, show the evidence-span highlighting in the real abstract. Click
"Check against OpenAlex": a live external lookup counts how many OpenAlex works
already combine both sides' key concepts — low count here, so this isn't just
an artifact of a small corpus.

**2:00 — Draft a direction.** Click "Draft direction." While it generates
(a few seconds, one real LLM call): "this is grounded in both sides' real
papers, marked unverified, and editable — not a black box." Land on the
drafted pitch: a real, specific, citeable research direction naming both
clusters.

**2:30 — Shortlist.** Click "Keep." Add one line of notes. Click
"Export to Markdown" — show the formatted output: "this is what goes into the
supervisor meeting — the claim, the method sketch, the datasets, the papers to
read, and an explicit unverified disclaimer, not a polished-looking black box."

**2:55 — Close.** "One path end to end: pick a field — instant, over a shared
prebuilt corpus — see what's accelerating, find where the evidence is thin,
cross-check that against the wider literature, draft a direction, walk into
the meeting with it."

## If something breaks live

- **Backend down / a fetch fails:** every screen has a real error state
  ("Couldn't load…", a Retry button) — show it, say "this is the real error
  state, not a crash," and restart `uvicorn`.
- **LLM call fails or is slow:** the draft-direction step is the only live LLM
  call in the whole script; if it errors, the button shows the real error
  message inline. Have a second gap ready as a fallback (any of the 6 ranked
  gaps works — pick one with a round number of citations for easy narration).
- **OpenAlex check fails/times out:** it's on-demand and clearly separate from
  the core flow — note it ("that's a live external call, can be flaky") and
  move on to Draft direction.
- **Don't start a new field on an unfamiliar topic live** — most topics in this
  corpus's domain (retrieval/RAG/QA/LLM knowledge) resolve to a real field in
  under a second, but it's still worth rehearsing the exact topic string once
  beforehand rather than typing something untested in the room.
