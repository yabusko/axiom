# Axiom: 3-minute demo script

For interviews. Matches the confirmed demo scenario in `docs/product/PRODUCT.md`:
start from an already-built field, go straight to a gap, draft a direction, shortlist
it — no live build in the critical path. Uses the real "retrieval-augmented
generation" field (300 real OpenAlex papers) that's already in the running app; every
number below is real, pulled from that corpus, not staged.

## Setup (before the room, not during it)

1. Backend running: `uvicorn api.main:app --reload` (repo root, venv active).
2. Frontend running: `npm run dev` in `web/` → `http://localhost:5173`.
3. Open the app to `/` (Start) so the first thing on screen is the product's name
   and the real field in "Recent fields."

## The script (~3 minutes)

**0:00 — Start.** "Axiom isn't a paper search engine — it surfaces two things:
which concepts are accelerating in a field, and which related areas of work barely
cite each other despite being closely related. That second one is the interesting
part — those are candidate thesis directions." Click into the
"retrieval-augmented generation" field (300 papers, already built).

**0:20 — Landscape.** Point at the stat row: 300 papers, 1,699 in-corpus citations,
31 topic communities, 15 candidate gaps. "These aren't a second clustering —
they're the same communities the gaps are built from." Scroll to Rising concepts:
"Machine translation," "Human–computer interaction" rising from 0 prior mentions —
"shown as a ratio, not a raw percent, because a concept with zero prior mentions
still has somewhere to rise from." Key papers: SQuAD, BERT — ranked by PageRank
within this corpus's own citation graph, not raw citation count.

**0:50 — Research gaps.** Click through to the ranked list. Pick the top one:
"a 17-paper cluster on NLP/question-answering methods barely cites an 11-paper
machine-translation cluster, despite being 94.5% similar in embedding space —
zero citations cross between them." Point at the "Uncalibrated score" badge:
"the G-score is a ranking signal, not a verified measurement — we say that
out loud instead of hiding it in a tooltip."

**1:20 — The bridge view.** Open the gap detail. Walk the two-cluster diagram:
"side A, side B, this line is the gap." Scroll to the evidence: real papers on
each side (HotpotQA, Active Retrieval Augmented Generation on one side; mT5,
XL-Sum on the other), and — the part that makes this actionable — real
methods and datasets the agent pipeline actually extracted from those papers'
abstracts: "side A uses active retrieval and chain-of-thought prompting; here's
exactly which sentence that came from." Click a paper to open the panel, show
the evidence-span highlighting in the real abstract.

**2:00 — Draft a direction.** Click "Draft direction." While it generates
(a few seconds, one real LLM call): "this is grounded in both sides' real
papers, marked unverified, and editable — not a black box." Land on the
drafted pitch: a real, specific, citeable research direction naming both
clusters.

**2:30 — Shortlist.** Click "Keep." Add one line of notes. Click
"Export to Markdown" — show the formatted output: "this is what goes into the
supervisor meeting — the claim, the method sketch, the datasets, the papers to
read, and an explicit unverified disclaimer, not a polished-looking black box."

**2:55 — Close.** "Ten minutes ago this was a 6-tab Streamlit app with no
front door and an agent pipeline whose output never reached the UI. Now it's
one path: pick a field, see what's accelerating, find where the evidence is
thin, draft a direction, walk into the meeting with it."

## If something breaks live

- **Backend down / a fetch fails:** every screen has a real error state
  ("Couldn't load…", a Retry button) — show it, say "this is the real error
  state, not a crash," and restart `uvicorn`.
- **LLM call fails or is slow:** the draft-direction step is the only live LLM
  call in the whole script; if it errors, the button shows the real error
  message inline. Have a second gap ready as a fallback (any of the 15 ranked
  gaps works — pick one with a round number of citations for easy narration).
- **Don't start a new field live** — a real build takes several minutes and
  costs real money; it's demonstrated by description ("the Start screen is how
  you'd begin on a new field"), not performed.
