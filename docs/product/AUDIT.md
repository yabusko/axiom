# Axiom Streamlit UX/Product Audit (Phase A)

Read-only audit. No app/API code was changed. Grounded in 24 screenshots
(`docs/product/audit/{demo,real}/*_{desktop,mobile}.png`) plus
`app/streamlit_app.py` (1407 lines), `app/vector_tools.py` (670 lines),
`api/main.py` (631 lines), `axiom/graph.py`, `axiom/gaps.py`, `axiom/db.py`,
`docs/DECISIONS.md`, `docs/agentic/ARCHITECTURE.md`.

## Summary

Axiom is a single 1400-line Streamlit script with real, deliberately-built
custom dark/light theming (CSS variables, a hero banner, pill chips, styled
tabs/expanders — not default Streamlit chrome) and six tabs that each work
correctly for what they individually do. The product's actual defects are
structural, not cosmetic: there is no in-app way to start a new
field/topic (corpus choice is a CLI flag at ingest time), the FastAPI layer
in `api/main.py` is entirely dead code as far as the Streamlit app is
concerned (zero calls from `app/*` into `api/*` — every widget talks straight
to `axiom/*`/SQLite/Qdrant), the agentic extraction pipeline's output
(entities, methods, datasets, concept clusters) is wired into the API but
never read by the UI anywhere, and the six tabs are presented as equal
siblings with no indication of which one a first-time user should open first
even though the product's own hero line ("two signals: which concepts are
accelerating, and which related literatures barely cite each other") names
exactly two of the six. Mobile is functionally usable — nothing is
unreachable — but the hero/metadata chrome consumes the entire first
viewport (844px) before any tab content appears, and the tab bar overflows
horizontally with no scroll affordance.

---

## Tab 1 — Trending (📈)

**Screenshots:** `trending_desktop.png`, `trending_mobile.png` (demo: 30
papers, 2021–2025; real: 300 papers, 2020–2026). Both corpora render
identically in layout; only the data differs (demo: 3 risers shown, real: a
longer list with much larger % swings, e.g. "Graph" +315%).

**Widget → code map** (`app/streamlit_app.py`):
- Subheader/caption copy — `render_trending()` L1132–1144.
- "Heuristic signal" info banner — L1141–1144, static text every load.
- Venue selectbox + year-range (`render_year_filter`, floor pinned to
  `config.VELOCITY_MIN_YEAR`) — L1146–1155.
- Data: `get_velocity_analysis()` L437–444 → `axiom.velocity.get_top_velocity_keywords` directly (SQLite `conn`), no API call.
- Risers/faders bar charts — `_velocity_bar_chart()` L1087–1129, Altair, filtered to concepts with ≥`config.VELOCITY_MIN_CHART_COUNT` papers in both windows (L1182–1186).
- Full ranked list — custom HTML rows L1206–1220, "low volume" chip for `k.low_confidence`.

**Heuristic findings:**
- *Visibility of system status* (Low) — prior/recent window and paper counts are shown (L1169–1174), good: the velocity math is legible, not a black box.
- *Match with real world* (Med) — "velocity" and "log2 share ratio" are exposed as literal labels on the chart x-axis (L1117–1119) and in the caption; a non-technical thesis student has to parse `log2(recent_share/prior_share)` to read a bar chart. The `pct_label` (+N%) mitigates this but the axis title itself is still jargon.
- *Recognition over recall* (Low) — filters are visible selects, not typed state.

## Tab 2 — Research gaps (🕸️)

**Screenshots:** `research_gaps_desktop.png`/`_mobile.png`, both corpora.
Demo: 30 papers, 8 sub-topics, 6 candidate gaps, 29 in-corpus citations. Real:
300 papers, 30 sub-topics, 15 candidate gaps, 1699 in-corpus citations — same
layout, denser graph.

**Widget → code map:**
- Metrics row (Papers/In-corpus citations/Sub-topics/Candidate gaps) — `render_graph_view()` L923–927, from `graph.stats()` (`axiom/graph.py` L240–252) and `gaps.analyze()` (`axiom/gaps.py`), cached via `get_gap_analysis()` L427–434.
- View radio (Research gaps / Top influential) + candidate-gap selectbox — L932–960.
- 3D force graph — `_graph_html()`/`_GRAPH3D_TEMPLATE` L727–859, `3d-force-graph` loaded from `unpkg.com` CDN, rendered via `components.html` L988–991.
- Gap detail (`_render_gap_detail`, L867–894) — not visible in any static screenshot (no gap was pre-selected when the screenshot was taken); from code: shows semantic similarity, inter-citation count, the G-score + 4-component breakdown (similarity/disconnection/velocity/authority) with an "Uncalibrated defaults (OD17)" caveat (L881–883), and top-5 papers per side.
- "Generate hypothesis pitch" button — L1003–1029, calls `axiom.agents.synthesis.generate_hypothesis` (strong-tier LLM via `llm_providers`) synchronously via `asyncio.run`, writes to `review_queue` (SQLite) directly — bypasses `api/main.py`'s equivalent `POST /graph/gaps/{gap_index}/hypothesize` (L461–499) entirely, duplicating that logic client-side.
- "Unverified Candidate" banner — L997–1002, static `st.info`, same text regardless of the gap's actual G-score.
- Influence ranking — L1059–1083, `graph.influence()` (PageRank over corpus-only subgraph, `axiom/graph.py` L154–179).

**Heuristic findings:**
- *Visibility of system status* (Low) — metrics row + "X papers · Y citations in view" caption under the graph (L973–974) is good, concrete feedback.
- *User control and freedom* (Med) — once "Generate hypothesis pitch" is clicked there's no undo/cancel visible mid-spinner, and the produced pitch is stashed only in `st.session_state["last_pitch"]` keyed by `sel_gap_idx` (L1025, L1031–1032) — switching to another gap or tab and back loses the just-generated draft from view (it is still in the Review queue, but a user who doesn't know that may think it vanished).
- *Consistency* (Med) — the "⚠️ Unverified Candidate" wording here (L998) is near-identical to but not verbatim the same as the Review queue's banner (L1338) and `axiom/hypothesis.py`'s `disclaimer` string (L55) — three slightly different phrasings of the same warning, none scoped to the specific pitch's confidence.
- *Help/docs* (Low) — the 3D graph has an inline control legend ("drag rotate · +/–/▣ zoom · click node to fly · wheel scrolls page", L982–984) directly in the UI — good, discoverable.
- *Aesthetic/minimalist* (Low) — the "Show 3D graph" toggle (L964–965) is a thoughtful escape hatch for the known wheel-capture problem; well-reasoned per its own comment (L962–963).

## Tab 3 — Search (🔍)

**Screenshots:** `search_desktop.png`/`_mobile.png`, both corpora — static
empty-query state only ("Search the corpus" empty-state card). No result-card
screenshot exists.

**Widget → code map:**
- Venue multiselect, year filter, Top-K — `render_search()` L660–676.
- Hybrid/Dense radio, gated on `store.supports_hybrid()` — L679–688.
- Query text input + dispatch — L706–716: `get_encoder().encode_query(query)` (SPECTER2, `axiom/embed.py`) then `store.search_hybrid`/`store.search` (`axiom/qdrant_client.py`) directly — no API call.
- Results — `render_hits()` L611–651: expander cards with score chip, concepts, abstract, DOI link, **Add to reading list** (`toggle_bookmark` → `axiom.db`) and **Similar papers** pivot (sets `session_state["similar_to"]`, rerenders via `store.similar_papers`).
- From code (not screenshotted): a result card is an `st.expander` — collapsed by default, so a 10-result page shows 10 collapsed rows; abstract/DOI/actions only appear once expanded. The "Similar papers" pivot swaps the whole tab content for the pivot view with a "← Back to search" button (L701–703) that clears `similar_to` and returns to an **empty** query box (the typed query itself isn't persisted in session state) — a minor one-way trip: searching, pivoting to similar papers, then going back loses the original query text and results, not just the pivot.

**Heuristic findings:**
- *Error prevention* (Low) — "Top-K" is clamped `min_value=1, max_value=point_count` (L674), can't be set above corpus size.
- *User control/freedom* (Med) — see the back-from-pivot query loss above; no "restore last query" affordance.
- *Flexibility/efficiency* (Low) — hybrid vs dense toggle and an explicit tooltip explaining what hybrid adds (L682–684) is a good, honest touch.

## Tab 4 — Reading list (📚)

**Screenshots:** `reading_list_desktop.png`/`_mobile.png`, both corpora —
identical empty-state in both (no bookmarks exist in either seeded corpus for
this screenshot run).

**Widget → code map:**
- `render_reading_list()` L1224–1294: `db.list_bookmarks(conn)` (SQLite) directly.
- Per-paper expander: chips, DOI link, abstract, **Remove bookmark**, and (if no cached summary) **Summarize** button → `summarize.summarize_paper()` (local Ollama, `config.OLLAMA_MODEL`) → `db.save_summary`. Cached summaries render as a bullet list with `_(cites `{paper_id}`)_` — oddly, every bullet is annotated as "cites" the paper's own ID (L1274), which reads like a citation-grounding claim but is really just a provenance tag on the paper being summarized, not a citation relationship — mildly confusing phrasing.
- From code: this tab's only way to populate is the Search tab's bookmark button; there is no "browse all papers" or "bookmark from the influence ranking" path, so the Reading list is reachable only after a successful Search.

**Heuristic findings:**
- *Match with real world* (Low-Med) — "cites `{paper_id}`" bullet-suffix (above) misuses "cites."
- *Help and documentation* (Low) — caption clearly states the model and that it's local/Ollama-based (L1228–1230), good transparency about where the LLM call runs.
- *Empty states* (Low) — consistent, actionable empty-state card ("Bookmark papers from a **Search** result card…") matches the pattern used everywhere else.

## Tab 5 — Review queue (🗂️)

**Screenshots:** `review_queue_desktop.png`/`_mobile.png`, both corpora —
both show "No pending pitches" (neither screenshot run generated a pitch
first).

**Widget → code map:**
- `render_review_queue()` L1298–1356. Status radio (pending/approved/rejected/all) — L1306–1307 → `db.list_review_queue()`.
- Per-item expander: claim, method sketch, datasets, supporting paper IDs, the "⚠️ Unverified Candidate" `st.warning` (L1338, see Consistency finding above), Approve/Reject buttons → `db.set_review_status()` directly (SQLite), bypassing `api/main.py`'s equivalent `/review-queue/{id}/approve|reject` (L521–536).
- From code: Approve/Reject only flips a `status` column. There is no downstream consumer of "approved" anywhere in the app — no list of approved hypotheses surfaced elsewhere, no export. An approved pitch's only visible trace is that it now shows under the "approved" filter in this same tab. This matches OD16's explicit scope ("nothing is promoted automatically") but the product gives the reviewer no signal about what approval is *for*.

**Heuristic findings:**
- *User control/freedom* (Low) — Approve/Reject are not undoable from the UI (status flips with no confirm, no "revert to pending"); a misclick silently reclassifies an item. Low severity since it's an internal back-office queue, not public-facing.
- *Visibility of system status* (Med) — approving has no visible consequence beyond the status filter; a reviewer has no way to tell the approval accomplished anything (dead end, see below).

## Tab 6 — Advanced: vector geometry (🔬)

**Screenshots:** `advanced_vector_desktop.png`/`_mobile.png`, both corpora.
Desktop screenshots show the **Paper Similarity Analyzer** sub-feature only
(scroll cut off before Vector Gap Discovery); mobile shows only the header
and intro caption. Backed entirely by `app/vector_tools.py`.

**Widget → code map:**
- Tab dispatch — `render_vector_tools()` L1360–1388, both sub-features behind one tab.
- **Paper Similarity Analyzer** (`render_proximity_analyzer`, `vector_tools.py` L304–503): two paper selectboxes (scrolled from `_get_all_paper_titles`, a raw Qdrant `scroll`), cosine similarity metric + plain-language verdict (`_interpret`, L273–301), a PCA 2-D neighborhood scatter (Plotly), and a "bridge papers" search at the vector midpoint.
- **Vector Gap Discovery** (`render_vector_gap_discovery`, L510–669): KMeans topic count slider, "Rescan" button (clears 3 caches), bridgeable-similarity-band slider defaulted from the corpus's own p40/p88 percentiles, a PCA landscape plot with the top gap's midpoint marked "GAP", and a ranked gap feed with "Opportunity: HIGH/MODERATE" labels.
- Both sub-features talk to Qdrant directly via `store.client` (raw `qdrant_client`), independent of `axiom.qdrant_client.AxiomQdrant`'s higher-level search methods used elsewhere.

**Heuristic findings:**
- *Match between system and real world* (Med) — the demo screenshot shows cosine 0.901 labeled "🟢 Near-duplicate territory" for two arbitrarily-selected papers (selectbox defaults to index 0/1); the real-corpus screenshot shows 0.877 for two topically different papers ("(Comet-) Atomic 2020" vs. "A Critical Evaluation of Evaluations for Long-form QA") also labeled "Near-duplicate" — the `_interpret()` threshold (≥0.82, L275) appears to run hot on SPECTER2 embeddings of this corpus, so a first-time user comparing two unrelated-looking titles may distrust the "near-duplicate" framing. Worth recalibrating thresholds against this corpus's actual similarity distribution (the Vector Gap Discovery sub-feature already does this adaptively via `sim_pct` p40/p88 — the Similarity Analyzer's thresholds are hardcoded by contrast).
- *Consistency* (Low) — this tab reads Qdrant via a second, parallel code path (`vector_tools.py`'s raw client calls) rather than `axiom.qdrant_client.AxiomQdrant`'s methods used by Search/Research gaps — not visible to the user, but a maintenance/consistency smell flagged for completeness.
- *Recognition over recall* (Low) — the tab's own caption correctly self-identifies as "a complementary geometry check, not a replacement" (L1369–1370) for Research gaps — good, proactive disambiguation rather than leaving the user to infer the relationship.

---

## Two gap features

| | 🕸️ Research gaps (tab 2) | 🔬 Vector Gap Discovery (tab 6) |
|---|---|---|
| Module | `axiom/gaps.py` `analyze()` | `app/vector_tools.py` `_find_topic_gaps()` |
| Clustering | Louvain communities on the **citation graph** (undirected corpus subgraph) | KMeans (k chosen by user slider) on raw SPECTER2 vectors |
| Signal used | Citation structure (inter-community edge count) + semantic centroid cosine + (OD17) concept velocity + authority (log cited-by) | Pure embedding geometry: centroid cosine band + "bridge ratio" (closest paper's joint similarity to both centroids) |
| Score | `gap_score = sim/(1+inter_citations)` (OD9) plus a calibrated composite `g_score` blending 4 components (OD17); **weights/threshold explicitly uncalibrated**, flagged in-UI (`streamlit_app.py` L881–883) and in `docs/DECISIONS.md` OD17 | `gap_score = 1 - bridge_ratio`, `coverage = clip(bridge_ratio,0,1)`; thresholds (band) are percentile-adaptive per corpus, not calibrated against labels either |
| Output in UI | Ranked gap list + "Generate hypothesis pitch" (feeds Review queue) | Ranked gap feed with "Opportunity: HIGH/MODERATE" label; **no path to hypothesis generation or Review queue** |
| Does either reference the other? | Tab 2's caption: "The product's citation-backed gap view" (L904) — implicitly primary, no explicit pointer to tab 6 | Tab 6's caption explicitly says: "Second opinion from embedding space only — no citation evidence. The product gap view is **Research gaps**… This tab is a complementary geometry check, not a replacement" (`streamlit_app.py` L1367–1370) and its module docstring states both are "retained because they surface complementary signals" (`vector_tools.py` L1–14) |
| Provenance | Native to this project (OD9/OD17) | Ported wholesale from a separate side project `qdrant_ms/` (PR #3) during OD18, which unified the Qdrant collection so both tools could read the same data |
| Evidence comparing quality | None found. No eval script, DECISIONS entry, or RESULTS.md section benchmarks one against the other (e.g., against labeled gap data) — OD17's calibration scaffold (`scripts/calibrate_gap_thresholds.py`) only calibrates tab 2's `g_score`, nothing analogous exists for tab 6. | |

**Recommendation: keep both, but demote tab 6 to a secondary panel inside tab 2 (or clearly subordinate it), not drop either.** Reasoning: the two features are genuinely complementary signal sources exactly as `docs/DECISIONS.md` OD18 argues (citation-evidenced vs. geometry-only), and dropping the geometry-only view would remove the only gap signal that works when citation data is sparse (the exact failure mode OD9's own text flags: "on a sparse single-topic corpus… discrimination is limited"). But the current presentation — two unrelated top-level tabs, one feeding the product's actual workflow (hypothesis → review queue) and one a dead-end analysis panel — overstates tab 6's standing as a peer product surface. Concretely: fold "Vector Gap Discovery" into tab 2 as an expandable "cross-check with embedding-only geometry" section (or keep tab 6 but rename it to make clear it has no workflow — it currently already self-disclaims this in text, but the *information architecture* (a full top-level tab, equal visual weight, in a different numbered position from "Research gaps") still implies parity it doesn't have). At minimum, wire a "Generate hypothesis pitch" affordance from a Vector Gap Discovery result too, or explicitly state why that's intentionally withheld.

---

## Agent pipeline visibility

**Confirmed: the agentic extraction pipeline's output is never read by the Streamlit app, anywhere.**

- `app/streamlit_app.py` and `app/vector_tools.py` contain zero references to `paper_concepts`, `concept_cooccurrence`, `agent_runs`, `concept_graph`, or `agents_pipeline` (grep confirms only one agent-related import: `from axiom.agents import synthesis as synthesis_agent`, used solely for the Research-gaps tab's "Generate hypothesis pitch" button, L32, L1012 — this is the *synthesis* agent only, not the extraction pipeline).
- The extraction pipeline (coordinator → extraction → enrichment → linking, `axiom/agents/{coordinator,extraction,enrichment,linking}.py`, per `docs/agentic/ARCHITECTURE.md`) writes `paper_concepts`/`concept_cooccurrence` rows holding typed entities (`kind: method | task | dataset | metric`, `axiom/agents/contracts.py` L37) per paper, and `axiom/concept_graph.py` builds a Louvain-clustered, PageRanked co-occurrence graph over them (`ConceptGraphAnalysis`).
- That analysis is exposed **only** via `api/main.py`'s `POST /agents/run`, `GET /agents/runs/{id}`, `GET /concepts/communities`, `GET /concepts/central` (`api/main.py` L565–630) — none of which Streamlit calls (confirmed: `grep -rn "api\.main\|requests\|httpx" app/` returns nothing).
- Net effect: a user can never see which methods/datasets/tasks the agent extracted per paper, never see the concept clusters it formed, and never trigger or monitor an extraction run — from inside the product. The only place this is runnable/visible today is the FastAPI docs UI (`/docs`) or direct API calls, and `docs/agentic/RESULTS.md`/benchmark scripts.
- The task brief's framing ("the agent pipeline is invisible") is **confirmed as stated**, not merely plausible.

---

## Dead ends and duplicated features

- **FastAPI layer is wholesale dead code from the product's perspective.** Every one of `api/main.py`'s ~20 routes (search, papers, trends, graph/gaps, reading-list, review-queue, hypothesize, agents/*, concepts/*) has zero callers in `app/*`; Streamlit reimplements each data access directly against `axiom/*`/SQLite/Qdrant instead (confirmed by the empty grep above, and by `api/main.py`'s own docstring calling itself "the stable REST seam a future React frontend will use," L1–3 — i.e., it's explicitly known/intended to be unused by Streamlit, not an oversight — but it does mean today's FastAPI service is unreachable from the one product surface that exists).
- **Hypothesis-generation logic is duplicated.** `render_graph_view()`'s "Generate hypothesis pitch" button (`streamlit_app.py` L1003–1029) and `api/main.py`'s `POST /graph/gaps/{gap_index}/hypothesize` (L461–499) both call `synthesis_agent.generate_hypothesis` and both write to `review_queue` — two independent call sites doing the same thing with slightly different error handling (Streamlit catches `LLMError`/`VerificationError` as inline `st.error`; the API maps them to 503/422).
- **Approve/Reject in Review queue is a dead end.** Flipping status has no downstream effect visible anywhere in the product (see Tab 5 findings above) — by design (OD16: "nothing is promoted automatically"), but the UI gives no hint of what approval *is for*, so a first-time reviewer cannot tell the action did anything beyond relabeling.
- **Similar-papers pivot loses the originating query.** Search → "Similar papers" → "← Back to search" returns to an empty query box, not the original search (`streamlit_app.py` L697–723) — a one-way trip for the query text, though not for bookmarking state.
- **Two gap-finding features, one hypothesis path.** See "Two gap features" above — Vector Gap Discovery's ranked gaps have no "generate pitch" affordance, so a user who finds a compelling gap there has no in-product way to act on it; they'd have to find the analogous (if it exists) citation-based gap in tab 2 instead.
- **No corpus/topic switcher anywhere in the UI** — see below.

---

## What's actually true vs. what the task brief assumed

- **"The tabs follow the system, not the user."** Confirmed, with a caveat: the tab order (Trending → Research gaps → Search → Reading list → Review queue → Advanced) is explicitly documented as intentional in the file's own module docstring ("Demo tab order: Trending → Research gaps → Search → …", `streamlit_app.py` L3–5) — it is a deliberate "lead with insight, not search box" design choice, not an accident. But the brief's underlying complaint stands: nothing in the UI (no onboarding copy, no "start here" marker, no default-selected tab logic) tells a first-time user that this order is meaningful, or that Search — the one universally-expected entry point — is 3rd rather than 1st. The hero line does name two signals (trending/gaps) which does roughly match tabs 1–2, so the ordering has a latent logic a careful reader could reconstruct from the hero text, but that's a stretch for a cold user. **Confirmed at Medium-High severity, not blanket-disproven.**
- **"No front door" (no in-app corpus/topic switcher).** **Confirmed exactly as stated.** `scripts/ingest_openalex.py --topic "..."` (L31–33) and `scripts/bootstrap_synthetic.py` (argparse-free, hardcoded demo generator) are the only ways to seed a corpus; there is no in-app control, button, or form anywhere in `streamlit_app.py` that starts a new ingest or switches the active field. The app's only "which corpus am I looking at" signal is the static pill row (`ACL · EMNLP · COLING · NAACL · 2020–2025`, L552) which doesn't even change between the demo and real-corpus screenshots despite the real corpus actually being about "retrieval-augmented generation," not the venue-filtered ACL/EMNLP/COLING/NAACL set the pill implies. The string is a hardcoded literal (L552), not derived from `db.distinct_venues()` (which the Search/Trending venue filters do use, L464/L660/L1146) — and `ingest_openalex.py`'s own default venue allow-list (`config.RELEVANT_VENUE_PATTERNS`, `axiom/config.py` L89–96) actually also admits NeurIPS/ICLR/ICML/AAAI, which the pill doesn't mention either. **This pill is actively inaccurate on the real corpus**, worth flagging beyond the original brief.
- **"Two different 'gap' features."** Confirmed exactly as characterized in the brief, and better-disclosed in-app than the brief implied: both captions in the live UI already state the relationship explicitly (tab 2 "The product's citation-backed gap view," tab 6 "complementary geometry check, not a replacement") and `docs/DECISIONS.md` OD18 documents the decision to keep both. The gap isn't *awareness* (the app is honest about which is primary) — it's that tab 6 has no workflow connection (no hypothesis path) despite being presented as a full co-equal tab.
- **"The agent pipeline is invisible."** Confirmed exactly as stated — see dedicated section above.
- **"Honesty is handled by banners."** Confirmed exactly as stated — the "Unverified Candidate" banner appears in at least 3 near-duplicate forms (`streamlit_app.py` L998, L1048, L1338; `axiom/hypothesis.py` L55) and is identical regardless of a pitch's actual G-score or verification detail; the G-score's own "Uncalibrated defaults" caveat (L881–883) is a second, separate repeated disclaimer, also not scoped per-score (it's the same sentence attached to every gap regardless of how close to or far from calibrated the weights are known to be for that particular community pair).
- **Custom dark-theme styling.** The brief's framing implicitly treats this app as potentially default/unstyled Streamlit needing a redesign pass. **This is not accurate and the audit should say so plainly:** the app ships a genuinely non-trivial, deliberately engineered theming system — CSS custom properties for a full light/dark palette (`streamlit_app.py` L68–118), a persisted theme toggle synced to a URL query param (L44–57), a styled hero banner, pill chips, styled tabs/expanders/buttons/scrollbars, and per-component dark-mode-aware chart theming for both Altair (`_chart_theme()`) and Plotly (`vector_tools.py`'s `_plot_theme()`) and even the 3D graph (`_GRAPH3D_TEMPLATE`'s `THEMES` dict switches background/link/arrow/particle colors by app theme). All 24 screenshots confirm this renders correctly and consistently across both corpora and both breakpoints. **Any redesign stage should treat this as a real asset to preserve/extend, not something to bolt on from scratch.**
- **Mobile layout.** Not explicitly claimed broken in the brief, but worth stating precisely since the brief asked for comparison: nothing is cut off in the sense of being unreachable, but (a) the hero + 4-pill metadata row alone fills the entire 844px mobile viewport in every one of the 12 mobile screenshots, pushing all tab content below the fold with no visual cue that tabs exist until scrolled; (b) the tab bar (`stTabs`) overflows horizontally and the active tab is sometimes rendered flush-left and sometimes flush-right of the visible strip across different screenshots (e.g. "Research gaps" sits mid-strip while "Review queue"/"Advanced: vector geometry" sit flush-right) with no fade/arrow affordance indicating more tabs are off-screen in either direction — a user has to discover horizontal swipe by accident.
