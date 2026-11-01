# Axiom design system — "Field Map" (Phase D, Gate 2 pick: Direction B, 2026-10-31)

Canonical tokens live in `web/design/tokens.css`; a live demo of every component below in both
themes is `web/design/component-gallery.html`. This document is the spec; those two files are
the implementation of it.

## Direction

Cartographic/spatial, grounded in the subject (a map of a field, not a generic SaaS dashboard —
see `AXIOM_PRODUCT_TASK.md` §6, condensed into the task's own design principles since the
`frontend-design` plugin couldn't be loaded mid-session, see `.claude/product/PROGRESS.md`).
Topics render as sized map tiles, not a list; the gap "bridge" view sits on a graph-paper
backdrop; one accent color (coral) marks the signature element — the gap connector and the
single primary action per screen — and nowhere else. Sharp 0px corners throughout (map-tile
feel, deliberately not the generic rounded-card-with-soft-shadow look). No shadows anywhere;
elevation is communicated by a `--surface` fill against `--bg`, plus hairline borders.

## Tokens

### Color — light theme (default)

| Token | Value | Use | Contrast |
|---|---|---|---|
| `--bg` | `#F3F6F7` | page background | — |
| `--surface` | `#FFFFFF` | cards, header, panels | — |
| `--text` | `#20262B` | primary text | 14.08:1 on `--bg`, 15.29:1 on `--surface` |
| `--text-muted` | `#5C6B73` | secondary text, captions | 5.08:1 on `--bg`, 5.52:1 on `--surface` |
| `--border` | `#CBD5D8` | decorative dividers only | 1.38:1 — not AA-rated, not required (decorative) |
| `--border-strong` | `#7E8D93` | input/interactive-element outlines | 3.16:1 on `--bg` (meets the 3:1 non-text threshold, WCAG 1.4.11) |
| `--mark` | `#E2572B` | graphical only: gap connector line, marker dot, focus ring | 3.43:1 on `--bg` (meets 3:1 non-text; **never use as text or as a text-bearing fill — fails 4.5:1 at 3.43**) |
| `--action` | `#C74A24` | button/CTA fills (text sits on them) | 4.74:1 with white text — self-contained, so it's identical in both themes |
| `--region` | `#3E7C74` | topic-tile border-left accent, active-nav border | 4.46:1 on `--bg` — used as a border/wash, never as text |
| `--region-wash` | `rgba(62,124,116,.10)` | topic-tile background tint | — |

### Color — dark theme

| Token | Value | Use | Contrast |
|---|---|---|---|
| `--bg` | `#1B2226` | page background | — |
| `--surface` | `#20272B` | cards, header, panels | — |
| `--text` | `#E7ECEE` | primary text | 13.53:1 on `--bg`, 12.72:1 on `--surface` |
| `--text-muted` | `#9BA8AE` | secondary text | 6.60:1 on `--bg`, 6.21:1 on `--surface` |
| `--border` | `#3A444A` | decorative dividers | not AA-rated, not required |
| `--border-strong` | `#636D73` | input/interactive outlines | 3.04:1 on `--bg` |
| `--mark` | `#E2572B` (unchanged) | graphical only | 4.32:1 on `--bg` — stronger than on light, same hue, no shift needed |
| `--action` | `#C74A24` (unchanged) | button/CTA fills | 4.74:1 with white text, theme-independent |
| `--region` | `#5FA096` (brightened) | topic-tile/nav accent | 5.34:1 on `--bg` — the light theme's `#3E7C74` drops to 3.33:1 on the dark background, still technically passing the 3:1 non-text floor but uncomfortably close, so dark mode gets a brightened variant for real legibility margin, not just a passing number |

**The one hard rule the two accent tokens encode**: `--mark` (bright coral, 3.43–4.32:1) is for
graphical marks only — a line, a dot, a focus ring, an icon stroke. `--action` (darker coral,
4.74:1 with white text, verified with the WCAG relative-luminance formula, not eyeballed) is for
anything with text sitting directly on it. Using `--mark` as a button fill is the single most
likely contrast regression in implementation — it was Direction B's own exploration HTML's
actual behavior before this gate (`web/design/directions/b/gap-detail.html`'s `.btn-primary`
used the bright `--coral-marker` at 3.73:1 with white text, which fails AA text contrast) —
**Phase E's component code must use `--action`, not `--mark`, for every button/CTA fill.**

### Type

One family throughout (unified/technical direction, no serif): **Public Sans**, loaded via
Google Fonts. Headlines weight 800 (`--h1` 2.75rem down to `--h3` 1.25rem), body 400/500, button
labels 700. Numeric data uses `font-feature-settings: "tnum" 1` (tabular figures) rather than a
separate monospace family — keeps Direction B's "unified, technical" character instead of
Direction A's ledger-mono split.

### Spacing, radii, elevation

4px base unit (`--space-1` 4px … `--space-12` 48px) — tighter than a typical 8px grid, matching
the "map legend" density called for in the direction brief. `--radius: 0` everywhere, no
exceptions. No `box-shadow` anywhere in the system; a card is `--surface` on `--bg` plus a
`1px solid var(--border)` and, where it needs a semantic color cue (a topic tile, an active nav
item), a `border-left: 3px solid var(--region)` or `background: var(--region-wash)` — never a
shadow standing in for elevation.

### Motion

`prefers-reduced-motion: reduce` disables all transitions/animations (`web/design/tokens.css`,
base rules) — the only motion this system uses at all is the paper-panel slide-in and the
build-progress step transitions, both already motion that "shows what changed" per the
task's own principle, not decorative.

## Component specs

Each spec below names the tokens it uses (so a reviewer can audit a future implementation
against this doc without re-deriving contrast math) and is demoed live in
`web/design/component-gallery.html`, both themes.

1. **Gap card** (ranked list row, `Research gaps` screen). `--surface` background, `1px solid
   var(--border)`, `border-left: 3px solid var(--region)`. Headline (`h3`, `--text`) states the
   finding in a sentence, not a label ("an NLP/ML cluster barely cites the QA cluster next to
   it"). A `tnum`-styled stat row (similarity %, citation count, G-score) in `--text-muted`.
   `--region-wash` never applied to the whole card (that's the topic *tile*'s treatment,
   component 7) — a gap card stays white/surface so the ranked list reads as a list, not a map.

2. **Bridge view** (`Research gaps` detail, the signature element). The one place this system
   spends visual boldness: two labeled rectangles (`1.5px solid var(--region)`, `--surface`
   fill) connected by a line in `--mark`, sitting on a faint `var(--border)` grid-paper backdrop
   (a repeating 1px-line CSS background, opacity low enough to read as texture not noise —
   `web/design/directions/b/gap-detail.html` has the working implementation, carry it over
   unchanged). The connector line and its label ("32 citations — the gap") are the only `--mark`
   usage on the whole screen. Similarity/citation totals sit outside the diagram in `--text-muted`.

3. **Trend row + sparkline** (`Landscape` → Rising concepts). A row: concept name (`--text`),
   a minimal SVG line sparkline (`stroke: var(--text-muted)`, 1.5px, no fill, no axis chrome —
   two data points only at this corpus's real window granularity, don't fabricate intermediate
   points), the recent-vs-prior ratio in `tnum` on the right. No color-coding by direction
   (rising vs. fading) beyond the arrow-free trend line itself — color is reserved for `--mark`/
   `--region`, not spent on a third "this number went up" signal.

4. **Paper row** (evidence papers, key papers, shortlist's papers-to-read). Title (`--text`,
   links to the Paper panel), one metadata line (`venue · year · N citations`, `--text-muted`,
   `tnum` for the numbers), `1px solid var(--border)` as the row divider (not a boxed card —
   rows in a list share a container's single border, consistent with the "ledger-adjacent but
   spatial" identity: dense where Direction A was also dense, but without A's own hairline-per-row
   literalism carried over — Direction B uses one container border, not N row borders, to match
   its flatter, map-tile character. See the gallery for the exact rendering.)

5. **Evidence highlight** (Paper panel abstract). Three underline styles coding entity `kind` —
   solid = method, dashed = dataset, dotted = task — in `--text` (never a color-coded highlight;
   this is a case where a text-color accent would either fail contrast at small sizes or compete
   with `--mark`/`--region`'s reserved meanings). A legend line states the mapping once, in
   `--text-muted`, at the top of the entity list — not repeated per-entity.

6. **Badge** ("AI-generated", "unverified", "uncalibrated score"). `1px solid var(--text)`,
   `--text` label, `--surface` fill, no color-coding (a badge is not a status/severity signal in
   this system — `--mark`/`--action`/`--region` all stay reserved for their one meaning each).
   Always paired with a one-line plain-text explanation directly below or beside it — per
   PRODUCT.md's "Honesty by design," this explanation is **always visible at this fidelity**,
   not hidden behind hover/focus (a real hover/focus-triggered tooltip is a Phase E implementation
   upgrade once real interactivity exists; don't regress to "always visible" being the only state
   once hover exists — the explanation should still appear on focus for keyboard users even if
   hover becomes the primary trigger on pointer devices).

7. **Build-progress steps** (`Start` screen, loading state). The one place numbered steps are
   legitimate (a real sequence, per the task's own exception to "no 01/02/03 markers"). Three
   real steps per `docs/product/ARCHITECTURE_NOTE.md` §3 — "Finding papers," "Embedding papers,"
   "Extracting methods & datasets" — not five; "mapping topics"/"finding gaps" are lazy
   first-load reads, not build steps, and must not appear as progress-bar segments (that note's
   finding carries directly into this spec so the two docs don't silently diverge). Each step:
   a filled `--region` dot once complete, an outlined `--border-strong` dot while pending, a
   `--mark`-ringed dot (via `:focus-visible`-style outline, persistent not just on focus) for the
   active step. The paid extraction step shows its real cost estimate inline (~$0.12 / ~4 min for
   300 papers, from `ARCHITECTURE_NOTE.md` §3 — not a placeholder) before it starts, and
   `papers_done / papers_total` in `tnum` while running.

8. **Empty / error states**. Both use the same layout shell as the populated screen (not a
   special centered-illustration page — the existing wireframes already establish this pattern,
   carry it forward) with a `--text-muted` one-or-two-sentence message. Empty states name the
   next action in an active verb ("Type a topic above to start a field," not "No fields yet").
   Error states say what happened and what to do, never just "Error" or a raw exception string —
   e.g. "The build stalled fetching papers from OpenAlex. Try again, or start a narrower topic."
   A retry action uses `--action` (it's a real CTA with text on a fill), never `--mark`.

## WCAG AA verification

Every pairing above was computed with the WCAG 2.1 relative-luminance contrast formula (script
run locally, not eyeballed — see the Color tables above for each ratio) against the **4.5:1**
text threshold and the **3:1** non-text/large-text/UI-component threshold, per WCAG 1.4.3 and
1.4.11. The one real finding from this pass: Direction B's own exploration HTML used the bright
`--mark` coral as a button fill, which fails text contrast at 3.73:1 with white text — fixed in
these tokens by introducing `--action` (a darker coral, 4.74:1) as the dedicated text-bearing-fill
token. Every other pairing in both tables above already passes its threshold without
modification from Direction B's original exploration values.

## What this phase did not do

Did not re-skin all 5 wireframe screens in Direction B's real tokens (that's Phase E's "build
screen by screen" in React, per the task's own phase split) — this phase's scope is the token
system, its WCAG verification, and the 8 component specs demoed in isolation
(`component-gallery.html`), not a second full pass over every screen.
