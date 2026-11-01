# Axiom web

React + Vite + TypeScript SPA for Axiom's primary flow (stage-3 product redesign,
OD33 — reverses OD15). Talks to `api/main.py`'s `/fields/*` endpoints
(`docs/product/ARCHITECTURE_NOTE.md`). Streamlit (`app/streamlit_app.py`) stays the
internal lab console; this is the product surface.

## Run it

Needs the FastAPI backend running first (it owns the data):

```bash
# from the repo root, with the project's Python venv active
AXIOM_OS_TRUSTSTORE=1 uvicorn api.main:app --reload   # http://localhost:8000
```

Then, in `web/`:

```bash
npm install
npm run dev   # http://localhost:5173
```

The API base URL defaults to `http://localhost:8000`; override with a `VITE_API_BASE`
env var (e.g. in `web/.env.local`) if FastAPI runs elsewhere.

### Static demo build (no backend)

`VITE_DEMO=1 npm run build` routes every API call to `src/demoData.ts` instead of
fetching -- the whole app works with no backend at all, for a static deploy (this
repo's GitHub Pages workflow, `.github/workflows/pages.yml`, builds this way).

## Structure

- `src/api.ts` — typed client for every `/fields/*` endpoint; types mirror
  `api/main.py`'s Pydantic response models 1:1.
- `src/pages/` — Start, Landscape, Gaps, GapDetail, Shortlist, and Map (3D citation
  graph / embedding view, lazy-loaded, outside the main flow).
- `src/demoData.ts` — static response data for a `VITE_DEMO=1` build (below); keep it
  in sync with `e2e/fixtures.ts` if you refresh one.
- `src/components/` — shared components per `docs/product/DESIGN_SYSTEM.md`'s specs
  (gap card, bridge view, trend row, paper row, evidence highlight, badge,
  build-progress steps, empty/error states) plus the global Paper panel.
- `src/index.css` imports `../design/tokens.css` directly — the design tokens are
  NOT duplicated here; `web/design/` stays the one source of truth for both the
  static design-system gallery and this app.

## Tests

```bash
npm run lint       # oxlint
npm run build      # tsc -b && vite build -- type-checks and production-builds
npm run test:e2e   # Playwright: the happy-path flow + an axe accessibility scan per screen
```

`e2e/fixtures.ts` mocks every `/fields/*` call (scoped to `http://localhost:8000` —
do not use a bare path glob like `**/fields/...` in a route mock here, it will also
intercept the SPA's own same-shaped page navigations and serve raw JSON instead of
the app shell; a real bug this suite caught once already). The e2e suite runs
against the dev server with no live backend and no LLM spend. Backend endpoint
tests live in `tests/test_fields.py` at the repo root (pytest), not here.

Playwright is configured with a single worker (`playwright.config.ts`) — this
laptop's dev server doesn't reliably serve several concurrent page loads at once;
parallel runs intermittently hit `page.goto` timeouts.
