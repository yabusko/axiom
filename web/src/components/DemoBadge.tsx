import { DEMO_MODE } from "../api";

// B8: shown in both app headers (App.tsx's FieldNav and Start.tsx) when the
// build was made with VITE_DEMO=1 -- the GitHub Pages deploy, which serves
// recorded real responses from web/src/demoData.ts with no live backend.
export function DemoBadge() {
  if (!DEMO_MODE) return null;
  return (
    <span className="demo-badge" title="Static build serving recorded real API responses -- no live backend.">
      Demo data
    </span>
  );
}
