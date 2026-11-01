import { Suspense, lazy } from "react";
import { NavLink, Route, Routes, useParams } from "react-router-dom";
import Start from "./pages/Start";
import Landscape from "./pages/Landscape";
import Gaps from "./pages/Gaps";
import GapDetail from "./pages/GapDetail";
import Shortlist from "./pages/Shortlist";
import { PaperPanel } from "./components/PaperPanel";
import { LoadingState } from "./components/States";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

// Outside the primary flow and pulls in a 3D rendering library -- code-split
// so its weight never lands in the main-flow bundle (first-load JS budget).
const MapScreen = lazy(() => import("./pages/Map"));

function FieldNav() {
  const { fieldId } = useParams<{ fieldId: string }>();
  const { data } = useQuery({
    queryKey: ["field-status", fieldId],
    queryFn: () => api.fieldStatus(fieldId!),
    enabled: !!fieldId,
  });

  return (
    <header className="app-header">
      <NavLink to="/" className="brand">
        <span className="wordmark">Axiom</span>
        <span className="tagline">Find under-explored directions, with evidence.</span>
      </NavLink>
      {fieldId && (
        <nav className="primary-nav" aria-label="Primary">
          <NavLink to={`/fields/${fieldId}/landscape`}>Landscape</NavLink>
          <NavLink to={`/fields/${fieldId}/gaps`}>Research gaps</NavLink>
          <NavLink to={`/fields/${fieldId}/shortlist`}>Shortlist</NavLink>
          <NavLink to={`/fields/${fieldId}/map`}>Map</NavLink>
        </nav>
      )}
      {data && (
        <div className="field-context">
          <strong>{data.topic}</strong> · {data.papers_total ?? "…"} papers
        </div>
      )}
    </header>
  );
}

function AppFooter() {
  const { data } = useQuery({ queryKey: ["corpus-manifest"], queryFn: api.corpusManifest });
  return (
    <footer className="app-footer">
      Axiom — a research-direction finder, not a paper search engine.
      {data && (
        <span className="corpus-manifest">
          {" "}
          · Base corpus: {data.papers_count.toLocaleString()} papers, {data.from_year}–{data.to_year}, updated{" "}
          {data.updated_at.slice(0, 10)}
        </span>
      )}
    </footer>
  );
}

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Start />} />
        <Route
          path="/fields/:fieldId/*"
          element={
            <>
              <FieldNav />
              <main className="app-main">
                <Routes>
                  <Route path="landscape" element={<Landscape />} />
                  <Route path="gaps" element={<Gaps />} />
                  <Route path="gaps/:gapIndex" element={<GapDetail />} />
                  <Route path="shortlist" element={<Shortlist />} />
                  <Route
                    path="map"
                    element={
                      <Suspense fallback={<LoadingState label="Loading the map…" />}>
                        <MapScreen />
                      </Suspense>
                    }
                  />
                </Routes>
              </main>
              <PaperPanel />
            </>
          }
        />
      </Routes>
      <AppFooter />
    </>
  );
}
