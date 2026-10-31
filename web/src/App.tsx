import { NavLink, Route, Routes, useParams } from "react-router-dom";
import Start from "./pages/Start";
import Landscape from "./pages/Landscape";
import Gaps from "./pages/Gaps";
import GapDetail from "./pages/GapDetail";
import Shortlist from "./pages/Shortlist";
import { PaperPanel } from "./components/PaperPanel";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

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
                </Routes>
              </main>
              <PaperPanel />
            </>
          }
        />
      </Routes>
      <footer className="app-footer">Axiom — a research-direction finder, not a paper search engine.</footer>
    </>
  );
}
