import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { api, type EntityOut } from "../api";
import { usePaperPanel } from "../hooks/usePaperPanel";

const KIND_CLASS: Record<EntityOut["kind"], string> = {
  method: "ev-method",
  dataset: "ev-dataset",
  task: "ev-task",
  metric: "ev-metric",
};

function highlightAbstract(abstract: string, entities: EntityOut[]) {
  type Match = { start: number; end: number; kind: EntityOut["kind"] };
  const matches: Match[] = [];
  for (const e of entities) {
    if (!e.evidence_span) continue;
    const idx = abstract.indexOf(e.evidence_span);
    if (idx === -1) continue; // evidence is the title, not in the abstract body
    matches.push({ start: idx, end: idx + e.evidence_span.length, kind: e.kind });
  }
  matches.sort((a, b) => a.start - b.start);
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  for (const m of matches) {
    if (m.start < cursor) continue; // skip overlapping spans, keep it simple/legible
    if (m.start > cursor) nodes.push(abstract.slice(cursor, m.start));
    nodes.push(
      <span className={KIND_CLASS[m.kind]} key={`${m.start}-${m.end}`}>
        {abstract.slice(m.start, m.end)}
      </span>,
    );
    cursor = m.end;
  }
  if (cursor < abstract.length) nodes.push(abstract.slice(cursor));
  return nodes;
}

export function PaperPanel() {
  const { fieldId } = useParams<{ fieldId: string }>();
  const { openPaperId, closePaper } = usePaperPanel();
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["paper", fieldId, openPaperId],
    queryFn: () => api.paper(fieldId!, openPaperId!),
    enabled: !!fieldId && !!openPaperId,
  });

  useEffect(() => {
    if (!openPaperId) return;
    closeButtonRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") closePaper();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [openPaperId, closePaper]);

  if (!openPaperId) return null;

  return (
    <>
      <div className="panel-backdrop" onClick={closePaper} />
      <aside className="paper-panel" role="dialog" aria-label="Paper detail" aria-modal="true">
        <div className="paper-panel-head">
          <h2>{isLoading ? "Loading…" : (data?.title ?? openPaperId)}</h2>
          <button className="panel-close" onClick={closePaper} type="button" ref={closeButtonRef}>
            Close
          </button>
        </div>
        {isError && <p className="muted">Couldn't load this paper.</p>}
        {data && (
          <>
            <p className="paper-meta">
              {[data.venue, data.year, `${data.cited_by_count.toLocaleString()} citations`].filter(Boolean).join(" · ")}
            </p>
            {data.entities.length > 0 && (
              <div className="badge-row">
                <span className="badge">Unverified</span>
                <span className="badge-note">
                  Entities below are extracted by an LLM pipeline from this abstract, not confirmed by a human.
                </span>
              </div>
            )}
            {data.abstract ? (
              <>
                <h3>Abstract</h3>
                <p className="abstract">{highlightAbstract(data.abstract, data.entities)}</p>
              </>
            ) : (
              <p className="muted">No abstract available for this paper.</p>
            )}
            {data.entities.length > 0 ? (
              <>
                <p className="legend">
                  <span>
                    <span className="ev-method">solid underline</span> = method
                  </span>
                  <span>
                    <span className="ev-dataset">dashed</span> = dataset
                  </span>
                  <span>
                    <span className="ev-task">dotted</span> = task
                  </span>
                </p>
                <h3>Extracted entities</h3>
                <div className="badge-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 6 }}>
                  {data.entities.map((e) => (
                    <span key={`${e.kind}-${e.canonical}`} className="badge-note">
                      <strong>{e.kind}</strong>: {e.canonical}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <p className="muted">No extracted entities for this paper yet.</p>
            )}
            {data.doi && (
              <p>
                <a href={data.doi} target="_blank" rel="noreferrer">
                  Open paper (DOI)
                </a>
              </p>
            )}
          </>
        )}
      </aside>
    </>
  );
}
