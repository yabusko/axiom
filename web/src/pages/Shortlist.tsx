import { useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type ShortlistItemOut, type ShortlistStatus } from "../api";
import { EmptyState, ErrorState, LoadingState } from "../components/States";

function DirectionCard({ item, fieldId }: { item: ShortlistItemOut; fieldId: string }) {
  const queryClient = useQueryClient();
  const [notes, setNotes] = useState(item.notes ?? "");

  const patch = useMutation({
    mutationFn: (body: Partial<Pick<ShortlistItemOut, "status" | "notes" | "papers_to_read">>) =>
      api.patchShortlist(fieldId, item.id, body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["shortlist", fieldId] }),
  });

  return (
    <div className="card" style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)" }}>
      <h3 style={{ marginTop: 0 }}>{item.title}</h3>
      <p>{item.claim}</p>
      <p className="muted" style={{ fontSize: "var(--small)" }}>
        <strong>Method sketch:</strong> {item.method_sketch}
      </p>
      {item.datasets.length > 0 && (
        <p className="muted" style={{ fontSize: "var(--small)" }}>
          <strong>Datasets:</strong> {item.datasets.join(", ")}
        </p>
      )}
      {item.supporting_paper_ids.length > 0 && (
        <p className="muted" style={{ fontSize: "var(--small)" }}>
          <strong>Papers to read:</strong> {item.supporting_paper_ids.join(", ")}
        </p>
      )}
      <div className="badge-row">
        <span className="badge">AI-generated</span>
        <span className="badge-note">This pitch was written by a model, not verified — review before relying on it.</span>
      </div>

      <label style={{ display: "block", marginTop: "var(--space-4)", fontWeight: 700, fontSize: "var(--small)" }}>
        Notes
      </label>
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        onBlur={() => {
          if (notes !== (item.notes ?? "")) patch.mutate({ notes });
        }}
        rows={2}
        style={{ width: "100%", boxSizing: "border-box", padding: 8, border: "1px solid var(--border-strong)", background: "var(--bg)", color: "var(--text)", fontFamily: "inherit" }}
        placeholder="Why this direction, or what to check before the meeting…"
      />

      <div style={{ display: "flex", gap: "var(--space-2)", marginTop: "var(--space-4)" }}>
        {item.status !== "kept" && (
          <button className="btn btn-primary" onClick={() => patch.mutate({ status: "kept" as ShortlistStatus })} type="button">
            Keep
          </button>
        )}
        {item.status !== "discarded" && (
          <button className="btn" onClick={() => patch.mutate({ status: "discarded" as ShortlistStatus })} type="button">
            Discard
          </button>
        )}
        {item.status !== "draft" && (
          <button className="btn-quiet" onClick={() => patch.mutate({ status: "draft" as ShortlistStatus })} type="button">
            Back to draft
          </button>
        )}
      </div>
    </div>
  );
}

function ExportView({ fieldId }: { fieldId: string }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["shortlist-export", fieldId],
    queryFn: () => api.exportMarkdown(fieldId),
  });

  const copy = () => {
    if (data) navigator.clipboard?.writeText(data);
  };

  return (
    <div className="card" style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)" }}>
      <h3 style={{ marginTop: 0 }}>Markdown export</h3>
      {isLoading && <p className="muted">Loading…</p>}
      {isError && <p className="muted">Couldn't load the export.</p>}
      {data && (
        <>
          <pre style={{ whiteSpace: "pre-wrap", background: "var(--bg)", padding: "var(--space-4)", border: "1px solid var(--border)", maxHeight: 360, overflow: "auto" }}>
            {data}
          </pre>
          <button className="btn" onClick={copy} type="button">
            Copy to clipboard
          </button>
        </>
      )}
    </div>
  );
}

export default function Shortlist() {
  const { fieldId } = useParams<{ fieldId: string }>();
  const [showExport, setShowExport] = useState(false);
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["shortlist", fieldId],
    queryFn: () => api.shortlist(fieldId!),
    enabled: !!fieldId,
  });

  if (isLoading) return <LoadingState label="Loading the shortlist…" />;
  if (isError || !data) {
    return (
      <ErrorState title="Couldn't load the shortlist" onRetry={() => refetch()}>
        Try again.
      </ErrorState>
    );
  }

  const groups: { status: ShortlistStatus; label: string }[] = [
    { status: "draft", label: "Draft" },
    { status: "kept", label: "Kept" },
    { status: "discarded", label: "Discarded" },
  ];

  return (
    <>
      <div className="page-head">
        <h1>Shortlist</h1>
        <p className="lede">Directions you've drafted from research gaps. Keep the ones worth bringing to a supervisor meeting.</p>
      </div>

      {data.length === 0 ? (
        <EmptyState title="Nothing drafted yet">
          Open a research gap and draft a direction — it'll show up here.
        </EmptyState>
      ) : (
        <>
          {groups.map((g) => {
            const items = data.filter((i) => i.status === g.status);
            if (items.length === 0) return null;
            return (
              <div className="section" style={{ marginBottom: "var(--space-8)" }} key={g.status}>
                <h2>{g.label}</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
                  {items.map((item) => (
                    <DirectionCard item={item} fieldId={fieldId!} key={item.id} />
                  ))}
                </div>
              </div>
            );
          })}

          <button className="btn" onClick={() => setShowExport((v) => !v)} type="button">
            {showExport ? "Hide export" : "Export to Markdown"}
          </button>
          {showExport && (
            <div style={{ marginTop: "var(--space-4)" }}>
              <ExportView fieldId={fieldId!} />
            </div>
          )}
        </>
      )}
    </>
  );
}
