import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type FieldStatusOut } from "../api";
import { BuildProgress } from "../components/BuildProgress";
import { EmptyState, ErrorState, LoadingState } from "../components/States";

function BuildingFieldCard({ fieldId, estCostUsd, estSeconds }: { fieldId: string; estCostUsd?: number; estSeconds?: number }) {
  const navigate = useNavigate();
  const { data: status, isError } = useQuery({
    queryKey: ["field-status", fieldId],
    queryFn: () => api.fieldStatus(fieldId),
    refetchInterval: (query) => {
      const s = query.state.data as FieldStatusOut | undefined;
      return s && (s.status === "ready" || s.status === "failed") ? false : 2000;
    },
  });

  if (isError || !status) {
    return <ErrorState title="Couldn't load build status">Try refreshing the page.</ErrorState>;
  }
  if (status.status === "failed") {
    return (
      <ErrorState title="The build failed" onRetry={undefined}>
        {status.error ?? "Something went wrong during the build. Try again, or start a narrower topic."}
      </ErrorState>
    );
  }

  return (
    <div className="card" style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)" }}>
      <h3 style={{ marginTop: 0 }}>Building “{status.topic}”</h3>
      <BuildProgress status={status} estCostUsd={estCostUsd} estSeconds={estSeconds} />
      {status.status === "ready" && (
        <p style={{ marginTop: "var(--space-4)" }}>
          <button className="btn btn-primary" onClick={() => navigate(`/fields/${fieldId}/landscape`)} type="button">
            View landscape
          </button>
        </p>
      )}
    </div>
  );
}

export default function Start() {
  const [topic, setTopic] = useState("");
  const [buildingFieldId, setBuildingFieldId] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<{ est_cost_usd: number; est_seconds: number } | null>(null);
  const queryClient = useQueryClient();

  const fieldsQuery = useQuery({ queryKey: ["fields"], queryFn: api.listFields });

  const createMutation = useMutation({
    mutationFn: () => api.createField(topic.trim()),
    onSuccess: (res) => {
      setBuildingFieldId(res.field_id);
      setEstimate({ est_cost_usd: res.est_cost_usd, est_seconds: res.est_seconds });
      setTopic("");
      queryClient.invalidateQueries({ queryKey: ["fields"] });
    },
  });

  return (
    <>
      <header className="app-header">
        <a href="/" className="brand" style={{ textDecoration: "none" }}>
          <span className="wordmark">Axiom</span>
          <span className="tagline">Find under-explored directions, with evidence.</span>
        </a>
      </header>
      <main className="app-main">
        <div className="page-head">
          <h1>Start a field</h1>
          <p className="lede">
            Type a topic and Axiom builds a corpus for it: fetching papers, mapping topics, extracting methods and
            datasets, and finding research gaps. One field is active at a time.
          </p>
        </div>

        <div
          className="card"
          style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)", marginBottom: "var(--space-8)" }}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (topic.trim()) createMutation.mutate();
            }}
            style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", alignItems: "center" }}
          >
            <label htmlFor="topic-input" style={{ width: "100%", fontWeight: 700 }}>
              Topic
            </label>
            <input
              id="topic-input"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. retrieval-augmented generation"
              style={{
                flex: "1 1 320px",
                padding: "11px 14px",
                border: "1px solid var(--border-strong)",
                background: "var(--bg)",
                color: "var(--text)",
                fontFamily: "inherit",
                fontSize: "1rem",
              }}
            />
            <button className="btn btn-primary" type="submit" disabled={!topic.trim() || createMutation.isPending}>
              {createMutation.isPending ? "Starting…" : "Build field"}
            </button>
          </form>
          <p className="muted" style={{ marginTop: "var(--space-4)", marginBottom: 0, fontSize: "var(--small)" }}>
            Building the full corpus takes a few minutes and makes paid extraction calls. You'll see a cost and time
            estimate before the extraction step runs.
          </p>
          {createMutation.isError && (
            <p style={{ color: "var(--action)", marginTop: "var(--space-4)" }}>
              Couldn't start the build: {(createMutation.error as Error).message}
            </p>
          )}
        </div>

        {buildingFieldId && (
          <div style={{ marginBottom: "var(--space-8)" }}>
            <BuildingFieldCard fieldId={buildingFieldId} estCostUsd={estimate?.est_cost_usd} estSeconds={estimate?.est_seconds} />
          </div>
        )}

        <h2>Recent fields</h2>
        {fieldsQuery.isLoading && <LoadingState label="Loading recent fields…" />}
        {fieldsQuery.isError && <ErrorState title="Couldn't load recent fields">Try refreshing the page.</ErrorState>}
        {fieldsQuery.data && fieldsQuery.data.length === 0 && (
          <EmptyState title="No fields yet">
            Type a topic above to start a field — Axiom builds its corpus and finds research gaps in a few minutes.
          </EmptyState>
        )}
        {fieldsQuery.data && fieldsQuery.data.length > 0 && (
          <div className="paper-list">
            {fieldsQuery.data.map((f) => (
              <div className="paper-row" key={f.id}>
                <div className="paper-row-main">
                  {f.status === "ready" ? (
                    <a href={`/fields/${f.id}/landscape`}>{f.topic}</a>
                  ) : (
                    <span style={{ fontWeight: 700 }}>{f.topic}</span>
                  )}
                  <div className="paper-meta">
                    {f.papers_total ?? "…"} papers · {f.status}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </>
  );
}
