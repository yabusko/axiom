import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type FieldCreateOut } from "../api";
import { EmptyState, ErrorState, LoadingState } from "../components/States";

function ThinCoverageCard({ result, topic }: { result: FieldCreateOut; topic: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div
      className="card"
      style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)" }}
    >
      <h3 style={{ marginTop: 0 }}>Thin coverage for &ldquo;{topic}&rdquo;</h3>
      <p className="muted">
        Only {result.total_relevant} papers in the corpus are relevant to this topic — too few for a useful
        landscape or gap analysis. The corpus can grow around this topic specifically:
      </p>
      <pre
        style={{
          background: "var(--bg)", border: "1px solid var(--border)", padding: "var(--space-4)",
          overflowX: "auto", fontSize: "var(--small)",
        }}
      >
        {result.extend_command}
      </pre>
      <button
        type="button"
        className="btn-quiet"
        onClick={() => {
          navigator.clipboard?.writeText(result.extend_command ?? "");
          setCopied(true);
        }}
      >
        {copied ? "Copied" : "Copy command"}
      </button>
      <p className="muted" style={{ fontSize: "var(--small)" }}>
        Run it from the repo root (fetches, embeds, and extracts the new papers — takes a few minutes and a small
        LLM spend), then try this topic again.
      </p>
    </div>
  );
}

export default function Start() {
  const [topic, setTopic] = useState("");
  const [thinResult, setThinResult] = useState<{ result: FieldCreateOut; topic: string } | null>(null);
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const manifestQuery = useQuery({ queryKey: ["corpus-manifest"], queryFn: api.corpusManifest });
  const fieldsQuery = useQuery({ queryKey: ["fields"], queryFn: api.listFields });

  const deleteMutation = useMutation({
    mutationFn: (fieldId: string) => api.deleteField(fieldId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["fields"] }),
  });

  const createMutation = useMutation({
    mutationFn: () => api.createField(topic.trim()),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["fields"] });
      if (res.status === "ready") {
        setThinResult(null);
        navigate(`/fields/${res.field_id}/landscape`);
      } else {
        setThinResult({ result: res, topic: topic.trim() });
      }
      setTopic("");
    },
  });

  const manifest = manifestQuery.data;
  const headline = manifest
    ? `Instant search over ${manifest.papers_count.toLocaleString()} papers (retrieval & QA, ` +
      `${manifest.from_year}–${manifest.to_year}, updated ${manifest.updated_at.slice(0, 10)}).`
    : "Type a topic and Axiom finds it instantly in the prebuilt corpus.";

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
          <p className="lede">{headline}</p>
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
              {createMutation.isPending ? "Searching…" : "Find this topic"}
            </button>
          </form>
          <p className="muted" style={{ marginTop: "var(--space-4)", marginBottom: 0, fontSize: "var(--small)" }}>
            A slice of the prebuilt corpus, ranked by relevance — seconds, not minutes. No per-topic download or
            extraction.
          </p>
          {createMutation.isError && (
            <p style={{ color: "var(--action)", marginTop: "var(--space-4)" }}>
              Couldn't find this topic: {(createMutation.error as Error).message}
            </p>
          )}
        </div>

        {thinResult && (
          <div style={{ marginBottom: "var(--space-8)" }}>
            <ThinCoverageCard result={thinResult.result} topic={thinResult.topic} />
          </div>
        )}

        <h2>Recent fields</h2>
        {fieldsQuery.isLoading && <LoadingState label="Loading recent fields…" />}
        {fieldsQuery.isError && <ErrorState title="Couldn't load recent fields">Try refreshing the page.</ErrorState>}
        {fieldsQuery.data && fieldsQuery.data.length === 0 && (
          <EmptyState title="No fields yet">Type a topic above — Axiom finds it in the prebuilt corpus instantly.</EmptyState>
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
                {f.id !== "default" && (
                  <button
                    type="button"
                    className="btn-quiet"
                    onClick={() => {
                      if (confirm(`Delete the field "${f.topic}"? This can't be undone.`)) {
                        deleteMutation.mutate(f.id);
                      }
                    }}
                    disabled={deleteMutation.isPending}
                  >
                    Delete
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </>
  );
}
