import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type SideDetailOut } from "../api";
import { BridgeDiagram } from "../components/BridgeDiagram";
import { PaperList } from "../components/PaperRow";
import { Badge } from "../components/Badge";
import { ErrorState, LoadingState } from "../components/States";

function headline(a: SideDetailOut, b: SideDetailOut): string {
  return `${a.size}-paper ${a.labels[0] ?? "cluster"} cluster barely cites the ${b.size}-paper ${b.labels[0] ?? "cluster"} cluster right next to it`;
}

function SideCard({ side, label }: { side: SideDetailOut; label: string }) {
  return (
    <div className="card" style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)" }}>
      <h3 style={{ marginTop: 0 }}>{label}</h3>
      <p className="muted" style={{ marginTop: 0 }}>{side.size} papers</p>
      <div className="badge-row" style={{ marginBottom: "var(--space-2)" }}>
        {side.labels.map((l) => (
          <span className="badge" key={l}>{l}</span>
        ))}
      </div>
      <p className="muted" style={{ fontSize: "var(--small)" }}>
        These are raw OpenAlex concept tags — a rough description of the cluster, not the pipeline's own extraction.
      </p>
      {side.top_papers.length > 0 && (
        <>
          <h4>Evidence papers</h4>
          <PaperList papers={side.top_papers} />
        </>
      )}
      {(side.methods.length > 0 || side.datasets.length > 0) && (
        <>
          <h4>Methods &amp; datasets extracted from these papers</h4>
          <div className="badge-row">
            {side.methods.map((m) => (
              <span className="badge" key={`m-${m}`}>method: {m}</span>
            ))}
            {side.datasets.map((d) => (
              <span className="badge" key={`d-${d}`}>dataset: {d}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function GapDetail() {
  const { fieldId, gapIndex } = useParams<{ fieldId: string; gapIndex: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [crossCheckOpen, setCrossCheckOpen] = useState(false);
  const index = Number(gapIndex);

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["gap-detail", fieldId, index],
    queryFn: () => api.gapDetail(fieldId!, index),
    enabled: !!fieldId && Number.isFinite(index),
  });

  const draftMutation = useMutation({
    mutationFn: () => api.draftDirection(fieldId!, index),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["shortlist", fieldId] });
      navigate(`/fields/${fieldId}/shortlist`);
    },
  });

  if (isLoading) return <LoadingState label="Loading this gap…" />;
  if (isError || !data) {
    return (
      <ErrorState title="Couldn't load this gap" onRetry={() => refetch()}>
        This gap's detail didn't load. Try again, or go back to the research gaps list.
      </ErrorState>
    );
  }

  return (
    <>
      <p>
        <a href={`/fields/${fieldId}/gaps`}>← Back to research gaps</a>
      </p>
      <h1>{headline(data.a, data.b)}</h1>
      <p className="lede" style={{ color: "var(--text-muted)", maxWidth: "70ch" }}>
        {data.a.size} papers centered on {data.a.labels[0] ?? "this cluster"} sit at{" "}
        {(data.semantic_similarity * 100).toFixed(1)}% semantic similarity to a {data.b.size}-paper{" "}
        {data.b.labels[0] ?? "cluster"}, but only {data.inter_citations} citations cross between them.
      </p>

      <Badge
        label="Uncalibrated score"
        note={`G-score ${data.g_score.toFixed(3)} blends similarity, disconnection, concept velocity and authority with default weights not yet calibrated against labeled gap data — read it as a ranking signal, not a verified measurement.`}
      />

      <div style={{ margin: "var(--space-6) 0" }}>
        <BridgeDiagram
          aLabel={data.a.labels[0] ?? "Side A"}
          aSize={data.a.size}
          bLabel={data.b.labels[0] ?? "Side B"}
          bSize={data.b.size}
          similarityPct={data.semantic_similarity * 100}
          interCitations={data.inter_citations}
          totalCitations={data.total_in_corpus_citations}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "var(--space-6)", marginBottom: "var(--space-6)" }}>
        <SideCard side={data.a} label="Side A" />
        <SideCard side={data.b} label="Side B" />
      </div>

      <div className="card" style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-4) var(--space-6)", marginBottom: "var(--space-6)" }}>
        <button
          type="button"
          className="btn-quiet"
          style={{ fontWeight: 700, cursor: "pointer" }}
          onClick={() => setCrossCheckOpen((v) => !v)}
          aria-expanded={crossCheckOpen}
        >
          {crossCheckOpen ? "−" : "+"} Cross-check with embedding geometry
        </button>
        {crossCheckOpen && (
          <p className="muted" style={{ marginTop: "var(--space-4)", marginBottom: 0 }}>
            An independent, citation-free read on the same two clusters: {(data.semantic_similarity * 100).toFixed(1)}%
            centroid similarity in embedding space alone, with no citation graph involved. Useful as a second opinion,
            especially on corpora too citation-sparse for the method above — but on its own it has no path to a
            direction pitch. (A deeper geometry breakdown — bridge ratio, nearest-neighbor papers — isn't computed for
            this view; see "Under the hood" for the full Vector Gap Discovery analysis.)
          </p>
        )}
      </div>

      <div className="card" style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-6)" }}>
        <h3 style={{ marginTop: 0 }}>Draft direction</h3>
        <p className="muted">
          Axiom can draft a short, citeable pitch that names both sides of this gap and proposes a direction
          connecting them. It's generated, not verified — you'll review and edit it before it goes on your shortlist.
        </p>
        <button className="btn btn-primary" onClick={() => draftMutation.mutate()} disabled={draftMutation.isPending} type="button">
          {draftMutation.isPending ? "Drafting…" : "Draft direction"}
        </button>
        {draftMutation.isError && (
          <p style={{ color: "var(--action)", marginTop: "var(--space-4)" }}>
            Couldn't draft a direction: {(draftMutation.error as Error).message}
          </p>
        )}
      </div>
    </>
  );
}
