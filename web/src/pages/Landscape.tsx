import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { TrendRow } from "../components/TrendRow";
import { PaperList } from "../components/PaperRow";
import { EmptyState, ErrorState, LoadingState } from "../components/States";

export default function Landscape() {
  const { fieldId } = useParams<{ fieldId: string }>();
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["overview", fieldId],
    queryFn: () => api.overview(fieldId!),
    enabled: !!fieldId,
  });

  if (isLoading) return <LoadingState label="Loading the landscape…" />;
  if (isError || !data) {
    return (
      <ErrorState title="Couldn't load the landscape" onRetry={() => refetch()}>
        This field's data didn't load. Try again.
      </ErrorState>
    );
  }
  if (data.papers === 0) {
    return <EmptyState title="This field has no papers yet">Its build may still be running — check back shortly.</EmptyState>;
  }

  const sortedTopics = [...data.topics].sort((a, b) => b.size - a.size);
  const shown = sortedTopics.slice(0, 4);
  const rest = sortedTopics.slice(4);
  const restCount = rest.reduce((sum, t) => sum + t.size, 0);

  return (
    <>
      <div className="page-head">
        <h1>Landscape</h1>
        <p className="lede">An overview of this field's topics, what's accelerating, and the papers other work in this corpus leans on most.</p>
      </div>

      <div className="stat-row">
        <div className="stat-tile">
          <div className="num">{data.papers.toLocaleString()}</div>
          <div className="label">Papers</div>
        </div>
        <div className="stat-tile">
          <div className="num">{data.in_corpus_citations.toLocaleString()}</div>
          <div className="label">In-corpus citations</div>
        </div>
        <div className="stat-tile">
          <div className="num">{data.topic_communities}</div>
          <div className="label">Topic communities</div>
        </div>
        <div className="stat-tile">
          <div className="num">{data.candidate_gaps}</div>
          <div className="label">Candidate research gaps</div>
        </div>
      </div>

      <div className="section" style={{ marginBottom: "var(--space-12)" }}>
        <h2>Topics</h2>
        <p className="muted">
          Ranked by size. These are the same communities the research gaps are built from — there's one clustering
          everywhere, not a separate map per screen.
        </p>
        <div className="topic-grid">
          {shown.map((t) => (
            <div className="topic-tile" key={t.cid} style={{ flexGrow: t.size }}>
              <div className="count">{t.size} papers</div>
              <div className="label">{t.labels[0] ?? `Topic ${t.cid}`}</div>
              <div className="tags">{t.labels.join(", ")}</div>
            </div>
          ))}
          {rest.length > 0 && (
            <div className="topic-tile" style={{ opacity: 0.7, flexGrow: restCount }}>
              <div className="count">{restCount} papers</div>
              <div className="label">{rest.length} more topic communities</div>
              <div className="tags">Not itemized in this view.</div>
            </div>
          )}
        </div>
      </div>

      <div className="section" style={{ marginBottom: "var(--space-12)" }}>
        <h2>Rising concepts</h2>
        <p className="muted">Comparing recent vs. prior mentions. Shown as a ratio, not a raw percent — a concept with zero prior mentions still has somewhere to rise from.</p>
        {data.rising.length === 0 ? (
          <EmptyState title="No rising concepts met the minimum sample size">Check back once this field has more papers.</EmptyState>
        ) : (
          <div className="paper-list">
            {data.rising.map((k) => (
              <TrendRow k={k} key={k.concept} />
            ))}
          </div>
        )}
      </div>

      <div className="section">
        <h2>Key papers</h2>
        <p className="muted">Ranked by influence within this corpus's citation graph (PageRank), not raw citation count alone.</p>
        {data.key_papers.length === 0 ? (
          <EmptyState title="No key papers yet">This field's citation graph may still be building.</EmptyState>
        ) : (
          <PaperList papers={data.key_papers} showPagerank />
        )}
      </div>
    </>
  );
}
