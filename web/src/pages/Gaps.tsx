import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { GapCard } from "../components/GapCard";
import { EmptyState, ErrorState, LoadingState } from "../components/States";

export default function Gaps() {
  const { fieldId } = useParams<{ fieldId: string }>();
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["gaps", fieldId],
    queryFn: () => api.gaps(fieldId!),
    enabled: !!fieldId,
  });

  if (isLoading) return <LoadingState label="Detecting communities & research gaps…" />;
  if (isError || !data) {
    return (
      <ErrorState title="Couldn't load research gaps" onRetry={() => refetch()}>
        This field's gap analysis didn't load. Try again.
      </ErrorState>
    );
  }

  // gaps.analyze() already ranks by g_score server-side (docs/agentic/ARCHITECTURE.md) --
  // not re-sorted here.
  const ranked = data.gaps;

  return (
    <>
      <div className="page-head">
        <h1>Research gaps</h1>
        <p className="lede">
          The product's citation-backed gap view. Sub-topics (communities) come from the citation graph; a candidate
          research gap is a pair of communities close in meaning yet barely citing each other.
        </p>
      </div>
      {ranked.length === 0 ? (
        <EmptyState title="No candidate gaps found for this field">
          This can happen on a small or very citation-sparse corpus. Try a broader topic, or check back once the
          field has more papers.
        </EmptyState>
      ) : (
        <div className="gap-list">
          {ranked.map((gap) => (
            <GapCard gap={gap} fieldId={fieldId!} key={gap.gap_id} />
          ))}
        </div>
      )}
    </>
  );
}
